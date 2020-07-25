# Imports
import os
import imageio
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchvision import datasets, transforms

from skimage.color import lab2rgb
from skimage.io import imread
from skimage.color import rgb2lab, rgb2gray
from skimage import img_as_ubyte

import cv2
import numpy as np
from PIL import Image
import scipy.misc

from model import Color_model
from training_layers import decode

scale_transform = transforms.Compose([
    transforms.Resize((224,224),2),
    #transforms.RandomCrop(224),
])

def load_image(image_path,transform=None):
    # Load the test image
    rgb_image = Image.open(image_path)
    
    # Apply the transforms that compose the 'scale_transform' 
    if transform is not None:
        rgb_image_resized = transform(rgb_image)

    rgb_image_ori_size = rgb_image
    rgb_image_ori_size = np.asarray(rgb_image_ori_size)
    ori_w, ori_h, num_channels = rgb_image_ori_size.shape # Get the original image shape and number of channels
    ori_size = (ori_w, ori_h) # Store the shape of the original image 
    
    rgb_image_resized = np.asarray(rgb_image_resized)
    lab_image_resized = rgb2lab(rgb_image_resized) # Convert the image colorspace from RGB to CIE Lab
    lab_image_resized = lab_image_resized.transpose(2,0,1)    
    img_l_resized = lab_image_resized[0,:,:] # L channel
    img_l_resized = (np.round(img_l_resized)).astype(np.int32)
    img_l_resized = img_l_resized - 50
    img_l_resized = torch.from_numpy(img_l_resized).unsqueeze(0) # img_l_resized -> L channel image with shape (224x224) that will be the model input
    
    img_rgb_original = np.array(rgb_image)
    img_lab_original = rgb2lab(img_rgb_original) # Convert the image colorspace from RGB to CIE Lab
    img_lab_original = img_lab_original.transpose(2, 0, 1)
    img_l_original = (np.round(img_lab_original[0,:,:])).astype(np.int32) # L channel
    img_l_original = torch.from_numpy(img_l_original) # img_l_original -> This image will be combined to the predicted (a,b) channels. Both 'img_l_original' and predicted (a,b) channels are in original image shape

    return img_l_resized, img_l_original, ori_size

def main(args):

    data_dir = args.test_images_dir
    dirs = os.listdir(data_dir)
    print(dirs)

    # Model instance whose architecture was configured in the 'model.py' file
    color_model = Color_model().cuda().eval()
    # Loading a pretrained model weights to the model instance archtecture
    color_model.load_state_dict(torch.load(args.model_file))

    tensor_to_PIL = transforms.ToPILImage() # Create a transform that convert a tensor to a PIL Object

    # Annealed Mean
    T = 0.38 # Temperature
    soft = nn.Softmax(dim=1) # Instance of softmax function
    Q_bins = np.load(args.Q_bins_file) # 313 bins of quantized colors
    #print(Q_bins.shape)
    #print(Q_bins) 
    
    for file in dirs:
        
        # The 'load_image()' function loads the test dataset images to be submitted to the trained model.
        # This function returns the following data:
        # - img_l_resized -> L channel image with shape (224x224) that will be the model input
        # - img_l_original -> This image will be combined to the predicted (a,b) channels. Both 'img_l_original' and predicted (a,b) channels are in original image shape
        # - ori_size -> Shape of original image that will be used to resized the final colored image.
        img_l_resized, img_l_original, ori_size = load_image(data_dir+'/'+file, scale_transform)

        img_l_resized = img_l_resized.unsqueeze(0).float().cuda()
        
        # Submitting L channel image with shape (224x224) as input to the model. The model output is a tensor with shape (313x56x56)
        img_ab_313 = color_model(img_l_resized)
        #img_ab_313 = abs(img_ab_313)
        img_ab_313 = img_ab_313.cpu()

        # Resize the tensor with shape (313x56x56) to the original image shape
        img_ab_313 = F.upsample(img_ab_313, size = ori_size, mode = 'nearest') 
        
        # The Annealed Mean is applied to the output tensor with shape (313x[Original Width]x[Original Height])
        img_ab_313_log_t = (torch.log10(img_ab_313+0.001))/T
        
        # Apply softmax function over the 313 layers of the tensor. This will generate a probability distribution across the 313 layers for each pixel
        soft_image_log_t = soft(img_ab_313_log_t)
        # print(soft_image_log_t.shape)
        # print(soft_image_log_t)
        # print(soft_image_log_t.max())
        # print(soft_image_log_t.min())
        # print(torch.sum(soft_image_log_t,dim=1))
        
        # - bins_matrix_values -> Matrix containing the highest probability value of the distribution for each pixel
        # - bins_matrix_indexes -> Matrix containing the index of the probability distribution over the 313 layers that has the highest value of the probability distribution
        bins_matrix_values, bins_matrix_indexes = torch.max(soft_image_log_t, dim=1, keepdim=True )
        # print(bins_matrix_values.shape)
        # print(bins_matrix_values.min())
        # print(bins_matrix_values.max())
        # print(bins_matrix_indexes.shape)
        # print(bins_matrix_indexes.min())
        # print(bins_matrix_indexes.max())


        bins_matrix_indexes_flatten = torch.flatten(bins_matrix_indexes)
        #print(set(bins_matrix_indexes_flatten))
        
        # Use the indexes of highest probabilities values of the distribution for each pixel to select the bins(a and b values) of the 313 quantized colors stored in 'Q_bins' 
        predicted_ab_channels = Q_bins[bins_matrix_indexes[:,:]]
        predicted_ab_channels = predicted_ab_channels.squeeze(0).squeeze(0).transpose(2,0,1)

        predicted_ab_channels = predicted_ab_channels.transpose(1, 2, 0)
        img_l_original = img_l_original.numpy()
        # print(predicted_ab_channels.shape)
        # print(predicted_ab_channels)

        # Create the final image matrix structure to receive the L channel and the (a,b) predicted channels.
        # The shape of this final image is ([Original Width]x[Original Height]x3) and this structure is filled initialy with zeros.
        img_lab_final = np.zeros((ori_size[0],ori_size[1], 3)) 

        # Assign the L channel image values to the first channel of the 'img_lab_final' image and assign the (a,b) predicted channels to the second and third channels of the 'img_lab_final' image
        img_lab_final[:,:,0] = img_l_original
        img_lab_final[:,:,1:] = predicted_ab_channels

        #print(img_lab_final.transpose(2,0,1))
        
        # Converting the final colorized imagem colorspace from CIE Lab to RGB
        img_rgb_final = lab2rgb(img_lab_final)

        # Saving the colorized image
        #imageio.imwrite(args.output_images_dir + file, img_as_ubyte(abs(img_rgb_final)))
        imageio.imwrite(args.output_images_dir + file, img_rgb_final)                  

if __name__ == '__main__':

    # Set all configurations and directories here in this section
    parser = argparse.ArgumentParser()

    # Files and directories parameters
    parser.add_argument('--test_images_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/dataset/test/images', help = 'Directory of test dataset images')
    parser.add_argument('--output_images_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/output/', help = 'Directory where the output images will be saved')

    parser.add_argument('--model_file', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/model-1000-360.ckpt', help = 'Specific trained model to be loaded')

    parser.add_argument('--Q_bins_file', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/code/resources/pts_in_hull.npy', help = 'pts_in_hull.npy file with 313 quantized color to be loaded')
   
    args = parser.parse_args()
    print(args)
    main(args)
    
