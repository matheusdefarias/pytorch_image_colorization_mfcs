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
    rgb_image = Image.open(image_path)
    
    if transform is not None:
        rgb_image_resized = transform(rgb_image)

    rgb_image_ori_size = rgb_image
    rgb_image_ori_size = np.asarray(rgb_image_ori_size)
    ori_w, ori_h, num_channels = rgb_image_ori_size.shape
    ori_size = (ori_w, ori_h)
    
    rgb_image_resized = np.asarray(rgb_image_resized)
    lab_image_resized = rgb2lab(rgb_image_resized)
    lab_image_resized = lab_image_resized.transpose(2,0,1)    
    img_l_resized = lab_image_resized[0,:,:]
    img_l_resized = (np.round(img_l_resized)).astype(np.int) # L channel
    img_l_resized = img_l_resized - 50
    img_l_resized = torch.from_numpy(img_l_resized).unsqueeze(0)
    
    img_l_56 = transforms.Resize(ori_size,2)(rgb_image) 
    img_l_56 = np.array(img_l_56)
    img_l_56 = rgb2lab(img_l_56)
    img_l_56 = img_l_56.transpose(2, 0, 1)
    img_l_56 = (np.round(img_l_56[0,:,:])).astype(np.int)
    img_l_56 = torch.from_numpy(img_l_56)
        
    return img_l_resized, img_l_56, ori_size

def main(args):

    data_dir = args.test_images_dir
    dirs=os.listdir(data_dir)
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
    
        img_l_resized, img_l_56, ori_size = load_image(data_dir+'/'+file, scale_transform)

        img_l_resized = img_l_resized.unsqueeze(0).float().cuda()
        
        img_ab_313 = color_model(img_l_resized)
        #img_ab_313 = abs(img_ab_313)
        img_ab_313 = img_ab_313.cpu()

        img_ab_313 = F.upsample(img_ab_313, size = ori_size, mode = 'nearest') 
        
        img_ab_313_log_t = (torch.log10(img_ab_313+0.001))/T
        
        soft_image_log_t = soft(img_ab_313_log_t)
        #print(soft_image_log_t.max())
        #print(torch.sum(soft_image_log_t,dim=1))
        
        bins_matrix_values, bins_matrix_indexes = torch.max(soft_image_log_t, dim=1, keepdim=True )
        # print(bins_matrix_values.shape)
        # print(bins_matrix_values.min())
        # print(bins_matrix_values.max())
        # print(bins_matrix_indexes.shape)
        # print(bins_matrix_indexes.min())
        # print(bins_matrix_indexes.max())

        bins_matrix_indexes_flatten = torch.flatten(bins_matrix_indexes)
        #print(set(bins_matrix_indexes_flatten))
        
        predicted_ab_channels = Q_bins[bins_matrix_indexes[:,:]]
        predicted_ab_channels = predicted_ab_channels.squeeze(0).squeeze(0).transpose(2,0,1)

        predicted_ab_channels = predicted_ab_channels.transpose(1, 2, 0)
        img_l_56 = img_l_56.numpy()
        # print(predicted_ab_channels.shape)
        # print(predicted_ab_channels)

        img_lab_final = np.zeros((ori_size[0],ori_size[1], 3)) 

        img_lab_final[:,:,0] = img_l_56
        img_lab_final[:,:,1:] = predicted_ab_channels

        #print(img_lab_final.transpose(2,0,1))
        
        img_rgb_final = lab2rgb(img_lab_final)

        imageio.imwrite(args.output_images_dir + file, img_rgb_final*255)                

if __name__ == '__main__':

    # Set all configurations and directories here in this section
    parser = argparse.ArgumentParser()

    # Files and directories parameters
    parser.add_argument('--test_images_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/dataset/test/images', help = 'Directory of test dataset images')
    parser.add_argument('--output_images_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/output/', help = 'Directory where the output images will be saved')

    parser.add_argument('--model_file', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/model-1000-25.ckpt', help = 'Specific trained model to be loaded')

    parser.add_argument('--Q_bins_file', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/code/resources/pts_in_hull.npy', help = 'pts_in_hull.npy file with 313 quantized color to be loaded')
   
    args = parser.parse_args()
    print(args)
    main(args)
    
