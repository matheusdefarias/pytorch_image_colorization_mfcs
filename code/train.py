# Imports
import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer
from data_loader import CustomDataset
from model import Color_model

original_transform = transforms.Compose([
    transforms.Resize((224,224),2),
    #transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

def main(args):
    
    # Instance of the class that will preprocess and generate proper images for training
    train_set = CustomDataset(args.image_dir, original_transform)

    # Data loader that will generate the proper batches of images for training
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    # Model instance whose architecture was configured in the 'model.py' file
    model = Color_model().cuda()
    # model.load_state_dict(torch.load(args.load_model))
    
    # Loss function used
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Model parameters and optimizer used during the training step
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)

    # Instance of 'NNEncLayer' class that is responsable for to return a probability distribution for each pixel of the (a,b) channels. This class is in 'training_layers.py' file 
    encode_ab_layer = NNEncLayer()   

    #####################################################################
    #----------------------->> TRAINING STEP <<-------------------------#
    #####################################################################
    print("====>> Training step started!! <<====")

    # Number of batches
    total_batch = len(data_loader)

    # Store the loss of every epoch for training dataset.
    running_loss_history = []

    # Start time to measure time of training
    start_time = time.time()
    
    # Main loop, loop for each epoch
    for epoch in range(args.num_epochs):        

        # Every loss per batch is summed to get the final loss for each epoch for training dataset.
        running_loss = 0.0    

        # Loop for each batch of images
        for i, (images, img_ab, filename) in enumerate(data_loader):
            #print(filename)
            
            # Grayscale images represented by L channel
            images = images.unsqueeze(1).float().cuda() # Unsqueeze(1) add one more dimension to the tensor in position 1, than converted to float and loaded to the GPU
            # Ground truth represented by (a,b) channels
            img_ab = img_ab.float() 

            # 'encode_ab' -> represents a probability distribution for each pixel of the (a,b) channels
            # 'max_encode_ab' -> represents the indexes that have the highest values of probability along each pixel layers
            encode_ab, max_encode_ab = encode_ab_layer.forward(img_ab)
            encode_ab = torch.from_numpy(encode_ab).long().cuda()

            #with open('encode_ab.txt', 'w') as file:
            #    file.write(str(encode_ab))

            # 'max_encode_ab' is used as targets. So it is converted to long data type and then loaded to the GPU
            targets = torch.Tensor(max_encode_ab).long().cuda()
            # The input grayscale images are submitted to the model and the result tensor with shape [Bx313xWxH] is stored in 'output'
            outputs = model(images)

            #print(encode_ab.shape)
            #print(outputs.shape)
            #output=outputs[0].cpu().data.numpy()
            #out_max=np.argmax(output,axis=0)

            #print('set',set(out_max.flatten()))

            #loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()

            # The loss is performed for each batch(image)
            loss = criterion(outputs,targets)

            # Every loss per batch is summed to get the final loss for each epoch. 
            running_loss += loss.item() 
            
            #multi=loss*boost_nongray.squeeze(1)

            model.zero_grad()            
            loss.backward()
            optimizer.step()

            # Print info about the training according to the log_step value
            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Batch [{}/{}]'.format(epoch+1, args.num_epochs, i+1, total_batch))

            # Save the model according to the checkpoints configured
            if epoch in args.checkpoint_step and i == (args.trainDataset_length/args.batch_size)-1:
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            
        # Average Loss of an epoch for training dataset.
        epoch_loss = running_loss/len(data_loader) # Acumulated Loss divided by number of images batches on training dataset.
        running_loss_history.append(epoch_loss)

        print('--------->>> Epoch [{}/{}], Epoch Loss: {:.4f}'.format(epoch, args.num_epochs, epoch_loss))
    
    print('Loss History: {}'.format(running_loss_history))
    print("{:.2f} minutes".format((time.time() - start_time)/60))
    print("                                                    ")

    plt.plot(np.arange(0,args.num_epochs), running_loss_history, label='Training Loss')
    
    ax = plt.gca()
    ax.set_facecolor((0.85, 0.85, 0.85))
    plt.grid(color='w', linestyle='solid')
    ax.set_axisbelow(True)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.save_lossCurve)                   
            

if __name__ == '__main__':

    # Set all configurations and directories here in this section
    parser = argparse.ArgumentParser()

    # Files and directories parameters
    parser.add_argument('--image_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/dataset/train/images', help = 'Directory of train dataset images')
    parser.add_argument('--trainDataset_length', type = int, default = 10000, help = 'Number of images in train dataset')
    parser.add_argument('--model_path', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models', help = 'Path where partial and final trained models will be saved')
    parser.add_argument('--load_model', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/model-1-25.ckpt', help = 'Specific trained model to be loaded')
    parser.add_argument('--save_lossCurve', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/loss_curve.jpg', help = 'Path where the loss curve image will be saved')
    
    parser.add_argument('--log_step', type = int, default = 100, help = 'Step size for printing info about the training progress')
    parser.add_argument('--checkpoint_step', type = list, default = [24, 49, 74, 99], help = 'Checkpoints for saving partial and final trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 100, help ='Number of epochs')
    parser.add_argument('--batch_size', type = int, default = 25, help ='Number of images in each batch')
    parser.add_argument('--learning_rate', type = float, default = 1e-3, help ='Learning rate, the step size at each iteration while moving toward a minimum of a loss function')
    parser.add_argument('--num_workers', type = int, default = 8, help ='Number of cores working')
    
    args = parser.parse_args()
    print(args)
    print("                                                    ")
    main(args)

    #100 [24, 49, 74, 99]
    #300 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299]
    #350 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349]
    #400 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399]
    #500 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399]
    #800 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399, 424, 449, 474, 499, 524, 549, 574, 599, 624, 649, 674, 699, 724, 749, 774, 799]
    #1000 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399, 424, 449, 474, 499, 524, 549, 574, 599, 624, 649, 674, 699, 724, 749, 774, 799, 824, 849, 874, 899, 924, 949, 974, 999]    
