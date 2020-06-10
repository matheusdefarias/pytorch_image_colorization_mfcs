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
from data_loader import TrainImageFolder, CustomDataset
from model import Color_model

#np.set_printoptions(threshold=np.inf)

original_transform = transforms.Compose([
    transforms.Resize((224,224),2),
    #transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

def main(args):
    
    train_set = CustomDataset(args.image_dir, original_transform)

    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    model = Color_model().cuda()
    # model.load_state_dict(torch.load(args.load_model))
    criterion = nn.CrossEntropyLoss().cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)

    encode_ab_layer = NNEncLayer()   

    #######################
    ### Train the model ###
    #######################

    total_batch = len(data_loader)

    # Analyze the losses an every epoch for training dataset.
    running_loss_history = []

    # Measure time of training
    start_time = time.time()
    
    for epoch in range(args.num_epochs):        

        # Every loss per batch is summed to get the final loss for each epoch for training dataset.
        running_loss = 0.0    

        for i, (images, img_ab, filename) in enumerate(data_loader):
            #print(filename)
            images = images.unsqueeze(1).float().cuda()
            img_ab = img_ab.float()
                        
            encode_ab, max_encode_ab = encode_ab_layer.forward(img_ab)
            encode_ab = torch.from_numpy(encode_ab).long().cuda()

            #with open('encode_ab.txt', 'w') as file:
            #    file.write(str(encode_ab))

            targets=torch.Tensor(max_encode_ab).long().cuda()
            outputs = model(images)
            #print(encode_ab.shape)
            #print(outputs.shape)
            #output=outputs[0].cpu().data.numpy()
            #out_max=np.argmax(output,axis=0)

            #print('set',set(out_max.flatten()))

            #loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
            loss = criterion(outputs,targets)
            # Every loss per batch is summed to get the final loss for each epoch. 
            running_loss += loss.item() 
            
            #multi=loss*boost_nongray.squeeze(1)

            model.zero_grad()
            
            loss.backward()
            optimizer.step()

            # Print every batch
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Batch [{}/{}]'.format(epoch, args.num_epochs, i, total_batch))

            # Save the model checkpoints
            if epoch in args.checkpoint_step and i == args.batch_size - 1:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/dataset/train/images', help = 'directory for resized images')
    parser.add_argument('--model_path', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models', help = 'path for saving trained models')
    parser.add_argument('--load_model', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/model-1-25.ckpt', help = 'model to be loaded')
    parser.add_argument('--save_lossCurve', type = str, default = '/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/models/loss_x_epochs.jpg', help = 'path to save loss_x_epochs.jpg image')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    parser.add_argument('--log_step', type = int, default = 50, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 5, help = 'step size for saving trained models')
    
    
    parser.add_argument('--checkpoint_step', type = list, default = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399, 424, 449, 474, 499, 524, 549, 574, 599, 624, 649, 674, 699, 724, 749, 774, 799, 824, 849, 874, 899, 924, 949, 974, 999], help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 25)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    args = parser.parse_args()
    print(args)
    print("                                                    ")
    main(args)

    #100 [24, 49, 74, 99]
    #300 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299]
    #500 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399]
    #800 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399, 424, 449, 474, 499, 524, 549, 574, 599, 624, 649, 674, 699, 724, 749, 774, 799]
    #1000 [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299, 324, 349, 374, 399, 424, 449, 474, 499, 524, 549, 574, 599, 624, 649, 674, 699, 724, 749, 774, 799, 824, 849, 874, 899, 924, 949, 974, 999]    
