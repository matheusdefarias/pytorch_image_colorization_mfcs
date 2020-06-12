# Imports
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.utils.data as data

from skimage.color import rgb2lab, rgb2gray
from skimage import io

class CustomDataset(Dataset):
    """Custom Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list=os.listdir(root_dir)
        self.tensor_to_PIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.root_dir+'/'+self.file_list[idx]).convert('RGB') # Reads the image and converts it to RGB just for security
        
        if self.transform is not None:
            img_rgb_resized = transforms.Resize((56,56),2)(img)
            img_rgb_ori = np.array(img)
            
            img_lab_ori = rgb2lab(img_rgb_ori) # Convert to CIE Lab color space
            img_lab_resized = rgb2lab(img_rgb_resized) # Convert to CIE Lab color space
            
            img_rgb_transposed = img_rgb_ori.transpose(2, 0, 1) #(C, W, H) - Original Image in RGB Color Space   
            img_lab_transposed = img_lab_ori.transpose(2, 0, 1) #(C, W, H) - Original Image in CIE Lab Color Space
            img_lab_resized_transposed = img_lab_resized.transpose(2, 0, 1) #(C, W, H) - Resized Image in CIE Lab Color Space
            
            img_l = (np.round(img_lab_transposed[0,:,:])).astype(np.int32) # L channel
            img_l = self.tensor_to_PIL(img_l) # Convert to PIL object image to be possible apply the following transform
            img_l_resized = self.transform(img_l)
            img_l_resized = np.array(img_l_resized) # Convert to numpy array
            img_l_resized = img_l_resized - 50 # Centering the L channel values in zero
            img_l_resized = torch.from_numpy(img_l_resized) # Convert to torch tensor
            
            img_ab_resized = (np.round(img_lab_resized_transposed[1:3, :, :])).astype(np.int32) # (a,b) channels with int intensity values            
            img_ab_resized = np.array(img_ab_resized) # Convert to numpy array            
            img_ab_resized = torch.from_numpy(img_ab_resized) # Convert to torch tensor

            filename = self.root_dir+'/'+self.file_list[idx]
            
            return img_l_resized, img_ab_resized, filename # img_l_resized -> 1x224x224 and img_ab_resized -> 2x56x56