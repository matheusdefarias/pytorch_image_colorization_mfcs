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

scale_transform = transforms.Compose([
    transforms.Resize((224,224),2),
    #transforms.RandomCrop(224),
    #transforms.ToTensor()
])

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
        img = Image.open(self.root_dir+'/'+self.file_list[idx]).convert('RGB') # Read the image
        
        if self.transform is not None:
            img_rgb_resized = transforms.Resize((56,56),2)(img)
            img_rgb_ori = np.array(img)
            
            img_lab_ori = rgb2lab(img_rgb_ori) # Convert to CIE Lab color space
            img_lab_resized = rgb2lab(img_rgb_resized) # Convert to CIE Lab color space
            
            img_rgb_transposed = img_rgb_ori.transpose(2, 0, 1) #(C, W, H)    
            img_lab_transposed = img_lab_ori.transpose(2, 0, 1) #(C, W, H)
            img_lab_resized_transposed = img_lab_resized.transpose(2, 0, 1) #(C, W, H)
            
            img_l = (np.round(img_lab_transposed[0,:,:])).astype(np.int32) # L channel
            img_l = self.tensor_to_PIL(img_l) # Convert to PIL object image to be possible apply the following transform
            img_l_resized = self.transform(img_l)
            img_l_resized = np.array(img_l_resized) # Convert to numpy array
            img_l_resized = img_l_resized - 50 # Centering the L channel values in zero
            img_l_resized = torch.from_numpy(img_l_resized) # Convert to torch tensor
            
            img_ab_resized = (np.round(img_lab_resized_transposed[1:3, :, :])).astype(np.int) # (a,b) channels with int intensity values            
            img_ab_resized = np.array(img_ab_resized) # Convert to numpy array            
            img_ab_resized = torch.from_numpy(img_ab_resized) # Convert to torch tensor

            filename = self.root_dir+'/'+self.file_list[idx]
            
            return img_l_resized, img_ab_resized, filename # img_l_resized -> 1x224x224 and img_ab_resized -> 2x56x56
        
class TrainImageFolder(data.Dataset):
    def __init__(self, data_dir, transform):
       self.file_list=os.listdir(data_dir)
       self.transform=transform
       self.data_dir=data_dir
    def __getitem__(self, index):
        try:
            img=Image.open(self.data_dir+'/'+self.file_list[index])
            if self.transform is not None:
                img_original = self.transform(img)
                img_resize=transforms.Resize(56)(img_original)
                img_original = np.asarray(img_original)
                img_lab = rgb2lab(img_resize)
                #img_lab = (img_lab + 128) / 255
                img_ab = img_lab[:, :, 1:3]
                img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                #print('img_ori',img_original.shape)
                #print('img_ab',img_ab.size())
                img_original = rgb2lab(img_original)[:,:,0]-50.
                img_original = torch.from_numpy(img_original)
                return img_original, img_ab
        except:
            pass
    def __len__(self):
        return len(self.file_list)


class ValImageFolder(data.Dataset):
    def __init__(self,data_dir):
        self.file_list=os.listdir(data_dir)
        self.data_dir=data_dir

    def __getitem__(self, index):
        img=Image.open(self.data_dir+'/'+self.file_list[index])
        img_scale = scale_transform(img)
        img_scale = np.asarray(img_scale)
        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        return img_scale

    def __len__(self):
        return len(self.file_list)

class CustomDataset2(Dataset):
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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.root_dir+'/'+self.file_list[idx])
        
        if self.transform is not None:
            img_original = self.transform(img)
            img_ori_resized = transforms.Resize((64,64),2)(img_original)

            img_ori_lab = rgb2lab(img_original)
            img_ori_lab = img_ori_lab.transpose(2, 0, 1)            
            img_ori_lab = np.asarray(img_ori_lab)
            img_l = img_ori_lab[0,:,:]
            img_l = np.asarray(img_l, dtype=np.float32)
            img_l = torch.from_numpy(img_l)
            
            img_resized_lab = rgb2lab(img_ori_resized)
            img_resized_lab = img_resized_lab.transpose(2, 0, 1)
            img_resized_lab = np.asarray(img_resized_lab)
            img_ab = img_resized_lab[1:3, :, :]
            img_ab = np.asarray(img_ab, dtype=np.float32)
            img_ab = torch.from_numpy(img_ab)
            
            return img_l, img_ab
