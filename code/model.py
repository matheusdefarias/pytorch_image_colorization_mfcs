# Imports
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class Color_model(nn.Module):
    def __init__(self):
        super(Color_model, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 64),
        )
        
        self.conv2 = nn.Sequential(
            # conv2
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 128),
        )
        
        self.conv3 = nn.Sequential(
            # conv3
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 256),
        )

        self.conv4 = nn.Sequential(
            # conv4
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv5 = nn.Sequential(
            # conv5
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv6 = nn.Sequential(
            # conv6
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv7 = nn.Sequential(
            # conv7
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features = 256),
        )

        self.conv8 = nn.Sequential(
            # conv8
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias=True),
            nn.ReLU(True),
        )

        self.conv8_313 = nn.Sequential(
            # conv8_313
            nn.Conv2d(in_channels = 128, out_channels = 313, kernel_size = 1, stride = 1, dilation = 1, bias=True),
            nn.ReLU(True)
        )

        #self.conv8_313 = nn.Conv2d(in_channels = 128, out_channels = 313, kernel_size = 1, stride = 1, dilation = 1)
        
        self.apply(weights_init)

    def forward(self, gray_image):
        conv1=self.conv1(gray_image)
        conv2=self.conv2(conv1)
        conv3=self.conv3(conv2)
        conv4=self.conv4(conv3)
        conv5=self.conv5(conv4)
        conv6=self.conv6(conv5)
        conv7=self.conv7(conv6)
        conv8=self.conv8(conv7)
        features = self.conv8_313(conv8)

        return features

        #print("conv1",conv1.shape)
        #print("conv2",conv2.shape)
        #print("conv3",conv3.shape)
        #print("conv4",conv4.shape)
        #print("conv5",conv5.shape)
        #print("conv6",conv6.shape)
        #print("conv7",conv7.shape)
        #print("conv8",conv8.shape)
        #print("features",features.shape)
        #print(features)       

        # conv1 torch.Size([1, 64, 112, 112])
        # conv2 torch.Size([1, 128, 56, 56])
        # conv3 torch.Size([1, 256, 28, 28])
        # conv4 torch.Size([1, 512, 28, 28])
        # conv5 torch.Size([1, 512, 28, 28])
        # conv6 torch.Size([1, 512, 28, 28])
        # conv7 torch.Size([1, 256, 28, 28])
        # conv8 torch.Size([1, 128, 56, 56])
        # features torch.Size([1, 313, 56, 56])
