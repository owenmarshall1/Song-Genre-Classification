import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#CONV Neural data 
class ImageConv2d(nn.Module):
    def __init__(self, num_classes, input_channels=3 ,out_channels=64, kernel_size=2, pool_size=3, activation=None, img_size=(128, 128)):
        super().__init__()
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels= input_channels, 
            out_channels= out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels= out_channels , 
            out_channels= out_channels*2, 
            kernel_size=kernel_size, 
            stride=1, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
            in_channels= out_channels*2 , 
            out_channels= out_channels*4, 
            kernel_size=kernel_size, 
            stride=1, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(0.2)

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d( in_channels= out_channels*4, 
            out_channels= out_channels*8, 
            kernel_size=kernel_size, 
            stride=1, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(0.4)
        )
        final_size = self._get_conv_output(img_size, input_channels)
        self.fc = nn.Sequential(
            nn.Linear(final_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _get_conv_output(self, img_size, input_channels):

            dummy_input = torch.rand(1, input_channels, img_size[0], img_size[1])
            output = self.conv1(dummy_input)
            output = self.conv2(output)
            output = self.conv3(output)
            n_size = output.view(output.size(0), -1).size(1)
            
            print(f"Calculated Final Size (H*W*C): {n_size}")
            return n_size
    def forward(self , X): 
        X= self.conv1(X)
        X= self.conv2(X)
        X= self.conv3(X)
   
        X= torch.flatten(X,1)

        X = self.fc(X)
        return X