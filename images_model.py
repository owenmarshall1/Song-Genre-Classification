import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#CONV Neural data 
class ImageConv2d(nn.Module):
    def __init__(self, num_classes, input_channels ,out_channels, kernel_size=2, pool_size=3, activation=None, img_size=(128, 128)):
        super().__init__()
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels= input_channels, 
            out_channels= out_channels,
            
            kernel_size=kernel_size, 
            stride= 1, 
            padding = 'same'
            ),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        
       )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size=kernel_size, 
            stride=1, 
            padding = 'same' 

            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Dropout(0.3),

        )
    def forward(self , X): 
        X= self.conv1(X)
        X= self.conv2(X)
        return X
    
class Conv2dLayers(nn.Module):
    def __init__(self,num_classes, img_size=(128,128)):
        super().__init__()

        self.conv = nn.Sequential(
            ImageConv2d(num_classes, input_channels=3, out_channels=64),
            ImageConv2d(num_classes, input_channels=64, out_channels=128),
        )

        # Compute flattened size after conv layers
        dummy_input = torch.rand(1, 3, img_size[0], img_size[1])
        out = self.conv(dummy_input) 
        final_channels = out.view(out.size(0), -1).size(1)
        print(f"Final channels before pooling: {final_channels}")

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        

    def forward(self, X):
        X= self.conv(X)
        X = X.view(X.size(0), -1) 
        X = self.fc(X)
        return X