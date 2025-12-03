import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummaryX import summary
import numpy as np
from torchvision import datasets 
from torchvision import transforms
from torchinfo import summary # <-- Add this

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class ImageGenreDataset:
    def __init__(self, data_path, img_size=(128, 128), batch_size=32, split_ratio=0.9):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load dataset with transform
        self.data = datasets.ImageFolder(self.data_path, transform=self.transform)
        self.classes = self.data.classes
        self.num_classes = len(self.classes)
        
        # Split dataset
        train_len = int(len(self.data) * self.split_ratio)
        test_len  = len(self.data) - train_len

        self.train_data, self.test_data = random_split(self.data, [train_len, test_len])

        # Create loaders
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.test_data,  batch_size=self.batch_size, shuffle=True)

    def get_loaders(self):
        return self.train_loader, self.test_loader

    def get_classes(self):
        return self.classes

#CONV Neural data 
class ImageConv2d(nn.Module):
    def __init__(self, num_classes, input_channels=3 ,out_channels=16, kernel_size=3, pool_size=2, activation=None):
        super().__init__()
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels= input_channels, 
            out_channels= out_channels, 
            kernel_size=kernel_size, 
            stride=2, 
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
            stride=2, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
            in_channels= out_channels*2 , 
            out_channels= out_channels*4, 
            kernel_size=2, 
            stride=2, 
            padding =0
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)

        )
        final_size = out_channels*4*1*1

        self.fc = nn.Sequential(
            nn.Linear(final_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self , X): 
        X= self.conv1(X)
        X= self.conv2(X)
        X= self.conv3(X)

        X= torch.flatten(X,1)

        X = self.fc(X)
        return X