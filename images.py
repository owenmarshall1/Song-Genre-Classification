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

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
###Loading data And Data preperation taking 10 of each for testing as unseen data
data = datasets.ImageFolder("data/images_original", transform=transform)
num_classes = len(data.classes)
print(data.classes)

train_data, test_data = random_split(data, (0.9,0.1))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=True)


#CONV Neural data 
class ImageConv2d(nn.Module):
    def __init__(self, num_classes, input_channels ,out_channels=16, kernel_size=3, pool_size=2, activation=None):
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

input_channels = 3 
model = ImageConv2d(num_classes=num_classes, input_channels=input_channels)

print("Model Summary (Input 3x128x128):")
summary(model, input_size=(1, input_channels, 128, 128))