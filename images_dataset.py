from torchsummaryX import summary
import numpy as np
from torchvision import datasets 
from torchvision import transforms
from torchinfo import summary 
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 


class ImageGenreDataset:
    def __init__(self, data_path, img_size=(128, 128), batch_size=32, split_ratio=0.9):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.crop_borders =(35,35,54,42)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: F.crop(
                img,
                self.crop_borders[0],  # top
                self.crop_borders[2],  # left
                img.height - self.crop_borders[0] - self.crop_borders[1],  # height
                img.width - self.crop_borders[2] - self.crop_borders[3]   # width
            )),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=128, scale=(0.85, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset with transform
        self.data = datasets.ImageFolder(self.data_path, transform=self.transform)
        self.classes = self.data.classes
        self.num_classes = len(self.classes)
        
        # Split dataset
        self.train_data, self.test_data = random_split(self.data,(split_ratio, 1-split_ratio))

        # Create loaders
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.test_data,  batch_size=self.batch_size, shuffle=False)


    def get_loaders(self):
        return self.train_loader, self.test_loader

    def get_classes(self):
        return self.classes
