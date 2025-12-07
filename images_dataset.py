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

        train_size = self.img_size[0] if isinstance(self.img_size, tuple) else self.img_size
        self.train_transform = transforms.Compose([
            transforms.Lambda(lambda img: F.crop(
                img,
                self.crop_borders[0],  # top
                self.crop_borders[2],  # left
                img.height - self.crop_borders[0] - self.crop_borders[1],  # height
                img.width - self.crop_borders[2] - self.crop_borders[3]   # width
            )),
            transforms.RandomResizedCrop(train_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Lambda(lambda img: F.crop(
                img,
                self.crop_borders[0],
                self.crop_borders[2],
                img.height - self.crop_borders[0] - self.crop_borders[1],
                img.width - self.crop_borders[2] - self.crop_borders[3]
            )),
            transforms.Resize((train_size, train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        base_data = datasets.ImageFolder(self.data_path, transform=None)
        self.classes = base_data.classes
        self.num_classes = len(self.classes)

        total = len(base_data)
        train_len = int(self.split_ratio * total)
        test_len = total - train_len
        train_subset, test_subset = random_split(base_data, [train_len, test_len])

        class TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                x, y = self.subset[idx]
                if self.transform is not None:
                    x = self.transform(x)
                return x, y

        self.train_data = TransformedSubset(train_subset, self.train_transform)
        self.test_data = TransformedSubset(test_subset, self.test_transform)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader  = DataLoader(self.test_data,  batch_size=self.batch_size, shuffle=False, num_workers=0)


    def get_loaders(self):
        return self.train_loader, self.test_loader

    def get_classes(self):
        return self.classes
