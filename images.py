import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummaryX import summary
import numpy as np
from torchvision import datasets 


data = datasets.ImageFolder("data/images_original")
train_loader = DataLoader(data, batch_size=32, shuffle=True)
num_classes = len(data.classes)
print(data.classes)


class Conv2d(nn.Module):
    