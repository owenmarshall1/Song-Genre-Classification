import torch
import torch.nn.functional as F

def test_model(model, test_loader, device):
    