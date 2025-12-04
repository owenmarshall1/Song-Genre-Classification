import torch
import torch.nn.functional as F
import torch.nn as nn


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0 
    lossfunction = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            X= X.to(device)
            y= y.to(device)

            y_predicted = model (X)
            test_loss = lossfunction(y_predicted, y)
            predicted = torch.argmax(y_predicted, dim=1)
            total +=1 
            correct += (predicted == y).sum().item()
        accuracy = correct/ total 
    return test_loss, accuracy