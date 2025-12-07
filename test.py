import torch
import torch.nn.functional as F
import torch.nn as nn


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0 
    total_loss = 0.0
    lossfunction = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_predicted = model(X)
            batch_loss = lossfunction(y_predicted, y)
            total_loss += batch_loss.item() * y.size(0)
            predicted = torch.argmax(y_predicted, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = (correct / total) * 100 if total > 0 else 0.0

    return avg_loss, accuracy