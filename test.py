import torch
import torch.nn.functional as F
import torch.nn as nn

#####Testing models accuracy and loss on unseen data
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0 
    total_loss = 0.0
    lossfunction = nn.CrossEntropyLoss()

    #Computing loss and accuracy in batches
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_predicted = model(X)
            batch_loss = lossfunction(y_predicted, y)
            total_loss += batch_loss.item() * y.size(0)
            
            # get predicted class labels
            predicted = torch.argmax(y_predicted, dim=1)
            
            #update counters
            total += y.size(0)
            correct += (predicted == y).sum().item()
    #computing average loss and accuracy of batches
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = (correct / total) * 100 if total > 0 else 0.0

    return avg_loss, accuracy