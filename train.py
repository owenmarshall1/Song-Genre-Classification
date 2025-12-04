import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from audio_dataset import AudioGenreDataset, LABELS
from audio_model import AudioCNN

from images_dataset import ImageGenreDataset
from images_model import ImageConv2d
from test import test_model
from tqdm import tqdm

def train(model, loader, device, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}" ):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            predicted = preds.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = correct/total
        print(f"Epoch {epoch+1} | Accuracy: {accuracy*100:.2f}%")
        if accuracy> 0.97:
            break
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audio", "spec"], required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "audio":
        dataset = AudioGenreDataset(
            root_dir="data/genres_original",
            labels=LABELS,
            sample_rate=16000,
            duration=4
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        model = AudioCNN()
    else:
        data_handler = ImageGenreDataset("data/images_original")
        train_loader = data_handler.train_loader
        test_loader = data_handler.test_loader
        model = ImageConv2d(num_classes=data_handler.num_classes)
        

    train(model, train_loader, device)
    test_acc = test_model(model, test_loader, device)
    test_loss, test_acc = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
if __name__ == "__main__":
    main()