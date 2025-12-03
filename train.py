import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from audio_dataset import AudioGenreDataset, LABELS
from audio_model import AudioCNN

def train(model, loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            predicted = preds.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f"Epoch {epoch+1} | Accuracy: {100*correct/total:.2f}%")

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
        dataset = ImageGenerDataset(
            
        )
        return

    train(model, loader, device)

if __name__ == "__main__":
    main()