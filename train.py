import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from audio_dataset import AudioGenreDataset, LABELS
from audio_model import AudioCNN

from images_dataset import ImageGenreDataset
from images_model import ImageViT
from test import test_model
from tqdm import tqdm


def train(model, train_loader, device, epochs, test_loader=None, base_lr=3e-4, weight_decay=0.05, warmup_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        # simple linear warmup for first few epochs
        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
            for g in optimizer.param_groups:
                g["lr"] = base_lr * warmup_factor

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}" ):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            predicted = preds.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss += loss.item() * y.size(0)

        if epoch >= warmup_epochs:
            scheduler.step()

        epoch_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audio", "spec"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = None
    test_loader = None
    model = None

    if args.mode == "audio":
        dataset = AudioGenreDataset(
            root_dir="data/genres_original",
            labels=LABELS,
            sample_rate=16000,
            duration=4
        )
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = AudioCNN()
    else:
        data_handler = ImageGenreDataset("data/images_original")
        train_loader = data_handler.train_loader
        test_loader = data_handler.test_loader
        model = ImageViT(num_classes=data_handler.num_classes)

    train(model, train_loader, device, epochs=args.epochs, test_loader=test_loader)

    if test_loader is not None:
        test_loss, test_acc = test_model(model, test_loader, device)
        print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()