from torch.utils.data import DataLoader, random_split
from audio_dataset import AudioGenreDataset, LABELS

def create_dataloaders(batch_size=16, trainsplit=0.8):
    dataset = AudioGenreDataset(
        root_dir="data/genres_original",
        labels = LABELS,
        sample_rate=16000,
        duration=4
    )

    total_len = len(dataset)
    train_len = int(total_len * trainsplit)
    val_len = total_len - train_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_len, val_len