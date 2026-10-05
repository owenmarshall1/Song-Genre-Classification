# Song-Genre-Classification

A PyTorch project that trains convolutional neural networks to predict the genre of a music track. It is a supervised, multi-class (10-genre) classification problem built on the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset.

Two approaches are implemented:

| Mode    | Input                                   | Model                          |
|---------|-----------------------------------------|--------------------------------|
| `audio` | Raw waveform (mono, 16 kHz, first 4 s)  | 1D CNN (`AudioCNN`)            |
| `spec`  | Mel-spectrogram images (PNG)            | 2D CNN (`Conv2dLayers`)        |

## Genres

`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

## Project structure

```
.
├── audio_dataset.py   # AudioGenreDataset: loads .wav files, resamples, pads/truncates, skips corrupted files
├── audio_model.py     # AudioCNN: 3× (Conv1d → BatchNorm → ReLU → MaxPool) + fully connected head
├── images_dataset.py  # ImageGenreDataset: crops plot borders, resizes, normalizes, 90/10 train/test split
├── images_model.py    # ImageConv2d block and Conv2dLayers: stacked Conv2d blocks + fully connected head
├── dataloaders.py     # Helper that builds train/val loaders for the audio dataset
├── train.py           # Training entry point (--mode audio | spec)
├── test.py            # test_model(): computes loss and accuracy on a held-out loader
└── data/              # GTZAN dataset (see below)
```

## Dataset

The dataset is not stored in this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and extract the contents of its `Data/` folder into `data/` at the project root:

```
data/
├── genres_original/<genre>/<genre>.000NN.wav   # 30 s audio clips, 100 per genre
├── images_original/<genre>/<genre>000NN.png    # Mel-spectrogram image per clip
├── features_30_sec.csv                         # Pre-extracted features (not used yet)
└── features_3_sec.csv
```

GTZAN contains at least one corrupted file (`jazz.00054.wav`). `AudioGenreDataset` detects unreadable files and skips them at load time.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically when available; otherwise training runs on the CPU.

## Usage

Train and evaluate the spectrogram model:

```bash
python train.py --mode spec
```

Train the raw-audio model:

```bash
python train.py --mode audio
```

Pass `--seed N` (default `42`) to change the random seed. The same seed always produces the same train/test split, so runs can be compared.

Training uses Adam (lr `1e-4`) with cross-entropy loss for up to 100 epochs. It stops early once training accuracy exceeds 90%. Afterwards, test loss and accuracy are printed for the held-out split.

## Results

| Model                     | Train accuracy | Test accuracy |
|---------------------------|----------------|---------------|
| `Conv2dLayers` (`spec`)   | ~90%           | ~66%          |

The gap between train and test accuracy shows the spectrogram model is overfitting. Improving generalization is the current focus.
