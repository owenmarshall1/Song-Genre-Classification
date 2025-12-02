import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
import io

LABELS = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

class AudioGenreDataset(Dataset):
    def __init__(self, root_dir, labels=LABELS, sample_rate=16000, duration=4):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration  # total samples per audio

        self.files = []
        for genre, label in labels.items():
            genre_dir = os.path.join(root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
            for f in os.listdir(genre_dir):
                if f.lower().endswith(".wav"):
                    self.files.append((os.path.join(genre_dir, f), label))
        
    # i had an issue with them loading so this filters out corrupted files    
        self.files = self._filter_valid_files()   
    def _filter_valid_files(self):
        valid_files = []
        invalid_count = 0
        for i, (filepath, label) in enumerate(self.files):
            if (i + 1) % 10 == 0:
                print(f"Checking file {i + 1}/{len(self.files)}...")
            try:
                audio = AudioSegment.from_wav(filepath)
                if len(audio) > 0:
                    valid_files.append((filepath, label))
            except Exception:
                # skip corrupted files
                invalid_count += 1
                pass
        print(f"Loaded {len(valid_files)} valid audio files (filtered {invalid_count} corrupted from {len(self.files)})")
        return valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath, label = self.files[index]

        try:
            # Load audio using pydub
            audio = AudioSegment.from_wav(filepath)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Handle stereo to mono conversion
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            # Normalize to [-1, 1]
            samples = samples.astype(np.float32) / 32768.0
            
            # Resample if needed
            original_sr = audio.frame_rate
            if original_sr != self.sample_rate:
                samples = librosa.resample(samples, orig_sr=original_sr, target_sr=self.sample_rate)
            
            waveform = torch.from_numpy(samples).float().unsqueeze(0)

            # Truncate or pad to fixed length
            if waveform.size(1) > self.num_samples:
                waveform = waveform[:, :self.num_samples]
            elif waveform.size(1) < self.num_samples:
                pad_amount = self.num_samples - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_amount))

            return waveform, label

        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}: {e}")
