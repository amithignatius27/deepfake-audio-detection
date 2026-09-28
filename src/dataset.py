import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

# -----------------------------
# AUDIO CONFIG
# -----------------------------
SR = 16000
N_MELS = 128
MAX_FRAMES = 400   # fixed length for batching


# -----------------------------
# MEL SPECTROGRAM
# -----------------------------
def compute_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160,
        win_length=400,
        power=2.0
    )
    mel = np.log1p(mel)
    return mel.astype(np.float32)


def pad_or_truncate(mel):
    """Ensure fixed time dimension"""
    if mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel


# -----------------------------
# DATASET
# -----------------------------
class SpoofDataset(Dataset):
    """
    Folder structure expected:

    base_dir/
      ├── real/
      │     ├── *.wav / *.mp3 / *.mp4
      └── fake/
            ├── *.wav / *.mp3 / *.mp4
    """

    def __init__(self, base_dir):
        self.files = []
        self.labels = []

        real_dir = os.path.join(base_dir, "real")
        fake_dir = os.path.join(base_dir, "fake")

        # REAL = 0
        for f in os.listdir(real_dir):
            if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a")):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(0)

        # FAKE = 1
        for f in os.listdir(fake_dir):
            if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a")):
                self.files.append(os.path.join(fake_dir, f))
                self.labels.append(1)

        print(f"Loaded {len(self.files)} samples from {base_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load audio (librosa supports wav/mp3/mp4)
        wav, _ = librosa.load(path, sr=SR, mono=True)

        # Compute mel
        mel = compute_mel(wav)
        mel = pad_or_truncate(mel)

        # Global normalization (important for stability)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Shape: [1, 128, 400]
        mel = torch.tensor(mel).unsqueeze(0)

        return mel, label
