import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

SR = 16000
N_MELS = 128
MAX_FRAMES = 400

def compute_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160,
        win_length=400
    )
    mel = np.log1p(mel).astype(np.float32)
    return mel

def pad_or_truncate(mel):
    if mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel

class SpoofDataset(Dataset):
    def __init__(self, base_dir):
        self.files = []
        self.labels = []

        real_dir = os.path.join(base_dir, "real")
        fake_dir = os.path.join(base_dir, "fake")

        # LOAD REAL FIRST (label = 0)
        
        for f in os.listdir(real_dir):
            if f.endswith(".wav") or f.endswith(".flac"):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(0)  # REAL = 0

        # LOAD FAKE (label = 1)
    
        for f in os.listdir(fake_dir):
            if f.endswith(".wav") or f.endswith(".flac"):
                self.files.append(os.path.join(fake_dir, f))
                self.labels.append(1)  # FAKE = 1


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        wav, sr = librosa.load(path, sr=SR)
        mel = compute_mel(wav)
        mel = pad_or_truncate(mel)

        # -----------------------------
        # GLOBAL NORMALIZATION
        # consistent across ALL data
        
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        mel = torch.tensor(mel).unsqueeze(0).float()
        return mel, label
