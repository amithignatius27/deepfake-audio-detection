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
        y=wav, sr=SR, n_mels=N_MELS,
        n_fft=512, hop_length=160, win_length=400
    )
    mel = np.log1p(mel)
    return mel.astype(np.float32)

def pad_or_truncate(mel):
    if mel.shape[1] < MAX_FRAMES:
        mel = np.pad(mel, ((0,0),(0,MAX_FRAMES-mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel

class StereoSpoofDataset(Dataset):
    def __init__(self, base_dir):
        self.files, self.labels = [], []

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(base_dir, cls)
            for f in os.listdir(folder):
                if f.lower().endswith((".wav",".mp3",".mp4",".m4a")):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.files[idx], sr=SR, mono=False)

        if wav.ndim == 1:  # mono → fake stereo
            wav = np.stack([wav, wav])

        mel_l = pad_or_truncate(compute_mel(wav[0]))
        mel_r = pad_or_truncate(compute_mel(wav[1]))

        mel_l = (mel_l - mel_l.mean()) / (mel_l.std() + 1e-6)
        mel_r = (mel_r - mel_r.mean()) / (mel_r.std() + 1e-6)

        mel = torch.tensor(np.stack([mel_l, mel_r]))  # [2,128,400]
        label = torch.tensor(self.labels[idx])

        return mel, label
