import torch
import librosa
import numpy as np
import os
from model import ResNetAudio

# -----------------------------
# CONFIG
# -----------------------------
SR = 16000
N_MELS = 128
MAX_FRAMES = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_model_finetuned1.pth"


# -----------------------------
# MEL FUNCTIONS (SAME AS TRAIN)
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
    mel = np.log1p(mel).astype(np.float32)
    return mel


def pad_or_truncate(mel):
    if mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel


def preprocess_audio(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)

    mel = compute_mel(wav)
    mel = pad_or_truncate(mel)

    # same normalization as dataset.py
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    # shape → [1, 1, 128, 400]
    mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
    return mel.float()


# -----------------------------
# MAIN INFERENCE
# -----------------------------
def main():
    print("\n=== Deepfake Audio Detection ===\n")

    audio_path = input("Enter path to audio file: ").strip()

    if not os.path.exists(audio_path):
        print("❌ File not found")
        return

    # Load model
    model = ResNetAudio().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    # Preprocess
    mel = preprocess_audio(audio_path).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "REAL" if pred == 0 else "FAKE"

    print("\n------------------------------")
    print(f" Prediction : {label}")
    print(f" Confidence : {confidence * 100:.2f}%")
    print("------------------------------\n")


if __name__ == "__main__":
    main()
