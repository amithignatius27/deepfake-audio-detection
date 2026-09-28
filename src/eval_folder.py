import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from model import ResNetAudio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
SR = 16000
N_MELS = 128
MAX_FRAMES = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_DIR = "C:/Users/STANLEY ERNEST/OneDrive/Desktop/eicas_detector/eval"
MODEL_PATH = "checkpoints/best_model1.pth"

# -----------------------------
# AUDIO PROCESSING
# -----------------------------
def compute_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160,
        win_length=400
    )
    return np.log1p(mel).astype(np.float32)

def pad_or_truncate(mel):
    if mel.shape[1] < MAX_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, MAX_FRAMES - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :MAX_FRAMES]
    return mel

def preprocess(path):
    wav, _ = librosa.load(path, sr=SR)
    mel = compute_mel(wav)
    mel = pad_or_truncate(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

# -----------------------------
# MAIN EVALUATION
# -----------------------------
def main():
    print("\n=== FULL EVAL (FOLDER-BASED) ===\n")

    model = ResNetAudio().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    y_true, y_pred = [], []

    for label_name, label_value in [("real", 0), ("fake", 1)]:
        folder = os.path.join(EVAL_DIR, label_name)

        for fname in tqdm(os.listdir(folder), desc=f"Evaluating {label_name}"):
            if not fname.endswith((".wav", ".flac")):
                continue

            path = os.path.join(folder, fname)
            mel = preprocess(path)

            with torch.no_grad():
                out = model(mel)
                pred = torch.argmax(out, dim=1).item()

            y_true.append(label_value)
            y_pred.append(pred)

    # -----------------------------
    # METRICS
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n==============================")
    print(f" Accuracy  : {acc*100:.2f}%")
    print(f" Precision : {prec*100:.2f}%")
    print(f" Recall    : {rec*100:.2f}%")
    print(f" F1-score  : {f1*100:.2f}%")
    print("\n Confusion Matrix:")
    print(cm)
    print("==============================\n")

if __name__ == "__main__":
    main()
