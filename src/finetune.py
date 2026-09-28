import torch
from torch.utils.data import DataLoader
from dataset import SpoofDataset
from model import ResNetAudio
import os

# -----------------------------
# CONFIG
# -----------------------------
EPOCHS = 8
BATCH_SIZE = 16

HEAD_LR = 5e-4      # phase 1
FINETUNE_LR = 1e-4  # phase 2

DATA_DIR = "data_finetune/train"
CHECKPOINT_OUT = "checkpoints/best_model_finetuned1.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # DATASET
    # -----------------------------
    train_ds = SpoofDataset(DATA_DIR)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = ResNetAudio().to(device)

    print("Loading base model weights...")
    model.load_state_dict(
        torch.load("checkpoints/best_model1.pth", map_location=device),
        strict=False
    )

    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------
    # PHASE 1 — TRAIN HEAD ONLY
    # -----------------------------
    print("\n=== PHASE 1: Training classifier head only ===")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        model.backbone.fc.parameters(),
        lr=HEAD_LR
    )

    for epoch in range(3):
        model.train()
        correct, total = 0, 0

        for mel, label in train_dl:
            mel, label = mel.to(device), label.to(device)

            out = model(mel)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total
        print(f"Phase1 Epoch {epoch+1}/3 | Train Acc: {acc:.4f}")

    # -----------------------------
    # PHASE 2 — UNFREEZE layer4 + head
    # -----------------------------
    print("\n=== PHASE 2: Fine-tuning layer4 + head ===")

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR
    )

    for epoch in range(3, EPOCHS):
        model.train()
        correct, total = 0, 0

        for mel, label in train_dl:
            mel, label = mel.to(device), label.to(device)

            out = model(mel)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total
        print(f"Phase2 Epoch {epoch+1}/{EPOCHS} | Train Acc: {acc:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_OUT)

    print("\n✅ Fine-tuning complete")
    print("Saved model to:", CHECKPOINT_OUT)


if __name__ == "__main__":
    main()
