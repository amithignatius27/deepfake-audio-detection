import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader import SpoofDataset
from model import ResNetAudio
from tqdm import tqdm
import os

EPOCHS = 20
BATCH = 16
LR = 1e-4

def main():
    print("Loading datasets...")

    train_ds = SpoofDataset("data/train")
    dev_ds   = SpoofDataset("data/dev")

    # -----------------------------
    # 1. FIX CLASS IMBALANCE
    # -----------------------------
    labels = train_ds.labels
    labels_tensor = torch.tensor(labels)
    class_counts = torch.bincount(labels_tensor)  # [num_real, num_fake]

    # inverse frequency weight
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_tensor]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
    dev_dl   = DataLoader(dev_ds, batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetAudio().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=2, factor=0.5, verbose=True
    )

    # ensure checkpoint folder exists
    os.makedirs("checkpoints", exist_ok=True)

    best_val = 0

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        correct = 0
        running_loss = 0

        pbar = tqdm(train_dl)
        for mel, label in pbar:
            mel = mel.to(device)
            label = label.to(device)

            out = model(mel)
            loss = loss_fn(out, label)
            running_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == label).sum().item()
            total += len(label)

            pbar.set_description(
                f"Epoch [{epoch+1}] Loss: {loss.item():.4f}"
            )

        train_acc = correct / total
        avg_loss = running_loss / total
        print(f"Train Acc: {train_acc:.4f}")

        # ---- VALIDATION ----
        model.eval()
        total = 0
        correct = 0

        val_loss = 0
        with torch.no_grad():
            for mel, label in dev_dl:
                mel = mel.to(device)
                label = label.to(device)

                out = model(mel)
                val_loss += loss_fn(out, label).item()

                pred = out.argmax(1)
                correct += (pred == label).sum().item()
                total += len(label)

        val_acc = correct / total
        print(f"Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model1.pth")
            print("Saved best model")

if __name__ == "__main__":
    main()
