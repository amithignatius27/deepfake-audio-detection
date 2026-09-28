import torch
from torch.utils.data import DataLoader
from data_loader import SpoofDataset
from model import ResNetAudio

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetAudio().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model1.pth", map_location=device))
    model.eval()

    eval_ds = SpoofDataset("data/eval")
    dl = DataLoader(eval_ds, batch_size=16)

    total = 0
    correct = 0

    with torch.no_grad():
        for mel, label in dl:
            mel = mel.to(device)
            label = label.to(device)

            out = model(mel)
            pred = out.argmax(1)

            correct += (pred == label).sum().item()
            total += len(label)

    print("Eval Accuracy:", correct / total)

if __name__ == "__main__":
    main()
