import torch
import torch.nn as nn
import torchvision.models as models


class ResNetAudio(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------------------
        # Backbone: ResNet18
        # -----------------------------
        self.backbone = models.resnet18(weights=None)

        # Modify first conv layer for 1-channel (mel spectrogram)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # -----------------------------
        # Strong classifier head (CRITICAL)
        # -----------------------------
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)  # REAL (0) / FAKE (1)
        )

        # -----------------------------
        # Weight initialization (important)
        # -----------------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: [B, 1, 128, 400]
        """
        return self.backbone(x)
