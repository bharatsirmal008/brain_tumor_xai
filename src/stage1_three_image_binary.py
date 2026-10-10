import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class Stage1ThreeImageBinary(nn.Module):
    """
    Input:
        [B, 3, 224, 224]

    View order:
        axial, coronal, sagittal

    Output:
        [B] tumor logits

    Positive label:
        1 = tumor
        0 = no tumor

    No BatchNorm running statistics.
    """

    def __init__(self):
        super().__init__()

        # Shared encoder for all three views
        self.encoder = nn.Sequential(
            ConvGNBlock(1, 16),
            ConvGNBlock(16, 32),
            ConvGNBlock(32, 64),
            ConvGNBlock(64, 96),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Learn view importance
        self.view_attention = nn.Sequential(
            nn.Linear(96, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Expected input [B,3,H,W], got "
                + str(tuple(x.shape))
            )

        batch_size = x.shape[0]

        # Small internal resolution for efficient training
        x = F.interpolate(
            x.float(),
            size=(112, 112),
            mode="bilinear",
            align_corners=False
        )

        # Treat three anatomical views independently,
        # then fuse learned features.
        x = x.reshape(
            batch_size * 3, 1, 112, 112
        )

        features = self.encoder(x)

        features = features.reshape(
            batch_size, 3, 96
        )

        scores = self.view_attention(features)

        weights = torch.softmax(scores, dim=1)

        fused = (features * weights).sum(dim=1)

        logits = self.classifier(fused)

        return logits.squeeze(1)