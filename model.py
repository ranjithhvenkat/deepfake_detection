# model.py

import torch
import torch.nn as nn

class CNNViT(nn.Module):
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        super(CNNViT, self).__init__()

        # CNN Feature Extractor (5 blocks)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )

        # Transformer prep
        self.flatten_dim = 4 * 4  # after 5 max-pools: 128 → 4
        self.embedding_dim = 512

        self.pos_embedding = nn.Parameter(torch.randn(1, self.flatten_dim, self.embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=4 * self.embedding_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, P, C)

        x = x + self.pos_embedding[:, :x.size(1), :]  # Match positional embedding length
        x = self.transformer(x)

        x = x.permute(0, 2, 1)  # Shape: (B, C, P)
        x = self.classifier(x)

        return x
