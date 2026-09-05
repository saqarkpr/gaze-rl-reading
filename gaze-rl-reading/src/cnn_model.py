"""
cnn_model.py

A small CNN that reads the synthetic gaze-heatmap image (gaze_heatmap.py)
and outputs two things:

  1. `embedding` -- an 8-number learned feature vector, later fed into the
     RL agent's state alongside the hand-crafted text features. This is
     the CNN-based visual modeling half of the project: instead of
     hand-engineered numeric features, a CNN looks at the image and
     learns what matters.
  2. `logit` -- a single struggle-probability score, used ONLY to
     pretrain the CNN (see pretrain_cnn.py) with a simple supervised
     step, before the embedding is handed to the RL agent.

Kept deliberately small (2 conv layers, ~28x28 MNIST-sized input) so it
is easy to read top to bottom and easy to retrain if you want to try
different image sizes or window widths later.
"""

import torch
import torch.nn as nn


class GazeCNN(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1x28x28 -> 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 8x14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # -> 16x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 16x7x7
        )
        self.embed = nn.Linear(16 * 7 * 7, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, 1, 28, 28), values in [0, 1]
        feats = self.conv(x)
        feats = feats.flatten(1)
        embedding = torch.relu(self.embed(feats))
        logit = self.classifier(embedding).squeeze(-1)
        return embedding, logit
