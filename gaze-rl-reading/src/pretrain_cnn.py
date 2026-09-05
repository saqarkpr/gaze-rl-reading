"""
pretrain_cnn.py

Plain supervised training -- no RL here. The CNN looks at a gaze-heatmap
image patch and predicts the weak struggle label (from weak_supervision.py,
the same programmatic proxy signal used everywhere else in this project;
still never the hidden true_difficulty).

This step is what makes the RL agent's later input a genuine LEARNED
visual feature instead of a hand-picked number: we are training a
classifier first, then throwing away its final classification head and
reusing its second-to-last layer as an embedding (a very common, simple
transfer-learning pattern -- train small model on an easy proxy task,
reuse its internal representation elsewhere).

If you are learning this: this file is the simplest one to start with.
It's a standard "load images, load labels, forward, loss, backward" loop,
same shape as any basic PyTorch image classifier tutorial.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import GazeCNN
from gaze_heatmap import make_patch
from weak_supervision import struggle_labels

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def build_xy(passages):
    """Flatten every (token, patch, weak_label) triple across all passages
    into one big supervised dataset -- same idea as flattening pixels into
    one training set in a standard image classification tutorial."""
    X, y = [], []
    for passage in passages:
        weak_bin, _ = struggle_labels(passage)
        n = len(passage["tokens"])
        for t in range(n):
            X.append(make_patch(passage, t))
            y.append(weak_bin[t])
    X = np.stack(X).astype(np.float32)[:, None, :, :]  # (N, 1, 28, 28)
    y = np.array(y, dtype=np.float32)
    return X, y


def main(n_epochs=8, batch_size=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = load_split("train")
    val_data = load_split("val")

    print("building image patches (this is the slow-ish one-time step)...")
    X_train, y_train = build_xy(train_data)
    X_val, y_val = build_xy(val_data)
    print(f"train patches: {X_train.shape}, val patches: {X_val.shape}")

    model = GazeCNN(embed_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    n = X_train_t.shape[0]
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            _, logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            _, val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_acc = ((torch.sigmoid(val_logits) > 0.5).float() == y_val_t).float().mean().item()

        print(f"epoch {epoch:2d} | train_loss {epoch_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "gaze_cnn.pt"))
    print(f"saved CNN weights to {OUT_DIR}/gaze_cnn.pt")


if __name__ == "__main__":
    main()
