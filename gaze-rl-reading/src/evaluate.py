"""
evaluate.py

Compares four things against held-out `true_difficulty` (never used in
any training step in this project):

  1. RL agent (text-only)        -- from train.py
  2. RL + CV agent (text + CNN)  -- from train_cv.py
  3. Word-length baseline        -- naive non-learned heuristic
  4. Weak label itself           -- the noisy gaze-derived proxy signal
     everything else was trained on (upper-reference, not a real "method")
"""

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn import QNetwork
from env import token_features, ACTION_CONTINUE
from env_cv import STATE_DIM_CV
from cnn_model import GazeCNN
from gaze_heatmap import make_patch
from weak_supervision import struggle_labels

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
STATE_DIM = 4
N_ACTIONS = 3
TRUE_DIFF_THRESHOLD = 0.5


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def agent_predict_passage(q_net, passage):
    tokens = passage["tokens"]
    n = len(tokens)
    preds, recent = [], []
    for t in range(n):
        hist = np.mean(recent[-5:]) if recent else 0.0
        feat = token_features(tokens[t], t, n, hist)
        with torch.no_grad():
            q = q_net(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        struggle = 1 if action != ACTION_CONTINUE else 0
        preds.append(struggle)
        recent.append(struggle)
    return np.array(preds)


def agent_cv_predict_passage(q_net_cv, cnn, passage):
    tokens = passage["tokens"]
    n = len(tokens)
    preds, recent = [], []
    for t in range(n):
        hist = np.mean(recent[-5:]) if recent else 0.0
        text_feat = token_features(tokens[t], t, n, hist)
        patch = make_patch(passage, t)
        x = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cv_feat, _ = cnn(x)
            feat = np.concatenate([text_feat, cv_feat.squeeze(0).numpy()])
            q = q_net_cv(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        struggle = 1 if action != ACTION_CONTINUE else 0
        preds.append(struggle)
        recent.append(struggle)
    return np.array(preds)


def baseline_predict_passage(passage, length_threshold=7):
    return np.array([1 if len(w) > length_threshold else 0 for w in passage["tokens"]])


def prf(pred, true):
    pred, true = np.array(pred), np.array(true)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    test_data = load_split("test")

    q_net = QNetwork(STATE_DIM, N_ACTIONS)
    q_net.load_state_dict(torch.load(os.path.join(OUT_DIR, "q_net.pt"), map_location="cpu"))
    q_net.eval()

    q_net_cv = QNetwork(STATE_DIM_CV, N_ACTIONS)
    q_net_cv.load_state_dict(torch.load(os.path.join(OUT_DIR, "q_net_cv.pt"), map_location="cpu"))
    q_net_cv.eval()

    cnn = GazeCNN(embed_dim=8)
    cnn.load_state_dict(torch.load(os.path.join(OUT_DIR, "gaze_cnn.pt"), map_location="cpu"))
    cnn.eval()

    preds_by_method = {"RL agent (text-only)": [], "RL + CV agent (text+CNN)": [],
                        "Word-length baseline": [], "Weak label itself (upper ref.)": []}
    true_all = []

    for passage in test_data:
        true_diff = np.array(passage["true_difficulty"])
        true_bin = (true_diff >= TRUE_DIFF_THRESHOLD).astype(int)

        preds_by_method["RL agent (text-only)"].extend(agent_predict_passage(q_net, passage).tolist())
        preds_by_method["RL + CV agent (text+CNN)"].extend(agent_cv_predict_passage(q_net_cv, cnn, passage).tolist())
        preds_by_method["Word-length baseline"].extend(baseline_predict_passage(passage).tolist())
        weak_bin, _ = struggle_labels(passage)
        preds_by_method["Weak label itself (upper ref.)"].extend(weak_bin.tolist())
        true_all.extend(true_bin.tolist())

    results = {}
    for name, preds in preds_by_method.items():
        p, r, f1 = prf(preds, true_all)
        results[name] = {"precision": p, "recall": r, "f1": f1}
        print(f"{name:34s} | precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f}")

    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUT_DIR, "history.json")) as f:
        hist = json.load(f)
    with open(os.path.join(OUT_DIR, "history_cv.json")) as f:
        hist_cv = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist["epoch"], hist["val_reward"], label="RL agent (text-only) val reward")
    ax.plot(hist_cv["epoch"], hist_cv["val_reward"], label="RL+CV agent val reward")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean episode reward")
    ax.set_title("DQN training: text-only vs. text+CNN state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = list(results.keys())
    f1s = [results[n]["f1"] for n in names]
    ax.bar(names, f1s, color=["#4C72B0", "#55A868", "#DD8452", "#999999"])
    ax.set_ylabel("F1 vs. held-out true difficulty")
    ax.set_ylim(0, 1)
    ax.set_title("Struggle-point detection: 4-way comparison")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nsaved plots + results to {OUT_DIR}")


if __name__ == "__main__":
    main()
