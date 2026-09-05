"""
train_cv.py

Same DQN training loop as train.py, but using ReadingEnvCV -- the state
now includes the CNN's learned gaze embedding, not just hand-crafted text
features. The CNN itself is frozen (already pretrained in pretrain_cnn.py)
so only the DQN agent is learning here.
"""

import json
import os
import random

import numpy as np
import torch

from cnn_model import GazeCNN
from dqn import DQNAgent
from env_cv import ReadingEnvCV, STATE_DIM_CV
from env import N_ACTIONS
from weak_supervision import struggle_labels

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def load_frozen_cnn():
    cnn = GazeCNN(embed_dim=8)
    cnn.load_state_dict(torch.load(os.path.join(OUT_DIR, "gaze_cnn.pt"), map_location="cpu"))
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False
    return cnn


def run_episode(agent, passage, cnn, train=True, batch_size=64):
    weak_bin, _ = struggle_labels(passage)
    env = ReadingEnvCV(passage, weak_bin, cnn)
    state = env.reset()
    total_reward = 0.0
    losses = []

    done = False
    while not done:
        action = agent.act(state, greedy=not train)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.buffer.push(
                state, action, reward,
                next_state if next_state is not None else np.zeros(STATE_DIM_CV, dtype=np.float32),
                float(done),
            )
            loss = agent.update(batch_size=batch_size)
            if loss is not None:
                losses.append(loss)
        total_reward += reward
        state = next_state if next_state is not None else state
    return total_reward, losses


def main(n_epochs=15, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = load_split("train")
    val_data = load_split("val")
    cnn = load_frozen_cnn()

    agent = DQNAgent(state_dim=STATE_DIM_CV, n_actions=N_ACTIONS)

    history = {"epoch": [], "train_reward": [], "val_reward": [], "loss": []}

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_data)
        epoch_rewards, epoch_losses = [], []
        for passage in train_data:
            r, losses = run_episode(agent, passage, cnn, train=True)
            epoch_rewards.append(r)
            epoch_losses.extend(losses)
        agent.sync_target()

        val_rewards = []
        for passage in val_data:
            r, _ = run_episode(agent, passage, cnn, train=False)
            val_rewards.append(r)

        mean_train = float(np.mean(epoch_rewards))
        mean_val = float(np.mean(val_rewards))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"epoch {epoch:2d} | train_reward {mean_train:6.2f} | val_reward {mean_val:6.2f} | loss {mean_loss:.4f} | eps {agent.epsilon():.3f}")

        history["epoch"].append(epoch)
        history["train_reward"].append(mean_train)
        history["val_reward"].append(mean_val)
        history["loss"].append(mean_loss)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(agent.q_net.state_dict(), os.path.join(OUT_DIR, "q_net_cv.pt"))
    with open(os.path.join(OUT_DIR, "history_cv.json"), "w") as f:
        json.dump(history, f)
    print(f"saved model + history to {OUT_DIR}")


if __name__ == "__main__":
    main()
