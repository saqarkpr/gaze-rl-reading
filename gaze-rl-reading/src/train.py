"""
train.py

Trains the DQN reading agent on the training split, using ONLY the weak
struggle labels derived from simulated gaze (weak_supervision.py). Ground
truth difficulty is never touched here -- it is reserved for evaluate.py.
"""

import json
import os
import random

import numpy as np
import torch

from dqn import DQNAgent
from env import ReadingEnv, N_ACTIONS
from weak_supervision import struggle_labels

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
STATE_DIM = 4  # matches token_features() in env.py


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def run_episode(agent, passage, train=True, batch_size=64):
    weak_bin, _ = struggle_labels(passage)
    env = ReadingEnv(passage, weak_bin)
    state = env.reset()
    total_reward = 0.0
    losses = []

    done = False
    while not done:
        action = agent.act(state, greedy=not train)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.buffer.push(state, action, reward, next_state if next_state is not None else np.zeros(STATE_DIM, dtype=np.float32), float(done))
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

    agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)

    history = {"epoch": [], "train_reward": [], "val_reward": [], "loss": []}

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_data)
        epoch_rewards = []
        epoch_losses = []
        for passage in train_data:
            r, losses = run_episode(agent, passage, train=True)
            epoch_rewards.append(r)
            epoch_losses.extend(losses)
        agent.sync_target()

        val_rewards = []
        for passage in val_data:
            r, _ = run_episode(agent, passage, train=False)
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
    torch.save(agent.q_net.state_dict(), os.path.join(OUT_DIR, "q_net.pt"))
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"saved model + history to {OUT_DIR}")


if __name__ == "__main__":
    main()
