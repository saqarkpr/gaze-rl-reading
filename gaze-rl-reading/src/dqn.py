"""
dqn.py

A small DQN agent, deliberately kept close in structure to the
Symbolic-RL-Fredholm-Solver thesis code (same algorithm family, applied
to a new problem domain): a Q-network, epsilon-greedy exploration with
decay, a replay buffer, and a target network for stability.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=2000, device="cpu"):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        self.q_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def epsilon(self):
        # Exponential decay -- fixes the non-decaying-epsilon bug flagged
        # in the thesis code review, applied correctly here from the start.
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)

    def act(self, state, greedy=False):
        self.steps_done += 1
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        non_final_mask = ~dones.bool()
        non_final_next = [s for s, d in zip(batch.next_state, batch.done) if not d]

        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        next_q = torch.zeros(batch_size, device=self.device)
        if non_final_next:
            next_states_t = torch.tensor(np.array(non_final_next), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_q[non_final_mask] = self.target_net(next_states_t).max(dim=1)[0]

        target = rewards + self.gamma * next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
