"""
env.py

A minimal sequential-decision environment modeling a reader's path
through a passage, in the same spirit as the DQN-over-a-structured-problem
framing used in the M.Sc. thesis (Symbolic-RL-Fredholm-Solver).

At each token position, the agent observes ONLY text-derivable features
(word length, log-frequency proxy, position in passage, a short rolling
history) -- never the gaze signal itself. It picks one of three actions:

    0 = CONTINUE  (reader moves on smoothly)
    1 = REGRESS   (reader jumps back to re-read -- struggling)
    2 = PAUSE     (reader pauses/re-fixates in place -- struggling)

It is rewarded by how well its action matches the WEAK struggle label
derived from that passage's simulated gaze data (weak_supervision.py).
At deployment, gaze would not be available in advance for a *new* reader
-- the trained policy is meant to predict likely struggle points from
text alone, having learned the association from many (text, gaze) pairs
during training. This is the same shift from "supervised on hand labels"
to "supervised on a programmatic/behavioral proxy signal" as the RLVF
project, applied to sequential reading data instead of symbolic math.
"""

import numpy as np

ACTION_CONTINUE = 0
ACTION_REGRESS = 1
ACTION_PAUSE = 2
N_ACTIONS = 3

WORD_FREQ_RANK = {  # smaller = more frequent/common (rough hand-set proxy)
    "the": 1, "a": 1, "is": 2, "of": 1, "and": 1, "cat": 5, "dog": 5,
    "ran": 6, "sat": 6, "big": 5, "reading": 15, "sentence": 20,
    "teacher": 12, "school": 8, "quantum": 90, "photosynthesis": 95,
    "ergodic": 98, "metacognitive": 96, "hypothesis": 60,
    "juxtaposition": 92, "however": 10, "nevertheless": 40,
    "consequently": 35, "ambiguous": 55, "syntax": 45, "morphology": 65,
    "children": 10, "quickly": 12, "carefully": 14, "understand": 18,
    "comprehend": 50, "articulate": 52, "because": 3, "although": 20,
    "therefore": 22,
}


def token_features(word, position, passage_len, history_struggle):
    """
    Build a small feature vector the agent can observe for a token.
    history_struggle: rolling mean of the agent's own recent struggle
    actions (proxy for reader fatigue), purely derived from the agent's
    own past actions -- not from gaze.
    """
    length = len(word) / 15.0  # normalize roughly
    freq_rank = WORD_FREQ_RANK.get(word, 50) / 100.0
    pos_frac = position / max(passage_len - 1, 1)
    return np.array([length, freq_rank, pos_frac, history_struggle], dtype=np.float32)


class ReadingEnv:
    """One episode = one passage, walked token by token."""

    def __init__(self, passage, weak_labels):
        self.passage = passage
        self.tokens = passage["tokens"]
        self.weak_labels = weak_labels  # binary array, same length as tokens
        self.n = len(self.tokens)
        self.reset()

    def reset(self):
        self.t = 0
        self._recent_struggles = []
        return self._obs()

    def _obs(self):
        hist = np.mean(self._recent_struggles[-5:]) if self._recent_struggles else 0.0
        return token_features(self.tokens[self.t], self.t, self.n, hist)

    def step(self, action):
        expected_struggle = self.weak_labels[self.t]
        took_struggle_action = 1 if action in (ACTION_REGRESS, ACTION_PAUSE) else 0

        # Reward: agreement with the weak label. Small asymmetric penalty
        # discourages the agent from spamming REGRESS/PAUSE to game recall.
        if took_struggle_action == expected_struggle:
            reward = 1.0
        else:
            reward = -1.0 if expected_struggle == 1 else -0.3

        self._recent_struggles.append(took_struggle_action)
        self.t += 1
        done = self.t >= self.n
        obs = self._obs() if not done else None
        return obs, reward, done, {"predicted_struggle": took_struggle_action}
