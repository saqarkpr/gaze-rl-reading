"""
env_cv.py

Same reading environment as env.py (agent still picks CONTINUE / REGRESS
/ PAUSE, still rewarded by agreement with the weak label), but the
observation the agent sees is now:

    [ text features (4) ]  +  [ CNN embedding of the gaze heatmap (8) ]
                                          = 12 numbers total

The CNN embedding comes from the frozen, already-pretrained GazeCNN
(pretrain_cnn.py) looking at the image patch around the current token
(gaze_heatmap.py). This is the actual RL+CV fusion: the agent's decision
is now informed by a learned visual representation, not just hand-picked
numeric features.

We freeze the CNN here (no gradient) to keep training simple and stable:
the CV part and the RL part are trained in two clear, separate, easy to
debug stages, instead of one large end-to-end model.
"""

import numpy as np
import torch

from env import token_features, ACTION_CONTINUE, ACTION_REGRESS, ACTION_PAUSE, N_ACTIONS  # noqa: F401
from gaze_heatmap import make_patch

CV_EMBED_DIM = 8
STATE_DIM_CV = 4 + CV_EMBED_DIM  # = 12


class ReadingEnvCV:
    def __init__(self, passage, weak_labels, cnn):
        self.passage = passage
        self.tokens = passage["tokens"]
        self.weak_labels = weak_labels
        self.n = len(self.tokens)
        self.cnn = cnn  # frozen, pretrained GazeCNN
        self.reset()

    def _cnn_embedding(self, t):
        patch = make_patch(self.passage, t)
        x = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        with torch.no_grad():
            embedding, _ = self.cnn(x)
        return embedding.squeeze(0).numpy()

    def reset(self):
        self.t = 0
        self._recent_struggles = []
        return self._obs()

    def _obs(self):
        hist = np.mean(self._recent_struggles[-5:]) if self._recent_struggles else 0.0
        text_feat = token_features(self.tokens[self.t], self.t, self.n, hist)
        cv_feat = self._cnn_embedding(self.t)
        return np.concatenate([text_feat, cv_feat]).astype(np.float32)

    def step(self, action):
        expected_struggle = self.weak_labels[self.t]
        took_struggle_action = 1 if action in (ACTION_REGRESS, ACTION_PAUSE) else 0

        if took_struggle_action == expected_struggle:
            reward = 1.0
        else:
            reward = -1.0 if expected_struggle == 1 else -0.3

        self._recent_struggles.append(took_struggle_action)
        self.t += 1
        done = self.t >= self.n
        obs = self._obs() if not done else None
        return obs, reward, done, {"predicted_struggle": took_struggle_action}
