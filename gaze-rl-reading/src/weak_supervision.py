"""
weak_supervision.py

This is the "verifier" of the project -- the piece that plays the same
role the programmatic math verifier plays in the RLVF project. Instead of
a human annotator labeling "this word was hard for the reader," we derive
a proxy struggle score directly from the gaze signal itself:

    struggle_score = normalized(fixation_ms) + normalized(dwell_ms) + regressed

This score is WEAK and NOISY (it is a proxy, not ground truth), but it is
fully programmatic and requires zero human labeling -- the same "verifier
instead of human preference labels" structure used in the RLVF project,
applied to gaze data instead of a symbolic-math checker.

The RL agent (env.py) is rewarded using ONLY this weak signal. The
`true_difficulty` field in the dataset is reserved for evaluate.py and is
never passed into training.
"""

import numpy as np


def compute_struggle_scores(passage):
    """
    Given one passage dict (as produced by data_gen.py), compute a
    per-token weak struggle score in [0, 1] using only observable gaze
    features (fixation_ms, dwell_ms, regressed). No ground truth used.
    """
    fixation = np.array(passage["fixation_ms"])
    dwell = np.array(passage["dwell_ms"])
    regressed = np.array(passage["regressed"])

    def normalize(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-6:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    fixation_n = normalize(fixation)
    dwell_n = normalize(dwell)

    # Weighted combination: dwell and fixation duration matter most,
    # a regression event is a strong-but-sparse extra signal.
    struggle = 0.4 * fixation_n + 0.4 * dwell_n + 0.2 * regressed
    return np.clip(struggle, 0, 1)


def struggle_labels(passage, threshold=0.5):
    """Binarize the weak struggle score into a proxy struggle/no-struggle label."""
    scores = compute_struggle_scores(passage)
    return (scores >= threshold).astype(int), scores
