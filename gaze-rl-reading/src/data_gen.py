"""
data_gen.py

Generates synthetic reading passages with a HIDDEN per-token difficulty
ground truth, and simulates noisy gaze features (fixation duration,
regression probability, dwell time) that correlate with that hidden
difficulty -- the way real eye-tracking corpora (e.g. GECO, Provo Corpus)
correlate gaze behavior with comprehension difficulty.

The RL agent and the weak-supervision signal (weak_supervision.py) only
ever see the noisy gaze features, never the hidden difficulty. Ground
truth is kept aside purely for evaluation, mirroring how a real system
would be validated against held-out human comprehension labels.

Why synthetic data: this project is a fast, self-contained proof of
concept for the ZHAW/UZH LitAI PhD position. It intentionally reuses the
same "verifiable signal instead of human labels" structure as the RLVF
project, applied to gaze-style sequential data instead of symbolic math.
"""

import numpy as np
import json
import os

RNG_SEED = 42

# A small vocabulary with an intrinsic "true difficulty" score in [0, 1],
# loosely standing in for real predictors of reading difficulty:
# word length, frequency (rarity), and syntactic role.
VOCAB = {
    "the": 0.05, "a": 0.05, "is": 0.08, "of": 0.08, "and": 0.05,
    "cat": 0.10, "dog": 0.10, "ran": 0.12, "sat": 0.10, "big": 0.12,
    "reading": 0.25, "sentence": 0.30, "teacher": 0.20, "school": 0.18,
    "quantum": 0.85, "photosynthesis": 0.90, "ergodic": 0.92,
    "metacognitive": 0.88, "hypothesis": 0.70, "juxtaposition": 0.87,
    "however": 0.35, "nevertheless": 0.55, "consequently": 0.50,
    "ambiguous": 0.65, "syntax": 0.55, "morphology": 0.75,
    "children": 0.15, "quickly": 0.20, "carefully": 0.22,
    "understand": 0.28, "comprehend": 0.60, "articulate": 0.62,
    "because": 0.15, "although": 0.40, "therefore": 0.38,
}
WORDS = list(VOCAB.keys())
DIFFICULTY = np.array([VOCAB[w] for w in WORDS])


def make_passage(rng, length=25):
    """Sample a token sequence with mixed easy/hard words."""
    # Bias toward a plausible reading passage: mostly easy words with
    # occasional hard ones, not uniform random -- real text is bursty.
    idx = []
    i = 0
    while i < length:
        if rng.random() < 0.22:
            # short burst of 1-2 hard words
            hard_pool = [j for j, d in enumerate(DIFFICULTY) if d > 0.5]
            idx.append(rng.choice(hard_pool))
        else:
            easy_pool = [j for j, d in enumerate(DIFFICULTY) if d <= 0.5]
            idx.append(rng.choice(easy_pool))
        i += 1
    return np.array(idx)


def simulate_gaze(rng, token_idx):
    """
    Simulate per-token gaze features conditioned on hidden difficulty,
    with realistic noise so the signal is a WEAK proxy, not a clean label.

    Returns dict of arrays: fixation_ms, regression_prob, dwell_ms
    """
    true_diff = DIFFICULTY[token_idx]

    # Fixation duration (ms): baseline + difficulty effect + noise.
    # Real fixation durations for easy words ~200-250ms, hard words ~350-500ms.
    fixation_ms = 200 + 300 * true_diff + rng.normal(0, 40, size=len(token_idx))
    fixation_ms = np.clip(fixation_ms, 80, None)

    # Regression probability: chance the reader jumps back to re-read.
    regression_prob = np.clip(0.05 + 0.5 * true_diff + rng.normal(0, 0.08, size=len(token_idx)), 0, 1)
    regressed = rng.random(len(token_idx)) < regression_prob

    # Dwell time: total time spent on token including any re-reads.
    dwell_ms = fixation_ms * (1 + regressed * rng.uniform(0.5, 1.2, size=len(token_idx)))

    return {
        "fixation_ms": fixation_ms,
        "regressed": regressed.astype(int),
        "dwell_ms": dwell_ms,
        "true_difficulty": true_diff,  # kept ONLY for evaluation, never for training
    }


def generate_dataset(n_passages=200, passage_len=25, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    passages = []
    for _ in range(n_passages):
        tok_idx = make_passage(rng, length=passage_len)
        gaze = simulate_gaze(rng, tok_idx)
        passages.append({
            "tokens": [WORDS[i] for i in tok_idx],
            "token_idx": tok_idx.tolist(),
            "fixation_ms": gaze["fixation_ms"].tolist(),
            "regressed": gaze["regressed"].tolist(),
            "dwell_ms": gaze["dwell_ms"].tolist(),
            "true_difficulty": gaze["true_difficulty"].tolist(),
        })
    return passages


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    train = generate_dataset(n_passages=300, seed=RNG_SEED)
    val = generate_dataset(n_passages=60, seed=RNG_SEED + 1)
    test = generate_dataset(n_passages=60, seed=RNG_SEED + 2)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"wrote {len(data)} passages to {path}")
