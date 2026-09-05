"""
gaze_heatmap.py

Converts numeric gaze features into a small image -- the way real
eye-tracking research turns raw fixation coordinates into a scanpath /
heatmap image before feeding it to a CNN. This is the piece that makes
the project a genuine RL + Computer Vision hybrid, not RL alone.

For each token position t, we render a tiny 28x28 grayscale image of the
gaze pattern in a small window around t: each nearby token becomes a
Gaussian "blob" whose brightness = fixation duration and whose size =
dwell time; a regression event nudges the blob down slightly (a simple
stand-in for "the eye jumped back to an earlier point").

28x28 was chosen on purpose -- same size as MNIST digits. If you have
seen any basic CNN/MNIST tutorial, this CNN is the same shape of problem.
"""

import numpy as np

IMG_SIZE = 28


def make_patch(passage, t, window=2, img_size=IMG_SIZE):
    """
    Build one 28x28 grayscale image representing the gaze scanpath in a
    window of tokens around position t.

    Returns: np.ndarray, shape (img_size, img_size), values in [0, 1]
    """
    n = len(passage["tokens"])
    fixation = np.array(passage["fixation_ms"])
    dwell = np.array(passage["dwell_ms"])
    regressed = np.array(passage["regressed"])

    img = np.zeros((img_size, img_size), dtype=np.float32)
    yy, xx = np.mgrid[0:img_size, 0:img_size]

    span = 2 * window + 1
    xs = np.linspace(3, img_size - 4, span)  # evenly spaced blob centers, left to right

    for k, pos in enumerate(range(t - window, t + window + 1)):
        if pos < 0 or pos >= n:
            continue
        cx = xs[k]
        cy = img_size / 2 + (5.0 if regressed[pos] else 0.0)
        intensity = float(np.clip(fixation[pos] / 500.0, 0.15, 1.0))
        radius = float(np.clip(dwell[pos] / 400.0, 1.2, 4.0))
        blob = intensity * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2)))
        img += blob

    return np.clip(img, 0.0, 1.0)


def make_all_patches(passage, window=2):
    """Convenience: build the image patch for every token position in a passage."""
    n = len(passage["tokens"])
    return np.stack([make_patch(passage, t, window=window) for t in range(n)])
