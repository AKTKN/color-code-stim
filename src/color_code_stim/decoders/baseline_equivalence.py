"""Exact equality classes for three corrections in original X/Z DEM order."""

import numpy as np


COLORS = ("r", "g", "b")


def classify_baselines(baselines):
    """Return multiplicity class and ordered equality groups (r, g, b)."""
    if isinstance(baselines, dict):
        vectors = [np.asarray(baselines[color], dtype=bool) for color in COLORS]
    else:
        vectors = np.asarray(baselines, dtype=bool)
        if vectors.ndim != 2 or vectors.shape[0] != 3:
            raise ValueError("baseline must have three original-DEM corrections")
        vectors = list(vectors)
    if len({vector.shape for vector in vectors}) != 1:
        raise ValueError("baseline corrections must have the same shape")
    groups = []
    for color, vector in zip(COLORS, vectors):
        for group in groups:
            if np.array_equal(vector, vectors[COLORS.index(group[0])]):
                group.append(color)
                break
        else:
            groups.append([color])
    return len(groups) - 1, tuple(tuple(group) for group in groups)
