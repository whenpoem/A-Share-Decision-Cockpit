from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence

import numpy as np

from astock_prob.utils import brier_score, multiclass_log_loss


def simplex_weights(step: float) -> List[tuple[float, float, float]]:
    values = np.arange(0.0, 1.0 + step / 2.0, step)
    combos = []
    for a, b in product(values, repeat=2):
        c = 1.0 - a - b
        if c < -1e-9:
            continue
        c = max(c, 0.0)
        if abs(a + b + c - 1.0) <= 1e-9:
            combos.append((float(a), float(b), float(c)))
    return combos


def fit_multiclass_weights(prob_matrices: Sequence[np.ndarray], y_true: np.ndarray, step: float) -> tuple[float, float, float]:
    best = (1.0, 0.0, 0.0)
    best_loss = float("inf")
    for weights in simplex_weights(step):
        blended = blend_multiclass(prob_matrices, weights)
        loss = multiclass_log_loss(y_true, blended)
        if loss < best_loss:
            best_loss = loss
            best = weights
    return best


def fit_binary_weights(prob_vectors: Sequence[np.ndarray], y_true: np.ndarray, step: float) -> tuple[float, float, float]:
    best = (1.0, 0.0, 0.0)
    best_score = float("inf")
    for weights in simplex_weights(step):
        blended = blend_binary(prob_vectors, weights)
        score = brier_score(y_true, blended)
        if score < best_score:
            best_score = score
            best = weights
    return best


def blend_multiclass(prob_matrices: Sequence[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    output = np.zeros_like(prob_matrices[0], dtype=float)
    for weight, matrix in zip(weights, prob_matrices):
        output += weight * matrix
    output = np.clip(output, 1e-12, None)
    return output / output.sum(axis=1, keepdims=True)


def blend_binary(prob_vectors: Sequence[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    output = np.zeros_like(prob_vectors[0], dtype=float)
    for weight, vector in zip(weights, prob_vectors):
        output += weight * vector
    return np.clip(output, 0.0, 1.0)
