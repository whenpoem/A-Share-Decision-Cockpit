from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from astock_prob.utils import multiclass_log_loss


@dataclass
class TemperatureScaler:
    temperatures: List[float]
    selected_temperature: float = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        best_temp = 1.0
        best_loss = float("inf")
        for temp in self.temperatures:
            scaled = self.transform(probs, temperature=temp)
            loss = multiclass_log_loss(y_true, scaled)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        self.selected_temperature = best_temp
        return self

    def transform(self, probs: np.ndarray, temperature: float | None = None) -> np.ndarray:
        temp = self.selected_temperature if temperature is None else temperature
        adjusted = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temp)
        return adjusted / adjusted.sum(axis=1, keepdims=True)


class BinaryCalibrator:
    method: str

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BinaryCalibrator":
        raise NotImplementedError

    def transform(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class IsotonicCalibrator(BinaryCalibrator):
    x_: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    y_: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    method: str = "isotonic"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        order = np.argsort(scores)
        x = scores[order].astype(float)
        y = labels[order].astype(float)
        blocks = [[x[idx], x[idx], 1, y[idx]] for idx in range(len(x))]
        pointer = 0
        while pointer < len(blocks) - 1:
            left_mean = blocks[pointer][3] / blocks[pointer][2]
            right_mean = blocks[pointer + 1][3] / blocks[pointer + 1][2]
            if left_mean <= right_mean:
                pointer += 1
                continue
            merged = [
                blocks[pointer][0],
                blocks[pointer + 1][1],
                blocks[pointer][2] + blocks[pointer + 1][2],
                blocks[pointer][3] + blocks[pointer + 1][3],
            ]
            blocks[pointer : pointer + 2] = [merged]
            pointer = max(pointer - 1, 0)
        self.x_ = np.array([(block[0] + block[1]) / 2.0 for block in blocks], dtype=float)
        self.y_ = np.array([block[3] / block[2] for block in blocks], dtype=float)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if len(self.x_) == 0:
            return np.asarray(scores, dtype=float)
        return np.interp(scores, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


@dataclass
class PlattCalibrator(BinaryCalibrator):
    coef_: float = 1.0
    intercept_: float = 0.0
    method: str = "platt"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        x = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        logits = np.log(x / (1.0 - x))
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=42, max_iter=500)
            model.fit(logits.reshape(-1, 1), y.astype(int))
            self.coef_ = float(model.coef_.ravel()[0])
            self.intercept_ = float(model.intercept_.ravel()[0])
        except Exception:
            self.coef_ = 1.0
            self.intercept_ = float(np.log((y.mean() + 1e-6) / max(1.0 - y.mean(), 1e-6)))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        logits = np.log(x / (1.0 - x))
        output = self.coef_ * logits + self.intercept_
        return 1.0 / (1.0 + np.exp(-output))


@dataclass
class MeanCalibrator(BinaryCalibrator):
    mean_: float = 0.5
    method: str = "mean"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "MeanCalibrator":
        label_arr = np.asarray(labels, dtype=float)
        self.mean_ = float(label_arr.mean()) if len(label_arr) else 0.5
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(scores)), self.mean_, dtype=float)


def fit_binary_calibrator(scores: np.ndarray, labels: np.ndarray, min_samples: int) -> BinaryCalibrator:
    label_arr = np.asarray(labels, dtype=int)
    if len(label_arr) < min_samples or len(np.unique(label_arr)) < 2:
        return PlattCalibrator().fit(scores, labels) if len(np.unique(label_arr)) >= 2 else MeanCalibrator().fit(scores, labels)
    return IsotonicCalibrator().fit(scores, labels)
