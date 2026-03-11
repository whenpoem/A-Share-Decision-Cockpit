from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FeaturePreprocessor:
    medians_: Optional[pd.Series] = None
    scales_: Optional[pd.Series] = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        self.medians_ = frame.median(numeric_only=True).fillna(0.0)
        centered = frame.fillna(self.medians_)
        scales = centered.std(ddof=0).replace(0.0, 1.0)
        self.scales_ = scales.fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.scales_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform.")
        filled = frame.fillna(self.medians_)
        scaled = (filled - self.medians_) / self.scales_
        return scaled.to_numpy(dtype=float)


class DistanceWeightedKNNClassifier:
    def __init__(self, neighbors: int = 40) -> None:
        self.neighbors = neighbors
        self.preprocessor = FeaturePreprocessor()
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "DistanceWeightedKNNClassifier":
        self.preprocessor.fit(frame)
        self.X_train = self.preprocessor.transform(frame)
        self.y_train = target.to_numpy()
        self.classes_ = np.unique(self.y_train)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
        X = self.preprocessor.transform(frame)
        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for row_idx, row in enumerate(X):
            distances = np.linalg.norm(self.X_train - row, axis=1)
            nearest_idx = np.argsort(distances)[: self.neighbors]
            nearest_dist = distances[nearest_idx]
            nearest_labels = self.y_train[nearest_idx]
            weights = 1.0 / (nearest_dist + 1e-6)
            for class_idx, label in enumerate(self.classes_):
                mask = nearest_labels == label
                probs[row_idx, class_idx] = weights[mask].sum()
            total = probs[row_idx].sum()
            if total <= 0:
                probs[row_idx] = 1.0 / len(self.classes_)
            else:
                probs[row_idx] /= total
        return probs


class SklearnCompatibleWrapper:
    def __init__(self, objective: str, random_seed: int = 42, expected_classes: Optional[np.ndarray] = None) -> None:
        self.objective = objective
        self.random_seed = random_seed
        self.expected_classes = expected_classes
        self.model = None
        self.preprocessor = FeaturePreprocessor()
        self.classes_: Optional[np.ndarray] = None
        self.backend = "distance_knn"
        self.constant_class_: Optional[int] = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "SklearnCompatibleWrapper":
        observed_classes = np.unique(target.to_numpy())
        self.classes_ = self.expected_classes if self.expected_classes is not None else observed_classes
        if len(observed_classes) == 1:
            self.backend = "constant"
            self.constant_class_ = int(observed_classes[0])
            self.model = None
            return self
        model = self._build_model(len(observed_classes))
        if self.backend == "lightgbm":
            model.fit(frame, target)
        elif self.backend == "sklearn":
            self.preprocessor.fit(frame)
            model.fit(self.preprocessor.transform(frame), target.to_numpy())
        else:
            model.fit(frame, target)
        self.model = model
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.backend == "constant":
            return self._constant_probs(len(frame))
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        if self.backend == "lightgbm":
            probs = np.asarray(self.model.predict_proba(frame), dtype=float)
            model_classes = np.asarray(self.model.classes_, dtype=int)
        elif self.backend == "sklearn":
            probs = np.asarray(self.model.predict_proba(self.preprocessor.transform(frame)), dtype=float)
            model_classes = np.asarray(self.model.classes_, dtype=int)
        else:
            probs = np.asarray(self.model.predict_proba(frame), dtype=float)
            model_classes = np.asarray(self.model.classes_, dtype=int)
        if self.classes_ is None:
            return probs
        aligned = np.zeros((len(frame), len(self.classes_)), dtype=float)
        for class_idx, label in enumerate(self.classes_):
            if label in model_classes:
                aligned[:, class_idx] = probs[:, np.where(model_classes == label)[0][0]]
        row_sums = aligned.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze() == 0
        if np.any(zero_rows):
            aligned[zero_rows] = 1.0 / aligned.shape[1]
            row_sums = aligned.sum(axis=1, keepdims=True)
        return aligned / row_sums

    def _constant_probs(self, row_count: int) -> np.ndarray:
        if self.classes_ is None or self.constant_class_ is None:
            raise RuntimeError("Constant model is not initialized.")
        probs = np.zeros((row_count, len(self.classes_)), dtype=float)
        class_idx = int(np.where(self.classes_ == self.constant_class_)[0][0])
        probs[:, class_idx] = 1.0
        return probs

    def _build_model(self, class_count: int):
        try:
            import lightgbm as lgb

            self.backend = "lightgbm"
            if self.objective == "multiclass":
                return lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=class_count,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_seed,
                )
            return lgb.LGBMClassifier(
                objective="binary",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
            )
        except ImportError:
            pass
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.backend = "sklearn"
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=5,
                max_iter=250,
                random_state=self.random_seed,
            )
        except ImportError:
            self.backend = "distance_knn"
            return DistanceWeightedKNNClassifier()
