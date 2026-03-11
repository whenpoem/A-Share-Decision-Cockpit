from __future__ import annotations

from typing import Dict, Iterable


def enforce_monotonic_touch_probabilities(probs: Dict[str, float], thresholds: Iterable[float], direction: str) -> Dict[str, float]:
    thresholds = sorted(float(item) for item in thresholds)
    adjusted = dict(probs)
    running = 1.0
    for threshold in thresholds:
        key = _touch_key(threshold, direction)
        current = min(max(adjusted.get(key, 0.0), 0.0), 1.0)
        running = min(running, current)
        adjusted[key] = running
    return adjusted


def _touch_key(threshold: float, direction: str) -> str:
    pct = int(round(threshold * 100))
    return f"touch_{direction}_{pct}"
