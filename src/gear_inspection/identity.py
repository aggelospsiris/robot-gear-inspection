"""Sampled type confirmation and hysteresis, independent of anomaly voting."""

from collections import deque

import numpy as np


class AccumulatedTypeEvidence:
    """Confidence-weighted winning-class evidence, retained through valid gaps."""

    lock_after_confirmation = True

    def __init__(self, min_observations=15, minimum_agreement=0.8, min_span=0.3, **_):
        if min_observations < 2 or not 0.5 <= minimum_agreement <= 1 or min_span < 0:
            raise ValueError("Invalid cumulative confirmation settings")
        self.min_observations = min_observations
        self.agreement = minimum_agreement
        self.min_span = min_span
        self.history = deque(maxlen=128)
        self.confirmed = None
        self.confidence = 0.0
        self.observations_count = 0
        self.weights = np.zeros(2, float)
        self.first_sample = None
        self.last_sample = None
        self.last_hash = None
        self.last_frame = None

    def update_detection(self, gear, confidence, timestamp, crop_hash, frame_index):
        if self.confirmed is not None:
            return self.confirmed, False
        if (
            gear not in ("A", "B")
            or not np.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid detector evidence")
        if not np.isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if self.last_frame is not None and (
            frame_index <= self.last_frame or timestamp <= self.last_sample
        ):
            raise ValueError("Evidence must be fresh and ordered")
        if crop_hash == self.last_hash:
            return None, False
        if self.first_sample is None:
            self.first_sample = timestamp
        self.last_sample, self.last_hash, self.last_frame = (
            timestamp,
            crop_hash,
            frame_index,
        )
        self.weights["AB".index(gear)] += confidence
        self.observations_count += 1
        self.history.append((timestamp, gear, confidence))
        share = self.weights / max(float(self.weights.sum()), 1e-12)
        self.confidence = float(share.max())
        if (
            self.observations_count >= self.min_observations
            and timestamp - self.first_sample >= self.min_span - 1e-9
            and self.confidence >= self.agreement - 1e-9
            and abs(float(share[0] - share[1])) > 1e-9
        ):
            self.confirmed = "AB"[int(share.argmax())]
        return self.confirmed, False


class TypeEvidence:
    def __init__(
        self,
        min_observations=3,
        min_span=0.3,
        sample_interval=0.15,
        switch_observations=5,
        switch_span=0.6,
        lock_after_confirmation=False,
        **_,
    ):
        if min_observations < 2 or switch_observations < min_observations:
            raise ValueError(
                "Type confirmation needs multiple observations and stronger switching evidence"
            )
        self.min_observations = min_observations
        self.min_span = min_span
        self.sample_interval = sample_interval
        self.switch_observations = switch_observations
        self.switch_span = switch_span
        self.lock_after_confirmation = lock_after_confirmation
        self.history = deque(maxlen=max(min_observations, switch_observations))
        self.confirmed = None
        self.last_sample = None
        self.last_hash = None
        self.confidence = 0.0

    def update(self, probability, timestamp, crop_hash, minimum):
        p = np.asarray(probability, dtype=float)
        if (
            p.shape != (2,)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("Invalid A/B probability vector")
        if self.lock_after_confirmation and self.confirmed is not None:
            return self.confirmed, False
        candidate = "AB"[int(np.argmax(p))] if p.max() >= minimum else None
        if candidate is None:
            # Ambiguous observations interrupt a proposed identity change.
            self.history.clear()
            return None, False
        if crop_hash == self.last_hash or (
            self.last_sample is not None
            and timestamp - self.last_sample < self.sample_interval - 1e-9
        ):
            return (self.confirmed if candidate == self.confirmed else None), False
        self.last_sample, self.last_hash = timestamp, crop_hash
        if self.history and self.history[-1][1] != candidate:
            self.history.clear()
        self.history.append((timestamp, candidate, float(p.max())))
        n = (
            self.min_observations
            if self.confirmed is None
            else self.switch_observations
        )
        span = self.min_span if self.confirmed is None else self.switch_span
        changed = False
        if candidate == self.confirmed:
            self.confidence = float(p.max())
        elif (
            len(self.history) >= n
            and self.history[-1][0] - self.history[-n][0] >= span - 1e-9
        ):
            changed = self.confirmed is not None
            self.confirmed = candidate
            self.confidence = float(np.mean([v[2] for v in list(self.history)[-n:]]))
        return (self.confirmed if candidate == self.confirmed else None), changed


def geometry_drift(previous, current, max_area_ratio=2.5, min_box_iou=0.1):
    if previous is None:
        return False
    a, b = previous, current
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    iou = intersection / max(1, area_a + area_b - intersection)
    return max(area_a / area_b, area_b / area_a) > max_area_ratio or iou < min_box_iou
