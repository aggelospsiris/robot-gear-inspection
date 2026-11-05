"""Class-independent causal object identity across short detector/SAM gaps.

Uses background camera motion, local residual velocity, position/scale and HOG
appearance. It never receives A/B predictions, class labels or evaluation truth.
Ambiguous reappearances start a fresh ID instead of inheriting a possibly wrong
immutable label. This is a bounded short-gap association, not general long-term
re-identification across cameras or sessions.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box):
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2


def overlap(a, b):
    size = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    return intersection / max(
        1.0, float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection
    )


class CameraMotion:
    def __init__(self):
        self.gray = None
        self.last_report = {}

    def reset(self):
        self.gray = None

    def update(self, rgb, boxes):
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5)
        matrix = np.eye(3)
        report = dict(method="identity", inliers=0)
        if self.gray is not None:
            mask = np.full(self.gray.shape, 255, np.uint8)
            for box in boxes:
                x0, y0, x1, y1 = np.rint(np.asarray(box) * 0.5).astype(int)
                cv2.rectangle(
                    mask, (max(0, x0 - 5), max(0, y0 - 5)), (x1 + 5, y1 + 5), 0, -1
                )
            points = cv2.goodFeaturesToTrack(self.gray, 500, 0.01, 8, mask=mask)
            src = dst = None
            if points is not None and len(points) >= 16:
                tracked, ok, _ = cv2.calcOpticalFlowPyrLK(
                    self.gray, gray, points, None, winSize=(31, 31), maxLevel=3
                )
                back, ok_back, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.gray, tracked, None, winSize=(31, 31), maxLevel=3
                )
                valid = (
                    (ok[:, 0] > 0)
                    & (ok_back[:, 0] > 0)
                    & (np.linalg.norm(back - points, axis=(1, 2)) < 1.5)
                )
                src, dst = points[valid], tracked[valid]
            if src is not None and len(src) >= 12:
                h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
                if (
                    h is not None
                    and inliers is not None
                    and inliers.sum() >= 12
                    and inliers.mean() > 0.45
                ):
                    corners = np.float32(
                        [
                            [0, 0],
                            [gray.shape[1], 0],
                            [gray.shape[1], gray.shape[0]],
                            [0, gray.shape[0]],
                        ]
                    )[None]
                    warped = cv2.perspectiveTransform(corners, h)[0]
                    area = abs(cv2.contourArea(warped)) / (
                        gray.shape[0] * gray.shape[1]
                    )
                    if np.isfinite(warped).all() and 0.4 < area < 2.5:
                        scale = np.diag([0.5, 0.5, 1.0])
                        matrix = np.linalg.inv(scale) @ h @ scale
                        report = dict(
                            method="background_lk_homography",
                            inliers=int(inliers.sum()),
                        )
        self.gray = gray
        self.last_report = report
        return matrix


@dataclass
class Memory:
    box: np.ndarray
    appearance: np.ndarray
    last_detected: float
    velocity: np.ndarray


class ShortGapIdentity:
    def __init__(
        self,
        max_missing_seconds=3.0,
        minimum_appearance=0.65,
        maximum_center_distance=1.5,
        ambiguity_margin=0.08,
        max_objects=12,
    ):
        if (
            max_missing_seconds <= 0
            or not 0 <= minimum_appearance <= 1
            or maximum_center_distance <= 0
            or ambiguity_margin < 0
        ):
            raise ValueError("Invalid reappearance settings")
        self.max_missing = max_missing_seconds
        self.minimum_appearance = minimum_appearance
        self.max_distance = maximum_center_distance
        self.margin = ambiguity_margin
        self.max_objects = max_objects
        self.motion = CameraMotion()
        self.reset()

    def reset(self):
        self.memory = {}
        self.next_id = 1
        self.timestamp = None
        self.motion.reset()
        self.events = []

    def forget(self, key):
        self.memory.pop(key, None)

    def appearance(self, rgb, box):
        x0, y0, x1, y1 = np.rint(box).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
        crop = rgb[y0:y1, x0:x1]
        if not crop.size:
            raise ValueError("Empty appearance crop")
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (48, 48))
        # The installed minimal OpenCV build has no objdetect/HOGDescriptor.
        # Fixed 6x6 cells, nine unsigned gradient bins, normalized 2x2 blocks.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        magnitude = np.hypot(gx, gy)
        bins = (np.mod(np.arctan2(gy, gx), np.pi) * (9 / np.pi)).astype(int) % 9
        yy, xx = np.indices(gray.shape)
        hist = np.zeros((6, 6, 9), np.float32)
        np.add.at(hist, (yy // 8, xx // 8, bins), magnitude)
        blocks = []
        for y in range(5):
            for x in range(5):
                block = hist[y : y + 2, x : x + 2].reshape(-1)
                block = np.minimum(block / max(float(np.linalg.norm(block)), 1e-9), 0.2)
                blocks.append(block / max(float(np.linalg.norm(block)), 1e-9))
        feature = np.concatenate(blocks)
        return feature / max(float(np.linalg.norm(feature)), 1e-9)

    def begin_frame(self, rgb, timestamp):
        if not math.isfinite(timestamp) or (
            self.timestamp is not None and timestamp <= self.timestamp
        ):
            raise ValueError("Timestamps must increase")
        dt = 0 if self.timestamp is None else timestamp - self.timestamp
        h = self.motion.update(rgb, [m.box for m in self.memory.values()])
        self.timestamp = timestamp
        self.events = []
        for key, m in list(self.memory.items()):
            if timestamp - m.last_detected > self.max_missing:
                del self.memory[key]
                self.events.append(dict(kind="expired", gear_id=key))
                continue
            x0, y0, x1, y1 = m.box
            points = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None]
            warped = cv2.perspectiveTransform(points, h)[0]
            if np.isfinite(warped).all():
                m.box = np.r_[warped.min(0), warped.max(0)]
            drift = m.velocity * min(dt, 0.15)
            m.box += np.r_[drift, drift]
            m.velocity *= math.exp(-dt * 2)

    def associate(self, rgb, boxes):
        if self.timestamp is None:
            raise RuntimeError("Call begin_frame first")
        boxes = [np.asarray(box, float) for box in boxes]
        features = [self.appearance(rgb, box) for box in boxes]
        keys = list(self.memory)
        cost = np.full((len(boxes), len(keys)), 1e6)
        similarities = np.zeros_like(cost)
        for i, (box, feature) in enumerate(zip(boxes, features)):
            for j, key in enumerate(keys):
                m = self.memory[key]
                size = np.maximum(m.box[2:] - m.box[:2], 1)
                distance = float(np.linalg.norm(center(box) - center(m.box))) / max(
                    float(np.linalg.norm(size)), 1
                )
                area_ratio = float(
                    np.prod(np.maximum(box[2:] - box[:2], 1)) / np.prod(size)
                )
                similarity = float(np.clip(feature @ m.appearance, 0, 1))
                similarities[i, j] = similarity
                active = self.timestamp - m.last_detected <= 0.15
                geometric = overlap(box, m.box)
                appearance_ok = similarity >= self.minimum_appearance or (
                    active and geometric > 0.35
                )
                limit = self.max_distance if not active else min(self.max_distance, 0.9)
                if appearance_ok and distance <= limit and 0.35 < area_ratio < 2.8:
                    cost[i, j] = (
                        0.5 * distance + 0.3 * (1 - similarity) + 0.2 * (1 - geometric)
                    )
        matches = {}
        if cost.size:
            rr, cc = linear_sum_assignment(cost)
            for i, j in zip(rr, cc):
                if cost[i, j] >= 1e6:
                    continue
                key = keys[j]
                dormant = self.timestamp - self.memory[key].last_detected > 0.15
                alternatives = [*np.delete(cost[i], j), *np.delete(cost[:, j], i)]
                if (
                    dormant
                    and alternatives
                    and min(alternatives) - cost[i, j] < self.margin
                ):
                    continue
                matches[int(i)] = key
                if dormant:
                    self.events.append(
                        dict(
                            kind="reconnected",
                            gear_id=key,
                            gap_seconds=self.timestamp - self.memory[key].last_detected,
                            appearance_similarity=float(similarities[i, j]),
                        )
                    )
        for i, (box, feature) in enumerate(zip(boxes, features)):
            if i not in matches:
                if len(self.memory) >= self.max_objects:
                    continue
                key = self.next_id
                self.next_id += 1
                matches[i] = key
                self.memory[key] = Memory(
                    box.copy(), feature, self.timestamp, np.zeros(2)
                )
                self.events.append(dict(kind="created", gear_id=key))
            else:
                m = self.memory[matches[i]]
                elapsed = max(self.timestamp - m.last_detected, 1 / 60)
                velocity = np.clip((center(box) - center(m.box)) / elapsed, -300, 300)
                m.velocity = 0.8 * m.velocity + 0.2 * velocity
                if float(feature @ m.appearance) > 0.75:
                    m.appearance = 0.9 * m.appearance + 0.1 * feature
                    m.appearance /= max(float(np.linalg.norm(m.appearance)), 1e-9)
                m.box = box.copy()
                m.last_detected = self.timestamp
        return matches

    def mask_supported(self, key, box):
        m = self.memory.get(key)
        return m is not None and overlap(np.asarray(box, float), m.box) >= 0.15
