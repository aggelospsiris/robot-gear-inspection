"""Native-resolution perceptual blur and visibility gate."""

import json
import numpy as np, cv2
from scipy.ndimage import uniform_filter1d

FEATURES = [
    "retention5_min",
    "retention5_max",
    "retention11_min",
    "retention11_max",
    "edge_width_median",
    "edge_width_p75",
    "relative_edge_width",
]


def measure(gray, mask):
    inside = cv2.erode(mask, np.ones((13, 13), "uint8")) > 0
    ys0, xs0 = np.where(mask)
    short_side = (
        int(min(xs0.max() - xs0.min() + 1, ys0.max() - ys0.min() + 1))
        if len(xs0)
        else 0
    )
    result = dict(
        interior_pixels=int(inside.sum()), edge_count=0, object_short_side=short_side
    )
    if inside.sum() < 50:
        return dict(result, **{k: 0.0 for k in FEATURES})
    grad = [
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)),
    ]
    for h in [5, 11]:
        for axis, name in [(1, "x"), (0, "y")]:
            blurred = uniform_filter1d(gray, h, axis=axis, mode="reflect")
            g = grad[0 if axis == 1 else 1]
            gb = np.abs(
                cv2.Sobel(blurred, cv2.CV_32F, int(axis == 1), int(axis == 0), ksize=3)
            )
            den = float(g[inside].sum())
            result[f"retention{h}_{name}"] = float(
                np.clip(1 - np.maximum(0, g - gb)[inside].sum() / max(den, 1e-9), 0, 1)
            )
    for h in [5, 11]:
        result[f"retention{h}_min"] = min(
            result[f"retention{h}_x"], result[f"retention{h}_y"]
        )
        result[f"retention{h}_max"] = max(
            result[f"retention{h}_x"], result[f"retention{h}_y"]
        )
    # FWHM of intensity-gradient profiles sampled along local edge normals.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    norm = np.maximum(mag, 1e-9)
    nx = gx / norm
    ny = gy / norm
    yy, xx = np.indices(gray.shape, dtype="float32")
    forward = cv2.remap(mag, xx + nx, yy + ny, cv2.INTER_LINEAR)
    back = cv2.remap(mag, xx - nx, yy - ny, cv2.INTER_LINEAR)
    threshold = max(0.04, float(np.quantile(mag[inside], 0.75)))
    candidates = (
        inside
        & (mag >= threshold)
        & (mag >= forward)
        & (mag >= back)
        & (gray < 0.97)
        & (gray > 0.03)
    )
    ys, xs = np.where(candidates)
    if len(xs) > 256:
        take = np.linspace(0, len(xs) - 1, 256, dtype=int)
        xs = xs[take]
        ys = ys[take]
    widths = []
    distance = np.arange(-9, 9.01, 0.5, dtype="float32")
    if len(xs):
        mx = xs[:, None].astype("float32") + nx[ys, xs, None] * distance
        my = ys[:, None].astype("float32") + ny[ys, xs, None] * distance
        profiles = cv2.remap(
            gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
        )
        pg = np.abs(np.gradient(profiles, 0.5, axis=1))
        for values in pg:
            peak = 16 + int(np.argmax(values[16:21]))
            half = values[peak] / 2
            if values[peak] < 0.005:
                continue
            left = peak
            right = peak
            while left > 0 and values[left] > half:
                left -= 1
            while right < len(values) - 1 and values[right] > half:
                right += 1
            if left == 0 or right == len(values) - 1:
                continue
            l = left + (half - values[left]) / max(
                values[left + 1] - values[left], 1e-9
            )
            rr = (
                right
                - 1
                + (values[right - 1] - half)
                / max(values[right - 1] - values[right], 1e-9)
            )
            widths.append(float((rr - l) * 0.5))
    result.update(
        edge_count=len(widths),
        edge_width_median=float(np.median(widths)) if widths else 0.0,
        edge_width_p75=float(np.quantile(widths, 0.75)) if widths else 0.0,
    )
    result["relative_edge_width"] = result["edge_width_median"] / max(short_side, 1)
    return result


def score(features, cfg):
    x = np.array([features[k] for k in FEATURES])
    z = (x - np.array(cfg["mean"])) / np.array(cfg["scale"])
    v = float(z @ np.array(cfg["coefficients"]) + cfg["intercept"])
    return float(1 / (1 + np.exp(-np.clip(v, -50, 50))))


class QualityGate:
    def __init__(self, path):
        self.config = json.loads(open(path).read())

    def evaluate(self, rgb, mask, gear=None):
        if mask.sum() < 100:
            return {
                "status": "insufficient",
                "usable": False,
                "reason": "too_few_foreground_pixels",
            }
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            return {
                "status": "clipped",
                "usable": False,
                "reason": "gear_touches_frame_boundary",
            }
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("float32") / 255
        m = measure(gray, mask.astype("uint8"))
        p = score(m, self.config)
        minimum = self.config["minimum_short_side"].get(
            gear, min(self.config["minimum_short_side"].values())
        )
        insufficient = (
            m["edge_count"] < 8
            or m["interior_pixels"] < 50
            or m["object_short_side"] < minimum
        )
        status = (
            "uncertain"
            if insufficient
            else "blurred"
            if p >= self.config["reject_threshold"]
            else "sharp"
            if p < self.config["accept_threshold"]
            else "uncertain"
        )
        fg = gray[mask]
        exposure = float(((fg < 0.02) | (fg > 0.98)).mean())
        if exposure > 0.8:
            status = "exposure"
        return dict(
            status=status,
            usable=status == "sharp",
            blur_score=p,
            reason="insufficient_visual_evidence" if insufficient else status,
            **m,
        )
