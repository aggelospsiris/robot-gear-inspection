"""Versioned, shared training/inference preprocessing for gear identity heads.

This never changes the anomaly encoder input, reference bank or its calibration.
"""

import cv2
import numpy as np

PREPROCESSING = "tight_foreground_square_v1"
METAL_PREPROCESSING = "tight_metal_core_square_v2"
SHAPE_NAMES = [
    "minor_major_ratio",
    "solidity",
    "rotated_extent",
    "component_fraction",
    "width_q10",
    "width_q50",
    "width_q90",
    "central_end_width_ratio",
]


def compact_foreground(rgb, mask):
    """Remove tray appendages from a SAM proposal using its compact dark core.

    Camera-specific metal/tabletop prior; the intensity cap and minimum core
    depth match the existing proposal baseline. It never modifies anomaly input.
    Specular highlights are retained by a small dilation of the core convex hull.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None, dict(reason="empty_foreground")
    full_mask = mask
    yy, xx = np.where(mask)
    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1
    rgb = rgb[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    level, _ = cv2.threshold(gray[mask], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = (mask & (gray <= min(125, level))).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), "uint8"))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    candidates = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < 50:
            continue
        component = (labels == j).astype("uint8")
        depth = float(cv2.distanceTransform(component, cv2.DIST_L2, 5).max())
        if depth >= 3.5:
            candidates.append((depth * np.sqrt(stats[j, cv2.CC_STAT_AREA]), j, depth))
    if not candidates:
        return None, dict(
            reason="no_compact_foreground", intensity_threshold=float(min(125, level))
        )
    _, j, depth = max(candidates)
    ys, xs = np.where(labels == j)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype("int32"))
    support = np.zeros(mask.shape, "uint8")
    cv2.fillConvexPoly(support, hull, 1)
    support = cv2.dilate(support, np.ones((5, 5), "uint8")) > 0
    local_refined = mask & support
    refined = np.zeros_like(full_mask)
    refined[y0:y1, x0:x1] = local_refined
    return refined, dict(
        reason="compact_foreground",
        intensity_threshold=float(min(125, level)),
        core_depth=depth,
        retained_mask_fraction=float(refined.sum() / full_mask.sum()),
    )


def identity_inputs(rgb, mask, preprocessing):
    info = {}
    if preprocessing == METAL_PREPROCESSING:
        mask, info = compact_foreground(rgb, mask)
        if mask is None:
            return None, None, info
    elif preprocessing != PREPROCESSING:
        raise ValueError("Unsupported identity preprocessing")
    return classifier_crop(rgb, mask), shape_features(mask), info


def classifier_crop(rgb, mask):
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if rgb.shape[:2] != mask.shape or not len(xs):
        raise ValueError("Classifier needs a nonempty mask aligned with RGB")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.where(mask[y0:y1, x0:x1, None], rgb[y0:y1, x0:x1], 127).astype("uint8")
    h, w = crop.shape[:2]
    size = max(h, w)
    margin = max(2, round(size * 0.05))
    out = np.full((size + 2 * margin, size + 2 * margin, 3), 127, dtype="uint8")
    y, x = (out.shape[0] - h) // 2, (out.shape[1] - w) // 2
    out[y : y + h, x : x + w] = crop
    return out


def shape_features(mask):
    """Scale/rotation invariant silhouette evidence; no absolute object-size cue."""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        raise ValueError("Insufficient foreground for shape features")
    points = np.stack([xs, ys], 1).astype("float32")
    values, vectors = np.linalg.eigh(np.cov(points.T))
    principal = (points - points.mean(0)) @ vectors
    major = principal[:, 1]
    minor = principal[:, 0]
    lo, hi = major.min(), major.max()
    widths = []
    for a, b in zip(np.linspace(lo, hi, 13)[:-1], np.linspace(lo, hi, 13)[1:]):
        v = minor[(major >= a) & (major <= b)]
        widths.append(float(v.max() - v.min() + 1) if len(v) else 0)
    widths = np.asarray(widths) / max(float(minor.max() - minor.min() + 1), 1)
    hull = cv2.convexHull(points)
    hull_area = max(cv2.contourArea(hull), 1)
    rect = cv2.minAreaRect(points)[1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype="uint8"))
    area = len(xs)
    result = np.array(
        [
            np.sqrt(max(values[0], 0) / max(values[1], 1e-9)),
            min(area / hull_area, 1),
            min(area / max(rect[0] * rect[1], 1), 1),
            stats[1:, cv2.CC_STAT_AREA].max() / area,
            *np.quantile(widths, [0.1, 0.5, 0.9]),
            np.mean(widths[4:8]) / max(np.mean(widths[[0, 1, 10, 11]]), 0.05),
        ],
        dtype="float32",
    )
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite silhouette features")
    return result


def head_features(global_vector, shape, spec):
    q = np.asarray(global_vector, dtype="float64").reshape(-1)
    q /= max(np.linalg.norm(q), 1e-9)
    geometry = np.clip(
        (np.asarray(shape) - spec["shape_mean"])
        / np.maximum(spec["shape_scale"], 1e-6),
        -5,
        5,
    )
    return np.r_[q, float(spec["shape_weight"]) * geometry]


def head_probability(global_vector, shape, spec):
    x = head_features(global_vector, shape, spec)
    z = (x @ np.asarray(spec["coefficients"]) + spec["intercept"]) / spec.get(
        "temperature", 1
    )
    return float(1 / (1 + np.exp(-np.clip(z, -50, 50))))


def validate_head(spec):
    if spec.get("version") != "clip_shape_identity_v1" or spec.get(
        "preprocessing"
    ) not in [PREPROCESSING, METAL_PREPROCESSING]:
        raise ValueError("Unsupported identity head")
    if spec.get("shape_names") != SHAPE_NAMES:
        raise ValueError("Identity silhouette feature order differs")
    for name in ["shape_mean", "shape_scale"]:
        x = np.asarray(spec[name], dtype=float)
        if x.shape != (len(SHAPE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid identity normalization")
    if np.any(np.asarray(spec["shape_scale"]) <= 0):
        raise ValueError("Identity feature scale must be positive")
    coef = np.asarray(spec["coefficients"], dtype=float)
    if (
        coef.shape != (spec["global_dimensions"] + len(SHAPE_NAMES),)
        or not np.isfinite(coef).all()
    ):
        raise ValueError("Invalid identity coefficients")
    if (
        not np.isfinite(
            [
                spec["temperature"],
                spec["intercept"],
                spec["shape_weight"],
                spec["minimum_probability"],
            ]
        ).all()
        or spec["temperature"] <= 0
        or not 0 < spec["minimum_probability"] < 1
    ):
        raise ValueError("Invalid identity probability policy")
