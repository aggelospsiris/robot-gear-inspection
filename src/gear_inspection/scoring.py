"""Versioned image-score pooling, shared by calibration and runtime."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


LEGACY_SCORING = {
    "version": "heatmap_max_v1",
    "pooling": "heatmap_max",
    "smoothing_sigma": 4,
    "global_fusion": "three_head_arithmetic_mean",
    "image_fusion": "harmonic_mean_global_local",
    "spatial_support": "entire_padded_crop",
}
TOP10_SCORING = {
    **LEGACY_SCORING,
    "version": "top10_percent_patch_mean_v1",
    "pooling": "top_fraction_patch_mean",
    "fraction": 0.10,
    "patch_grid": [37, 37],
    "patch_values": "area_average_of_smoothed_heatmap",
    "rounding": "ceil",
}


def load_scoring(assets):
    path = Path(assets) / "scoring.json"
    rule = json.loads(path.read_text()) if path.exists() else dict(LEGACY_SCORING)
    if rule not in (LEGACY_SCORING, TOP10_SCORING):
        raise ValueError(
            "Unsupported image scoring rule; select a versioned rule and recalibrate"
        )
    return rule


def scoring_sha256(rule):
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()


def validate_scoring(thresholds, rule):
    if thresholds.get("scoring_rule", LEGACY_SCORING) != rule:
        raise ValueError(
            "Image scoring rule changed; recalibrate thresholds before inference"
        )


def pool_local_score(maps, rule):
    """Pool B×H×W smoothed heatmaps into one local score per image.

    A 518×518 heatmap consists of 37×37 cells of 14×14 pixels. Area averaging
    gives equal weight to each patch before selecting ceil(0.10*1369)=137.
    The CLS token is absent from these spatial maps.
    """
    values = torch.as_tensor(maps)
    if values.ndim != 3 or not torch.isfinite(values).all():
        raise ValueError("Expected finite B×H×W heatmaps")
    if rule == LEGACY_SCORING:
        return values.flatten(1).max(1).values
    if rule != TOP10_SCORING:
        raise ValueError("Unsupported local pooling rule")
    patches = F.adaptive_avg_pool2d(values[:, None], tuple(rule["patch_grid"])).flatten(
        1
    )
    k = max(1, math.ceil(rule["fraction"] * patches.shape[1]))
    return patches.topk(k, dim=1, sorted=False).values.mean(1)
