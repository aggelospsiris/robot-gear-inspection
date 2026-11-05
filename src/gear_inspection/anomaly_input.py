"""Versioned masked anomaly input shared by preparation and live inference."""

import hashlib
import json
from pathlib import Path

import numpy as np


def masked_gear(crop, mask, background=127):
    if (
        crop.dtype != np.uint8
        or crop.ndim != 3
        or crop.shape[2] != 3
        or mask.shape != crop.shape[:2]
        or not mask.any()
    ):
        raise ValueError("Expected RGB crop and a nonempty matching mask")
    return np.where(mask[:, :, None], crop, background).astype(np.uint8)


def load_preprocessing(assets):
    path = Path(assets) / "preprocessing.json"
    if not path.exists():
        return None
    spec = json.loads(path.read_text())
    if (
        spec["version"] != "sam3_masked_gear_v1"
        or spec["background_rgb"] != [127, 127, 127]
        or spec["crop_padding"] != 0.18
    ):
        raise ValueError("Unsupported anomaly input preprocessing")
    return spec


def preprocessing_sha(assets):
    return hashlib.sha256(
        (Path(assets) / "preprocessing.json").read_bytes()
    ).hexdigest()
