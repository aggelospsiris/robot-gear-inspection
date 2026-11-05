"""Bind corrected calibration labels to their resulting thresholds."""

import hashlib
import json
from pathlib import Path


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_taxonomy(assets, thresholds):
    profile = Path(assets).parent
    path = profile / "taxonomy.json"
    if not path.is_file():
        if "taxonomy_sha256" in thresholds:
            raise ValueError("Thresholds require a missing taxonomy file")
        return
    taxonomy = json.loads(path.read_text())
    if (
        thresholds.get("taxonomy_sha256") != file_sha256(path)
        or thresholds.get("taxonomy_version") != taxonomy["version"]
    ):
        raise ValueError(
            "Taxonomy changed or is not calibrated; recalibrate thresholds"
        )
    manifest = profile / "manifest.csv"
    if not manifest.is_file() or thresholds.get(
        "calibration_manifest_sha256"
    ) != file_sha256(manifest):
        raise ValueError("Calibration labels changed; recalibrate thresholds")
