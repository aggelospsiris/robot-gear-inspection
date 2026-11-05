"""Check the portable layout without loading CUDA models."""

import json
from pathlib import Path
import sys
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gear_inspection.runtime import resolve_paths, validate_assets
from gear_inspection.taxonomy import validate_taxonomy


def test_config_resolves_independently_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = json.loads((ROOT / "configs/stream.json").read_text())
    result = resolve_paths(ROOT / "configs/stream.json", cfg)
    assert result["weights"] == ROOT / "weights"
    assert result["calibration"] == ROOT / "assets/calibration"
    assert result["reference_bank"] == ROOT / "assets/reference_bank"


def test_only_five_neural_checkpoints_belong_in_weights():
    expected = {
        "sam3.pt",
        "ftfsod.pth",
        "clip.pt",
        "adaptclip_s2.pth",
        "bert.safetensors",
    }
    checksums = json.loads((ROOT / "ASSETS_SHA256.json").read_text())
    assert {Path(p).name for p in checksums if p.startswith("weights/")} == expected
    assert all(
        p.name == ".gitkeep" or p.name in expected for p in (ROOT / "weights").iterdir()
    )
    assert len(list((ROOT / "assets/reference_bank").glob("*.pt"))) == 18
    assert (ROOT / "assets/tokenizer/bert-base-uncased/tokenizer.json").is_file()


def test_calibration_fingerprints_survive_layout_change():
    cal = ROOT / "assets/calibration"
    threshold = json.loads((cal / "thresholds.json").read_text())
    validate_taxonomy(cal, threshold)
    assert threshold["thresholds"] == {"A": 0.062799371778965, "B": 0.12884044647216797}
    # Missing or edited metadata must still fail, even without CUDA.
    mutated = dict(threshold, calibration_manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="Calibration labels"):
        validate_taxonomy(cal, mutated)


def test_bank_index_records_are_resolved_inside_reference_directory():
    bank = ROOT / "assets/reference_bank"
    data = json.loads((bank / "index.json").read_text())
    assert len(data["records"]) == 72
    for row in data["records"]:
        path = bank / row["file"]
        assert path.is_file() and path.resolve().is_relative_to(bank.resolve())
