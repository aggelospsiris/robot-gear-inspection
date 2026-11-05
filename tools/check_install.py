"""Verify required assets; optionally stop before checking GPU environments."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify_assets(root):
    entries = json.loads((root / "ASSETS_SHA256.json").read_text())
    for name, expected in entries.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}; extract the checkpoint ZIP into weights/"
            )
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Asset changed: {name}")
    print(f"All {len(entries)} model and asset checksums match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    verify_assets(ROOT)
    if args.assets_only:
        return
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    subprocess.run(
        [
            str(ROOT / ".venv-ftfsod/bin/python"),
            "-c",
            "import torch, mmcv, transformers; import mmcv.ops; "
            'assert torch.cuda.is_available(); print("Detector environment OK")',
        ],
        check=True,
    )
    print("Weights, CUDA and inference environments OK")


if __name__ == "__main__":
    main()
