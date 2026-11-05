"""Create and verify a checkpoint-only ZIP without bundling reference data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    "sam3.pt",
    "ftfsod.pth",
    "clip.pt",
    "adaptclip_s2.pth",
    "bert.safetensors",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = json.loads((ROOT / "ASSETS_SHA256.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name in CHECKPOINTS:
            archive.write(ROOT / "weights" / name, name)
    with zipfile.ZipFile(args.output) as archive:
        assert set(archive.namelist()) == set(CHECKPOINTS)
        for name in CHECKPOINTS:
            with archive.open(name) as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected[f"weights/{name}"]:
                raise RuntimeError(f"Checkpoint ZIP failed checksum: {name}")
    print(
        f"Verified checkpoint-only ZIP: {args.output} ({args.output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
