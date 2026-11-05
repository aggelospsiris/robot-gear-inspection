# Gear inspection for the robot

Input: one **RGB frame and timestamp in seconds** at a time. Output: gear IDs,
SAM 3 masks, A/B type, anomaly score, normal/abnormal decision and optional heatmap
overlay. No runtime training, dataset download or reference-bank building is needed.
All required model weights and the prebuilt normal bank are included in `weights/`.
They are shipped in the handover archive and intentionally excluded from Git
history; a source-only Git clone does not contain the weights.

## Install once

Use Linux x86-64 with an NVIDIA CUDA GPU, a compatible NVIDIA driver, Python 3.12
and Python 3.10, and enough disk space for two environments plus about 6 GB of
weights. Installation needs internet; inference uses local files. The detector
needs its own Python 3.10 environment for compiled MMCV compatibility.

From this folder:

```bash
bash tools/setup.sh
```

The setup installs the pinned Python 3.12 inference environment and Python 3.10
FT-FSOD worker environment. It uses the CUDA 12.1/PyTorch 2.1 MMCV wheel index for
the detector. If the partner platform differs, those compiled packages must be
matched to that platform; do not replace MMCV with mmcv-lite.

## Connect to the robot

Run from this folder after installation:

```python
from pathlib import Path
from gear_inspection import Inspector

config = Path("configs/stream.json").resolve()
with Inspector.from_config(config) as inspector:
    for rgb, timestamp_seconds in camera_receiver:
        result = inspector.process_frame(rgb, timestamp_seconds, render=True)
        for gear in result.gears:
            send_to_robot(
                gear.gear_id, gear.mask, gear.gear_type,
                gear.is_defective, gear.anomaly_score,
            )
        display(result.overlay_rgb)
```

`camera_receiver`, `send_to_robot` and `display` are placeholders for your camera,
robot interface and display. Each `rgb` must be a NumPy `uint8` array, H×W×3,
**RGB channel order**. OpenCV cameras supply BGR: convert with
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. Timestamps must be finite, strictly
increasing seconds, preferably from the camera. Keep resolution fixed within a
session and use one Inspector per stream. Use a fresh Inspector for a new session.
The context manager releases the detector worker and tracking state.

- `gear.mask`: boolean H×W instance mask in original frame coordinates.
- `gear.gear_id`: session-local tracking ID; it may change after a long loss.
- `gear.gear_type`: `"A"`, `"B"`, or `None` while collecting type evidence.
- `gear.is_defective`: `False` = normal, `True` = abnormal, `None` = no decision yet.
  **Never turn `None` into normal.**
- `gear.anomaly_score`: current score or `None` if not scored; not a probability.
- `gear.heatmap`: H×W float map, zero outside its mask, or `None` when not scored.
- `gear.quality`, `normal_votes`, `anomalous_votes`: quality and evidence details.
- `result.metadata()`: JSON-compatible results without image arrays.

A rejected view can retain a previous temporal decision but contributes no new
score or vote. Zero heatmap arrays in the file example indicate missing maps when
the corresponding JSON score is null; they are not zero anomaly scores.

## Test your own saved frames

Create `frames.csv` with the header and one image per row, in timestamp order:

```csv
path,timestamp_seconds
frames/0001.png,0.000
frames/0002.png,0.033
```

```bash
.venv/bin/python examples/inspect_frames.py frames.csv --output output
```

The output contains annotated JPEGs, IDs/masks/heatmaps in compressed NumPy files,
and scores/decisions/votes in `predictions.jsonl`. No sample datasets are bundled.

## Fixed inference settings

FT-FSOD detects A/B boxes every frame at confidence ≥0.30. SAM 3 segments those
boxes; separate motion/appearance tracking maintains IDs across short gaps.
After ≥15 usable detections and ≥50% confidence-weighted agreement, with a strict
winning class, the type locks for that ID. The model never assigns type from
left/right position. This is SAM 3 image segmentation with external ID tracking,
not SAM 3 native video-memory tracking or a text-prompt detector.

Blurred, uncertain, clipped and overlapping views do not create anomaly evidence.
Only masked gear crops enter AdaptCLIP S2; background becomes neutral gray.
The bank uses 11 normal A and seven normal B originals, each at four rotations
(0/90/180/270): **72 reference views**. No extra augmentations are enabled.
The model retrieves 16 same-type views. The local score averages the highest
10% of a smoothed 37×37 patch grid across the padded crop, then is harmonically
fused with the mean of the three global heads. Thresholds are **A 0.0627993718**,
**B 0.1288404465**. At least five distinct usable anomaly samples spanning 0.6s,
sampled ≥0.15s apart, are required; cumulative normal/abnormal majority decides,
with mean score breaking a vote tie. Type locking and anomaly voting are separate.

`configs/stream.json` is the single deployment configuration. Paths are relative
to that file, so the folder can be moved. Use `device` to select a GPU. Preserve
the `.venv-ftfsod/bin/python` executable path: resolving its symlink loses its
virtual environment. Weights/calibration fingerprints are checked at load time;
do not edit thresholds, bank or scoring independently.

## Validation status

This is a partner testing prototype. On the 709-frame known-normal development
sequence it produced 15.0% false alarms among decided repeated gear observations
and one incorrect type lock. Damaged-video recall and production tracking are
not validated. Initial/blurred views may have no decision. Test with representative
robot-camera normal and damaged sequences before using decisions for robot action.

`weights/manifest.csv` and `weights/taxonomy.json` are small integrity records
required by calibration checks; they are not image datasets and their historical
paths are not read at inference. `WEIGHTS_SHA256.json` records included asset hashes.
No training images, videos, experiment reports or alternative model weights are
included. This repository is local; nothing has been published or sent externally.
See `THIRD_PARTY_NOTICES.md` and vendor licenses for redistribution terms.
