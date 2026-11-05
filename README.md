# Gear inspection for the robot

## Overview

This package inspects multiple gears in sequential camera frames. Supply one
**RGB frame and its timestamp** per call. For each visible gear, it returns an
ID, segmentation mask, A/B type, anomaly score and, after collecting enough
usable observations, a **normal or abnormal** decision. It can also produce an
overlay showing masks, heatmaps and vote counts.

The pipeline runs in this order:

1. **Detect:** FT-FSOD finds gears and predicts their A/B type.
2. **Segment and track:** SAM 3 produces masks; motion/appearance tracking maintains IDs.
3. **Confirm type:** collect usable detections, then lock A/B for each retained ID.
4. **Inspect:** reject poor-quality views and score the masked gear with AdaptCLIP S2.
5. **Vote:** combine observations over time before returning an anomaly decision.

The supplied version uses the selected rotation-only normal-reference bank.
All required weights and the prebuilt bank are included in the handover archive.
**No training, dataset download or bank building is required to run inference.**
Weights are excluded from Git history, so a source-only clone is insufficient.

Follow steps 1–6 below to install, test and integrate it. Detailed inference
settings and validation limits are at the end.

## Step 1 — Extract the package

Copy `robot_gear_inspection_handover.tar` to the robot computer, then run:

```bash
tar -xf robot_gear_inspection_handover.tar
cd robot_gear_inspection
```

Run the remaining commands from this folder. It contains:

| Path | Purpose |
| --- | --- |
| `configs/stream.json` | Single inference configuration |
| `weights/` | Detector, SAM 3, AdaptCLIP and normal-reference bank |
| `src/gear_inspection/` | Frame-processing API |
| `examples/inspect_frames.py` | Saved-frame example |
| `tools/setup.sh` | Environment installation |
| `tools/check_install.py` | Weights and environment verification |

## Step 2 — Check the computer

Use Linux x86-64 with an NVIDIA CUDA GPU and compatible driver. Both Python
**3.12** and **3.10** must already be installed. Check them:

```bash
nvidia-smi
python3.12 --version
python3.10 --version
```

Allow space for about 6 GB of weights plus two Python environments. Installation
needs internet; inference uses local files. Python 3.12 runs the main pipeline.
The detector runs in a separate Python 3.10 environment for compiled MMCV
compatibility.

## Step 3 — Install and verify

```bash
bash tools/setup.sh
```

This creates `.venv` for inference and `.venv-ftfsod` for the detector, installs
pinned dependencies, and checks the supplied weights and CUDA environments.
Wait for **“Weights, CUDA and inference environments OK”** before continuing.
To repeat that check later:

```bash
.venv/bin/python tools/check_install.py
```

The detector installation uses the CUDA 12.1/PyTorch 2.1 MMCV wheel index. On a
different platform, its compiled packages must be matched to that platform;
`mmcv-lite` cannot replace MMCV's required compiled operators.

## Step 4 — Test a sequence of saved frames

Prepare your own RGB image sequence. For example:

```text
robot_gear_inspection/
  frames/
    0001.png
    0002.png
    ...
  frames.csv
```

Create `frames.csv` with one image and capture timestamp per row, in time order:

```csv
path,timestamp_seconds
frames/0001.png,0.000
frames/0002.png,0.033
```

Paths in this CSV are relative to the CSV file, or can be absolute. Use enough
frames to collect type and anomaly evidence; two rows only illustrate the format.
No example image dataset is bundled.

Run:

```bash
.venv/bin/python examples/inspect_frames.py frames.csv --output output
```

Inspect the generated files:

| Output | Contents |
| --- | --- |
| `output/frame_000001.jpg` | Overlay with masks, heatmaps, scores and votes |
| `output/frame_000001.npz` | Corresponding `ids`, `masks` and `heatmaps` arrays |
| `output/predictions.jsonl` | Per-frame types, scores, decisions, quality and votes |

Initial frames may have no decision while evidence is collected. This is expected.

## Step 5 — Connect the camera and robot

Create an integration script using the following API. Run that script with
`.venv/bin/python your_script.py` from this folder.

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

Replace `camera_receiver`, `send_to_robot` and `display` with your camera, robot
interface and display functions. They are placeholders, not supplied functions.
Use `render=False` when no visual overlay is needed.

Before calling `process_frame`, ensure:

1. The frame is a NumPy `uint8` array with shape **H×W×3 in RGB order**.
   For an OpenCV BGR frame, use `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.
2. The timestamp is finite and in **seconds**, increasing strictly with each call.
   Prefer camera timestamps; convert milliseconds to seconds if necessary.
3. Frame resolution stays fixed within the session.
4. Each camera stream has its own Inspector, called sequentially. Start a fresh
   Inspector for a new session.

The context manager releases the detector worker and tracking state when finished.

## Step 6 — Read the results correctly

| Field | Meaning |
| --- | --- |
| `gear.gear_id` | Session-local tracking ID; may change after a long loss |
| `gear.mask` | Boolean H×W mask in original-frame coordinates |
| `gear.gear_type` | `"A"`, `"B"`, or `None` while collecting evidence |
| `gear.is_defective` | `False`: normal; `True`: abnormal; `None`: no decision yet |
| `gear.anomaly_score` | Current image score, or `None` when not scored; not a probability |
| `gear.heatmap` | H×W float map, zero outside the mask, or `None` when not scored |
| `gear.quality` | Quality status and whether this observation is usable |
| `gear.normal_votes`, `gear.anomalous_votes` | Accumulated anomaly votes |
| `result.metadata()` | JSON-compatible results without image arrays |

**Never interpret `None` as normal.** Wait for a decision before using it.
A rejected view can retain a previous temporal decision, but adds no new score
or vote. In the saved-frame example, a missing heatmap is represented by a zero
array and a null JSON score; it is not a zero anomaly score.

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
included. Source code is published at https://github.com/aggelospsiris/robot-gear-inspection.
The weights are distributed separately in the handover archive.
See `THIRD_PARTY_NOTICES.md` and vendor licenses for redistribution terms.
