# Timestamped RGB Gear Inspection

## Overview

Inspect multiple gears from sequential **RGB frames and timestamps**. The pipeline
returns tracked IDs, masks, A/B types, anomaly scores and temporal normal/abnormal
decisions, with optional heatmap overlays.

**FT-FSOD detection → SAM 3 segmentation → motion/appearance tracking → AdaptCLIP anomaly scoring → temporal voting.**

This configuration targets two specific gear types using 18 normal references
with rotation augmentation. Add the separately supplied weights to `weights/`,
then follow the steps below. No training or bank building is required.

## Step 1 — Get the source and model assets

Clone this repository:

```bash
git clone https://github.com/aggelospsiris/robot-gear-inspection.git
cd robot-gear-inspection
```

The repository includes configuration, tokenizer files, calibration and the
prebuilt normal-reference bank. **Only neural-network checkpoints are excluded
from Git.** Obtain the matching `robot_gear_inspection_weights.zip` from the
maintainer through [GitHub Issues](https://github.com/aggelospsiris/robot-gear-inspection/issues).
No public checkpoint download URL is currently provided.

Extract the ZIP into the empty `weights/` folder from the repository root:

```bash
unzip /path/to/robot_gear_inspection_weights.zip -d weights
```

The ZIP contains exactly five checkpoint files, with no enclosing directory.
The result must be `weights/sam3.pt`, not `weights/weights/sam3.pt`.
**Use the checkpoint-only ZIP matching this layout; older asset ZIPs have a
different directory structure.** Check all files after installing Python dependencies
with `.venv/bin/python tools/check_install.py --assets-only`.

### Repository layout

```text
robot-gear-inspection/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── requirements.lock.txt
├── requirements.ftfsod.lock.txt
├── ASSETS_SHA256.json               # Checksums for checkpoints and runtime assets
├── VERIFICATION.json                # Scope of the local validation
├── configs/
│   ├── stream.json                 # Pipeline settings and relative asset paths
│   └── detector/
│       ├── inference.py            # FT-FSOD/BERT architecture and local paths
│       └── manifest.json           # Detector class mapping and integrity bindings
├── assets/                         # Included in Git; no source images
│   ├── reference_bank/
│   │   ├── index.json              # 18 normal originals, 72 rotation views
│   │   ├── globals.npy
│   │   └── <parent_id>.pt          # 18 precomputed reference-feature files
│   ├── calibration/
│   │   ├── thresholds.json
│   │   ├── scoring.json
│   │   ├── preprocessing.json
│   │   ├── blur_filter.json
│   │   ├── taxonomy.json
│   │   └── manifest.csv            # Calibration lineage, not image data
│   └── tokenizer/
│       └── bert-base-uncased/
│           ├── config.json         # BERT architecture, no learned tensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── weights/                        # Checkpoints only; populated from the ZIP
│   ├── .gitkeep                    # Empty Git placeholder
│   ├── sam3.pt
│   ├── ftfsod.pth
│   ├── clip.pt
│   ├── adaptclip_s2.pth
│   └── bert.safetensors
├── src/gear_inspection/             # Inference API and pipeline
├── vendor/                         # Required upstream inference source/licenses
├── examples/
│   └── inspect_frames.py
├── tools/
│   ├── setup.sh
│   ├── check_install.py
│   ├── package_weights.py
│   └── ftfsod_worker.py
└── tests/                          # Layout and integrity regression checks
```

The reference bank contains **features**, not neural-network weights or original
images. It belongs in `assets/reference_bank/` and is needed for inference.
The repository is approximately 600 MB because those features are included.
`weights/` contains no tokenizer, reference bank, configuration or calibration.
The `.venv`, `.venv-ftfsod` and `output/` folders are created locally and ignored
by Git. Run subsequent commands from the repository root.

## Step 2 — Check the computer

Use Linux x86-64 with an NVIDIA CUDA GPU and compatible driver. Both Python
**3.12** and **3.10** must already be installed. Check them:

```bash
nvidia-smi
python3.12 --version
python3.10 --version
```

Allow space for about 5.8 GB of checkpoints, 600 MB of reference assets plus two Python environments. Installation
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
robot-gear-inspection/
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

## Step 5 — Integrate a camera stream

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
            handle_gear(
                gear.gear_id, gear.mask, gear.gear_type,
                gear.is_defective, gear.anomaly_score,
            )
        display(result.overlay_rgb)
```

Replace `camera_receiver`, `handle_gear` and `display` with your frame source,
application callback and display functions. They are placeholders, not supplied functions.
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

## Evaluation and limitations

This is a research inference prototype. On the 709-frame known-normal development
sequence it produced 15.0% false alarms among decided repeated gear observations
and one incorrect type lock. Damaged-video recall and production tracking are
not validated. Initial/blurred views may have no decision. Test with representative
normal and damaged sequences from your camera before integrating decisions into
automated inspection or control.

`assets/calibration/manifest.csv` and `assets/calibration/taxonomy.json` are small integrity records
required by calibration checks; they are not image datasets and their historical
paths are not read at inference. `ASSETS_SHA256.json` records included asset hashes.
No training images, videos, experiment reports or alternative model weights are
included. Source code is published at https://github.com/aggelospsiris/robot-gear-inspection.
Only model checkpoints are distributed separately; reference assets and calibration
are included in Git.
See `THIRD_PARTY_NOTICES.md` and vendor licenses for redistribution terms.


## Scope and licensing

This repository provides inference for the supplied A/B gear configuration.
Training scripts and tools for building new reference banks or calibrating other
products are not included. Applying the pipeline to new gear types or camera
conditions requires suitable detector weights, references and calibration.

See [LICENSE](LICENSE) for this project's license and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream components.
Model weights and dependencies retain their own license terms.

To create a checkpoint-only ZIP from a populated installation:

```bash
python tools/package_weights.py /path/to/robot_gear_inspection_weights.zip
```
