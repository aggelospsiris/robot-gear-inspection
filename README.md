#Gear Inspection

Multi-object gear inspection from sequential RGB frames and timestamps. Combines
[FT-FSOD](https://github.com/Intellindust-AI-Lab/FT-FSOD) detection, [SAM 3](https://github.com/facebookresearch/sam3) segmentation, motion/appearance tracking and [AdaptCLIP](https://github.com/gaobb/AdaptCLIP)
anomaly scoring with temporal voting.

Returns instance IDs, masks, A/B gear types, normal/abnormal decisions and heatmaps.
The released configuration uses 18 normal reference images with rotation
augmentation (72 views). Reference features and calibration are included;
no training is required for inference.

## Installation

Requirements: NVIDIA CUDA GPU, Python 3.12 and Python 3.10. The detector
uses a separate environment for its compiled MMCV operators.

```bash
git clone https://github.com/aggelospsiris/robot-gear-inspection.git
cd robot-gear-inspection
```

## Checkpoints

Download the checkpoints from the [weights folder on SharePoint](https://itmlgr.sharepoint.com/:f:/r/sites/tornado.eu/Shared%20Documents/06.%20TORNADO%20-%20Work%20Packages/TORNADO%20-%20WP4/UC1_anomaly_detection_weights?d=we4be14988f324919a5b9802534849f2e&csf=1&web=1&e=VJnnKT)
and place the five files directly in `weights/`. Access is subject to the folder's
SharePoint permissions.

Then install dependencies:

```bash
bash tools/setup.sh
```

```text
weights/
├── sam3.pt
├── ftfsod.pth
├── clip.pt
├── adaptclip_s2.pth
└── bert.safetensors
```

Only model checkpoints belong in `weights/`. Reference features, calibration and
tokenizer files are provided in `assets/`; pipeline settings are in
[`configs/stream.json`](configs/stream.json).

## Inference

Prepare a CSV containing image paths and increasing timestamps in seconds:

```csv
path,timestamp_seconds
frames/0001.png,0.000
frames/0002.png,0.033
```

```bash
.venv/bin/python examples/inspect_frames.py frames.csv --output output
```

Outputs are annotated images, masks/heatmaps in `.npz` files, and per-frame
scores, decisions and votes in `predictions.jsonl`.

For a camera stream:

```python
from gear_inspection import Inspector

with Inspector.from_config("configs/stream.json") as inspector:
    for rgb, timestamp in frame_source:  # Your frame iterator
        result = inspector.process_frame(rgb, timestamp, render=True)
        for gear in result.gears:
            print(gear.gear_id, gear.gear_type, gear.is_defective)
```

Input frames must be H×W×3 `uint8` RGB with fixed resolution and strictly
increasing timestamps. `is_defective` is `False` for normal, `True` for abnormal,
and `None` while collecting evidence. Do not treat `None` as normal.
Run integration scripts with `.venv/bin/python` from the repository root.

See [usage details](docs/usage.md) for the API, folder layout and inference settings.

## Evaluation

On a 709-frame known-normal development sequence, the supplied configuration
produced 15.0% false alarms among decided gear observations and one wrong type
lock. These are correlated observations, not independent test accuracy.
Damaged-video recall is unverified. The configuration targets the supplied A/B
gear types; training and recalibration for other products are outside this release.

## Acknowledgements

Built on [FT-FSOD](https://github.com/Intellindust-AI-Lab/FT-FSOD),
[SAM 3](https://github.com/facebookresearch/sam3) and
[AdaptCLIP](https://github.com/gaobb/AdaptCLIP).
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
