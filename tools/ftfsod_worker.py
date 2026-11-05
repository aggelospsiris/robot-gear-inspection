"""Persistent, inference-only FT-FSOD worker (run with .venv-ftfsod Python).

JSON lines on stdin/stdout; an mmap RGB file carries one frame at a time.
OpenMMLab messages go to stderr. No dataset, optimizer, bank building, or future
frames are needed at inference. Errors are explicit; there is no CLIP fallback.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/FT-FSOD"))


class Detector:
    def __init__(self, config, checkpoint, device, threshold=0.3, nms_iou=0.4):
        import torch
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        torch.set_num_threads(3)
        register_all_modules()
        self.model = init_detector(config, checkpoint, device=device, palette="coco")
        if tuple(self.model.dataset_meta["classes"]) != ("gear type A", "gear type B"):
            raise ValueError("Checkpoint class order must be gear type A, gear type B")
        self.pipeline = Compose(
            [
                dict(type="LoadImageFromNDArray"),
                dict(
                    type="FixScaleResize",
                    scale=(800, 1333),
                    keep_ratio=True,
                    backend="pillow",
                ),
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "text",
                        "custom_entities",
                    ),
                ),
            ]
        )
        self.threshold = threshold
        self.nms_iou = nms_iou

    def predict(self, rgb):
        import torch
        from mmdet.apis import inference_detector
        from torchvision.ops import nms

        with torch.inference_mode():
            # mmdetection NDArray API takes BGR, preprocessing converts it to RGB.
            result = inference_detector(
                self.model,
                rgb[:, :, ::-1].copy(),
                test_pipeline=self.pipeline,
                text_prompt=("gear type A", "gear type B"),
                custom_entities=True,
            ).pred_instances
        boxes = result.bboxes.detach().float().cpu()
        scores = result.scores.detach().float().cpu()
        labels = result.labels.detach().cpu()
        h, w = rgb.shape[:2]
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        valid = (
            (scores >= self.threshold)
            & (labels < 2)
            & torch.isfinite(boxes).all(1)
            & ((boxes[:, 2:] - boxes[:, :2]) > 1).all(1)
        )
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        # Suppress cross-class duplicates as well as duplicates within a class.
        keep = nms(boxes, scores, self.nms_iou).tolist()
        return [
            dict(
                bbox=[float(v) for v in boxes[i]],
                gear_type="AB"[int(labels[i])],
                score=float(scores[i]),
                class_name=("gear type A", "gear type B")[int(labels[i])],
            )
            for i in keep
        ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.4)
    args = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        detector = Detector(
            args.config, args.checkpoint, args.device, args.threshold, args.nms_iou
        )
    print(
        json.dumps(dict(ready=True, classes=["gear type A", "gear type B"])), flush=True
    )
    import numpy as np

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close"):
                break
            shape = tuple(request["shape"])
            if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 16:
                raise ValueError("Invalid RGB shape")
            rgb = np.memmap(request["rgb_file"], dtype="uint8", mode="r", shape=shape)
            with contextlib.redirect_stdout(sys.stderr):
                detections = detector.predict(rgb)
            del rgb
            print(
                json.dumps(
                    dict(request_id=request["request_id"], detections=detections)
                ),
                flush=True,
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=str(error))), flush=True)


if __name__ == "__main__":
    main()
