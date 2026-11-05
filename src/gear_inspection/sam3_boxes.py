"""SAM3 box-prompt segmentation on ordered RGB frames with external ID memory.

SAM3's image box predictor segments every prompted frame. Public IDs come from
ShortGapIdentity, not a SAM2 model or SAM3 native video-memory implementation.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .reidentification import ShortGapIdentity, overlap


def mask_bbox(mask):
    y, x = np.where(mask)
    return (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))


class SAM3BoxPredictor:
    def __init__(self, checkpoint, device="cuda:0", image_size=1008):
        from ultralytics import SAM

        self.device = device
        self.image_size = image_size
        self.checkpoint = str(Path(checkpoint).resolve())
        self.load_report = []
        original = torch.nn.Module.load_state_dict

        def capture(module, state, *args, **kwargs):
            result = original(module, state, *args, **kwargs)
            if type(module).__name__ == "SAM3Model":
                self.load_report.append(
                    {
                        "missing": list(result.missing_keys),
                        "unused_checkpoint_tensors": len(result.unexpected_keys),
                    }
                )
            return result

        with patch.object(torch.nn.Module, "load_state_dict", capture):
            self.model = SAM(self.checkpoint)
        if len(self.load_report) != 1 or self.load_report[0]["missing"]:
            raise RuntimeError(
                f"Incomplete SAM3 interactive weights: {self.load_report}"
            )

    @torch.inference_mode()
    def predict(self, rgb, boxes):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB")
        if not len(boxes):
            return np.zeros((0, *rgb.shape[:2]), bool), []
        boxes = np.asarray(boxes, float)
        if boxes.shape != (len(boxes), 4) or not np.isfinite(boxes).all():
            raise ValueError("Invalid box prompts")
        result = self.model.predict(
            source=rgb[:, :, ::-1].copy(),
            bboxes=boxes.tolist(),
            device=self.device,
            imgsz=self.image_size,
            conf=0.0,
            quantize=16,
            retina_masks=True,
            save=False,
            verbose=False,
        )[0]
        if result.masks is None:
            raise RuntimeError("SAM3 returned no masks for valid box prompts")
        order = np.argsort(result.boxes.cls.cpu().numpy())
        prompt_ids = result.boxes.cls.cpu().numpy()[order]
        if not np.array_equal(prompt_ids, np.arange(len(boxes))):
            raise RuntimeError("SAM3 changed box-prompt correspondence")
        masks = result.masks.data.cpu().numpy()[order].astype(bool)
        scores = result.boxes.conf.float().cpu().numpy()[order]
        if masks.shape != (len(boxes), *rgb.shape[:2]) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid SAM3 masks or mask scores")
        return masks, scores.tolist()


class SAM3BoxStream:
    def __init__(
        self,
        checkpoint,
        device="cuda:0",
        max_objects=12,
        proposal=None,
        reidentification=None,
        image_size=1008,
    ):
        if not getattr(proposal, "typed", False) or reidentification is None:
            raise ValueError(
                "SAM3 box stream needs typed proposals and identity memory"
            )
        self.predictor = SAM3BoxPredictor(checkpoint, device, image_size)
        self.proposal = proposal
        self.reidentifier = ShortGapIdentity(
            max_objects=max_objects, **reidentification
        )
        self.reset()

    def reset(self):
        self.index = -1
        self.next_id = 1
        self.timestamp = None
        self.last_detection = {}
        self.typed_detections = {}
        self.mask_scores = {}
        self.reidentifier.reset()
        if hasattr(self.proposal, "calls"):
            self.proposal.calls = 0
        if hasattr(self.proposal, "last_detection_timestamp"):
            self.proposal.last_detection_timestamp = None

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def step(self, rgb, boxes=None):
        if boxes is not None:
            raise ValueError("Use the typed proposal source for SAM3 prompts")
        self.index += 1
        self.reidentifier.begin_frame(rgb, self.timestamp)
        self.reidentifier.next_id = max(self.reidentifier.next_id, self.next_id)
        candidates = self.proposal(rgb, timestamp=self.timestamp)
        for d in candidates:
            d.validate(rgb.shape)
        matches = self.reidentifier.associate(rgb, [d.bbox for d in candidates])
        self.next_id = self.reidentifier.next_id
        fresh_ids = set()
        for i, key in matches.items():
            detection = candidates[i]
            fresh_ids.add(key)
            self.last_detection[key] = self.index
            self.typed_detections[key] = {
                "frame_index": self.index,
                "bbox": list(detection.bbox),
                "gear_type": detection.gear_type,
                "score": detection.score,
            }
        prompts = []
        keys = []
        for key, memory in self.reidentifier.memory.items():
            box = memory.box.copy()
            box[0::2] = np.clip(box[0::2], 0, rgb.shape[1])
            box[1::2] = np.clip(box[1::2], 0, rgb.shape[0])
            if min(box[2:] - box[:2]) < 3:
                continue
            if key not in fresh_ids:
                feature = self.reidentifier.appearance(rgb, box)
                if feature @ memory.appearance < self.reidentifier.minimum_appearance:
                    continue
            keys.append(key)
            prompts.append(box)
        masks, scores = self.predictor.predict(rgb, prompts)
        result = {}
        self.mask_scores = {}
        for key, mask, score, box in zip(keys, masks, scores, prompts):
            if (
                mask.sum() < 500
                or overlap(np.asarray(mask_bbox(mask), float), np.asarray(box, float))
                < 0.15
            ):
                continue
            result[key] = mask
            self.mask_scores[key] = score
        for key in list(self.typed_detections):
            if key not in self.reidentifier.memory:
                self.typed_detections.pop(key, None)
                self.last_detection.pop(key, None)
        return result

    def remove(self, key):
        self.reidentifier.forget(key)
        self.typed_detections.pop(key, None)
        self.last_detection.pop(key, None)

    def close(self):
        self.reset()
        if hasattr(self.proposal, "close"):
            self.proposal.close()
