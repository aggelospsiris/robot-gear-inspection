"""Typed detection, one-to-one association and an isolated FT-FSOD client."""

from dataclasses import dataclass
import atexit
import json
from pathlib import Path
import selectors
import subprocess
import tempfile
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TypedDetection:
    bbox: tuple
    gear_type: str
    score: float

    def validate(self, shape):
        h, w = shape[:2]
        b = self.bbox
        if (
            len(b) != 4
            or not all(np.isfinite(b))
            or not np.isfinite(self.score)
            or not 0 <= self.score <= 1
            or self.gear_type not in ("A", "B")
            or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h)
        ):
            raise ValueError("Invalid typed gear detection")


def associate_detections(detections, track_boxes, minimum_iou=0.15):
    """Each typed detection can confirm at most one track (class independent).

    Maximize valid match count before IoU; prevents two detections lending
    contradictory labels to one SAM mask. A/B class must not determine IDs.
    """
    from .segmentation import iou

    ids = list(track_boxes)
    if not detections or not ids:
        return {}, list(range(len(detections)))
    overlap = np.array([[iou(d.bbox, track_boxes[k]) for k in ids] for d in detections])
    cardinality_bonus = min(overlap.shape) + 1
    weights = (overlap >= minimum_iou) * (cardinality_bonus + overlap)
    rows, cols = linear_sum_assignment(-weights)
    matches = {
        int(r): ids[int(c)] for r, c in zip(rows, cols) if overlap[r, c] >= minimum_iou
    }
    return matches, [i for i in range(len(detections)) if i not in matches]


class FTFSODProposals:
    """One synchronous worker per Inspector; RGB mmap never accumulates frames."""

    typed = True

    def __init__(
        self,
        *,
        python,
        worker,
        config,
        checkpoint,
        device="cuda:0",
        threshold=0.3,
        nms_iou=0.4,
        interval=5,
        timeout=180,
        log=None,
        new_track_threshold=None,
        max_interval_seconds=0.5,
    ):
        if interval < 1 or not 0 < threshold < 1 or not 0 < nms_iou < 1:
            raise ValueError("Invalid detector settings")
        for name, value in dict(
            python=python, worker=worker, config=config, checkpoint=checkpoint
        ).items():
            if not Path(value).is_file():
                raise FileNotFoundError(f"FT-FSOD {name} missing: {value}")
        self.interval = interval
        if not np.isfinite(max_interval_seconds) or max_interval_seconds <= 0:
            raise ValueError("Detection refresh time must be positive")
        self.max_interval_seconds = max_interval_seconds
        self.last_detection_timestamp = None
        self.new_track_threshold = (
            threshold if new_track_threshold is None else new_track_threshold
        )
        if not threshold <= self.new_track_threshold <= 1:
            raise ValueError("New-track cutoff must be at least the detection cutoff")
        self.calls = 0
        self.timeout = timeout
        self.temp = tempfile.TemporaryDirectory(prefix="gear-ftfsod-")
        self.rgb_path = Path(self.temp.name) / "frame.rgb"
        self.process = None
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log, "a") if log else tempfile.TemporaryFile(mode="w+")
        command = [
            str(python),
            "-u",
            str(worker),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--threshold",
            str(threshold),
            "--nms-iou",
            str(nms_iou),
        ]
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="3",
        )
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_file,
                text=True,
                bufsize=1,
                env=env,
            )
            ready = self._read()
            if ready.get("classes") != ["gear type A", "gear type B"] or not ready.get(
                "ready"
            ):
                raise RuntimeError(f"FT-FSOD startup failed: {ready}")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("FT-FSOD worker timed out; inspect detector log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("FT-FSOD worker exited; inspect detector log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"FT-FSOD inference failed: {result['error']}")
        return result

    def __call__(self, rgb, timestamp=None):
        if self.process is None:
            raise RuntimeError("FT-FSOD client is closed")
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError("Detector input must be uint8 RGB")
        self.calls += 1
        if timestamp is not None and not np.isfinite(timestamp):
            raise ValueError("Detector timestamp must be finite")
        time_due = timestamp is not None and (
            self.last_detection_timestamp is None
            or timestamp - self.last_detection_timestamp >= self.max_interval_seconds
        )
        if (self.calls - 1) % self.interval and not time_due:
            return []
        np.ascontiguousarray(rgb).tofile(self.rgb_path)
        self.process.stdin.write(
            json.dumps(
                dict(
                    request_id=self.calls, rgb_file=str(self.rgb_path), shape=rgb.shape
                )
            )
            + "\n"
        )
        self.process.stdin.flush()
        reply = self._read()
        if reply.get("request_id") != self.calls:
            raise RuntimeError("FT-FSOD response does not match requested frame")
        results = [
            TypedDetection(tuple(r["bbox"]), r["gear_type"], r["score"])
            for r in reply["detections"]
        ]
        for d in results:
            d.validate(rgb.shape)
        self.last_detection_timestamp = timestamp
        return results

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"close":true}\n')
                    process.stdin.flush()
                    process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass
                process.stdout.close()
        self.temp.cleanup()
        self.log_file.close()
