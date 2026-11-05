from dataclasses import dataclass, field
from collections import deque
import hashlib, math, threading, uuid, time
import numpy as np, cv2
from .identity import TypeEvidence, AccumulatedTypeEvidence, geometry_drift

DECISION_POLICY = "cumulative_majority_binary_v2"


@dataclass
class GearResult:
    gear_id: int
    mask: np.ndarray
    bbox: tuple
    gear_type: str | None = None
    type_confidence: float = 0.0
    quality: dict = field(default_factory=dict)
    anomaly_score: float | None = None
    frame_is_defective: bool | None = None
    is_defective: bool | None = None
    status: str = "collecting_evidence"
    evidence_frames: int = 0
    normal_votes: int = 0
    anomalous_votes: int = 0
    mean_anomaly_score: float | None = None
    heatmap: np.ndarray | None = None
    identity: dict = field(default_factory=dict)

    @property
    def label(self):
        if self.is_defective is None:
            return None
        return "abnormal" if self.is_defective else "normal"


@dataclass
class FrameResult:
    session_id: str
    timestamp: float
    frame_index: int
    gears: list
    finalized: list
    overlay_rgb: np.ndarray | None = None
    processing_seconds: float = 0.0
    scoring_rule: dict | None = None
    rejected_instances: list = field(default_factory=list)
    identity_policy: dict | None = None

    def metadata(self):
        return dict(
            session_id=self.session_id,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            processing_seconds=self.processing_seconds,
            decision_policy=DECISION_POLICY,
            scoring_rule=self.scoring_rule,
            rejected_instances=self.rejected_instances,
            identity_policy=self.identity_policy,
            gears=[
                dict(
                    gear_id=g.gear_id,
                    bbox=g.bbox,
                    gear_type=g.gear_type,
                    type_confidence=g.type_confidence,
                    identity=g.identity,
                    quality=g.quality,
                    anomaly_score=g.anomaly_score,
                    frame_is_defective=g.frame_is_defective,
                    is_defective=g.is_defective,
                    label=g.label,
                    status=g.status,
                    evidence_frames=g.evidence_frames,
                    normal_votes=g.normal_votes,
                    anomalous_votes=g.anomalous_votes,
                    mean_anomaly_score=g.mean_anomaly_score,
                )
                for g in self.gears
            ],
            finalized=self.finalized,
        )


@dataclass
class Track:
    last_seen: float
    last_detected: float
    probabilities: np.ndarray | None = None
    gear_type: str | None = None
    evidence: deque = field(default_factory=lambda: deque(maxlen=128))
    last_hash: str | None = None
    normal_votes: int = 0
    anomalous_votes: int = 0
    first_evidence_at: float | None = None
    score_sum: float = 0.0
    type_evidence: TypeEvidence | None = None
    previous_box: tuple | None = None
    last_type_detection: int | None = None
    type_disputed: bool = False

    @property
    def evidence_frames(self):
        return self.normal_votes + self.anomalous_votes

    @property
    def mean_anomaly_score(self):
        return self.score_sum / self.evidence_frames if self.evidence_frames else None

    def add_evidence(self, timestamp, score, threshold):
        if self.first_evidence_at is None:
            self.first_evidence_at = timestamp
        self.evidence.append((timestamp, score))
        self.score_sum += score
        if score >= threshold:
            self.anomalous_votes += 1
        else:
            self.normal_votes += 1

    def clear_evidence(self):
        self.evidence.clear()
        self.normal_votes = self.anomalous_votes = 0
        self.first_evidence_at = None
        self.score_sum = 0.0
        self.last_hash = None


def crop_gear(rgb, mask, padding=0.18):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Empty gear mask")
    bx = (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))
    px = round((bx[2] - bx[0]) * padding)
    py = round((bx[3] - bx[1]) * padding)
    box = (
        max(0, bx[0] - px),
        max(0, bx[1] - py),
        min(rgb.shape[1], bx[2] + px),
        min(rgb.shape[0], bx[3] + py),
    )
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), box


class Inspector:
    """One ordered RGB stream per Inspector. No training or bank building at runtime."""

    def __init__(
        self,
        segmenter,
        backend,
        quality,
        thresholds,
        *,
        type_confidence=0.8,
        min_frames=5,
        min_span=0.6,
        sample_interval=0.15,
        lost_seconds=1.0,
        max_gap_seconds=2.0,
        identity=None,
        inspect_anomalies=True,
    ):
        self.segmenter = segmenter
        self.backend = backend
        self.quality = quality
        self.thresholds = thresholds
        self.type_confidence = type_confidence
        if (
            isinstance(min_frames, bool)
            or not isinstance(min_frames, int)
            or min_frames < 2
        ):
            raise ValueError(
                "min_frames must be an integer >= 2; a first-frame decision is forbidden"
            )
        self.min_frames = min_frames
        self.min_span = min_span
        self.sample_interval = sample_interval
        self.lost_seconds = lost_seconds
        self.max_gap_seconds = max_gap_seconds
        self.identity_config = identity
        self.inspect_anomalies = bool(inspect_anomalies)
        if identity and identity.get("aggregation", "consecutive_agreement") not in (
            "consecutive_agreement",
            "confidence_weighted_cumulative",
        ):
            raise ValueError("Unknown identity aggregation")
        if (
            identity
            and identity.get("aggregation") == "confidence_weighted_cumulative"
            and not getattr(segmenter, "reidentifier", None)
        ):
            raise ValueError(
                "Cumulative gap-tolerant identity requires reappearance tracking"
            )
        if (
            getattr(backend, "identity_v1", False)
            or getattr(backend, "detector_identity", False)
        ) and identity is None:
            raise ValueError("Identity v1 artifacts require an identity configuration")
        self._lock = threading.Lock()
        self.start_session()

    def start_session(self):
        self.segmenter.reset()
        self.tracks = {}
        self.session_id = uuid.uuid4().hex
        self.last_timestamp = None
        self.shape = None
        self.frame_index = 0
        self.active = True

    def _decision(self, t):
        if (
            t.evidence_frames < self.min_frames
            or t.first_evidence_at is None
            or t.evidence[-1][0] - t.first_evidence_at < self.min_span
        ):
            return None, "collecting_evidence"
        if t.anomalous_votes > t.normal_votes:
            return True, "anomalous_so_far"
        if t.normal_votes > t.anomalous_votes:
            return False, "normal_so_far"
        # A completed warm-up always yields one of two labels, including ties.
        abnormal = t.mean_anomaly_score >= self.thresholds[t.gear_type]
        return abnormal, "anomalous_so_far" if abnormal else "normal_so_far"

    def _finalize(self, k, reason):
        t = self.tracks.pop(k)
        decision, status = self._decision(t)
        return dict(
            gear_id=k,
            gear_type=t.gear_type,
            is_defective=decision,
            label=None if decision is None else "abnormal" if decision else "normal",
            status={"normal_so_far": "normal", "anomalous_so_far": "anomalous"}.get(
                status, status
            ),
            evidence_frames=t.evidence_frames,
            normal_votes=t.normal_votes,
            anomalous_votes=t.anomalous_votes,
            mean_anomaly_score=t.mean_anomaly_score,
            decision_policy=DECISION_POLICY,
            reason=reason,
            temporal_rule="cumulative majority after warm-up; tied votes use mean score >= type threshold; provisional",
        )

    def process_frame(self, rgb, timestamp, *, render=False, proposal_boxes=None):
        with self._lock:
            return self._process(rgb, timestamp, render, proposal_boxes)

    def _process(self, rgb, timestamp, render, proposal_boxes):
        start = time.perf_counter()
        if not self.active:
            raise RuntimeError("Session ended; call start_session before more frames")
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 16
        ):
            raise ValueError("rgb must be H×W×3 uint8 RGB, minimum16×16")
        if (
            isinstance(timestamp, (bool, str))
            or not isinstance(timestamp, (float, int, np.floating, np.integer))
            or not math.isfinite(timestamp)
        ):
            raise ValueError("timestamp must be finite seconds")
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("timestamps must strictly increase within a session")
        if self.shape is not None and rgb.shape != self.shape:
            raise ValueError("RGB dimensions changed; call end_session/start_session")
        finalized = []
        if (
            self.last_timestamp is not None
            and timestamp - self.last_timestamp > self.max_gap_seconds
        ):
            finalized = [self._finalize(k, "timestamp_gap") for k in list(self.tracks)]
            next_id = getattr(self.segmenter, "next_id", None)
            self.segmenter.reset()
            if next_id is not None:
                self.segmenter.next_id = next_id
                reidentifier = getattr(self.segmenter, "reidentifier", None)
                if reidentifier is not None and hasattr(reidentifier, "next_id"):
                    reidentifier.next_id = next_id
        self.shape = rgb.shape
        self.last_timestamp = timestamp
        self.frame_index += 1
        if hasattr(self.segmenter, "set_timestamp"):
            self.segmenter.set_timestamp(timestamp)
        masks = self.segmenter.step(rgb, proposal_boxes)
        visible = {
            k: m.astype(bool)
            for k, m in masks.items()
            if m.shape == rgb.shape[:2] and m.sum() >= 100
        }
        gears = []
        rejected_instances = []
        for k, mask in visible.items():
            t = self.tracks.setdefault(k, Track(timestamp, timestamp))
            t.last_seen = timestamp
            confirmations = getattr(self.segmenter, "last_detection", None)
            if confirmations is None or confirmations.get(k) == getattr(
                self.segmenter, "index", None
            ):
                t.last_detected = timestamp
            crop, local, box = crop_gear(rgb, mask)
            g = GearResult(k, mask.copy(), box)
            if self.identity_config is not None:
                if geometry_drift(
                    t.previous_box,
                    box,
                    self.identity_config.get("max_area_ratio", 2.5),
                    self.identity_config.get("min_box_iou", 0.1),
                ):
                    t.clear_evidence()
                    # The user-selected lock belongs to this SAM ID, including
                    # geometry changes. Only ending the track removes its type.
                    if not (
                        t.type_evidence is not None
                        and t.type_evidence.lock_after_confirmation
                        and t.type_evidence.confirmed is not None
                    ) and not self.identity_config.get(
                        "preserve_pending_type_evidence", False
                    ):
                        t.gear_type = None
                        t.type_evidence = None
                        t.last_type_detection = None
                        t.type_disputed = False
                t.previous_box = box
                if t.type_evidence is None:
                    evidence_class = (
                        AccumulatedTypeEvidence
                        if self.identity_config.get("aggregation")
                        == "confidence_weighted_cumulative"
                        else TypeEvidence
                    )
                    t.type_evidence = evidence_class(**self.identity_config)
            overlap = any(
                j != k
                and np.logical_and(mask, m).sum() / max(1, min(mask.sum(), m.sum()))
                > 0.20
                for j, m in visible.items()
            )
            if overlap:
                g.quality = {
                    "status": "occluded",
                    "usable": False,
                    "reason": "overlapping_instance_masks",
                }
            else:
                if hasattr(self.backend, "is_gear"):
                    object_ok, object_metrics = self.backend.is_gear(crop, local)
                    if not object_ok:
                        rejected_instances.append(
                            dict(
                                gear_id=k,
                                bbox=box,
                                identity=dict(
                                    accepted=False,
                                    reason=object_metrics.get(
                                        "reason", "appearance_gate_rejected"
                                    ),
                                    preencoding=True,
                                    metrics=object_metrics,
                                ),
                            )
                        )
                        self.tracks.pop(k, None)
                        self.segmenter.remove(k)
                        continue
                # Reject blurred/uncertain observations before ANY CLIP encoding:
                # they must not update the type classifier or anomaly votes.
                detection = getattr(self.segmenter, "typed_detections", {}).get(k)
                candidate_type = detection["gear_type"] if detection else t.gear_type
                if (
                    t.type_evidence is not None
                    and t.type_evidence.lock_after_confirmation
                    and t.type_evidence.confirmed is not None
                ):
                    candidate_type = t.type_evidence.confirmed
                g.quality = self.quality.evaluate(crop, local, candidate_type)
                typed_identity = getattr(self.backend, "detector_identity", False)
                if g.quality["usable"] or typed_identity:
                    if getattr(self.backend, "detector_identity", False):
                        confidence = 0.0 if detection is None else detection["score"]
                        gear = None if t.type_disputed else t.type_evidence.confirmed
                        # Native detail requirements belong to defect inspection.
                        # A high-confidence detector may still recognize a small
                        # gear, provided the measured view is not blur-ambiguous.
                        type_usable = g.quality["usable"] or (
                            g.quality.get("reason") == "insufficient_visual_evidence"
                            and g.quality.get("blur_score", 1)
                            < getattr(self.quality, "config", {}).get(
                                "accept_threshold", 0
                            )
                        )
                        fresh = (
                            detection is not None
                            and detection["frame_index"] == self.segmenter.index
                            and detection["frame_index"] != t.last_type_detection
                        )
                        locked = (
                            t.type_evidence.lock_after_confirmation
                            and t.type_evidence.confirmed is not None
                        )
                        minimum_confidence = self.type_confidence
                        if fresh and type_usable and not locked:
                            t.last_type_detection = detection["frame_index"]
                            # TypeEvidence consumes one class vote from each fresh
                            # detector observation, never repeated SAM propagation.
                            probability = (
                                [1.0, 0.0] if candidate_type == "A" else [0.0, 1.0]
                            )
                            minimum_confidence = self.type_confidence
                            if (
                                t.type_evidence.confirmed is not None
                                and candidate_type != t.type_evidence.confirmed
                            ):
                                minimum_confidence = max(
                                    minimum_confidence,
                                    self.identity_config.get(
                                        "switch_confidence", minimum_confidence
                                    ),
                                )
                            if confidence >= minimum_confidence:
                                if isinstance(t.type_evidence, AccumulatedTypeEvidence):
                                    gear, changed = t.type_evidence.update_detection(
                                        candidate_type,
                                        confidence,
                                        timestamp,
                                        hashlib.sha256(crop.tobytes()).hexdigest(),
                                        detection["frame_index"],
                                    )
                                else:
                                    gear, changed = t.type_evidence.update(
                                        probability,
                                        timestamp,
                                        hashlib.sha256(crop.tobytes()).hexdigest(),
                                        0.99,
                                    )
                                t.type_disputed = gear is None
                                if changed:
                                    t.clear_evidence()
                            # Weak detections can maintain localization. They do
                            # not vote, erase a confirmed type, or start a switch.
                        g.identity = dict(
                            source="ftfsod_typed_box",
                            detection=detection,
                            fresh_detection=fresh,
                            confirmed_type=t.type_evidence.confirmed,
                            type_observation_usable=type_usable,
                            type_vote_qualified=bool(
                                fresh
                                and type_usable
                                and not locked
                                and confidence >= minimum_confidence
                            )
                            if fresh and type_usable
                            else False,
                            type_status="confirmed"
                            if gear
                            else "collecting_or_disputed",
                            confirmation_observations=getattr(
                                t.type_evidence,
                                "observations_count",
                                len(t.type_evidence.history),
                            ),
                            aggregation=self.identity_config.get(
                                "aggregation", "consecutive_agreement"
                            ),
                            weighted_agreement=getattr(
                                t.type_evidence, "confidence", None
                            ),
                            type_locked=bool(
                                t.type_evidence.lock_after_confirmation
                                and t.type_evidence.confirmed is not None
                            ),
                        )
                    elif getattr(self.backend, "identity_v1", False):
                        g.identity = self.backend.inspect_identity(crop, local)
                        if not g.identity["accepted"]:
                            rejected_instances.append(
                                dict(gear_id=k, bbox=box, identity=g.identity)
                            )
                            self.tracks.pop(k, None)
                            self.segmenter.remove(k)
                            continue
                        prob = np.array(g.identity["probabilities"])
                        confidence = float(prob.max())
                        gear, changed = t.type_evidence.update(
                            prob,
                            timestamp,
                            hashlib.sha256(crop.tobytes()).hexdigest(),
                            max(
                                self.type_confidence,
                                g.identity["minimum_type_probability"],
                            ),
                        )
                        if changed:
                            t.clear_evidence()
                        g.identity["policy"] = "sampled_type_confirmation_v1"
                        g.identity["confirmation_observations"] = len(
                            t.type_evidence.history
                        )
                        g.identity["type_status"] = (
                            "confirmed"
                            if gear is not None
                            else "collecting_or_disputed"
                        )
                        g.identity["candidate_type"] = "AB"[int(np.argmax(prob))]
                        g.identity["confirmed_type"] = t.type_evidence.confirmed
                    else:
                        appearance = np.where(local[:, :, None], crop, 127).astype(
                            "uint8"
                        )
                        type_feature = self.backend.encode(appearance)
                        prob = np.asarray(
                            self.backend.classify(type_feature), dtype=float
                        )
                        t.probabilities = (
                            prob
                            if t.probabilities is None
                            else 0.75 * t.probabilities + 0.25 * prob
                        )
                        confidence = float(np.max(t.probabilities))
                        gear = (
                            "AB"[int(np.argmax(t.probabilities))]
                            if confidence >= self.type_confidence
                            else None
                        )
                    g.type_confidence = confidence
                    if (
                        gear is not None
                        and t.gear_type is not None
                        and gear != t.gear_type
                    ):
                        t.clear_evidence()
                    if gear is not None:
                        t.gear_type = gear
                    g.gear_type = gear
                    g.quality = self.quality.evaluate(crop, local, gear)
                    if (
                        self.inspect_anomalies
                        and gear is not None
                        and g.quality["usable"]
                    ):
                        anomaly_input = (
                            self.backend.prepare_anomaly_input(crop, local)
                            if hasattr(self.backend, "prepare_anomaly_input")
                            else crop
                        )
                        feature = self.backend.encode(anomaly_input)
                        score, heat = self.backend.score(feature, gear)
                        g.anomaly_score = float(score)
                        g.frame_is_defective = bool(score >= self.thresholds[gear])
                        full = np.zeros(rgb.shape[:2], dtype="float32")
                        x0, y0, x1, y1 = box
                        full[y0:y1, x0:x1] = cv2.resize(heat, (x1 - x0, y1 - y0))
                        full[~mask] = 0
                        g.heatmap = full
                        key = hashlib.sha256(anomaly_input.tobytes()).hexdigest()
                        if key != t.last_hash and (
                            not t.evidence
                            or timestamp - t.evidence[-1][0] >= self.sample_interval
                        ):
                            t.add_evidence(
                                timestamp, float(score), self.thresholds[gear]
                            )
                            t.last_hash = key
            g.is_defective, g.status = self._decision(t)
            if not self.inspect_anomalies:
                g.status = (
                    "type_confirmed"
                    if g.gear_type is not None
                    else "collecting_type_evidence"
                )
            g.evidence_frames = t.evidence_frames
            g.normal_votes = t.normal_votes
            g.anomalous_votes = t.anomalous_votes
            g.mean_anomaly_score = t.mean_anomaly_score
            if g.gear_type is None:
                g.gear_type = t.gear_type
                if g.is_defective is None:
                    g.status = "type_uncertain"
            gears.append(g)
        for k in list(self.tracks):
            evidence = self.tracks[k].type_evidence
            locked_sam_track = (
                getattr(self.backend, "detector_identity", False)
                and evidence is not None
                and evidence.lock_after_confirmation
                and evidence.confirmed is not None
            )
            if timestamp - self.tracks[k].last_seen > self.lost_seconds or (
                not locked_sam_track
                and timestamp - self.tracks[k].last_detected > self.lost_seconds
            ):
                finalized.append(self._finalize(k, "track_lost_or_unconfirmed"))
                self.segmenter.remove(k)
                gears = [g for g in gears if g.gear_id != k]
        result = FrameResult(
            self.session_id,
            timestamp,
            self.frame_index,
            gears,
            finalized,
            processing_seconds=time.perf_counter() - start,
            scoring_rule=getattr(self.backend, "scoring_rule", None),
            rejected_instances=rejected_instances,
            identity_policy=getattr(self.backend, "identity_policy", None),
        )
        if render:
            from .visualize import overlay

            result.overlay_rgb = overlay(rgb, result)
        result.processing_seconds = time.perf_counter() - start
        return result

    def end_session(self):
        with self._lock:
            final = [self._finalize(k, "session_end") for k in list(self.tracks)]
            self.segmenter.reset()
            self.last_timestamp = None
            self.shape = None
            self.active = False
            return final

    def close(self):
        """Release this stream's SAM state and isolated detector worker."""
        if self.active:
            self.end_session()
        if hasattr(self.segmenter, "close"):
            self.segmenter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @classmethod
    def from_config(cls, path, device=None):
        from .runtime import create_inspector

        return create_inspector(path, device)
