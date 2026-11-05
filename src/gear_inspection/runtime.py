"""Load the selected FT-FSOD/SAM3/AdaptCLIP deployment from explicit paths."""

from pathlib import Path
import hashlib
import json

from .anomaly_input import load_preprocessing, masked_gear, preprocessing_sha
from .model import AdaptCLIP, ReferenceBank
from .pipeline import Inspector
from .quality import QualityGate
from .scoring import load_scoring, validate_scoring
from .taxonomy import validate_taxonomy


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_paths(config_path, cfg):
    """Resolve every asset directory relative to the config, independent of cwd."""
    base = Path(config_path).resolve().parent
    return {
        name: (base / cfg["paths"][name]).resolve()
        for name in ("weights", "calibration", "reference_bank")
    }


class Backend:
    detector_identity = True
    identity_v1 = False

    def __init__(self, paths, device, k):
        self.model = AdaptCLIP(paths["calibration"], paths["weights"], device)
        self.scoring_rule = self.model.scoring_rule
        self.bank = ReferenceBank(paths["reference_bank"], device, k)
        self.preprocessing = load_preprocessing(paths["calibration"])
        if self.preprocessing is None:
            raise ValueError("Masked preprocessing settings are required")
        if self.bank.index.get("preprocessing_sha256") != preprocessing_sha(
            paths["calibration"]
        ):
            raise ValueError("Reference preprocessing differs from inference")
        self.identity_policy = dict(
            version="ftfsod_typed_boxes_v1",
            type_source="FT-FSOD detector; no CLIP type or object heads",
            type_uncertainty="preserve_tracking_no_votes_v1",
        )

    def is_gear(self, crop, mask):
        return True, dict(reason="accepted_typed_detector_proposal")

    def prepare_anomaly_input(self, crop, mask):
        return masked_gear(crop, mask)

    def encode(self, crop):
        return self.model.encode([crop])

    def score(self, feature, gear, exclude_parent=None):
        q, qp = feature
        pg, pp, _ = self.bank.retrieve(q, gear, exclude_parent)
        scores, maps = self.model.score(q, qp, pg, pp)
        return float(scores[0]), maps[0]


def validate_assets(config_path, cfg):
    """Check model/calibration bindings before allocating models or workers."""
    paths = resolve_paths(config_path, cfg)
    base = Path(config_path).resolve().parent
    cal, weights, bank = (
        paths[k] for k in ("calibration", "weights", "reference_bank")
    )
    if cfg.get("detector") != "ftfsod" or cfg.get("segmenter") != "sam3_boxes":
        raise ValueError("This deployment supports FT-FSOD with SAM3 box masks")
    threshold = json.loads((cal / "thresholds.json").read_text())
    validate_scoring(threshold, load_scoring(cal))
    validate_taxonomy(cal, threshold)
    spec = load_preprocessing(cal)
    if spec is None or threshold.get("preprocessing_sha256") != preprocessing_sha(cal):
        raise ValueError("Masked preprocessing changed; recalibration required")
    if cfg.get("sam3_image_size", 1008) != spec["sam3_image_size"]:
        raise ValueError("SAM3 resolution differs from calibration")
    if file_sha(base / cfg["sam3_checkpoint"]) != spec["sam3_checkpoint_sha256"]:
        raise ValueError("SAM3 checkpoint differs from calibration")
    if file_sha(bank / "index.json") != threshold["bank_index_sha256"]:
        raise ValueError("Reference bank changed; recalibration required")
    if threshold["retrieval_k"] != cfg["retrieval_k"]:
        raise ValueError("Retrieval count differs from calibration")
    if file_sha(weights / "adaptclip_s2.pth") != threshold["checkpoint_sha256"]:
        raise ValueError("AdaptCLIP checkpoint differs from calibration")
    settings = cfg["ftfsod"]
    manifest = json.loads((base / settings["manifest"]).read_text())
    for key in ("checkpoint", "config", "worker"):
        if file_sha(base / settings[key]) != manifest[key + "_sha256"]:
            raise ValueError(f"FT-FSOD {key} differs from deployment manifest")
    if manifest["label_mapping"] != {"0": "A", "1": "B"}:
        raise ValueError("Detector class mapping differs from reference routing")
    return paths, threshold, manifest


def create_inspector(path, device=None):
    from .sam3_boxes import SAM3BoxStream
    from .typed_detection import FTFSODProposals

    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    paths, threshold, manifest = validate_assets(path, cfg)
    device = device or cfg.get("device", "cuda:0")
    settings = cfg["ftfsod"]

    def resolved(key):
        return (path.parent / settings[key]).resolve()

    backend = Backend(paths, device, cfg["retrieval_k"])
    proposal = FTFSODProposals(
        # Preserve the executable symlink so the virtualenv stays selected.
        python=(path.parent / settings["python"]).absolute(),
        worker=resolved("worker"),
        config=resolved("config"),
        checkpoint=resolved("checkpoint"),
        device=device,
        threshold=cfg["detector_threshold"],
        interval=cfg["discovery_interval_frames"],
        nms_iou=settings["nms_iou"],
        new_track_threshold=settings["new_track_threshold"],
        max_interval_seconds=settings["max_interval_seconds"],
        log=resolved("log"),
    )
    try:
        segmenter = SAM3BoxStream(
            (path.parent / cfg["sam3_checkpoint"]).resolve(),
            device,
            max_objects=cfg["max_gears"],
            proposal=proposal,
            reidentification=cfg["reidentification"],
            image_size=cfg["sam3_image_size"],
        )
    except BaseException:
        proposal.close()
        raise
    backend.identity_policy.update(
        detector_checkpoint_sha256=manifest["checkpoint_sha256"],
        detector_score_threshold=cfg["detector_threshold"],
        detector_nms_iou=settings["nms_iou"],
        discovery_interval_frames=cfg["discovery_interval_frames"],
        new_track_score=settings["new_track_threshold"],
        max_discovery_interval_seconds=settings["max_interval_seconds"],
        initial_type_score=cfg["temporal"]["type_confidence"],
        lock_after_confirmation=cfg["identity"]["lock_after_confirmation"],
        aggregation=cfg["identity"]["aggregation"],
        reidentification=cfg["reidentification"],
    )
    return Inspector(
        segmenter,
        backend,
        QualityGate(paths["calibration"] / "blur_filter.json"),
        threshold["thresholds"],
        identity=cfg["identity"],
        **cfg["temporal"],
    )
