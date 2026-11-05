from pathlib import Path
import json, hashlib
import numpy as np
from .pipeline import Inspector
from .model import AdaptCLIP, ReferenceBank
from .quality import QualityGate
from .scoring import load_scoring, validate_scoring
from .taxonomy import validate_taxonomy
from .anomaly_input import load_preprocessing, preprocessing_sha, masked_gear
from .type_features import (
    head_probability,
    validate_head,
    identity_inputs,
    compact_foreground,
    METAL_PREPROCESSING,
)


class Backend:
    def __init__(self, assets, device, k, detector_identity=False):
        self.model = AdaptCLIP(assets, device)
        self.scoring_rule = self.model.scoring_rule
        self.bank = ReferenceBank(Path(assets) / "reference_bank", device, k)
        self.preprocessing = load_preprocessing(assets)
        if self.preprocessing is not None and self.bank.index.get(
            "preprocessing_sha256"
        ) != preprocessing_sha(assets):
            raise ValueError("Reference bank input preprocessing differs from runtime")
        self.detector_identity = detector_identity
        if detector_identity:
            self.identity_v1 = False
            self.identity_policy = dict(
                version="ftfsod_typed_boxes_v1",
                type_source="FT-FSOD detector; no CLIP type or object heads",
                type_uncertainty="preserve_tracking_no_votes_v1",
            )
            return
        self.classifier = json.loads(
            (Path(assets) / "type_classifier.json").read_text()
        )
        self.object_gate = json.loads((Path(assets) / "object_gate.json").read_text())
        self.identity_v1 = self.classifier.get("version") == "clip_shape_identity_v1"
        self.identity_policy = dict(
            version=self.classifier.get("version", "legacy_masked_clip_logistic"),
            preprocessing=self.classifier.get("preprocessing", "masked_padded_crop"),
            type_head_sha256=hashlib.sha256(
                (Path(assets) / "type_classifier.json").read_bytes()
            ).hexdigest(),
            type_uncertainty="preserve_tracking_no_votes_v1",
        )
        if self.identity_v1:
            self.object_classifier = json.loads(
                (Path(assets) / "object_classifier.json").read_text()
            )
            for head in [self.classifier, self.object_classifier]:
                validate_head(head)
            self.identity_policy["object_head_sha256"] = hashlib.sha256(
                (Path(assets) / "object_classifier.json").read_bytes()
            ).hexdigest()
            if (
                self.classifier["preprocessing"]
                != self.object_classifier["preprocessing"]
            ):
                raise ValueError("Identity heads require identical preprocessing")

    def inspect_identity(self, crop, mask):
        appearance, shape, refinement = identity_inputs(
            crop, mask, self.classifier["preprocessing"]
        )
        if appearance is None:
            return dict(
                accepted=False,
                object_probability=0.0,
                probabilities=[0.5, 0.5],
                reason=refinement["reason"],
                refinement=refinement,
                minimum_type_probability=self.classifier["minimum_probability"],
            )
        feature = self.encode(appearance)
        q = feature[0].detach().float().cpu().numpy().reshape(-1)
        object_probability = head_probability(q, shape, self.object_classifier)
        p = head_probability(q, shape, self.classifier)
        gear = "B" if p >= 0.5 else "A"
        cosine = float(
            (q / max(np.linalg.norm(q), 1e-9))
            @ np.array(self.classifier["class_centroids"][gear])
        )
        accepted = object_probability >= self.object_classifier["minimum_probability"]
        reason = "gear" if accepted else "learned_background"
        supported = cosine >= self.classifier["minimum_cosine_similarity"][gear]
        if accepted and not supported:
            reason = "unsupported_gear_appearance"
        return dict(
            accepted=bool(accepted),
            object_probability=object_probability,
            probabilities=[1 - p, p] if supported else [0.5, 0.5],
            raw_probabilities=[1 - p, p],
            type_supported=bool(supported),
            appearance_similarity=cosine,
            reason=reason,
            minimum_type_probability=self.classifier["minimum_probability"],
            refinement=refinement,
        )

    def encode(self, crop):
        return self.model.encode([crop])

    def prepare_anomaly_input(self, crop, mask):
        return masked_gear(crop, mask) if self.preprocessing is not None else crop

    def is_gear(self, crop, mask):
        import cv2

        if getattr(self, "detector_identity", False):
            return True, {"reason": "accepted_typed_detector_proposal"}
        if self.identity_v1:
            if self.classifier["preprocessing"] == METAL_PREPROCESSING:
                foreground, metrics = compact_foreground(crop, mask)
                if foreground is None:
                    return False, metrics
            return True, {"reason": "deferred_until_quality_checked_learned_identity"}

        fg = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[mask]
        mean = float(fg.mean())
        dark = float((fg < 125).mean())
        ok = (
            mean <= self.object_gate["max_foreground_mean"]
            and dark >= self.object_gate["min_foreground_dark_fraction"]
        )
        return ok, {"foreground_mean": mean, "foreground_dark_fraction": dark}

    def classify(self, feature):
        q = feature[0].detach().cpu().numpy().reshape(-1)
        q /= max(np.linalg.norm(q), 1e-9)
        z = float(
            q @ np.array(self.classifier["coefficients"]) + self.classifier["intercept"]
        )
        p = float(1 / (1 + np.exp(-np.clip(z, -50, 50))))
        gear = "B" if p >= 0.5 else "A"
        if (
            "class_centroids" in self.classifier
            and float(q @ np.array(self.classifier["class_centroids"][gear]))
            < self.classifier["minimum_cosine_similarity"][gear]
        ):
            return np.array([0.5, 0.5])
        return np.array([1 - p, p])

    def score(self, feature, gear, exclude_parent=None):
        q, qp = feature
        pg, pp, _ = self.bank.retrieve(q, gear, exclude_parent)
        scores, maps = self.model.score(q, qp, pg, pp)
        return float(scores[0]), maps[0]


def create_inspector(path, device=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("detector") == "ftfsod" and cfg.get("segmenter", "sam2") not in (
        "sam2",
        "sam3_boxes",
    ):
        raise ValueError(
            "FT-FSOD typed boxes require the SAM2 or SAM3 box stream adapter"
        )
    assets = (path.parent / cfg["assets"]).resolve()
    device = device or cfg.get("device", "cuda:0")
    threshold = json.loads((assets / "thresholds.json").read_text())
    validate_scoring(threshold, load_scoring(assets))
    validate_taxonomy(assets, threshold)
    preprocessing = load_preprocessing(assets)
    if preprocessing is not None:
        if threshold.get("preprocessing_sha256") != preprocessing_sha(assets):
            raise ValueError("Masked input changed; rebuild references and calibration")
        if cfg.get("segmenter") != "sam3_boxes":
            raise ValueError("This anomaly profile was calibrated with SAM3 box masks")
        if cfg.get("sam3_image_size", 1008) != preprocessing["sam3_image_size"]:
            raise ValueError("SAM3 mask resolution differs from anomaly calibration")
        checkpoint = (path.parent / cfg["sam3_checkpoint"]).resolve()
        with checkpoint.open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest()
                != preprocessing["sam3_checkpoint_sha256"]
            ):
                raise ValueError("SAM3 checkpoint differs from anomaly calibration")
    index = assets / "reference_bank/index.json"
    if hashlib.sha256(index.read_bytes()).hexdigest() != threshold["bank_index_sha256"]:
        raise ValueError("Reference bank changed; recalibrate thresholds")
    if threshold["retrieval_k"] != cfg["retrieval_k"]:
        raise ValueError("Reference retrieval count changed; recalibrate thresholds")
    checkpoint_sha = hashlib.file_digest(
        (assets / "adaptclip_s2.pth").open("rb"), "sha256"
    ).hexdigest()
    if checkpoint_sha != threshold["checkpoint_sha256"]:
        raise ValueError("Adapter checkpoint changed; recalibrate thresholds")
    segmenter, manifest, settings = create_segmenter(cfg, path, device, assets)
    backend = Backend(
        assets,
        device,
        cfg["retrieval_k"],
        detector_identity=cfg.get("detector") == "ftfsod",
    )
    if cfg.get("detector") == "ftfsod":
        backend.identity_policy.update(
            detector_checkpoint_sha256=manifest["checkpoint_sha256"],
            detector_score_threshold=cfg.get("detector_threshold", 0.3),
            detector_nms_iou=settings.get("nms_iou", 0.4),
            discovery_interval_frames=cfg.get("discovery_interval_frames", 5),
            new_track_score=settings.get(
                "new_track_threshold", cfg.get("detector_threshold", 0.3)
            ),
            max_discovery_interval_seconds=settings.get("max_interval_seconds", 0.5),
            initial_type_score=cfg.get("temporal", {}).get("type_confidence", 0.8),
            lock_after_confirmation=cfg.get("identity", {}).get(
                "lock_after_confirmation", False
            ),
            aggregation=cfg.get("identity", {}).get(
                "aggregation", "consecutive_agreement"
            ),
            reidentification=cfg.get("reidentification"),
            switch_type_score=cfg.get("identity", {}).get(
                "switch_confidence", cfg.get("temporal", {}).get("type_confidence", 0.8)
            ),
        )
    quality = QualityGate(assets / "blur_filter.json")
    return Inspector(
        segmenter,
        backend,
        quality,
        threshold["thresholds"],
        identity=cfg.get("identity"),
        inspect_anomalies=cfg.get("inspect_anomalies", True),
        **cfg.get("temporal", {}),
    )


def create_segmenter(cfg, path, device, assets):
    """Share validated detector/SAM construction across anomaly backends."""
    manifest = settings = None
    kind = cfg.get("segmenter", "sam2")
    if kind == "sam3":
        from .sam3_stream import SAM3Stream

        checkpoint = cfg.get("sam3_checkpoint")
        if checkpoint:
            checkpoint = (path.parent / checkpoint).resolve()
        segmenter = SAM3Stream(
            assets,
            device,
            memory_frames=cfg.get("sam_memory_frames", 16),
            max_objects=cfg.get("max_gears", 12),
            prompt=cfg.get("sam3_prompt", "gear"),
            checkpoint=checkpoint,
        )
    elif kind in ("sam2", "sam3_boxes"):
        proposal = None
        if cfg.get("detector", "tabletop") == "grounding_dino":
            from .detection import GroundingDINOProposals

            proposal = GroundingDINOProposals(
                assets,
                device,
                threshold=cfg.get("detector_threshold", 0.1),
                interval=cfg.get("discovery_interval_frames", 5),
            )
        elif cfg.get("detector") == "ftfsod":
            from .typed_detection import FTFSODProposals

            settings = cfg["ftfsod"]

            def resolved(key):
                return (path.parent / settings[key]).resolve()

            manifest = json.loads(resolved("manifest").read_text())
            for key in ("checkpoint", "config"):
                with resolved(key).open("rb") as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != manifest[key + "_sha256"]:
                    raise ValueError(f"FT-FSOD {key} differs from deployment manifest")
            if "worker_sha256" in manifest:
                with resolved("worker").open("rb") as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != manifest["worker_sha256"]:
                    raise ValueError("FT-FSOD worker differs from deployment manifest")
            if manifest["label_mapping"] != {"0": "A", "1": "B"}:
                raise ValueError(
                    "FT-FSOD class mapping differs from A/B reference routing"
                )
            proposal = FTFSODProposals(
                # Preserve the venv executable symlink. Resolving it selects the
                # base interpreter and loses the environment's site-packages.
                python=(path.parent / settings["python"]).absolute(),
                worker=resolved("worker"),
                config=resolved("config"),
                checkpoint=resolved("checkpoint"),
                device=device,
                threshold=cfg.get("detector_threshold", 0.3),
                interval=cfg.get("discovery_interval_frames", 5),
                nms_iou=settings.get("nms_iou", 0.4),
                new_track_threshold=settings.get(
                    "new_track_threshold", cfg.get("detector_threshold", 0.3)
                ),
                max_interval_seconds=settings.get("max_interval_seconds", 0.5),
                log=resolved("log") if settings.get("log") else None,
            )
        if kind == "sam3_boxes":
            from .sam3_boxes import SAM3BoxStream

            segmenter = SAM3BoxStream(
                (path.parent / cfg["sam3_checkpoint"]).resolve(),
                device,
                max_objects=cfg.get("max_gears", 12),
                proposal=proposal,
                reidentification=cfg.get("reidentification"),
                image_size=cfg.get("sam3_image_size", 1008),
            )
        else:
            from .segmentation import SAM2Stream

            segmenter = SAM2Stream(
                assets,
                device,
                memory_frames=cfg.get("sam_memory_frames", 16),
                max_objects=cfg.get("max_gears", 12),
                proposal=proposal,
                reidentification=cfg.get("reidentification"),
            )
    else:
        raise ValueError(f"Unknown segmenter: {kind}")
    return segmenter, manifest, settings
