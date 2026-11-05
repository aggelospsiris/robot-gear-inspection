"""Frozen six-source S2 architecture with versioned image-score pooling."""

from pathlib import Path
import sys, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from .scoring import load_scoring, pool_local_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/AdaptCLIP"))
import adaptcliplib
from adaptcliplib import VisualAdapter, TextualAdapter, PQAdapter, fusion_fun
from tools.utils import get_transform


class AdaptCLIP:
    def __init__(self, assets, device="cuda:0", load_adapters=True):
        self.assets = Path(assets)
        self.scoring_rule = load_scoring(self.assets)
        self.device = torch.device(device)
        self.preprocess = get_transform(518)[0]
        self.encoder, _ = adaptcliplib.load(str(self.assets / "clip.pt"), device="cpu")
        self.encoder.visual.DAPM_replace(DPAM_layer=20)
        self.textual = TextualAdapter(self.encoder, 518, 12)
        self.visual = VisualAdapter(518, 14, input_dim=768, reduction=4)
        self.pq = PQAdapter(
            518, 14, context=True, input_dim=768, mid_dim=128, layers_num=4
        )
        if load_adapters:
            state = torch.load(
                self.assets / "adaptclip_s2.pth", map_location="cpu", weights_only=True
            )
            for key, mod in [
                ("textual_learner", self.textual),
                ("visual_learner", self.visual),
                ("pq_learner", self.pq),
            ]:
                mod.load_state_dict(state[key], strict=True)
        for mod in [self.encoder, self.textual, self.visual, self.pq]:
            mod.to(self.device).eval()
            mod.requires_grad_(False)
        self.textual.prepare_static_text_feature(self.encoder)
        with torch.inference_mode():
            prompts, tokens = self.textual()
            self.learned_text = self.encoder.encode_text_learn(prompts, tokens).float()
        self.static_text = self.textual.static_text_features

    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.preprocess(Image.fromarray(im)) for im in images]).to(
            self.device
        )
        return self.encoder.encode_image(x, [6, 12, 18, 24], DPAM_layer=20)

    @torch.inference_mode()
    def score(self, q, qp, pg, pp):
        gv, lv = self.visual(q, qp, self.static_text)
        gt, lt = self.textual.compute_global_local_score(q, qp, self.learned_text)
        gv = gv.softmax(-1)[:, 1]
        gt = gt.softmax(-1)[:, 1]
        gpl, lpl, align = self.pq(q, qp, pg, pp)
        gp = torch.stack([p.softmax(-1) for p in gpl], dim=-1).mean(-1)[:, 1]
        lp = torch.stack([p[:, 1] for p in lpl], dim=1).mean(1)
        raw = fusion_fun([lv[:, 1], lt[:, 1], lp], "average_mean")
        raw = fusion_fun(
            [raw, fusion_fun(align, "harmonic_mean")[:, 0]], "harmonic_mean"
        )
        maps = np.stack(
            [
                gaussian_filter(
                    p.cpu().numpy(), sigma=self.scoring_rule["smoothing_sigma"]
                )
                for p in raw
            ]
        )
        local_score = pool_local_score(maps, self.scoring_rule).to(self.device)
        score = fusion_fun(
            [fusion_fun([gv, gt, gp], "average_mean"), local_score], "harmonic_mean"
        )
        if not np.isfinite(maps).all() or not torch.isfinite(score).all():
            raise RuntimeError("Non-finite AdaptCLIP output")
        return score.cpu().numpy(), maps


class ReferenceBank:
    """Search all stored views globally; load only K closest views for official local PQ."""

    def __init__(self, root, device, k=16):
        import json

        self.root = Path(root)
        self.device = device
        self.k = k
        self.index = json.loads((self.root / "index.json").read_text())
        self.rows = self.index["records"]
        self.globals = np.load(self.root / "globals.npy", mmap_mode="r")
        self.cache = {}

    def retrieve(self, q, gear, exclude_parent=None):
        candidates = np.array(
            [
                i
                for i, r in enumerate(self.rows)
                if r["gear_type"] == gear and r["parent_id"] != exclude_parent
            ],
            dtype=int,
        )
        if not len(candidates):
            raise RuntimeError(f"No references for type{gear}")
        g = np.asarray(self.globals[candidates], dtype="float32")
        query = q.detach().cpu().numpy().reshape(-1)
        query = query / max(np.linalg.norm(query), 1e-9)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        selected = candidates[np.argsort(-(g @ query), kind="stable")[: self.k]]
        patches = [[] for _ in range(4)]
        globals = []
        for i in selected:
            r = self.rows[int(i)]
            p = self.root / r["file"]
            if str(p) not in self.cache:
                if len(self.cache) >= self.k * 2:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[str(p)] = torch.load(
                    p, map_location="cpu", weights_only=True
                )
            item = self.cache[str(p)]
            globals.append(item["global"][r["view"]])
            for layer in range(4):
                patches[layer].append(item["patch"][layer][r["view"]])
        return (
            torch.stack(globals)[None].float().to(self.device),
            [torch.stack(p)[None].float().to(self.device) for p in patches],
            [self.rows[int(i)] for i in selected],
        )
