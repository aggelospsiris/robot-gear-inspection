# Inference assets

These files are included in Git so adding model checkpoints is enough to run the
selected configuration. No original or augmented training images are included.

- `reference_bank/`: features from 18 selected normal originals, four rotations
  each (72 views). The 18 `.pt` files are feature tensors, not learned model weights.
  `globals.npy` supports retrieval; `index.json` preserves parent/type/view routing.
- `calibration/`: frozen scoring, thresholds, preprocessing and blur parameters,
  plus taxonomy and calibration lineage. Manifest paths are dataset-relative
  provenance only and are never opened during inference.
- `tokenizer/bert-base-uncased/`: BERT vocabulary, tokenizer and architecture config.
  Learned BERT tensors are stored separately at `weights/bert.safetensors`.

Reference features occupy approximately 600 MB. All model tensors stay in
`weights/`. Checkpoint and runtime asset checksums are listed in the root
`ASSETS_SHA256.json`. Preserve the matching reference bank, scoring and thresholds.
