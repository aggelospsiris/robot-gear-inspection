# Third-party inference components

- FT-FSOD: https://github.com/Intellindust-AI-Lab/FT-FSOD, commit
  c68cea00fac282ec526d59bca55f27b3203daf37. Required mmdet source and its LICENSE
  are in vendor/FT-FSOD. Includes OpenMMLab components; retain their notices.
- AdaptCLIP: runtime source and LICENSE in vendor/AdaptCLIP. Frozen six-source
  S2 adapter and CLIP ViT-L weights are included for this testing prototype.
- SAM 3: local checkpoint used through Ultralytics 8.4.192. Upstream:
  https://github.com/facebookresearch/sam3 and https://github.com/ultralytics/ultralytics.
  SAM weights and Ultralytics software have their respective upstream license
  terms; the project LICENSE does not replace them.
- BERT base uncased: local tokenizer/model files for the FT-FSOD text encoder.
- PyTorch, torchvision, OpenCV, SciPy and other pinned dependencies retain their
  own licenses. The software installed by setup is not included in this folder.

Retain upstream notices and review applicable model/software redistribution
terms when sharing or deploying this prototype. No public training datasets
are redistributed.
