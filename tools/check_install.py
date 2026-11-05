"""Check local files and both inference environments without starting a camera."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for name,expected in json.loads((ROOT/'WEIGHTS_SHA256.json').read_text()).items():
 p=ROOT/name
 if not p.is_file():raise FileNotFoundError(p)
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=expected:raise RuntimeError(f'File changed: {name}')
import torch,cv2,ultralytics
if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
subprocess.run([str(ROOT/'.venv-ftfsod/bin/python'),'-c','import torch, mmcv, transformers; import mmcv.ops; assert torch.cuda.is_available(); print("Detector environment OK")'],check=True)
print('Weights, CUDA and inference environments OK')
