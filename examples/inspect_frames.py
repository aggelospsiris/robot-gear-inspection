"""Process timestamped RGB images listed in a CSV: path,timestamp_seconds."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from gear_inspection import Inspector
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('frames',type=Path)
p.add_argument('--output',type=Path,default=ROOT/'output')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with Inspector.from_config(ROOT/'configs/stream.json') as inspector,(a.output/'predictions.jsonl').open('w') as log,a.frames.open() as source:
 for i,row in enumerate(csv.DictReader(source),1):
  path=Path(row['path']);path=path if path.is_absolute() else a.frames.parent/path
  rgb=np.array(Image.open(path).convert('RGB'))
  result=inspector.process_frame(rgb,float(row['timestamp_seconds']),render=True)
  log.write(json.dumps(result.metadata())+'\n');log.flush()
  Image.fromarray(result.overlay_rgb).save(a.output/f'frame_{i:06d}.jpg')
  np.savez_compressed(a.output/f'frame_{i:06d}.npz',
   ids=np.array([g.gear_id for g in result.gears],dtype=np.int64),
   masks=np.stack([g.mask for g in result.gears]) if result.gears else np.zeros((0,*rgb.shape[:2]),bool),
   heatmaps=np.stack([g.heatmap if g.heatmap is not None else np.zeros(rgb.shape[:2],np.float32) for g in result.gears]) if result.gears else np.zeros((0,*rgb.shape[:2]),np.float32))
  print(i,[(g.gear_id,g.gear_type,g.label) for g in result.gears])
