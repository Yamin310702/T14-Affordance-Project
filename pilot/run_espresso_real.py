"""One real-input feasibility run; not an official benchmark."""
import os, sys, json, time, traceback, hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(root/'hf_cache')
import numpy as np
import torch
sys.path.insert(0,str(root/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
out=root/'espresso_real';out.mkdir(exist_ok=True)
meta={'status':'running','implementation':'unofficial Espresso-3D with local device adaptations',
      'input':'OpenAD partial validation sample 452','query':'Point to the part you would use to pour.',
      'checkpoint':'LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',
      'device':'mps','seed':1,'note':'Cross-dataset feasibility case, not comparable benchmark evidence.'}
(out/'run.json').write_text(json.dumps(meta,indent=2))
try:
    assert torch.backends.mps.is_available()
    torch.manual_seed(1)
    sys.path.insert(0,'/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
    from dataset import AffordNetDataset
    ds=AffordNetDataset('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data','val',partial=True)
    points,_,labels,sid,cat=ds[452]
    meta.update(shape_id=sid,category=cat,points=len(points))
    np.savez(out/'input.npz',xyz=points,labels=labels)
    del ds
    print('Loading pretrained vision, text and affordance weights...',flush=True)
    model=load_espresso(meta['checkpoint'],device='cpu').eval().to('mps')
    print('All weights loaded. Starting real-input inference.',flush=True)
    xyz=torch.from_numpy(points).float().unsqueeze(0).to('mps')
    torch.mps.synchronize();t=time.monotonic()
    with torch.inference_mode():
        logits=model(xyz,[meta['query']])
        heat=logits.sigmoid().float().cpu().numpy()[0]
    torch.mps.synchronize()
    assert heat.shape==(len(points),) and np.isfinite(heat).all()
    np.save(out/'heatmap.npy',heat)
    meta.update(status='complete',inference_seconds=time.monotonic()-t,
                heatmap_min=float(heat.min()),heatmap_max=float(heat.max()),heatmap_mean=float(heat.mean()),
                mps_current_allocated_bytes=torch.mps.current_allocated_memory())
except Exception as e:
    meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:
    (out/'run.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
