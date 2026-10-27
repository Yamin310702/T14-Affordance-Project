"""E08: fixed one-object development experiment; not a held-out benchmark."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.update(HF_HOME=str(root/'hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
sys.path.insert(0,str(root/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
out=root/'query_sensitivity';out.mkdir(exist_ok=True)
queries=['Point to the part you would use to pour.',
         'Point to the part used for pouring.',
         'Locate the region from which liquid is poured.',
         'Identify the pouring region.']
meta=dict(experiment='E08',status='running',implementation='unofficial Espresso-3D; CPU construction/MPS float32 adaptation',
          scope='One previously inspected partial-view vase; exploratory development only',seed=1,
          queries=queries,ensemble_members=[1,2,3],ensemble_rule='arithmetic mean of sigmoid heatmaps',
          metric_rule='source sample_metrics: aIoU averaged over 20 equally spaced thresholds in [0,1]; ROC AUC; SIM; MAE',
          diagnostic_threshold=0.5,device='mps',input_sha256=hashlib.sha256((root/'espresso_real/input.npz').read_bytes()).hexdigest(),
          prior_run=json.loads((root/'espresso_real/run.json').read_text()),
          controls='same input point order, checkpoint, preprocessing, cached vision features; eval mode; no prompt tuning')
(out/'run.json').write_text(json.dumps(meta,indent=2))
try:
 assert torch.backends.mps.is_available()
 torch.manual_seed(1);np.random.seed(1)
 data=np.load(root/'espresso_real/input.npz');xyz=data['xyz'];gt=(data['labels'].reshape(-1)==8).astype(np.float32)
 assert xyz.shape==(2048,3) and gt.sum()==563
 print('Loading cached weights offline...',flush=True)
 model=load_espresso(meta['prior_run']['checkpoint'],device='cpu').eval().to('mps')
 x=torch.from_numpy(xyz).float().unsqueeze(0).to('mps');maps=[]
 with torch.inference_mode():
  feat=model.vision(x)
  for q in queries:
   torch.mps.synchronize();t=time.monotonic()
   maps.append(model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0])
   print(f'{q} completed in {time.monotonic()-t:.3f}s',flush=True)
 maps=np.stack(maps);ensemble=maps[1:].mean(0)
 assert maps.shape==(4,2048) and np.isfinite(maps).all()
 all_maps=np.vstack([maps,ensemble]);names=['Original','Query 1','Query 2','Query 3','Ensemble']
 np.savez(out/'predictions.npz',xyz=xyz,ground_truth=gt,individual_heatmaps=maps,ensemble_heatmap=ensemble)
 rows=[]
 for name,h in zip(names,all_maps):
  aiou,auc,sim,mae=sample_metrics(h,gt)
  m=h>=.5;t=gt.astype(bool)
  rows.append(dict(name=name,aiou=float(aiou),auc=float(auc),sim=float(sim),mae=float(mae),
       iou_at_05=float((m&t).sum()/(m|t).sum()),recall_at_05=float(m[t].mean()),
       false_positive_rate_at_05=float(m[~t].mean()),predicted_positive_points=int(m.sum()),
       mean_target_score=float(h[t].mean()),mean_nontarget_score=float(h[~t].mean())))
 pairs=[]
 for i in range(1,4):
  for j in range(i+1,4):
   pairs.append(dict(queries=[i,j],mean_absolute_score_difference=float(np.abs(maps[i]-maps[j]).mean()),
       threshold05_disagreement=float(np.mean((maps[i]>=.5)!=(maps[j]>=.5)))))
 meta.update(status='complete',metrics=rows,pairwise_query_differences=pairs,
     mean_pointwise_query_std=float(maps[1:].std(0).mean()),
     original_rerun_mean_absolute_difference=float(np.abs(maps[0]-np.load(root/'espresso_real/heatmap.npy')).mean()))
 print(json.dumps(rows,indent=2),flush=True)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:
 (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
