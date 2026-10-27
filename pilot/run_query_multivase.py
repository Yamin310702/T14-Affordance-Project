"""E09: fixed 20-object pouring pilot. No prompt/threshold tuning."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.update(HF_HOME=str(root/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
sys.path.insert(0,str(root/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
sys.path.insert(0,'/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
from dataset import AffordNetDataset
out=root/'query_multivase';out.mkdir(exist_ok=True)
previous=json.loads((root/'query_sensitivity/run.json').read_text())
queries=previous['queries'];names=['Original','Query 1','Query 2','Query 3','Ensemble']
meta=dict(experiment='E09',status='preparing',scope='20 additional unique Vase objects; one target-present partial view each; development pilot only',
          queries=queries,seed=1,ensemble_members=[1,2,3],device='mps',
          provenance=previous['prior_run'],selection='SHA256 E09-object:UID order; one view ranked by SHA256 E09-view:UID:view_id; no model scores used',
          eligibility='Vase category; at least one pour point and one non-pour point in view; excludes E08 object',
          primary_metric='equal-object mean aIoU over 20 thresholds [0,1]',diagnostic_threshold=0.5,
          bootstrap='paired object bootstrap, 10000 resamples, seed 901; percentile 95% interval; exploratory',completed_objects=0)
rows=[]
def save(): (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 ds=AffordNetDataset('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data','val',partial=True)
 eligible={}
 for idx,info in enumerate(ds.all_data):
  sid=str(info['shape_id']);labels=np.asarray(info['data_info']['label']).reshape(-1)
  if str(info['semantic class']).lower()!='vase' or sid==previous['prior_run']['shape_id']:continue
  n=int((labels==8).sum())
  if not 0<n<len(labels):continue
  eligible.setdefault(sid,[]).append(dict(index=idx,shape_id=sid,view_id=str(info['view_id']),gt_points=n))
 rank=lambda s:hashlib.sha256(s.encode()).hexdigest()
 selected=[]
 for sid in sorted(eligible,key=lambda s:rank('E09-object:'+s))[:20]:
  selected.append(min(eligible[sid],key=lambda v:rank('E09-view:'+sid+':'+v['view_id'])))
 assert len(selected)==20,f'Only {len(selected)} eligible objects; no automatic replacement'
 (out/'selection.json').write_text(json.dumps(dict(eligible_objects=len(eligible),selected=selected),indent=2)+'\n')
 meta.update(status='running',eligible_objects=len(eligible));save()
 print(f'Selected {len(selected)} unique objects from {len(eligible)} eligible before inference.',flush=True)
 assert torch.backends.mps.is_available();torch.manual_seed(1);np.random.seed(1)
 model=load_espresso(meta['provenance']['checkpoint'],device='cpu').eval().to('mps')
 for item in selected:
  xyz,_,labels,sid,cat=ds[item['index']];gt=(np.asarray(labels).reshape(-1)==8).astype(np.float32)
  x=torch.from_numpy(xyz).float().unsqueeze(0).to('mps');t=time.monotonic()
  with torch.inference_mode():
   feat=model.vision(x)
   maps=np.stack([model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0] for q in queries])
  assert maps.shape==(4,len(gt)) and np.isfinite(maps).all()
  ensemble=maps[1:].mean(0);allmaps=np.vstack([maps,ensemble]);metrics=[]
  for name,h in zip(names,allmaps):
   aiou,auc,sim,mae=sample_metrics(h,gt);m=h>=.5;b=gt.astype(bool)
   metrics.append(dict(name=name,aiou=float(aiou),auc=float(auc),sim=float(sim),mae=float(mae),
                       recall05=float(m[b].mean()),fpr05=float(m[~b].mean()),positive_points=int(m.sum())))
  record=dict(**item,category=cat,metrics=metrics,seconds=time.monotonic()-t,
              query_score_std=float(maps[1:].std(0).mean()))
  rows.append(record)
  np.savez_compressed(out/f'{sid}.npz',xyz=xyz,ground_truth=gt,individual_heatmaps=maps,ensemble_heatmap=ensemble)
  (out/'per_object.json').write_text(json.dumps(rows,indent=2)+'\n')
  meta['completed_objects']=len(rows);save()
  print(f"{len(rows)}/20: {sid}, original {metrics[0]['aiou']:.4f}, ensemble {metrics[4]['aiou']:.4f}",flush=True)
 values=np.array([[m['aiou'] for m in row['metrics']] for row in rows]);rng=np.random.default_rng(901)
 indices=rng.integers(0,len(rows),size=(10000,len(rows)))
 summary=[]
 for i,name in enumerate(names):
  delta=values[:,i]-values[:,0];ci=np.quantile(delta[indices].mean(1),[.025,.975])
  means={k:float(np.mean([row['metrics'][i][k] for row in rows])) for k in ['aiou','auc','sim','mae','recall05','fpr05']}
  summary.append(dict(name=name,**means,mean_aiou_change_pp=float(100*delta.mean()),
                 paired_bootstrap95_pp=(ci*100).tolist(),improved=int((delta>1e-8).sum()),
                 worsened=int((delta< -1e-8).sum()),tied=int((np.abs(delta)<=1e-8).sum()),
                 empty_predictions05=sum(row['metrics'][i]['positive_points']==0 for row in rows)))
 meta.update(status='complete',summary=summary);save();print(json.dumps(summary,indent=2),flush=True)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');save();traceback.print_exc();raise
