"""E12: fixed query wording comparison across five OpenAD affordances."""
import os, sys, json, time, hashlib, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
os.environ.update(HF_HOME=str(ROOT/'hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np, torch
sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
sys.path.insert(0,str(REPO))
from dataset import AffordNetDataset
OUT=ROOT/'multiaffordance_prompt_study_mps'; OUT.mkdir(exist_ok=True)
AFFORDANCES={
 'grab':(0,['Point to the part you would grab.','Point to the part you would grasp.','Identify the region suitable for gripping.','Locate where you would grip the object.']),
 'pour':(8,['Point to the part you would use to pour.','Point to the part used for pouring.','Locate the region from which liquid is poured.','Identify the pouring region.']),
 'wrap':(7,['Point to the part you can wrap your hand around.','Identify the region you would wrap your fingers around.','Locate the part that provides a wrapping grip.','Point to the region designed to be held by wrapping your hand around it.']),
 'accommodate':(1,['Point to the part of the object that can hold something.','Identify the region used to contain or store items.','Locate the area that can accommodate contents.','Point to the part where you would place or keep something.']),
 'unlock':(3,['Point to the part you would use to unlock this object.','Locate the part used to open the object.','Identify the component that lets you open or release it.','Point to the part you would operate to gain access.']),
}
CONDITIONS=['original','phrase_1','phrase_2','phrase_3','ensemble']
DEVICE='mps'
if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; refusing CPU fallback for the MPS comparison')
meta={'experiment':'E12','status':'loading_dataset','model':'Unofficial Espresso-3D; Affogato-pretrained weights','device':DEVICE,'seed':1,'comparison_of':'E12 CPU run; identical dataset split, queries, weights, and code path with MPS device.',
      'scope':'Exploratory cross-dataset prompt wording study; OpenAD validation partial views; five affordance labels; 20 distinct selected objects per affordance.',
      'affordances':{k:{'label_id':v[0],'queries':v[1]} for k,v in AFFORDANCES.items()},
      'ensemble':'Arithmetic mean of the three paraphrase sigmoid score maps, pointwise.',
      'metric':'Affogato/Espresso source sample_metrics; aIoU over 20 thresholds, AUC, SIM, MAE.',
      'claim_limit':'Exploratory selected-object sample, not a representative benchmark. This unofficial Affogato-pretrained Espresso model is applied to OpenAD partial views. Query equivalence is not human-validated. No Affostruction inference.'}
def save(): (OUT/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
try:
 torch.manual_seed(1); np.random.seed(1)
 ds=AffordNetDataset(str(REPO/'data'),'val',partial=True)
 excluded=set()
 for name in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity']:
  prior=json.loads((ROOT/name/'run.json').read_text())
  excluded.update(x['shape_id'] for x in prior['objects'])
 selected={}; eligible_counts={}
 for aff,(label_id,queries) in AFFORDANCES.items():
  rows=[];seen=set()
  for i,d in enumerate(ds.all_data):
   sid=d['shape_id']
   if sid in seen: continue
   seen.add(sid)
   y=np.asarray(d['data_info']['label']).reshape(-1).astype(int)
   if np.any(y==label_id) and sid not in excluded:
    rows.append((hashlib.sha256(f'E12-{aff}:{sid}'.encode()).hexdigest(),i,sid,d['semantic class'],int((y==label_id).sum())))
  eligible_counts[aff]=len(rows)
  chosen=sorted(rows)[:20]
  if len(chosen)<20: raise RuntimeError(f'{aff}: only {len(chosen)} eligible new unique objects')
  selected[aff]=chosen
 meta.update(status='running',dataset_samples=len(ds),excluded_prior_shape_ids=len(excluded),eligible_unique_objects=eligible_counts,
  selection_rule='Within each affordance independently, retain the first validation partial view for each unique shape with positive GT label; exclude every shape used in E09, E10, or E11; rank by SHA256("E12-<affordance>:" + shape_id); take first 20. Selection uses GT availability only, not predictions.',
  objects={a:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in rows] for a,rows in selected.items()})
 save(); print(json.dumps({'status':meta['status'],'dataset_samples':len(ds),'eligible':eligible_counts,'objects_per_affordance':20},indent=2),flush=True)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to(DEVICE)
 results=[];saved={};start=time.monotonic()
 with torch.inference_mode():
  for aff,(label_id,queries) in AFFORDANCES.items():
   for ix,(sel,i,sid,cat,_) in enumerate(selected[aff],1):
    points,_,labels,got_sid,got_cat=ds[i]
    xyz=np.asarray(points,dtype=np.float32); gt=(np.asarray(labels).reshape(-1).astype(int)==label_id).astype(np.float32)
    assert sid==got_sid and len(xyz)==2048 and gt.sum()>0
    x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE); feat=model.vision(x)
    hmaps=[]
    for q in queries:
     h=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0]
     assert np.isfinite(h).all(); hmaps.append(h)
    hmaps=np.stack(hmaps); ens=hmaps[1:].mean(0)
    for cond,h in zip(CONDITIONS,[*hmaps,ens]):
     aiou,auc,sim,mae=sample_metrics(h,gt); mask=h>=.5;truth=gt.astype(bool)
     results.append({'affordance':aff,'shape_id':sid,'category':cat,'condition':cond,'gt_points':int(truth.sum()),'aiou':float(aiou),'auc':float(auc),'sim':float(sim),'mae':float(mae),'iou_at_05':float((mask&truth).sum()/(mask|truth).sum()),'recall_at_05':float(mask[truth].mean()),'false_positive_rate_at_05':float(mask[~truth].mean()),'predicted_positive_points':int(mask.sum())})
     key=f'{aff}__{sid}'
     if key not in saved: saved[key+'_xyz']=xyz; saved[key+'_gt']=gt
     saved[key+'__'+cond]=h
    print(f'Completed {aff} object {ix}/20: {cat}, {sid}; GT points={int(gt.sum())}',flush=True)
 # Means and paired, within-object bootstrap for each affordance/query.
 means=[];boot={}
 rng=np.random.default_rng(1)
 for aff in AFFORDANCES:
  objids=[x[2] for x in selected[aff]]; by={(x['shape_id'],x['condition']):x for x in results if x['affordance']==aff}
  for cond in CONDITIONS:
   rr=[x for x in results if x['affordance']==aff and x['condition']==cond]
   means.append({'affordance':aff,'condition':cond,'n_objects':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['aiou','auc','sim','mae','iou_at_05','recall_at_05','false_positive_rate_at_05']}})
  boot[aff]={}
  for cond in CONDITIONS[1:]:
   dif=np.array([by[(sid,cond)]['aiou']-by[(sid,'original')]['aiou'] for sid in objids])
   samples=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(10000)])
   boot[aff][cond]={'mean_delta_aiou':float(dif.mean()),'paired_bootstrap_95pct_ci':[float(z) for z in np.quantile(samples,[.025,.975])],'per_object_deltas':dif.tolist()}
 np.savez_compressed(OUT/'per_object_heatmaps.npz',**saved)
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,per_object_metrics=results,mean_metrics=means,paired_delta_vs_original=boot)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}'); traceback.print_exc()
finally:
 save(); print('Final status:',meta['status'],flush=True)
