"""E09 development set: fixed query sensitivity on distinct partial-view objects."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.update(HF_HOME=str(root/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
sys.path.insert(0,str(root/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
repo=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
sys.path.insert(0,str(repo))
from dataset import AffordNetDataset
out=root/'heldout_query_sensitivity';out.mkdir(exist_ok=True)
queries=['Point to the part used for pouring.',
 'Locate the region from which liquid is poured.',
 'Identify the pouring region.']
meta={'experiment':'E10','status':'loading_dataset','model':'Unofficial Espresso-3D; Affogato-pretrained weights',
 'device':'mps','seed':1,'scope':'Held-out exploratory evaluation, partial-view point clouds, pour affordance only',
 'query_conditions':queries,'ensemble':'Arithmetic mean of the three sigmoid score maps, pointwise',
 'sample_selection':'For each shape ID, take first partial view with any GT pour points; sort eligible IDs by SHA256("E09-pour:"+shape_id); next 20 after the E09 development set, excluding its 10 IDs. Selection uses GT availability only, not prediction outcomes.',
 'metric':'Affogato/Espresso source sample_metrics; aIoU 20 thresholds [0,1], AUC, SIM, MAE',
 'claim_limit':'Development set selected for positive pour labels; held-out from E09 development objects, but still a small pilot rather than a definitive benchmark.'}
(out/'run.json').write_text(json.dumps(meta,indent=2))
try:
 assert torch.backends.mps.is_available()
 torch.manual_seed(1);np.random.seed(1)
 ds=AffordNetDataset(str(repo/'data'),'val',partial=True)
 rows=[];seen=set()
 for i,d in enumerate(ds.all_data):
  sid=d['shape_id']
  if sid in seen:continue
  seen.add(sid)
  y=np.asarray(d['data_info']['label']).reshape(-1).astype(int)
  if np.any(y==8):rows.append((hashlib.sha256(('E09-pour:'+sid).encode()).hexdigest(),i,sid,d['semantic class'],int((y==8).sum())))
 chosen=sorted(rows)[10:30]
 if len(chosen)<20:raise RuntimeError(f'Only {len(chosen)} eligible unique objects')
 meta.update(status='running',dataset_samples=len(ds),eligible_unique_objects=len(rows),objects=[dict(index=i,shape_id=sid,category=cat,gt_pour_points=n,selection_hash=h) for h,i,sid,cat,n in chosen])
 (out/'run.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to('mps')
 results=[];save_rows=[];start=time.monotonic()
 with torch.inference_mode():
  for ix,(sel,i,sid,cat,_) in enumerate(chosen,1):
   points,_,labels,got_sid,got_cat=ds[i];xyz=np.asarray(points,dtype=np.float32);gt=(np.asarray(labels).reshape(-1).astype(int)==8).astype(np.float32)
   assert sid==got_sid and len(xyz)==2048 and gt.sum()>0
   x=torch.from_numpy(xyz).unsqueeze(0).to('mps');feat=model.vision(x)
   hmaps=[]
   # Original baseline query plus the three fixed paraphrases
   qs=['Point to the part you would use to pour.']+queries
   for q in qs:
    h=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0]
    assert np.isfinite(h).all();hmaps.append(h)
   hmaps=np.stack(hmaps);ens=hmaps[1:].mean(0)
   for cond,h in zip(['original','phrase_1','phrase_2','phrase_3','ensemble'],[*hmaps,ens]):
    aiou,auc,sim,mae=sample_metrics(h,gt);m=h>=.5;t=gt.astype(bool)
    row={'shape_id':sid,'category':cat,'condition':cond,'gt_points':int(t.sum()),'aiou':float(aiou),'auc':float(auc),'sim':float(sim),'mae':float(mae),
      'iou_at_05':float((m&t).sum()/(m|t).sum()),'recall_at_05':float(m[t].mean()),'false_positive_rate_at_05':float(m[~t].mean()),'predicted_positive_points':int(m.sum())}
    results.append(row);save_rows.append((sid,xyz,gt,cond,h))
   print(f'Completed object {ix}/20: {cat}, {sid}; GT pour points={int(gt.sum())}',flush=True)
 # per-object aggregate scores (each object receives equal weight)
 conditions=['original','phrase_1','phrase_2','phrase_3','ensemble'];summary=[]
 for cond in conditions:
  rr=[x for x in results if x['condition']==cond]
  summary.append({'condition':cond,'n_objects':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['aiou','auc','sim','mae','iou_at_05','recall_at_05','false_positive_rate_at_05']}})
 # paired bootstrap CI for mean aIoU difference vs original across selected objects
 objids=[x[2] for x in chosen];by={(x['shape_id'],x['condition']):x for x in results}
 rng=np.random.default_rng(1);boot={}
 for cond in conditions[1:]:
  dif=np.array([by[s,cond]['aiou']-by[s,'original']['aiou'] for s in objids])
  samples=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(10000)])
  boot[cond]={'mean_delta_aiou':float(dif.mean()),'paired_bootstrap_95pct_ci':[float(x) for x in np.quantile(samples,[.025,.975])], 'per_object_deltas':dif.tolist()}
 np.savez_compressed(out/'per_object_heatmaps.npz',**{f'{sid}_{cond}':h for sid,xyz,gt,cond,h in save_rows},**{f'{sid}_xyz':xyz for sid,xyz,gt,cond,h in save_rows},**{f'{sid}_gt':gt for sid,xyz,gt,cond,h in save_rows})
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,per_object_metrics=results,mean_metrics=summary,paired_delta_vs_original=boot)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:
 (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n');print('Final status:',meta['status'],flush=True)
