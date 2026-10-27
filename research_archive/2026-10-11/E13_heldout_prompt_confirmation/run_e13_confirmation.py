"""E13 preselected prompt confirmation on new, disjoint OpenAD partial objects."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent; REPO=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
os.environ.update(HF_HOME=str(ROOT/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; this MPS confirmation runner will not fall back to CPU')
DEVICE='mps'
sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
sys.path.insert(0,str(REPO))
from dataset import AffordNetDataset
OUT=ROOT/'E13_confirmation';OUT.mkdir(exist_ok=True)
# The single comparison prompt per affordance is locked using E12 means before E13 scores are examined.
AFF={
 'grab':(0,'Point to the part you would grab.','Point to the part you would grasp.'),
 'pour':(8,'Point to the part you would use to pour.','Point to the part used for pouring.'),
 'wrap':(7,'Point to the part you can wrap your hand around.','Point to the region designed to be held by wrapping your hand around it.'),
 'accommodate':(1,'Point to the part of the object that can hold something.','Point to the part where you would place or keep something.'),
 'unlock':(3,'Point to the part you would use to unlock this object.','Point to the part you would operate to gain access.'),
}
CONDITIONS=['original','selected_paraphrase']
meta={'experiment':'E13','status':'loading_dataset','model':'Unofficial Espresso-3D; Affogato-pretrained weights','device':'mps','seed':1,'scope':'Preselected prompt confirmation on fresh OpenAD validation partial views; 5 affordances; 40 objects per affordance.',
 'prompts_locked_from':'E12 mean aIoU; per affordance, selected the best of three tested paraphrases before evaluating E13 objects.',
 'affordances':{a:{'label_id':cid,'original_query':q0,'selected_paraphrase':q1} for a,(cid,q0,q1) in AFF.items()},
 'selection_rule':'Exclude every shape used in E09, E10, E11, or E12. For each affordance, hash eligible positive-GT unique shape IDs with SHA256("E13-confirm:<affordance>:" + shape_id), process affordances in the fixed order grab, pour, wrap, accommodate, unlock, and greedily select 40 IDs per affordance without reuse across E13 affordance sets. Selection uses GT label availability only, not predictions.',
 'primary_metric':'aIoU per object; paired mean difference selected paraphrase minus original, within-affordance bootstrap 95% CI.',
 'claim_limit':'Confirmatory follow-up on deterministic GT-positive samples; five comparisons without multiplicity correction; cross-dataset evaluation of unofficial Affogato-pretrained Espresso on OpenAD. Not a representative benchmark and not a test of Affostruction.'}
def save(): (OUT/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
try:
 torch.manual_seed(1);np.random.seed(1)
 ds=AffordNetDataset(str(REPO/'data'),'val',partial=True)
 excluded=set()
 for folder in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity','multiaffordance_prompt_study']:
  old=json.loads((ROOT/folder/'run.json').read_text()); objs=old['objects']
  if isinstance(objs,dict): objs=[x for rs in objs.values() for x in rs]
  excluded.update(x['shape_id'] for x in objs)
 selected={}; claimed=set(); eligible={}
 for aff,(cid,q0,q1) in AFF.items():
  seen=set();rows=[]
  for i,d in enumerate(ds.all_data):
   sid=d['shape_id']
   if sid in seen:continue
   seen.add(sid)
   y=np.asarray(d['data_info']['label']).reshape(-1).astype(int)
   if np.any(y==cid) and sid not in excluded:
    rows.append((hashlib.sha256(f'E13-confirm:{aff}:{sid}'.encode()).hexdigest(),i,sid,d['semantic class'],int((y==cid).sum())))
  eligible[aff]=len(rows)
  chosen=[]
  for row in sorted(rows):
   if row[2] in claimed:continue
   chosen.append(row);claimed.add(row[2])
   if len(chosen)==40:break
  if len(chosen)<40:raise RuntimeError(f'{aff}: only {len(chosen)} nonoverlapping fresh objects available')
  selected[aff]=chosen
 meta.update(status='running',dataset_samples=len(ds),excluded_prior_shape_ids=len(excluded),eligible_unique_objects=eligible,objects={a:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in rows] for a,rows in selected.items()},unique_shape_ids_across_E13=len(claimed))
 save();print(json.dumps({'status':'running','eligible':eligible,'objects_per_affordance':40,'unique_E13_shapes':len(claimed)},indent=2),flush=True)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to(DEVICE)
 results=[]; saved={};start=time.monotonic()
 with torch.inference_mode():
  for aff,(cid,q0,q1) in AFF.items():
   for ix,(h,i,sid,cat,_) in enumerate(selected[aff],1):
    points,_,labels,got_sid,got_cat=ds[i];xyz=np.asarray(points,dtype=np.float32);gt=(np.asarray(labels).reshape(-1).astype(int)==cid).astype(np.float32)
    assert sid==got_sid and len(xyz)==2048 and gt.sum()>0
    x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE);feat=model.vision(x)
    for cond,q in [('original',q0),('selected_paraphrase',q1)]:
     pred=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0]
     assert np.isfinite(pred).all()
     aiou,auc,sim,mae=sample_metrics(pred,gt);mask=pred>=.5;truth=gt.astype(bool)
     results.append({'affordance':aff,'shape_id':sid,'category':cat,'condition':cond,'query':q,'gt_points':int(truth.sum()),'aiou':float(aiou),'auc':float(auc),'sim':float(sim),'mae':float(mae),'iou_at_05':float((mask&truth).sum()/(mask|truth).sum()),'recall_at_05':float(mask[truth].mean()),'false_positive_rate_at_05':float(mask[~truth].mean())})
     key=f'{aff}__{sid}__';saved[key+cond]=pred;saved.setdefault(key+'xyz',xyz);saved.setdefault(key+'gt',gt)
    print(f'Completed {aff} {ix}/40: {cat}, {sid}; GT points={int(gt.sum())}',flush=True)
 means=[];boot={};rng=np.random.default_rng(1)
 for aff in AFF:
  ids=[x[2] for x in selected[aff]];by={(x['shape_id'],x['condition']):x for x in results if x['affordance']==aff}
  for cond in CONDITIONS:
   rr=[x for x in results if x['affordance']==aff and x['condition']==cond]
   means.append({'affordance':aff,'condition':cond,'n_objects':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['aiou','auc','sim','mae','iou_at_05','recall_at_05','false_positive_rate_at_05']}})
  dif=np.array([by[(sid,'selected_paraphrase')]['aiou']-by[(sid,'original')]['aiou'] for sid in ids])
  bs=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(20000)])
  boot[aff]={'mean_delta_aiou':float(dif.mean()),'paired_bootstrap_95pct_ci':[float(z) for z in np.quantile(bs,[.025,.975])],'n_positive_deltas':int((dif>0).sum()),'n_ties':int((dif==0).sum()),'per_object_deltas':dif.tolist()}
 np.savez_compressed(OUT/'predictions.npz',**saved)
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,per_object_metrics=results,mean_metrics=means,paired_results=boot)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:
 save();print('Final status:',meta['status'],flush=True)
