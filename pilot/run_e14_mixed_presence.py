"""E14: balanced stratified mixed-presence test on fresh objects using MPS."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
os.environ.update(HF_HOME=str(ROOT/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; refusing CPU fallback')
DEVICE='mps'
sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
from espresso.metrics import sample_metrics
sys.path.insert(0,str(REPO));from dataset import AffordNetDataset
OUT=ROOT/'E14_mixed_presence';OUT.mkdir(exist_ok=True)
AFF=['grab','pour','wrap','accommodate','unlock']
meta={'experiment':'E14','status':'loading_dataset','model':'Unofficial Espresso-3D; Affogato-pretrained weights','device':'mps','seed':1,'scope':'Fresh OpenAD validation partial views; balanced stratified test with positive and negative objects per affordance.',
 'sampling':'For each affordance, independently rank unique eligible shape IDs with SHA256 and randomly select 20 positive-GT and 20 zero-GT objects. This is stratified random sampling, not a natural-prevalence sample. All objects used in E09-E13 are excluded. Objects may occur under multiple affordance strata if they have different labels.',
 'prompts_locked_from':'E13 confirmation, itself selecting the best E12 paraphrase before E13. E14 does not change prompts.',
 'metric_definitions':{'positive_objects':'Espresso sample_metrics: aIoU, AUC, SIM, MAE; paired by object for original vs selected paraphrase.','negative_objects':'AUC/aIoU undefined because GT target is empty. Report object alarm rate (any predicted point >=0.5), predicted-positive point fraction at 0.5, mean predicted score, and max score.'},
 'claim_limit':'Small balanced selected sample (20 positive and 20 negative per affordance), not representative of natural object prevalence. Threshold 0.5 is diagnostic only. Five affordances and no multiplicity correction. Unofficial cross-dataset Espresso/Affogato weights on OpenAD. Not an Affostruction test.'}
def save(): (OUT/'run.json').write_text(json.dumps(meta,indent=2,allow_nan=True)+'\n')
save()
try:
 torch.manual_seed(1);np.random.seed(1)
 ds=AffordNetDataset(str(REPO/'data'),'val',partial=True)
 excluded=set()
 for folder in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity','multiaffordance_prompt_study','E13_confirmation']:
  old=json.loads((ROOT/folder/'run.json').read_text());objs=old['objects']
  if isinstance(objs,dict):objs=[x for vv in objs.values() for x in vv]
  excluded.update(x['shape_id'] for x in objs)
 e13=json.loads((ROOT/'E13_confirmation/run.json').read_text())['affordances']
 labels={'grab':0,'pour':8,'wrap':7,'accommodate':1,'unlock':3}
 data={};seen=set()
 for i,d in enumerate(ds.all_data):
  sid=d['shape_id']
  if sid in seen:continue
  seen.add(sid)
  if sid in excluded:continue
  y=np.asarray(d['data_info']['label']).reshape(-1).astype(int)
  data[sid]=(i,d['semantic class'],y)
 chosen={}; eligible={}
 for aff in AFF:
  cid=labels[aff]; pos=[];neg=[]
  for sid,(i,cat,y) in data.items():
   row=(hashlib.sha256(f'E14-{aff}-positive:{sid}'.encode()).hexdigest(),i,sid,cat,int((y==cid).sum()))
   if row[-1]>0:pos.append(row)
   else:neg.append((hashlib.sha256(f'E14-{aff}-negative:{sid}'.encode()).hexdigest(),i,sid,cat,0))
  eligible[aff]={'positive':len(pos),'negative':len(neg)}
  if len(pos)<20 or len(neg)<20:raise RuntimeError(f'{aff}: insufficient positive/negative eligible objects')
  chosen[aff]={'positive':sorted(pos)[:20],'negative':sorted(neg)[:20]}
 meta.update(status='running',dataset_samples=len(ds),excluded_prior_shape_ids=len(excluded),eligible_counts=eligible,
  objects={a:{g:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in groups] for g,groups in strata.items()} for a,strata in chosen.items()})
 save();print(json.dumps({'status':'running','eligible_counts':eligible,'sampled_per_affordance':'20 positive + 20 negative'},indent=2),flush=True)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to(DEVICE)
 results=[];saved={};start=time.monotonic()
 with torch.inference_mode():
  for aff in AFF:
   cid=labels[aff];q0=e13[aff]['original_query'];q1=e13[aff]['selected_paraphrase']
   for group,rows in chosen[aff].items():
    for ix,(h,i,sid,cat,_) in enumerate(rows,1):
     points,_,labels_arr,got_sid,got_cat=ds[i];xyz=np.asarray(points,dtype=np.float32);gt=(np.asarray(labels_arr).reshape(-1).astype(int)==cid).astype(np.float32)
     assert sid==got_sid and len(xyz)==2048 and ((gt.sum()>0)==(group=='positive'))
     x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE);feat=model.vision(x)
     for cond,q in [('original',q0),('selected_paraphrase',q1)]:
      pred=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0];assert np.isfinite(pred).all()
      aiou,auc,sim,mae=sample_metrics(pred,gt)
      mask=pred>=.5;truth=gt.astype(bool)
      results.append({'affordance':aff,'group':group,'shape_id':sid,'category':cat,'condition':cond,'query':q,'gt_points':int(truth.sum()),'aiou':float(aiou),'auc':float(auc),'sim':float(sim),'mae':float(mae),'object_alarm_at_05':bool(mask.any()),'predicted_positive_fraction_at_05':float(mask.mean()),'mean_score':float(pred.mean()),'max_score':float(pred.max()),'point_false_positive_rate_at_05':float(mask[~truth].mean()) if truth.any() else float(mask.mean())})
      key=f'{aff}__{sid}__';saved[key+cond]=pred;saved.setdefault(key+'xyz',xyz);saved.setdefault(key+'gt',gt)
     print(f'Completed {aff} {group} {ix}/20: {cat}, {sid}; GT points={int(gt.sum())}',flush=True)
 metrics=['aiou','auc','sim','mae','object_alarm_at_05','predicted_positive_fraction_at_05','mean_score','max_score','point_false_positive_rate_at_05']
 summary=[];paired={};rng=np.random.default_rng(1)
 for aff in AFF:
  paired[aff]={}
  for group in ['positive','negative']:
   for cond in ['original','selected_paraphrase']:
    rr=[x for x in results if x['affordance']==aff and x['group']==group and x['condition']==cond]
    row={'affordance':aff,'group':group,'condition':cond,'n_objects':len(rr)}
    for m in metrics:
     vals=[x[m] for x in rr if isinstance(x[m],(int,float,np.number)) and np.isfinite(x[m])]
     row[m]=float(np.mean(vals)) if vals else None
    summary.append(row)
   by={(x['shape_id'],x['condition']):x for x in results if x['affordance']==aff and x['group']==group}
   ids=[x[2] for x in chosen[aff][group]]
   primary='aiou' if group=='positive' else 'object_alarm_at_05'
   dif=np.array([float(by[(sid,'selected_paraphrase')][primary])-float(by[(sid,'original')][primary]) for sid in ids])
   bs=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(20000)])
   paired[aff][group]={'primary_metric':primary,'mean_delta':float(dif.mean()),'paired_bootstrap_95pct_ci':[float(z) for z in np.quantile(bs,[.025,.975])],'per_object_deltas':dif.tolist()}
 np.savez_compressed(OUT/'predictions.npz',**saved)
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,per_object_metrics=results,mean_metrics=summary,paired_results=paired)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally: save();print('Final status:',meta['status'],flush=True)
