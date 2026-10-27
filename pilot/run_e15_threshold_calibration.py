"""E15: learn per-affordance thresholds on train data; evaluate frozen thresholds on E14."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
os.environ.update(HF_HOME=str(ROOT/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; refusing CPU fallback')
DEVICE='mps';sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
sys.path.insert(0,str(REPO));from dataset import AffordNetDataset
OUT=ROOT/'E15_threshold_calibration';OUT.mkdir(exist_ok=True)
AFF=['grab','pour','wrap','accommodate','unlock'];IDS={'grab':0,'pour':8,'wrap':7,'accommodate':1,'unlock':3}
VAL=ROOT/'E14_mixed_presence'
meta={'experiment':'E15','status':'loading_datasets','model':'Unofficial Espresso-3D; Affogato-pretrained weights','device':'mps','seed':1,
 'scope':'Select threshold on OpenAD train split, apply frozen threshold to E14 validation objects; same threshold per affordance applied to original and selected paraphrase.',
 'prompts_source':'Exact E13 original/selected paraphrase pairs, locked before threshold experiment.',
 'calibration_design':'Per affordance select 40 unique positive-GT and 40 zero-GT training shapes by deterministic SHA256 rank; maximize macro per-object IoU over thresholds 0.00..1.00 in steps of 0.01 on original-query scores only. Empty-empty negative IoU=1; any predicted positive on a negative object gives IoU=0. Ties: closest to 0.50, then higher threshold.',
 'test_design':'Apply each affordance threshold unchanged to the E14 balanced validation samples (20 positive + 20 negative per affordance), for both original and selected paraphrase predictions. Do not refit on E14.',
 'claim_limit':'Small balanced calibration and test sets; threshold not population calibrated; five affordances; no multiplicity correction. Test set was used for prior E14 analyses but not for threshold selection. This is an exploratory threshold transfer test, not an independent untouched final benchmark.'}
def save(): (OUT/'run.json').write_text(json.dumps(meta,indent=2,allow_nan=True)+'\n')
save()
try:
 torch.manual_seed(1);np.random.seed(1)
 train=AffordNetDataset(str(REPO/'data'),'train',partial=True)
 val=AffordNetDataset(str(REPO/'data'),'val',partial=True)
 # Assert train and validation shape IDs are disjoint.
 train_unique={};val_unique={}
 for i,d in enumerate(train.all_data):train_unique.setdefault(d['shape_id'],(i,d))
 for i,d in enumerate(val.all_data):val_unique.setdefault(d['shape_id'],(i,d))
 assert not set(train_unique)&set(val_unique),'train/val shape leakage'
 val_run=json.loads((VAL/'run.json').read_text());e13=json.loads((ROOT/'E13_confirmation/run.json').read_text())
 queries={a:{'original':e13['affordances'][a]['original_query'],'selected_paraphrase':e13['affordances'][a]['selected_paraphrase']} for a in AFF}
 selections={};eligible={}
 for aff in AFF:
  cid=IDS[aff];pos=[];neg=[]
  for sid,(i,d) in train_unique.items():
   y=np.asarray(d['data_info']['label']).reshape(-1).astype(int);n=int((y==cid).sum())
   if n:pos.append((hashlib.sha256(f'E15-cal-{aff}-pos:{sid}'.encode()).hexdigest(),i,sid,d['semantic class'],n))
   else:neg.append((hashlib.sha256(f'E15-cal-{aff}-neg:{sid}'.encode()).hexdigest(),i,sid,d['semantic class'],0))
  eligible[aff]={'positive':len(pos),'negative':len(neg)}
  if len(pos)<40 or len(neg)<40:raise RuntimeError(f'insufficient train calibration shapes for {aff}')
  selections[aff]={'positive':sorted(pos)[:40],'negative':sorted(neg)[:40]}
 # Test IDs are disjoint from all training objects by dataset split.
 meta.update(status='running',train_unique_shapes=len(train_unique),validation_unique_shapes=len(val_unique),eligible_training=eligible,train_validation_shape_overlap=0,
  affordances={a:{'label_id':IDS[a],**queries[a]} for a in AFF},
  calibration_objects={a:{g:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in groups] for g,groups in st.items()} for a,st in selections.items()})
 save();print(json.dumps({'status':'running','eligible_training':eligible,'per_affordance':'40 positive+40 negative','train_val_overlap':0},indent=2),flush=True)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to(DEVICE)
 trainpred={};start=time.monotonic()
 with torch.inference_mode():
  for aff in AFF:
   for group,rows in selections[aff].items():
    for ix,(h,i,sid,cat,n) in enumerate(rows,1):
     points,_,labels,sid2,cat2=train[i];xyz=np.asarray(points,dtype=np.float32);gt=(np.asarray(labels).reshape(-1).astype(int)==IDS[aff]).astype(np.float32)
     assert sid==sid2 and len(xyz)==2048 and ((gt.sum()>0)==(group=='positive'))
     x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE);feat=model.vision(x)
     pred=model(x,[queries[aff]['original']],point_feat=feat).sigmoid().cpu().numpy()[0]
     trainpred[f'{aff}__{sid}__pred']=pred;trainpred[f'{aff}__{sid}__gt']=gt
    print(f'Calibrated-score collection: {aff} {group} {len(rows)} objects',flush=True)
 np.savez_compressed(OUT/'calibration_predictions.npz',**trainpred)
 # Optimize each threshold on balanced train objects; same threshold later applies to both queries.
 thresholds=[];grid=np.linspace(0,1,101)
 for aff in AFF:
  vals=[]
  for group in ['positive','negative']:
   for h,i,sid,cat,n in selections[aff][group]:
    pred=trainpred[f'{aff}__{sid}__pred'];gt=trainpred[f'{aff}__{sid}__gt'].astype(bool)
    for t in grid:
     mask=pred>=t; union=mask|gt
     iou=float((mask&gt).sum()/union.sum()) if union.any() else 1.0
     vals.append((group,sid,float(t),iou))
  means=[np.mean([v[3] for v in vals if v[2]==t]) for t in grid]
  best=max(means); candidates=[float(grid[j]) for j,z in enumerate(means) if abs(z-best)<1e-12]
  threshold=sorted(candidates,key=lambda t:(abs(t-.5),-t))[0]
  thresholds.append({'affordance':aff,'threshold':threshold,'train_balanced_macro_iou':float(best),'threshold_grid_step':.01,'tie_break':'closest to .50, then higher','n_positive':40,'n_negative':40})
 meta['calibrated_thresholds']=thresholds;save()
 # Evaluate frozen values on the saved E14 test predictions.
 e14=np.load(VAL/'predictions.npz');test_ids=val_run['objects'];results=[];paired={}
 rng=np.random.default_rng(1)
 for item in thresholds:
  aff=item['affordance'];t=item['threshold'];paired[aff]={}
  ids={g:[x['shape_id'] for x in test_ids[aff][g]] for g in ['positive','negative']}
  for group in ['positive','negative']:
   for cond in ['original','selected_paraphrase']:
    for sid in ids[group]:
     key=f'{aff}__{sid}__';pred=e14[key+cond].astype(np.float32);gt=e14[key+'gt'].astype(bool);mask=pred>=t
     union=mask|gt;iou=float((mask&gt).sum()/union.sum()) if union.any() else 1.0
     tp=int((mask&gt).sum());fp=int((mask&~gt).sum());fn=int((~mask&gt).sum())
     results.append({'affordance':aff,'group':group,'shape_id':sid,'condition':cond,'threshold':t,'iou_at_calibrated_threshold':iou,'object_alarm':bool(mask.any()),'positive_point_fraction':float(mask.mean()),'recall':float(tp/gt.sum()) if gt.any() else None,'precision':float(tp/mask.sum()) if mask.any() else 0.0,'mean_score':float(pred.mean()),'max_score':float(pred.max()),'gt_points':int(gt.sum())})
  by={(x['group'],x['shape_id'],x['condition']):x for x in results if x['affordance']==aff}
  for group in ['positive','negative','combined']:
   sample_ids=ids[group] if group!='combined' else ids['positive']+ids['negative']
   metric='iou_at_calibrated_threshold' if group!='negative' else 'object_alarm'
   dif=np.array([float(by[(g,sid,'selected_paraphrase')][metric])-float(by[(g,sid,'original')][metric]) for sid in sample_ids for g in ([group] if group!='combined' else ['positive' if sid in ids['positive'] else 'negative'])])
   bs=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(20000)])
   paired[aff][group]={'metric':metric,'mean_delta':float(dif.mean()),'paired_bootstrap_95pct_ci':[float(z) for z in np.quantile(bs,[.025,.975])],'per_object_deltas':dif.tolist()}
 summary=[]
 for aff in AFF:
  for group in ['positive','negative']:
   for cond in ['original','selected_paraphrase']:
    rr=[x for x in results if x['affordance']==aff and x['group']==group and x['condition']==cond]
    fields=['iou_at_calibrated_threshold','object_alarm','positive_point_fraction','recall','precision','mean_score','max_score']
    row={'affordance':aff,'group':group,'condition':cond,'n_objects':len(rr)}
    for f in fields:
     arr=[x[f] for x in rr if isinstance(x[f],(int,float)) and np.isfinite(x[f])]
     row[f]=float(np.mean(arr)) if arr else None
    summary.append(row)
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,test_metrics=results,test_summary=summary,paired_test_results=paired)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:save();print('Final status:',meta['status'],flush=True)
