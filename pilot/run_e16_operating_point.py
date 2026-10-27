"""E16 focused primary test: accommodate prompt threshold under a 10% alarm cap."""
import os, sys, json, time, hashlib, traceback, csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
OUT=Path(os.environ.get('E16_OUTPUT_DIR',str(ROOT/'E16_operating_point'))); OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HOME=str(ROOT/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np, torch
REQUIRE_MPS='--require-mps' in sys.argv
if REQUIRE_MPS and not torch.backends.mps.is_available():
 raise RuntimeError('MPS was explicitly requested but torch.backends.mps.is_available() is False; refusing to run on CPU.')
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
sys.path.insert(0,str(REPO))
from dataset import AffordNetDataset

AFF='accommodate'; CID=1; CAL_N=40; TEST_N=20; CAP=.10; GRID=np.linspace(0,1,101)
meta={'experiment':'E16','status':'loading','primary_affordance':AFF,
 'model':'Unofficial Espresso-3D reimplementation with Affogato-pretrained weights',
 'device':DEVICE,'mps_available':bool(torch.backends.mps.is_available()),'seed':16,
 'hypothesis':'The selected accommodation paraphrase may produce more false alarms than the original wording when scores are thresholded to the same pre-registered 10% calibration false-alarm budget.',
 'rule':'For each prompt separately, among thresholds with <=10% calibration negative-object alarms, maximize macro per-positive-object point recall; ties prefer lower negative-object alarm rate then higher threshold.',
 'calibration_design':'40 positive and 40 negative OpenAD train shapes for accommodate, matching E15 calibration IDs. Original-query score maps reused from E15; selected-paraphrase maps computed in E16.',
 'calibration_device_note':('Original-query calibration maps come from E15 (MPS); selected-paraphrase calibration and fresh validation test maps also used MPS.' if DEVICE=='mps' else 'Original-query calibration maps come from E15 (MPS); selected-paraphrase calibration and fresh validation test maps use CPU when MPS is unavailable. Report this mixed-device limitation.'),
 'test_design':'20 positive and 20 negative fresh OpenAD validation shapes, excluded from E09-E14 samples and disjoint from calibration IDs. Freeze prompt-specific cutoffs before scoring test shapes.',
 'alarm_definition':'A negative object is flagged when any point score is >= its frozen cutoff.',
 'sample_note':'Balanced stratified sample; not a natural-prevalence estimate. Current session reports MPS unavailable, so this run uses CPU. The model is fixed and was not fine-tuned on OpenAD; this is cross-dataset exploratory evaluation, not an official benchmark.'}
def save(): (OUT/'run.json').write_text(json.dumps(meta,indent=2,allow_nan=True)+'\n')
save()
try:
 torch.manual_seed(16);np.random.seed(16)
 train=AffordNetDataset(str(REPO/'data'),'train',partial=True)
 val=AffordNetDataset(str(REPO/'data'),'val',partial=True)
 e15=json.loads((ROOT/'E15_threshold_calibration/run.json').read_text())
 e13=json.loads((ROOT/'E13_confirmation/run.json').read_text())
 old=np.load(ROOT/'E15_threshold_calibration/calibration_predictions.npz')
 cal={g:e15['calibration_objects'][AFF][g] for g in ['positive','negative']}
 train_by_id={}
 for i,d in enumerate(train.all_data): train_by_id.setdefault(d['shape_id'],(i,d))
 assert all(r['shape_id'] in train_by_id for g in cal.values() for r in g)
 val_by_id={}
 for i,d in enumerate(val.all_data): val_by_id.setdefault(d['shape_id'],(i,d))
 excluded=set()
 for folder in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity','multiaffordance_prompt_study','E13_confirmation','E14_mixed_presence']:
  oldrun=json.loads((ROOT/folder/'run.json').read_text()); objs=oldrun['objects']
  if isinstance(objs,dict): objs=[x for rs in objs.values() for x in (rs.values() if isinstance(rs,dict) else rs)]
  for x in objs:
   if isinstance(x,list):
    for y in x:
     if isinstance(y,dict) and 'shape_id' in y: excluded.add(y['shape_id'])
   elif isinstance(x,dict) and 'shape_id' in x: excluded.add(x['shape_id'])
 # robustly collect E14's nested positive/negative records
 e14=json.loads((ROOT/'E14_mixed_presence/run.json').read_text())
 for groups in e14['objects'].values():
  for rows in groups.values(): excluded.update(x['shape_id'] for x in rows)
 for folder in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity','multiaffordance_prompt_study','E13_confirmation']:
  rr=json.loads((ROOT/folder/'run.json').read_text()); objs=rr['objects']
  if isinstance(objs,dict):
   for v in objs.values():
    if isinstance(v,list): excluded.update(x['shape_id'] for x in v if isinstance(x,dict) and 'shape_id' in x)
  elif isinstance(objs,list): excluded.update(x['shape_id'] for x in objs if isinstance(x,dict) and 'shape_id' in x)
 # Select fresh validation shapes by deterministic hash, based on GT eligibility only.
 pos=[];neg=[]
 for sid,(i,d) in val_by_id.items():
  if sid in excluded: continue
  y=np.asarray(d['data_info']['label']).reshape(-1).astype(int); n=int((y==CID).sum())
  row=(hashlib.sha256(f'E16-accommodate-{"positive" if n else "negative"}:{sid}'.encode()).hexdigest(),i,sid,d['semantic class'],n)
  (pos if n else neg).append(row)
 if len(pos)<TEST_N or len(neg)<TEST_N: raise RuntimeError(f'Not enough fresh val objects: positive={len(pos)}, negative={len(neg)}')
 test={'positive':sorted(pos)[:TEST_N],'negative':sorted(neg)[:TEST_N]}
 q0=e13['affordances'][AFF]['original_query']; q1=e13['affordances'][AFF]['selected_paraphrase']
 meta.update(status='running',eligible_fresh_validation={'positive':len(pos),'negative':len(neg)},
  excluded_prior_validation_shape_ids=len(excluded),
  calibration_objects=cal,
  test_objects={g:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in rows] for g,rows in test.items()},
  prompts={'original':q0,'selected_paraphrase':q1})
 save(); print(json.dumps({'status':'running','device':DEVICE,'eligible_fresh':meta['eligible_fresh_validation'],'test_per_group':TEST_N},indent=2),flush=True)
 scores={}
 for g,rows in cal.items():
  for r in rows:
   sid=r['shape_id']; k=f'{AFF}__{sid}__'
   scores[k+'cal_original']=old[k+'pred'].astype(np.float32)
   scores[k+'gt']=old[k+'gt'].astype(np.float32)
 model=load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors',device='cpu').eval().to(DEVICE)
 start=time.monotonic()
 with torch.inference_mode():
  for group,rows in cal.items():
   for ix,r in enumerate(rows,1):
    sid=r['shape_id'];i=r['index'];points,_,labels,sid2,_=train[i];assert sid==sid2
    gt=(np.asarray(labels).reshape(-1).astype(int)==CID).astype(np.float32)
    x=torch.from_numpy(np.asarray(points,dtype=np.float32)).unsqueeze(0).to(DEVICE);feat=model.vision(x)
    pred=model(x,[q1],point_feat=feat).sigmoid().cpu().numpy()[0].astype(np.float32)
    scores[f'{AFF}__{sid}__cal_selected_paraphrase']=pred
    if ix%10==0: print(f'Calibration {group}: {ix}/{len(rows)}',flush=True)
  for group,rows in test.items():
   for ix,(h,i,sid,cat,n) in enumerate(rows,1):
    points,_,labels,sid2,_=val[i];assert sid==sid2
    xyz=np.asarray(points,dtype=np.float32);gt=(np.asarray(labels).reshape(-1).astype(int)==CID).astype(np.float32)
    assert len(xyz)==2048 and ((gt.sum()>0)==(group=='positive'))
    x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE);feat=model.vision(x);k=f'{AFF}__{sid}__'
    scores[k+'xyz']=xyz;scores[k+'gt']=gt
    for cond,q in [('original',q0),('selected_paraphrase',q1)]:
     pred=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0].astype(np.float32)
     assert np.isfinite(pred).all();scores[k+cond]=pred
    print(f'Test {group}: {ix}/{len(rows)} {sid}',flush=True)
 # Pick prompt-specific cutoff on calibration scores.
 thresholds=[];curve=[]
 for cond,suffix in [('original','cal_original'),('selected_paraphrase','cal_selected_paraphrase')]:
  positives=[scores[f'{AFF}__{r["shape_id"]}__{suffix}'] for r in cal['positive']]
  gts=[scores[f'{AFF}__{r["shape_id"]}__gt'].astype(bool) for r in cal['positive']]
  negatives=[scores[f'{AFF}__{r["shape_id"]}__{suffix}'] for r in cal['negative']]
  feasible=[]
  for t in GRID:
   alarm=float(np.mean([np.any(p>=t) for p in negatives]))
   recall=float(np.mean([((p>=t)&g).sum()/g.sum() for p,g in zip(positives,gts)]))
   curve.append({'condition':cond,'threshold':float(t),'positive_object_macro_recall':recall,'negative_object_alarm_rate':alarm,'feasible':alarm<=CAP+1e-12})
   if alarm<=CAP+1e-12: feasible.append((recall,-alarm,float(t),alarm))
  if not feasible: raise RuntimeError(f'No feasible cutoff for {cond}')
  best=max(feasible,key=lambda x:(x[0],x[1],x[2]))
  thresholds.append({'condition':cond,'threshold':best[2],'cal_positive_object_macro_recall':best[0],
                     'cal_negative_object_alarm_rate':best[3],'n_positive':CAL_N,'n_negative':CAL_N,'alarm_cap':CAP})
 results=[]
 for item in thresholds:
  cond=item['condition'];t=item['threshold']
  for group,rows in test.items():
   for _,i,sid,cat,n in rows:
    p=scores[f'{AFF}__{sid}__{cond}'];gt=scores[f'{AFF}__{sid}__gt'].astype(bool);mask=p>=t
    inter=int((mask&gt).sum());union=int((mask|gt).sum())
    results.append({'affordance':AFF,'condition':cond,'group':group,'shape_id':sid,'category':cat,'threshold':t,
      'object_alarm':bool(mask.any()),'aiou':float(inter/union) if union else None,
      'recall':float(inter/gt.sum()) if gt.any() else None,'precision':float(inter/mask.sum()) if mask.any() else 0.0,
      'positive_point_fraction':float(mask.mean()),'point_fpr':float(mask[~gt].mean()) if (~gt).any() else None,
      'mean_score':float(p.mean()),'max_score':float(p.max())})
 rng=np.random.default_rng(16);paired={}
 for group,metric in [('positive','aiou'),('negative','object_alarm')]:
  by={cond:np.array([float(x[metric]) for x in results if x['group']==group and x['condition']==cond]) for cond in ['original','selected_paraphrase']}
  d=by['selected_paraphrase']-by['original'];bs=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)])
  paired[group]={'metric':metric,'original_mean':float(by['original'].mean()),'paraphrase_mean':float(by['selected_paraphrase'].mean()),
    'delta':float(d.mean()),'paired_object_bootstrap_95pct_ci':[float(x) for x in np.quantile(bs,[.025,.975])]}
 np.savez_compressed(OUT/'predictions.npz',**scores)
 for fname,rows in [('threshold_curve.csv',curve),('thresholds.csv',thresholds),('test_per_object.csv',results)]:
  with (OUT/fname).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meta.update(status='complete',elapsed_seconds=time.monotonic()-start,calibrated_thresholds=thresholds,
  test_metrics=results,paired_results=paired)
except Exception as e:
 meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally: save();print('Final status:',meta['status'],flush=True)
