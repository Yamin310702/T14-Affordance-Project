"""E16: calibrate thresholds for positive recall under a negative-object alarm cap."""
import os, sys, json, time, hashlib, traceback, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
OUT = ROOT / 'E16_operating_point'
OUT.mkdir(exist_ok=True)
os.environ.update(HF_HOME=str(ROOT/'hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
sys.path.insert(0, str(ROOT/'espresso-3d-reimplementation'))
from espresso.model import load_espresso
sys.path.insert(0, str(REPO))
from dataset import AffordNetDataset

AFF = ['grab', 'pour', 'wrap', 'accommodate', 'unlock']
LABEL = {'grab':0, 'pour':8, 'wrap':7, 'accommodate':1, 'unlock':3}
CAP = 0.10
GRID = np.linspace(0.0, 1.0, 101)
CAL_N = 40
TEST_N = 40
meta = {
  'experiment':'E16', 'status':'loading',
  'model':'Unofficial Espresso-3D with Affogato-pretrained weights',
  'device':DEVICE, 'mps_available':bool(torch.backends.mps.is_available()),
  'threshold_rule':('For each affordance and prompt separately, among thresholds with calibration negative-object alarm rate <= 0.10, maximize mean per-positive-object point recall; tie-break by lower negative alarm rate, then higher threshold.'),
  'calibration':f'{CAL_N} positive and {CAL_N} negative OpenAD train shapes per affordance; calibration IDs are the deterministic E15 IDs; original-query scores reused from E15 and selected-paraphrase scores generated in E16.',
  'test':f'{TEST_N} positive and {TEST_N} negative OpenAD train shapes per affordance, deterministically selected from IDs disjoint from all calibration IDs; query thresholds frozen before test scoring.',
  'scope_note':'The provided model is pretrained on Affogato and is not fine-tuned on OpenAD. E16 uses a new object-disjoint calibration/test split carved from OpenAD train. This is cross-dataset exploratory evaluation, not an official Espresso or Affogato benchmark.',
  'false_alarm_definition':'A negative object is flagged if any point score is >= threshold.',
  'sample_design':'Balanced, affordance-stratified object sample; does not estimate natural prevalence.',
  'seed':16,
}
def save():
    (OUT/'run.json').write_text(json.dumps(meta, indent=2, allow_nan=True)+'\n')

save()
try:
    t0 = time.monotonic()
    torch.manual_seed(16); np.random.seed(16)
    train = AffordNetDataset(str(REPO/'data'), 'train', partial=True)
    e15 = json.loads((ROOT/'E15_threshold_calibration/run.json').read_text())
    e13 = json.loads((ROOT/'E13_confirmation/run.json').read_text())
    old_npz = np.load(ROOT/'E15_threshold_calibration/calibration_predictions.npz')
    # Calibration IDs are exactly those preselected in E15, independent of scores.
    cal = {}
    cal_all_ids = set()
    for aff in AFF:
        cal[aff] = {}
        for group in ['positive', 'negative']:
            rows = e15['calibration_objects'][aff][group]
            cal[aff][group] = rows
            cal_all_ids.update(row['shape_id'] for row in rows)
    unique = {}
    for i,d in enumerate(train.all_data): unique.setdefault(d['shape_id'], (i,d))
    assert not cal_all_ids - set(unique), 'E15 calibration IDs missing from train dataset'

    # Pick a second, globally disjoint object sample for test. Labels only determine
    # positive/negative eligibility; predictions are not consulted during selection.
    test, claimed = {}, set(cal_all_ids)
    eligible = {}
    for aff in AFF:
        cid = LABEL[aff]; pos=[]; neg=[]
        for sid,(i,d) in unique.items():
            if sid in claimed: continue
            y = np.asarray(d['data_info']['label']).reshape(-1).astype(int)
            count = int((y == cid).sum())
            item = (hashlib.sha256(f'E16-test-{aff}-{sid}'.encode()).hexdigest(), i, sid, d['semantic class'], count)
            (pos if count else neg).append(item)
        eligible[aff] = {'positive':len(pos), 'negative':len(neg)}
        if len(pos) < TEST_N or len(neg) < TEST_N:
            raise RuntimeError(f'Insufficient disjoint test objects for {aff}: {eligible[aff]}')
        groups={'positive':sorted(pos)[:TEST_N], 'negative':sorted(neg)[:TEST_N]}
        test[aff] = groups
        for rows in groups.values(): claimed.update(row[2] for row in rows)
    assert not (cal_all_ids & {r[2] for a in AFF for g in test[a].values() for r in g})
    meta.update(status='running', train_unique_shapes=len(unique), calibration_unique_shape_ids=len(cal_all_ids),
                test_unique_shape_ids=sum(len(rows) for a in AFF for rows in test[a].values()),
                eligible_disjoint_test=eligible,
                affordances={a:{'label_id':LABEL[a], 'original_query':e13['affordances'][a]['original_query'],
                                'selected_paraphrase':e13['affordances'][a]['selected_paraphrase']} for a in AFF},
                calibration_objects={a:cal[a] for a in AFF},
                test_objects={a:{g:[{'index':i,'shape_id':sid,'category':cat,'gt_points':n,'selection_hash':h} for h,i,sid,cat,n in rows] for g,rows in test[a].items()} for a in AFF})
    save()
    print(json.dumps({'status':'running','device':DEVICE,'MPS available':torch.backends.mps.is_available(),
                      'calibration_shapes':len(cal_all_ids),'test_shapes':meta['test_unique_shape_ids'],'eligible_test':eligible},indent=2),flush=True)

    model = load_espresso('LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors', device='cpu').eval().to(DEVICE)
    # Store both prompt score maps for calibration/test; reuse E15 original maps.
    scores = {}
    for aff in AFF:
        for group in ['positive','negative']:
            for row in cal[aff][group]:
                sid=row['shape_id']; k=f'{aff}__{sid}__'
                scores[k+'cal_original']=old_npz[k+'pred'].astype(np.float32)
                scores[k+'gt']=old_npz[k+'gt'].astype(np.float32)
    start=time.monotonic()
    with torch.inference_mode():
        # Add selected paraphrase calibration maps and the independent test maps.
        for aff in AFF:
            cid=LABEL[aff]
            queries={'original':e13['affordances'][aff]['original_query'],
                     'selected_paraphrase':e13['affordances'][aff]['selected_paraphrase']}
            for group in ['positive','negative']:
                for row in cal[aff][group]:
                    sid=row['shape_id']; i=row['index']; points,_,labels,sid2,_=train[i]
                    assert sid==sid2
                    xyz=np.asarray(points,dtype=np.float32)
                    gt=(np.asarray(labels).reshape(-1).astype(int)==cid).astype(np.float32)
                    x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE); feat=model.vision(x)
                    q=queries['selected_paraphrase']
                    scores[f'{aff}__{sid}__cal_selected_paraphrase']=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0].astype(np.float32)
                print(f'Calibration paraphrase scores: {aff} {group} done',flush=True)
            for group in ['positive','negative']:
                for ix,row in enumerate(test[aff][group],1):
                    _,i,sid,cat,_=row
                    points,_,labels,sid2,_=train[i]; assert sid==sid2
                    xyz=np.asarray(points,dtype=np.float32)
                    gt=(np.asarray(labels).reshape(-1).astype(int)==cid).astype(np.float32)
                    assert len(xyz)==2048 and ((gt.sum()>0)==(group=='positive'))
                    x=torch.from_numpy(xyz).unsqueeze(0).to(DEVICE); feat=model.vision(x)
                    key=f'{aff}__{sid}__'; scores[key+'gt']=gt; scores[key+'xyz']=xyz
                    for cond,q in queries.items():
                        pred=model(x,[q],point_feat=feat).sigmoid().cpu().numpy()[0].astype(np.float32)
                        assert np.isfinite(pred).all()
                        scores[key+cond]=pred
                    print(f'Test predictions: {aff} {group} {ix}/{len(test[aff][group])}: {sid}',flush=True)

    # Select per-prompt cutoff on calibration data: max positive point recall
    # while at most 10% of calibration negative objects are flagged.
    thresholds=[]; calibration_rows=[]
    for aff in AFF:
        for cond,score_suffix in [('original','cal_original'),('selected_paraphrase','cal_selected_paraphrase')]:
            positives=[scores[f'{aff}__{r["shape_id"]}__{score_suffix}'] for r in cal[aff]['positive']]
            posgts=[scores[f'{aff}__{r["shape_id"]}__gt'].astype(bool) for r in cal[aff]['positive']]
            negatives=[scores[f'{aff}__{r["shape_id"]}__{score_suffix}'] for r in cal[aff]['negative']]
            feasible=[]
            for threshold in GRID:
                alarms=float(np.mean([np.any(p>=threshold) for p in negatives]))
                recall=float(np.mean([float(((p>=threshold)&g).sum()/g.sum()) for p,g in zip(positives,posgts)]))
                if alarms <= CAP+1e-12: feasible.append((recall,-alarms,float(threshold),alarms))
                calibration_rows.append({'affordance':aff,'condition':cond,'threshold':float(threshold),'positive_object_macro_recall':recall,'negative_object_alarm_rate':alarms,'feasible':alarms<=CAP+1e-12})
            if not feasible: raise RuntimeError(f'No feasible threshold for {aff}/{cond}')
            best=max(feasible,key=lambda x:(x[0],x[1],x[2]))
            thresholds.append({'affordance':aff,'condition':cond,'threshold':best[2],
                               'cal_positive_object_macro_recall':best[0], 'cal_negative_object_alarm_rate':best[3],
                               'n_positive':CAL_N,'n_negative':CAL_N,'alarm_cap':CAP,
                               'selection_rule':'maximize macro per-positive-object point recall; then lower negative alarm; then higher cutoff'})
    meta['calibrated_thresholds']=thresholds

    # Frozen test metrics; bootstrap by object within positive and negative groups.
    test_rows=[]; paired={}; rng=np.random.default_rng(16)
    for item in thresholds:
        aff=item['affordance']; cond=item['condition']; thr=item['threshold']
        for group in ['positive','negative']:
            for row in test[aff][group]:
                sid=row['shape_id']; key=f'{aff}__{sid}__'; pred=scores[key+cond]; gt=scores[key+'gt'].astype(bool)
                mask=pred>=thr
                inter=int((mask&gt).sum()); union=int((mask|gt).sum())
                test_rows.append({'affordance':aff,'condition':cond,'group':group,'shape_id':sid,'category':row['category'],
                    'threshold':thr,'object_alarm':bool(mask.any()),'positive_point_fraction':float(mask.mean()),
                    'recall':float((mask&gt).sum()/gt.sum()) if gt.any() else None,
                    'precision':float(inter/mask.sum()) if mask.any() else 0.0,
                    'aiou':float(inter/union) if union else None,'mean_score':float(pred.mean()),'max_score':float(pred.max()),
                    'point_fpr':float(mask[~gt].mean()) if (~gt).any() else None})
    for aff in AFF:
        paired[aff]={}
        for group,metric in [('positive','aiou'),('negative','object_alarm')]:
            vals={}
            for cond in ['original','selected_paraphrase']:
                rr=[x for x in test_rows if x['affordance']==aff and x['group']==group and x['condition']==cond]
                vals[cond]=np.array([float(x[metric]) for x in rr])
            d=vals['selected_paraphrase']-vals['original']
            bs=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)])
            paired[aff][group]={'metric':metric,'mean_delta_paraphrase_minus_original':float(d.mean()),
                                'paired_object_bootstrap_95pct_ci':[float(x) for x in np.quantile(bs,[.025,.975])],
                                'original_mean':float(vals['original'].mean()),'paraphrase_mean':float(vals['selected_paraphrase'].mean())}
    # Persist score arrays and human-readable tables.
    np.savez_compressed(OUT/'predictions.npz',**scores)
    for name,rows in [('calibration_threshold_curve.csv',calibration_rows),('test_per_object.csv',test_rows),('thresholds.csv',thresholds)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    meta.update(status='complete',elapsed_seconds=time.monotonic()-start,test_metrics=test_rows,paired_results=paired)
except Exception as e:
    meta.update(status='failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
finally:
    save(); print('Final status:',meta['status'],flush=True)
