#!/usr/bin/env python3
"""E19: frozen OpenAD, label-aligned words vs fixed paraphrase score maps.

No prompt selection, tuning or model training. See the dated PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
STUDY = ROOT / 'research_archive/2026-11-09/E19_prompt_strategies'
CONDITIONS = ('A_canonical', 'B_single_paraphrase', 'C_three_map_average')
LABELS = ['grasp','contain','lift','openable','layable','sittable','support','wrap_grasp',
          'pourable','move','displaY','pushable','pull','listen','wear','press','cut','stab']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load_prompts(path):
    spec = json.loads(Path(path).read_text())
    actions = spec['actions']
    if [a['canonical'] for a in actions] != LABELS:
        raise ValueError('Canonical label slots must exactly match the audited 18-slot order.')
    for a in actions:
        ps = a['paraphrases']
        if len(ps) != 3 or len(set(ps)) != 3 or any(not isinstance(s, str) or not s.strip() for s in ps):
            raise ValueError('Each action needs three distinct, nonempty fixed paraphrases.')
    if spec['none_query'] != 'none':
        raise ValueError('The none competitor is fixed across conditions.')
    vocabs = [[a['canonical'] for a in actions] + ['none']]
    vocabs += [[a['paraphrases'][k] for a in actions] + ['none'] for k in range(3)]
    return spec, vocabs


def condition_maps(maps):
    """Average complete 19-way probability maps, including the none competitor."""
    if len(maps) != 4 or any(x.shape != maps[0].shape for x in maps):
        raise ValueError('Expected four same-shaped maps: canonical and three paraphrase vocabularies.')
    return (maps[0], maps[1], np.mean(np.stack(maps[1:]), axis=0))


def confusion(gt, pred):
    gt = np.asarray(gt).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if gt.shape != pred.shape or np.any((gt < 0) | (gt > 18)) or np.any((pred < 0) | (pred > 18)):
        raise ValueError('Labels must align and be in the 19-slot vocabulary.')
    return np.bincount(gt.astype(np.int64)*19 + pred.astype(np.int64), minlength=361).reshape(19,19)


def counts(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return np.stack([tp, cm.sum(axis=-2)-tp, cm.sum(axis=-1)-tp], axis=-1)


def ious(ct):
    union = ct.sum(axis=-1)
    return np.divide(ct[...,0], union, out=np.full(union.shape, np.nan), where=union > 0)


def paired_intervals(per_object_counts, seed, n_boot):
    """Recompute global macro IoU after resampling whole objects with replacement."""
    rng = np.random.default_rng(seed)
    n = len(per_object_counts)
    samples = []
    for _ in range(n_boot):
        selected = rng.integers(0, n, n)
        scores = ious(per_object_counts[selected].sum(axis=0))[:,:18]
        samples.append(np.nanmean(scores, axis=-1))
    samples = np.asarray(samples)
    return {f'{CONDITIONS[k]}_minus_A': np.quantile(samples[:,k]-samples[:,0], [.025,.975]).tolist() for k in (1,2)}


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--split',choices=['partial','full'],required=True)
    ap.add_argument('--device',choices=['cpu','mps'],default='cpu')
    ap.add_argument('--max-shapes',type=int,default=0,help='0: all validation shapes; positive values are smoke checks')
    ap.add_argument('--prompts',type=Path,default=STUDY/'prompts.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--threads',type=int,default=4)
    args = ap.parse_args()
    if args.max_shapes < 0 or args.bootstrap < 1 or args.threads < 1:
        ap.error('max-shapes must be nonnegative; bootstrap and threads must be positive.')
    spec, vocabs = load_prompts(args.prompts)
    repo = args.repo.expanduser().resolve()
    checkpoint = (args.checkpoint or repo/'checkpoints/best_model_openad_pn2_partial.t7').resolve()
    out = args.output_dir.expanduser().resolve()
    # Refuse overwrites, including prior failed runs, to preserve provenance.
    out.mkdir(parents=True,exist_ok=False)
    meta = {'experiment':'E19_prompt_strategies','status':'preflight','split':args.split,
            'device':args.device,'seed':args.seed,'conditions':CONDITIONS,'prompts':spec,
            'prompts_sha256':digest(args.prompts),'runner_sha256':digest(__file__),
            'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
            'max_shapes':args.max_shapes,'bootstrap_resamples':args.bootstrap,
            'claim_scope':'smoke check only' if args.max_shapes else 'exploratory reused validation; not held-out confirmation',
            'negative_definition':'action absent from all stored full-shape validation labels; no negative inference from partial-view absence',
            'ensemble_rule':'mean of three complete 19-class softmax probability maps, then 19-way argmax; fixed none text',
            'primary_metric':'global 18-action macro IoU; absent-union slots undefined, explicitly excluded',
            'primary_contrast':'C minus A; object-paired bootstrap recomputes global macro IoU',
            'interpretation':'all 18 query slots change together in B/C; this is a whole-vocabulary strategy study, unlike E18 one-slot changes'}
    run_path = out/'run.json'
    def save():run_path.write_text(json.dumps(meta,indent=2)+'\n')
    save()
    try:
        os.environ['OMP_NUM_THREADS']=str(args.threads)
        os.environ['MKL_NUM_THREADS']=str(args.threads)
        sys.path.insert(0,str(repo));sys.path.insert(0,str(ROOT/'pilot'))
        import torch
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Requested MPS is unavailable; no fallback performed.')
        from dataset import AffordNetDataset
        from models.openad_pn2 import OpenAD_PN2, cls_encoder
        from run_seen_query_comparison import point_features
        device=torch.device(args.device)
        torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
        # Explicitly move the shared encoder as well as the point model.
        cls_encoder.clip_model.to(device);cls_encoder.device=device;cls_encoder.eval()
        model=OpenAD_PN2({},19).to(device).eval()
        model.load_state_dict(torch.load(checkpoint,map_location=device),strict=True)
        for param in model.parameters():param.requires_grad_(False)
        full=AffordNetDataset(str(repo/'data'),'val',partial=False)
        presence={}
        for row in full.all_data:
            label=np.asarray(row['data_info']['label']).astype(np.int64)
            if np.any((label < 0) | (label > 18)):
                raise ValueError('Unexpected full-shape label encoding.')
            presence[str(row['shape_id'])]=np.bincount(label.reshape(-1),minlength=19)[:18]>0
        dataset=AffordNetDataset(str(repo/'data'),'val',partial=True) if args.split=='partial' else full
        groups={}
        for idx,row in enumerate(dataset.all_data):groups.setdefault(str(row['shape_id']),[]).append(idx)
        shape_ids=sorted(groups)
        if args.max_shapes:shape_ids=shape_ids[:args.max_shapes]
        if not shape_ids:raise ValueError('No evaluation shapes.')
        if any(sid not in presence for sid in shape_ids):raise ValueError('Missing full-shape negative reference.')
        (out/'shape_ids.txt').write_text('\n'.join(shape_ids)+'\n')
        with torch.inference_mode():texts=[cls_encoder(v) for v in vocabs]
        meta.update(status='running',shape_count=len(shape_ids),view_count=sum(len(groups[sid]) for sid in shape_ids),
                    torch_version=torch.__version__,mps_available=torch.backends.mps.is_available(),
                    dataset_label_names=list(map(str,dataset.affordances)),
                    dataset_files={name:digest(repo/'data'/name) for name in dict.fromkeys(['full_shape_val_data.pkl',f'{"partial_view" if args.split=="partial" else "full_shape"}_val_data.pkl'])})
        save();started=time.monotonic();all_counts=[];object_rows=[];negative_rows=[]
        for n,sid in enumerate(shape_ids,1):
            samples=[dataset[i] for i in groups[sid]]
            xyz_np=np.stack([x[0] for x in samples]).astype(np.float32)
            gt=np.stack([x[2].reshape(-1) for x in samples]).astype(np.int64)
            # Shape-specific seeds make smoke/full runs share identical geometry sampling.
            shape_seed=(args.seed+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16))%(2**31)
            torch.manual_seed(shape_seed)
            with torch.inference_mode():
                feat=point_features(model,torch.from_numpy(xyz_np).permute(0,2,1).to(device))
                maps=[]
                for txt in texts:
                    den=torch.norm(feat,dim=2,keepdim=True)@torch.norm(txt,dim=0,keepdim=True)
                    logits=model.logit_scale*(feat@txt)/den
                    maps.append(torch.softmax(logits,dim=-1).cpu().numpy())
            if any(not np.isfinite(m).all() for m in maps):raise ValueError('Nonfinite probability map.')
            shape_counts=[]
            for condition,prob in zip(CONDITIONS,condition_maps(maps)):
                pred=prob.argmax(axis=-1);ct=counts(confusion(gt,pred));shape_counts.append(ct)
                for ci,label in enumerate(LABELS):
                    tp,fp,fn=map(int,ct[ci]);is_negative=not bool(presence[sid][ci])
                    row={'shape_id':sid,'category':str(samples[0][4]),'condition':condition,'action':label,
                         'view_count':len(samples),'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn) if tp+fp+fn else '',
                         'verified_label_negative':is_negative,'predicted_points':int((pred==ci).sum()),
                         'evaluated_points':int(gt.size)}
                    object_rows.append(row)
                    if is_negative:
                        if np.any(gt==ci):raise ValueError('Partial/full label conflict for a claimed negative pair.')
                        negative_rows.append(dict(row,false_positive_point_fraction=float(np.mean(pred==ci)),any_alarm=int(np.any(pred==ci))))
            all_counts.append(np.stack(shape_counts))
            if n%25==0 or n==len(shape_ids):print(f'{args.split} {n}/{len(shape_ids)} shapes; {time.monotonic()-started:.1f}s',flush=True)
        all_counts=np.stack(all_counts);np.savez_compressed(out/'object_counts.npz',counts=all_counts,shape_ids=shape_ids)
        write_csv(out/'per_object_action.csv',object_rows)
        if negative_rows:write_csv(out/'negative_object_action.csv',negative_rows)
        scores=ious(all_counts.sum(axis=0));summary=[]
        for k,condition in enumerate(CONDITIONS):
            for ci,label in enumerate(LABELS):
                neg=[r for r in negative_rows if r['condition']==condition and r['action']==label]
                summary.append({'condition':condition,'action':label,'global_iou':float(scores[k,ci]) if np.isfinite(scores[k,ci]) else '',
                                'negative_objects':len(neg),'any_alarm_rate':float(np.mean([r['any_alarm'] for r in neg])) if neg else '',
                                'mean_false_positive_point_fraction':float(np.mean([r['false_positive_point_fraction'] for r in neg])) if neg else ''})
        write_csv(out/'per_action_summary.csv',summary)
        macro=np.nanmean(scores[:,:18],axis=-1)
        meta.update(status='complete',inference_seconds=time.monotonic()-started,
                    macro_iou_18=dict(zip(CONDITIONS,map(float,macro))),
                    defined_action_count=dict(zip(CONDITIONS,map(int,np.isfinite(scores[:,:18]).sum(axis=-1)))),
                    paired_95_intervals=paired_intervals(all_counts,20261109,args.bootstrap),
                    auc_status='not computed in this initial runner; no AUC result claimed')
        # False-alarm uncertainty uses the same object-resampling unit, per action.
        rng=np.random.default_rng(20261109);alarm_ci={}
        for label in LABELS:
            neg_ids=[sid for sid in shape_ids if not presence[sid][LABELS.index(label)]]
            lookup={(r['shape_id'],r['condition']):r for r in negative_rows if r['action']==label}
            alarm_ci[label]={}
            for condition in CONDITIONS[1:]:
                for field in ('any_alarm','false_positive_point_fraction'):
                    delta=np.array([lookup[(sid,condition)][field]-lookup[(sid,CONDITIONS[0])][field] for sid in neg_ids])
                    if len(delta):
                        bs=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(args.bootstrap)]
                        alarm_ci[label][condition+'_'+field]={'delta':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
        meta['negative_paired_differences']=alarm_ci;save()
        print('Final status: complete',flush=True)
    except Exception:
        meta.update(status='failed',error=traceback.format_exc());save();raise

if __name__=='__main__':main()
