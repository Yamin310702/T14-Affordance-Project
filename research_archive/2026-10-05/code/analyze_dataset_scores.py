"""Point-weighted diagnostics using the unchanged OpenAD model and val loader."""
import argparse, csv, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
p.add_argument('--output', required=True)
p.add_argument('--start', type=int, default=0)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
repo = Path(a.repo).resolve(); os.chdir(repo); sys.path.insert(0, str(repo))
from gorilla.config import Config
from utils import build_model
from dataset import AffordNetDataset
config = repo/'config/openad_pn2/partial_view_open_vocab_cfg.py'
checkpoint = repo/'checkpoints/best_model_openad_pn2_partial.t7'
cfg = Config.fromfile(str(config))
random.seed(1); np.random.seed(1); torch.manual_seed(1)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = build_model(cfg).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.eval()
assert type(model).__name__ == 'OpenAD_PN2'
names = list(cfg.training_cfg.val_affordance); k = len(names); bg = names.index('none')
# Avoid build_dataset(): it unnecessarily loads the 2.5 GB training set too.
ds = AffordNetDataset(cfg.data.data_root, 'val', partial=True)
end = min(len(ds), a.start+a.limit) if a.limit else len(ds)
assert 0 <= a.start < end
cm = np.zeros((k,k), np.int64); sums = np.zeros((k,k), np.float64)
second = np.zeros(k, np.int64)
meta = dict(config=str(config), checkpoint=str(checkpoint), model=type(model).__name__, device=str(device),
            torch_version=torch.__version__, seed=1, batch_size=1, classes=names,
            dataset_samples=len(ds), start=a.start, end=end, completed_samples=0, status='running',
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            aggregation='point-weighted across views; repeated objects are not independent',
            note='Batch size 1; stochastic point sampling can differ from earlier batch-16 runs.')
print(json.dumps(meta, indent=2), flush=True)
t0 = time.monotonic()
def save():
    gt=cm.sum(1); pred=cm.sum(0); tp=cm.diagonal(); union=gt+pred-tp
    rows=[]
    for c,n in enumerate(names):
        den=int(gt[c]); avg=sums[c]/den if den else None
        rows.append(dict(affordance=n, gt_points=den, predicted_points=int(pred[c]),
            iou=float(tp[c]/union[c]) if union[c] else None,
            correct_percent=float(100*tp[c]/den) if den else None,
            predicted_none_percent=float(100*cm[c,bg]/den) if den else None,
            other_error_percent=float(100*(den-tp[c]-(cm[c,bg] if c!=bg else 0))/den) if den else None,
            mean_target_probability=float(avg[c]) if den else None,
            mean_none_probability=float(avg[bg]) if den else None,
            target_minus_none=float(avg[c]-avg[bg]) if den else None,
            per_point_target_rank2_percent=float(100*second[c]/den) if den else None,
            region_average_target_rank=int(1+np.sum(avg>avg[c])) if den else None))
    with (out/'per_affordance.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/'confusion_counts.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['GT / prediction']+names)
        for n,row in zip(names,cm): w.writerow([n]+row.tolist())
    meta['elapsed_seconds']=time.monotonic()-t0
    meta['miou_official_epsilon']=float(np.mean(tp/(union.astype(np.float32)+1e-6)))
    (out/'summary.json').write_text(json.dumps(meta,indent=2))
    np.savez(out/'aggregates.npz',confusion=cm,probability_sums=sums,target_rank2_counts=second)
with (out/'sample_regions.csv').open('w') as f:
    w=csv.writer(f); w.writerow(['index','shape_id','category','target','gt_points','correct_points','none_points','mean_target','mean_none'])
    save()
    with torch.inference_mode():
        for idx in range(a.start,end):
            points,_,labels,sid,cat=ds[idx]
            y=np.asarray(labels).reshape(-1).astype(np.int64)
            assert y.min()>=0 and y.max()<k
            x=torch.from_numpy(points).float().unsqueeze(0).permute(0,2,1).to(device)
            probs=model(x,names)[0].T.exp().cpu().numpy()
            assert probs.shape==(len(y),k) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(1),1,atol=1e-5)
            pred=probs.argmax(1)
            cm += np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
            for c in np.unique(y):
                mask=y==c; pp=probs[mask]; yp=pred[mask]
                sums[c]+=pp.sum(0,dtype=np.float64)
                ranks=1+(pp>pp[:,c,None]).sum(1)
                second[c]+=np.sum(ranks==2)
                w.writerow([idx,sid,cat,names[c],int(mask.sum()),int((yp==c).sum()),int((yp==bg).sum()),float(pp[:,c].mean()),float(pp[:,bg].mean())])
            meta['completed_samples']+=1
            if meta['completed_samples']%25==0 or idx==end-1:
                f.flush(); save()
                print(f"Processed {meta['completed_samples']}/{end-a.start}; elapsed {time.monotonic()-t0:.1f}s",flush=True)
meta['status']='complete'; save()
print('Saved to',out,flush=True)
