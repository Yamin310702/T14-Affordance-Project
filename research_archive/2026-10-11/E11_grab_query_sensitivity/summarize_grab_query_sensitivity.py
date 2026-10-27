import json,csv
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'grab_query_sensitivity';r=json.loads((p/'run.json').read_text())
conds=[x['condition'] for x in r['mean_metrics']]
fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
for n,cond in enumerate(conds):
 rows=[x for x in r['per_object_metrics'] if x['condition']==cond]
 ys=[100*x['aiou'] for x in rows]; ax.scatter([n]*len(ys),ys,s=30,alpha=.65)
 ax.plot([n-.18,n+.18],[100*r['mean_metrics'][n]['aiou']]*2,lw=5,label='mean' if n==0 else None)
ax.set_xticks(range(len(conds)),[x.replace('_','\n') for x in conds]);ax.set_ylabel('Per-object aIoU (%)');ax.set_title('Grab query wording sensitivity across 20 selected objects')
ax.grid(axis='y',alpha=.25);fig.savefig(p/'per_object_aiou.png',dpi=160)
with (p/'mean_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['mean_metrics'][0].keys());w.writeheader();w.writerows(r['mean_metrics'])
with (p/'per_object_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['per_object_metrics'][0].keys());w.writeheader();w.writerows(r['per_object_metrics'])
report=['# E11: grab-query sensitivity on new objects','',f"Status: {r['status']}. Objects: {len(r['objects'])}, unique shapes. Model: {r['model']}.",'', '## Equal-weight mean across selected development objects','', '| Condition | aIoU % | AUC % | SIM % | MAE |', '|---|---:|---:|---:|---:|']
for x in r['mean_metrics']: report.append(f"| {x['condition']} | {100*x['aiou']:.2f} | {100*x['auc']:.2f} | {100*x['sim']:.2f} | {x['mae']:.4f} |")
report+=['','## Paired aIoU difference from original query','', '| Condition | Mean change, percentage points | Paired bootstrap 95% interval |','|---|---:|---:|']
for k,v in r['paired_delta_vs_original'].items():report.append(f"| {k} | {100*v['mean_delta_aiou']:+.2f} | [{100*v['paired_bootstrap_95pct_ci'][0]:+.2f}, {100*v['paired_bootstrap_95pct_ci'][1]:+.2f}] |")
report+=['','The objects were selected deterministically from those with a positive ground-truth grab region; this is a selected pilot subset, not a random test set. The interval is descriptive and does not make this selected sample representative. The model is an unofficial Espresso-3D reimplementation. Affordance-query thresholds use the common repository metric; OpenAD multiclass mIoU is not directly comparable. The fixed 0.5 diagnostic can produce empty positive masks; read recall and false-positive rates together.','', 'See run.json for prompts, object IDs, per-object metric values and exact selection rule; per_object_heatmaps.npz contains the pointwise heatmaps and GT.']
(p/'RESULTS.md').write_text('\n'.join(report)+'\n')
print('\n'.join(report[:18]))
