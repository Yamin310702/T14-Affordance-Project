import json,csv,math
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'E13_confirmation';r=json.loads((p/'run.json').read_text())
assert r['status']=='complete' and r['device']=='mps' and len(r['per_object_metrics'])==400
assert all(len(v)==40 for v in r['objects'].values()) and r['unique_shape_ids_across_E13']==200
# Verify disjointness from all earlier scored experiments.
root=Path(__file__).resolve().parent; old=set()
for name in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity','multiaffordance_prompt_study']:
 x=json.loads((root/name/'run.json').read_text());objs=x['objects']
 if isinstance(objs,dict):objs=[z for vv in objs.values() for z in vv]
 old.update(z['shape_id'] for z in objs)
new={z['shape_id'] for rows in r['objects'].values() for z in rows}
assert not old&new
metrics=['aiou','auc','sim','mae','iou_at_05','recall_at_05','false_positive_rate_at_05']
means=[]
for aff in r['affordances']:
 for cond in ['original','selected_paraphrase']:
  rr=[x for x in r['per_object_metrics'] if x['affordance']==aff and x['condition']==cond]
  row={'affordance':aff,'condition':cond,'n_objects':len(rr),'n_auc_valid':sum(math.isfinite(x['auc']) for x in rr)}
  for m in metrics:
   vals=[x[m] for x in rr if math.isfinite(x[m])]
   row[m]=float(np.mean(vals)) if vals else None
  means.append(row)
r['mean_metrics_nan_safe']=means
r['validation_notes']=['One wrap object (shape fa2f25665c561e7be4179a13aff1e10) has all 2048 points labeled positive; ROC AUC is undefined for that object and excluded from wrap AUC means (n=39). Its aIoU, SIM and MAE remain valid and are included.','All E13 samples are disjoint from E09-E12 and are unique across the five E13 affordance sets.']
(p/'run.json').write_text(json.dumps(r,indent=2,allow_nan=True)+'\n')
with (p/'mean_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=means[0].keys());w.writeheader();w.writerows(means)
with (p/'per_object_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['per_object_metrics'][0].keys());w.writeheader();w.writerows(r['per_object_metrics'])
order=list(r['affordances']);boot=r['paired_results'];deltas=[100*boot[a]['mean_delta_aiou'] for a in order]
los=[100*(boot[a]['mean_delta_aiou']-boot[a]['paired_bootstrap_95pct_ci'][0]) for a in order]
his=[100*(boot[a]['paired_bootstrap_95pct_ci'][1]-boot[a]['mean_delta_aiou']) for a in order]
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained'); y=np.arange(len(order))
ax.errorbar(deltas,y,xerr=np.array([los,his]),fmt='o',capsize=4,color='#335d9f');ax.axvline(0,color='black',lw=1,ls='--')
ax.set_yticks(y,order);ax.invert_yaxis();ax.set_xlabel('Selected paraphrase minus original aIoU (percentage points)')
ax.set_title('E13 held-out prompt confirmation: 40 fresh objects per affordance');ax.grid(axis='x',alpha=.25)
fig.savefig(p/'paired_aiou_deltas.png',dpi=170);plt.close(fig)
q=r['affordances'];lines=['# E13: held-out confirmation of preselected query paraphrases','','**Status:** complete on MPS. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Data:** OpenAD validation partial views. **Design:** compare the original prompt to one paraphrase selected from E12 before examining E13 results. Each affordance has 40 fresh objects; 200 unique shapes total, disjoint from E09–E12 and from each other within this run. The query pair was fixed in advance, but selection was still limited to objects with a positive ground-truth label.','','## Results','','aIoU, AUC and SIM are percentages; MAE is lower-is-better. AUC is undefined for one wrap object whose every point is labeled positive, so wrap AUC uses 39 objects; other metrics use 40.','', '| Affordance | Original aIoU | Selected paraphrase aIoU | Change (pp) | Paired bootstrap 95% interval (pp) | Positive objects |', '|---|---:|---:|---:|---:|---:|']
for a in q:
 original=next(x for x in means if x['affordance']==a and x['condition']=='original'); para=next(x for x in means if x['affordance']==a and x['condition']=='selected_paraphrase');v=boot[a]
 lines.append(f"| {a} | {100*original['aiou']:.2f}% | {100*para['aiou']:.2f}% | {100*v['mean_delta_aiou']:+.2f} | [{100*v['paired_bootstrap_95pct_ci'][0]:+.2f}, {100*v['paired_bootstrap_95pct_ci'][1]:+.2f}] | {v['n_positive_deltas']}/40 |")
lines+=['','## Locked prompt pairs','','| Affordance | Original query | E12-selected paraphrase |','|---|---|---|']
for a,v in q.items():lines.append(f"| {a} | “{v['original_query']}” | “{v['selected_paraphrase']}” |")
lines+=['','## Interpretation','','The grab paraphrase improved mean aIoU by **5.85 percentage points** on this fresh set (paired interval +1.59 to +11.00). The wrap paraphrase reduced it by **0.73 points** (−1.35 to −0.19). Pour (+0.44), accommodate (+0.34), and unlock (+0.56) had intervals that include zero. This supports a tentative, affordance-specific wording effect: a paraphrase helped grab, hurt wrap, and did not produce a clear change for the other three labels.','','The intervals are unadjusted across five comparisons, and the evaluation set is a deterministic GT-positive subset from one dataset. Treat this as stronger preliminary evidence than E12, not a final universal claim. The model is unofficial Espresso-3D with Affogato-pretrained weights evaluated cross-dataset on OpenAD; this does not test Affostruction.','','## Artifacts','','- `run.json`: fixed prompts, selected object IDs, metrics, bootstrap intervals, and run metadata.\n- `predictions.npz`: point coordinates, ground truth, and both predicted heatmaps.\n- `mean_metrics.csv` and `per_object_metrics.csv`: tabular results.\n- `paired_aiou_deltas.png`: paired effect plot.\n- `run.log`: execution log.\n- `run_e13_confirmation.py` and `summarize_e13_confirmation.py`: scripts.']
(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
print('Verified no sample overlap with earlier experiments; 200 unique new objects.')
