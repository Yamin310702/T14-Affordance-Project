import json,csv,math
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
p=root/'E14_mixed_presence';r=json.loads((p/'run.json').read_text())
assert r['status']=='complete' and r['device']=='mps' and len(r['per_object_metrics'])==400
assert all(len(v['positive'])==20 and len(v['negative'])==20 for v in r['objects'].values())
metrics=['aiou','auc','sim','mae','object_alarm_at_05','predicted_positive_fraction_at_05','mean_score','max_score','point_false_positive_rate_at_05']
means=[]
for a in ['grab','pour','wrap','accommodate','unlock']:
 for group in ['positive','negative']:
  for cond in ['original','selected_paraphrase']:
   rr=[x for x in r['per_object_metrics'] if x['affordance']==a and x['group']==group and x['condition']==cond]
   row={'affordance':a,'group':group,'condition':cond,'n_objects':len(rr),'n_auc_valid':sum(math.isfinite(x['auc']) for x in rr)}
   for m in metrics:
    v=[x[m] for x in rr if isinstance(x[m],(int,float)) and math.isfinite(x[m])]
    row[m]=float(np.mean(v)) if v else None
   means.append(row)
r['mean_metrics_nan_safe']=means
r['validation_notes']=['All positive and negative groups contain 20 objects per affordance. Sample is balanced by GT-presence stratum and is not representative of natural prevalence.','Negative-object aIoU and AUC are undefined; negative-object detection uses the fixed 0.5 score threshold, which is diagnostic and not calibrated.']
(p/'run.json').write_text(json.dumps(r,indent=2,allow_nan=True)+'\n')
with (p/'mean_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=means[0].keys());w.writeheader();w.writerows(means)
with (p/'per_object_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['per_object_metrics'][0].keys());w.writeheader();w.writerows(r['per_object_metrics'])
# Paired effect plot: positive aIoU and negative object-level alarm rate.
affs=['grab','pour','wrap','accommodate','unlock'];fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,group,key,title in [(axes[0],'positive','aiou','Positive objects: aIoU change'),(axes[1],'negative','object_alarm_at_05','Negative objects: alarm-rate change at 0.5')]:
 vals=[];lo=[];hi=[]
 for a in affs:
  v=r['paired_results'][a][group];vals.append(v['mean_delta']*(100 if key=='aiou' else 1));ci=v['paired_bootstrap_95pct_ci'];scale=100 if key=='aiou' else 1;lo.append((v['mean_delta']-ci[0])*scale);hi.append((ci[1]-v['mean_delta'])*scale)
 y=np.arange(len(affs));ax.errorbar(vals,y,xerr=np.array([lo,hi]),fmt='o',capsize=4);ax.axvline(0,color='black',lw=1,ls='--');ax.set_yticks(y,affs);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.25)
axes[0].set_xlabel('Δ aIoU (percentage points)');axes[1].set_xlabel('Δ fraction of negative objects flagged')
fig.savefig(p/'paired_effects.png',dpi=170);plt.close(fig)
q=json.loads((root/'E13_confirmation/run.json').read_text())['affordances']; lines=['# E14: mixed-presence object evaluation','','**Status:** complete on MPS. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Data:** OpenAD validation partial views. **Design:** for each affordance, 20 objects were randomly selected from those with a positive ground-truth region and 20 from those with no ground-truth region. The same fixed original and E13-selected paraphrase were tested. Objects from E09–E13 were excluded. This stratified sample measures both localization and false alarms; it does not estimate natural prevalence.','','## Positive objects: localization','','Each affordance has 20 positive objects. Changes are selected paraphrase minus original aIoU; the intervals are paired bootstrap 95% intervals.','', '| Affordance | Original aIoU | Selected paraphrase aIoU | Change (pp) | 95% interval (pp) |','|---|---:|---:|---:|---:|']
for a in affs:
 x=next(x for x in means if x['affordance']==a and x['group']=='positive' and x['condition']=='original');y=next(x for x in means if x['affordance']==a and x['group']=='positive' and x['condition']=='selected_paraphrase');v=r['paired_results'][a]['positive'];ci=v['paired_bootstrap_95pct_ci']
 lines.append(f"| {a} | {100*x['aiou']:.2f}% | {100*y['aiou']:.2f}% | {100*v['mean_delta']:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] |")
lines+=['','## Negative objects: false alarms','','For objects with no target points, aIoU and AUC are undefined. The “alarm” metric is the share of objects with at least one score ≥0.5. The “positive points” metric is the average share of points with score ≥0.5. This threshold is a diagnostic, not calibrated.','', '| Affordance | Query | Negative objects flagged | Mean points flagged |','|---|---|---:|---:|']
for a in affs:
 for cond,label in [('original','Original'),('selected_paraphrase','Selected paraphrase')]:
  x=next(x for x in means if x['affordance']==a and x['group']=='negative' and x['condition']==cond)
  lines.append(f"| {a} | {label} | {100*x['object_alarm_at_05']:.0f}% ({int(round(20*x['object_alarm_at_05']))}/20) | {100*x['predicted_positive_fraction_at_05']:.3f}% |")
lines+=['','## Interpretation','','On the positive objects, the selected paraphrase had small or uncertain localization changes across all five affordances; each paired aIoU interval includes zero. On negative objects, the accommodate paraphrase raised the 0.5-threshold alarm rate from 30% to 65%, while wrap rose from 30% to 40% and pour stayed at 20%. Grab stayed at 15%; unlock decreased from 10% to 0%. The accommodated-region phrase therefore appears prone to false alarms under this threshold on this sample, despite its higher aIoU in earlier positive-only tests.','','This result shows why both presence and absence cases matter. It weakens any broad claim that choosing the best paraphrase from positive-only data improves real-world behavior. The findings remain preliminary: only 20 objects per stratum, deterministic hash selection, one dataset and model, class-balanced sampling rather than natural prevalence, and an uncalibrated threshold. Multiple comparisons are not adjusted.','','## Prompts','','The original and selected paraphrase text for each affordance is recorded in `run.json`; they are the exact E13 pairs.','','## Saved files','','- `run.json`: sample IDs, prompts, model, metrics, paired intervals, and eligibility counts.\n- `predictions.npz`: XYZ, labels, and score maps.\n- `mean_metrics.csv` and `per_object_metrics.csv`: tabular results.\n- `paired_effects.png`: paired changes on positive and negative groups.\n- `run.log`, `run_e14_mixed_presence.py`, and `summarize_e14_mixed_presence.py`: run log and scripts.']
(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:22]))
