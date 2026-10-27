import json,csv,hashlib
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'multiaffordance_prompt_study'; r=json.loads((p/'run.json').read_text())
assert r['status']=='complete' and len(r['per_object_metrics'])==500 and len(r['mean_metrics'])==25
# Check that all selected IDs are new relative to E09-E11.
root=Path(__file__).resolve().parent
old=set()
for n in ['multobject_query_sensitivity','heldout_query_sensitivity','grab_query_sensitivity']:
 old.update(x['shape_id'] for x in json.loads((root/n/'run.json').read_text())['objects'])
now={x['shape_id'] for rows in r['objects'].values() for x in rows}
assert not old & now
r['prior_experiment_ids_excluded']=['E09','E10','E11']
r['unique_shape_ids_across_all_affordances']=len(now)
r['overlap_between_affordance_sets_allowed']=True
r['analysis_note']='Bootstrap intervals are descriptive, within-affordance paired resamples of selected objects. They are not corrected for the many comparisons and are not population-level evidence.'
(p/'run.json').write_text(json.dumps(r,indent=2)+'\n')
with (p/'mean_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['mean_metrics'][0].keys());w.writeheader();w.writerows(r['mean_metrics'])
with (p/'per_object_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r['per_object_metrics'][0].keys());w.writeheader();w.writerows(r['per_object_metrics'])
affs=list(r['affordances']); conds=['original','phrase_1','phrase_2','phrase_3','ensemble']
means={(x['affordance'],x['condition']):x for x in r['mean_metrics']}
fig,axes=plt.subplots(1,len(affs),figsize=(16,5),sharey=True,layout='constrained')
for ax,aff in zip(axes,affs):
 vals=[100*means[(aff,c)]['aiou'] for c in conds]
 ax.bar(range(len(conds)),vals,color=['#4263eb','#20a779','#ef8c25','#b25db8','#777777'])
 ax.set_title(aff);ax.set_xticks(range(len(conds)),['base','P1','P2','P3','avg'],rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean aIoU (%)');fig.suptitle('E12 query wording results by affordance (20 selected objects each)')
fig.savefig(p/'per_affordance_aiou.png',dpi=170);plt.close(fig)
query_labels={'original':'Original query','phrase_1':'Phrase 1','phrase_2':'Phrase 2','phrase_3':'Phrase 3','ensemble':'Mean of paraphrase maps'}
report=['# E12: prompt wording across five affordances','','**Status:** complete. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Device:** CPU (MPS was unavailable during this run). **Data:** OpenAD validation partial views. **Selection:** 20 distinct positive-label objects for each affordance; all objects from E09, E10, and E11 were excluded. The five sets contain '+str(len(now))+' unique shape IDs overall; because selection is independent by affordance, some shapes can appear in more than one affordance set.','','## Mean results','','Each affordance uses 20 objects. aIoU, AUC and SIM are shown as percentages; MAE is lower-is-better.','', '| Affordance | Original | Phrase 1 | Phrase 2 | Phrase 3 | Paraphrase average |','|---|---:|---:|---:|---:|---:|']
for a in affs:
 report.append('| '+a+' | '+' | '.join(f"{100*means[(a,c)]['aiou']:.2f}%" for c in conds)+' |')
report+=['','## Paired aIoU change from the original query','','Changes are percentage points; intervals are paired bootstrap 95% intervals across the selected objects.','', '| Affordance | Phrase 1 | Phrase 2 | Phrase 3 | Paraphrase average |','|---|---:|---:|---:|---:|']
for a in affs:
 cells=[]
 for c in conds[1:]:
  v=r['paired_delta_vs_original'][a][c];lo,hi=v['paired_bootstrap_95pct_ci'];cells.append(f"{100*v['mean_delta_aiou']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
 report.append('| '+a+' | '+' | '.join(cells)+' |')
report+=['','## Interpretation','','Prompt wording effects differ by affordance in this pilot. For pouring, phrase 1 (“Point to the part used for pouring”) was +0.91 points, while phrase 3 (“Identify the pouring region”) was −5.40 points; the three-phrase mean was −1.59 points. For wrap, phrase 2 and the phrase average were lower than the original. Grab wording changes were small on this fresh set. Accommodate and unlock paraphrases had higher mean aIoU, but their paired intervals include zero.','','This is consistent with the earlier pouring and grab pilots in one useful way: no single paraphrase or averaging rule improves every affordance. It supports investigating affordance-specific prompt sensitivity, but it does not establish that wording reliably causes these changes.','','## Limits','','These are small deterministic subsets chosen because their ground-truth regions are present; they are not random or fully representative test samples. The bootstrap intervals are descriptive, apply only to the selected objects, and are not corrected for multiple comparisons. Region sizes vary widely (3 to over 1,800 points), and some labels/categories are concentrated in a few object types. The unofficial Affogato-pretrained model is being evaluated on OpenAD data, so this is a cross-dataset test. Prompts have not been human-validated. The run used CPU because MPS was unavailable; all E12 conditions share this device, but small numeric differences from earlier MPS runs are possible. Affostruction has not been run.','','## Artifacts','','- `run.json`: all prompts, label IDs, selected objects, metrics, paired intervals, and selection rules.\n- `per_object_heatmaps.npz`: point coordinates, ground truth, and predicted maps.\n- `mean_metrics.csv` and `per_object_metrics.csv`: scores in tabular form.\n- `per_affordance_aiou.png`: plot of mean aIoU by affordance and query condition.\n- `run_multiaffordance_prompt_study.py` and `summarize_multiaffordance_prompt_study.py`: scripts.']
(p/'RESULTS.md').write_text('\n'.join(report)+'\n')
print('\n'.join(report[:20]))
print(f'\nVerified: {len(r["per_object_metrics"])} rows, {len(now)} unique shapes, no overlap with prior experiments.')
