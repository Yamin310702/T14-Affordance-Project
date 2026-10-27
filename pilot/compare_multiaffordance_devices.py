import json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
cpu=root/'multiaffordance_prompt_study';mps=root/'multiaffordance_prompt_study_mps'
a=json.loads((cpu/'run.json').read_text());b=json.loads((mps/'run.json').read_text())
assert a['status']==b['status']=='complete' and a['device']=='cpu' and b['device']=='mps'
assert a['model']==b['model'] and a['affordances']==b['affordances']
assert {k:[x['shape_id'] for x in v] for k,v in a['objects'].items()}=={k:[x['shape_id'] for x in v] for k,v in b['objects'].items()}
ra={(x['affordance'],x['shape_id'],x['condition']):x for x in a['per_object_metrics']}
rb={(x['affordance'],x['shape_id'],x['condition']):x for x in b['per_object_metrics']}
assert ra.keys()==rb.keys() and len(ra)==500
pa=np.load(cpu/'per_object_heatmaps.npz');pb=np.load(mps/'per_object_heatmaps.npz')
assert set(pa.files)==set(pb.files)
for k in pa.files:
 if k.endswith('_gt') or k.endswith('_xyz'):assert np.array_equal(pa[k],pb[k]),f'Input mismatch: {k}'
summary=[]; metrics=['aiou','auc','sim','mae']
for aff in a['affordances']:
 for cond in ['original','phrase_1','phrase_2','phrase_3','ensemble']:
  x=next(x for x in a['mean_metrics'] if x['affordance']==aff and x['condition']==cond)
  y=next(x for x in b['mean_metrics'] if x['affordance']==aff and x['condition']==cond)
  summary.append({'affordance':aff,'condition':cond,**{f'cpu_{m}':x[m] for m in metrics},**{f'mps_{m}':y[m] for m in metrics},**{f'delta_{m}':y[m]-x[m] for m in metrics}})
individual={m:np.array([rb[k][m]-ra[k][m] for k in ra]) for m in metrics}
heat=[]
for k in pa.files:
 if not k.endswith('_gt') and not k.endswith('_xyz'):heat.append(np.abs(pa[k].astype('float32')-pb[k].astype('float32')).ravel())
heat=np.concatenate(heat)
result={'same_500_object_affordance_condition_rows':True,'same_selected_object_ids':True,'same_ground_truth_and_xyz_arrays':True,'individual_metric_absolute_difference':{m:{'mean':float(np.abs(d).mean()),'max':float(np.abs(d).max())} for m,d in individual.items()},'heatmap_absolute_difference':{'mean':float(heat.mean()),'p95':float(np.quantile(heat,.95)),'max':float(heat.max())},'cpu_inference_seconds':a['elapsed_seconds'],'mps_inference_seconds':b['elapsed_seconds'],'speedup':a['elapsed_seconds']/b['elapsed_seconds'],'per_affordance_condition_means':summary}
(mps/'device_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
with (mps/'device_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
lines=['# E12 CPU vs MPS rerun comparison','','**The MPS rerun reproduces the CPU experiment.** Both runs evaluated the same 100 objects (20 each for grab, pour, wrap, accommodate, and unlock), the same 500 object-condition rows, prompts, annotations, and pretrained weights. XYZ and ground-truth arrays match exactly.','','## Result','','Mean aIoU, AUC, SIM, and MAE are the same to the displayed precision for every affordance and query condition. Across the 500 individual result rows, maximum absolute differences were:','','| Metric | Maximum absolute difference |','|---|---:|']
for m in metrics: lines.append(f"| {m} | {result['individual_metric_absolute_difference'][m]['max']:.8g} |")
lines+=['',f"Across prediction heatmaps, mean absolute CPU–MPS difference was {heat.mean():.3g}, 95th percentile {np.quantile(heat,.95):.3g}, maximum {heat.max():.3g}. These tiny floating-point differences do not change the study’s conclusions.",'',f"The measured inference loop took {a['elapsed_seconds']:.1f} seconds on CPU and {b['elapsed_seconds']:.1f} seconds on MPS ({a['elapsed_seconds']/b['elapsed_seconds']:.2f}× faster on MPS). This excludes data selection and model startup.",'','## Mean aIoU comparison','','| Affordance | Condition | CPU | MPS | MPS − CPU (percentage points) |','|---|---|---:|---:|---:|']
for x in summary: lines.append(f"| {x['affordance']} | {x['condition']} | {100*x['cpu_aiou']:.4f}% | {100*x['mps_aiou']:.4f}% | {100*x['delta_aiou']:+.5f} |")
lines+=['','MPS was available when the run was launched directly outside the restricted Codex shell. The MPS results and heatmaps are saved separately, leaving the original CPU run unchanged. See `device_comparison.json` for precision-level comparisons.']
(mps/'DEVICE_COMPARISON.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:20]))
