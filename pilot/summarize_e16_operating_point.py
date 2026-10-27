"""Summarize E16 scores and generate object-level operating curves."""
import sys, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'espresso-3d-reimplementation'))
from espresso.metrics import sample_metrics

out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'E16_operating_point'
r=json.loads((out/'run.json').read_text()); z=np.load(out/'predictions.npz')
assert r['status']=='complete'
aff=r['primary_affordance']; conditions=['original','selected_paraphrase']
threshold={x['condition']:float(x['threshold']) for x in r['calibrated_thresholds']}
summary=[]; per=[]; curves=[]

for cond in conditions:
    # Calibration operating-point summary and full curve.
    pos=[]; neg=[]
    for obj in r['calibration_objects']['positive']:
        sid=obj['shape_id']; suffix='cal_original' if cond=='original' else 'cal_selected_paraphrase'
        pos.append((z[f'{aff}__{sid}__{suffix}'],z[f'{aff}__{sid}__gt'].astype(bool)))
    for obj in r['calibration_objects']['negative']:
        sid=obj['shape_id']; suffix='cal_original' if cond=='original' else 'cal_selected_paraphrase'
        neg.append(z[f'{aff}__{sid}__{suffix}'])
    t=threshold[cond]
    summary.append({'split':'calibration','condition':cond,'threshold':t,
        'positive_macro_recall':float(np.mean([((p>=t)&g).sum()/g.sum() for p,g in pos])),
        'negative_object_alarm_rate':float(np.mean([np.any(p>=t) for p in neg]))})
    for tcurve in np.linspace(0,1,101):
        curves.append({'split':'calibration','condition':cond,'threshold':float(tcurve),
            'positive_macro_recall':float(np.mean([((p>=tcurve)&g).sum()/g.sum() for p,g in pos])),
            'negative_object_alarm_rate':float(np.mean([np.any(p>=tcurve) for p in neg]))})

    # Fresh test results, frozen cutoff plus threshold-free official metrics.
    vals=[]
    for group in ['positive','negative']:
        for obj in r['test_objects'][group]:
            sid=obj['shape_id']; p=z[f'{aff}__{sid}__{cond}']; g=z[f'{aff}__{sid}__gt'].astype(bool)
            mask=p>=t; alarm=bool(mask.any())
            if group=='positive':
                aiou,auc,sim,mae=sample_metrics(p,g.astype(np.float32))
                row={'condition':cond,'group':group,'shape_id':sid,'category':obj['category'],'threshold':t,
                    'operating_point_aiou':float((mask&g).sum()/(mask|g).sum()) if (mask|g).any() else 0.,
                    'operating_point_recall':float((mask&g).sum()/g.sum()),'alarm':alarm,
                    'ranking_aiou':float(aiou),'auc':float(auc),'sim':float(sim),'mae':float(mae),
                    'max_score':float(p.max()),'score_at_05_alarm':bool((p>=.5).any())}
                vals.append((float(aiou),float(auc),float(sim),float(mae)))
            else:
                row={'condition':cond,'group':group,'shape_id':sid,'category':obj['category'],'threshold':t,
                    'operating_point_aiou':None,'operating_point_recall':None,'alarm':alarm,
                    'ranking_aiou':None,'auc':None,'sim':None,'mae':None,'max_score':float(p.max()),
                    'score_at_05_alarm':bool((p>=.5).any())}
            per.append(row)
        if group=='positive':
            v=np.nanmean(np.asarray(vals,dtype=float),axis=0)
            group_rows=[x for x in per if x['condition']==cond and x['group']=='positive']
            summary.append({'split':'fresh_validation','condition':cond,'threshold':t,
                'positive_macro_recall':float(np.mean([x['operating_point_recall'] for x in group_rows])),
                'positive_operating_aiou':float(np.mean([x['operating_point_aiou'] for x in group_rows])),
                'ranking_aiou':float(v[0]),'auc':float(v[1]),'sim':float(v[2]),'mae':float(v[3])})
        else:
            group_rows=[x for x in per if x['condition']==cond and x['group']=='negative']
            summary.append({'split':'fresh_validation','condition':cond,'threshold':t,
                'negative_object_alarm_rate':float(np.mean([x['alarm'] for x in group_rows])),
                'negative_objects_flagged':int(sum(x['alarm'] for x in group_rows)),'n_negative':len(group_rows)})

    # Held-out curve is descriptive; thresholds are not selected from it.
    for tc in np.linspace(0,1,101):
        rec=[]; alarms=[]
        for obj in r['test_objects']['positive']:
            sid=obj['shape_id'];p=z[f'{aff}__{sid}__{cond}'];g=z[f'{aff}__{sid}__gt'].astype(bool)
            rec.append(float(((p>=tc)&g).sum()/g.sum()))
        for obj in r['test_objects']['negative']:
            sid=obj['shape_id'];p=z[f'{aff}__{sid}__{cond}'];alarms.append(bool((p>=tc).any()))
        curves.append({'split':'fresh_validation','condition':cond,'threshold':float(tc),
            'positive_macro_recall':float(np.mean(rec)),'negative_object_alarm_rate':float(np.mean(alarms))})

paired={}
for metric in ['ranking_aiou','operating_point_recall']:
    a={x['shape_id']:x[metric] for x in per if x['group']=='positive' and x['condition']=='original'}
    b={x['shape_id']:x[metric] for x in per if x['group']=='positive' and x['condition']=='selected_paraphrase'}
    d=np.array([b[s]-a[s] for s in a]);rng=np.random.default_rng(1601)
    boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)])
    paired[metric]={'original_mean':float(np.mean(list(a.values()))),'paraphrase_mean':float(np.mean(list(b.values()))),
        'delta':float(d.mean()),'paired_object_bootstrap_95pct_ci':[float(x) for x in np.quantile(boot,[.025,.975])],
        'n_positive_objects':len(d)}
r['threshold_free_test_summary']=[x for x in summary if x['split']=='fresh_validation' and 'ranking_aiou' in x]
r['paired_threshold_free_effects']=paired
r['fixed_05_diagnostic']={c:{g:{'flagged':sum(x['score_at_05_alarm'] for x in per if x['condition']==c and x['group']==g),
    'n':sum(x['condition']==c and x['group']==g for x in per)} for g in ['positive','negative']} for c in conditions}
(out/'run.json').write_text(json.dumps(r,indent=2,allow_nan=True)+'\n')
for name,rows in [('summary.csv',summary),('test_per_object.csv',per),('operating_curves.csv',curves)]:
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(9,5.5));colors={'original':'#2563eb','selected_paraphrase':'#dc2626'}
for split,style in [('calibration','-'),('fresh_validation','--')]:
    for cond in conditions:
        rr=[x for x in curves if x['split']==split and x['condition']==cond]
        ax.plot([x['negative_object_alarm_rate'] for x in rr],[x['positive_macro_recall'] for x in rr],style,
            color=colors[cond],label=f'{cond.replace("_"," ")} — {split.replace("_"," ")}')
ax.axvline(.10,color='black',linestyle=':',label='10% calibration alarm cap')
ax.set(xlabel='Negative-object alarm rate',ylabel='Mean positive-object point recall',
    title='E16 accommodation: alarm–recall trade-off')
ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02);ax.grid(alpha=.25)
ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.02,.5));fig.tight_layout(rect=[0,0,.76,1])
fig.savefig(out/'operating_curve.png',dpi=180);plt.close(fig)
print(json.dumps({'device':r['device'],'thresholds':r['calibrated_thresholds'],
    'threshold_free_test':r['threshold_free_test_summary'],'paired':paired,'fixed_05':r['fixed_05_diagnostic']},indent=2))
