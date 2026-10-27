from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent/'query_sensitivity'
d=np.load(root/'predictions.npz');r=json.loads((root/'run.json').read_text());xyz=d['xyz']
maps=[d['ground_truth'],*d['individual_heatmaps'],d['ensemble_heatmap']]
titles=['Ground truth: pour']+[f"{m['name']} | aIoU {m['aiou']*100:.2f}%" for m in r['metrics']]
fig=plt.figure(figsize=(15,9),layout='constrained');axes=[]
for i,(h,title) in enumerate(zip(maps,titles)):
 ax=fig.add_subplot(2,3,i+1,projection='3d');axes.append(ax)
 s=ax.scatter(*xyz.T,c=h,cmap='coolwarm',vmin=0,vmax=1,s=3,depthshade=False)
 ax.view_init(elev=20,azim=45);ax.set_box_aspect(np.ptp(xyz,axis=0));ax.set_axis_off();ax.set_title(title,fontsize=13)
fig.colorbar(s,ax=axes,shrink=.65,pad=.02,label='Annotation / prediction score (fixed 0–1 scale)')
fig.suptitle('Pouring-query sensitivity — one partial-view vase\nUnofficial Espresso-3D; exploratory development case',fontsize=17)
fig.savefig(root/'comparison.png',dpi=150)
