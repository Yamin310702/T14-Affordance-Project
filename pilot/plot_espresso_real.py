from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent/'espresso_real'
a=np.load(root/'input.npz');xyz=a['xyz'];gt=a['labels'].reshape(-1)==8
heat=np.load(root/'heatmap.npy')
fig=plt.figure(figsize=(12,5),layout='constrained')
for i,(v,title) in enumerate([(gt.astype(float),'Ground truth: pour'),(heat,'Unofficial Espresso-3D: pouring query')]):
 ax=fig.add_subplot(1,2,i+1,projection='3d');s=ax.scatter(*xyz.T,c=v,cmap='coolwarm',vmin=0,vmax=1,s=5)
 ax.set_title(title);ax.set_box_aspect(np.ptp(xyz,axis=0));ax.view_init(elev=20,azim=45)
 ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
fig.colorbar(s,ax=fig.axes,shrink=.6,label='Heatmap score')
fig.suptitle('Vase sample 452 — single-case feasibility test, not a benchmark')
fig.savefig(root/'prediction.png',dpi=160)
