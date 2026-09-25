#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS=[('robotwin_dual_system_joint_cross_attention','Joint cross-attn'),('robotwin_dual_system_joint_cross_attention_detach','Joint cross-attn detach'),('robotwin_dual_system_joint_self_attention','Joint self-attn'),('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated'),('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--modality-dir',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args()
 rows=[]; labels=[]; nlayer=0
 for model,label in MODELS:
  f=a.modality_dir/model/'scores.csv'
  if not f.exists(): continue
  z=sorted(list(csv.DictReader(open(f))),key=lambda r:int(r['video_layer'])); rows.append([float(r['score']) for r in z]); labels.append(label); nlayer=max(nlayer,len(z))
 if not rows: raise RuntimeError('no scores')
 arr=np.full((len(rows),nlayer),np.nan); [arr.__setitem__((i,slice(0,len(r))),r) for i,r in enumerate(rows)]
 fig,ax=plt.subplots(figsize=(12,3.8),dpi=180); im=ax.imshow(arr,aspect='auto',interpolation='nearest',cmap='viridis'); ax.set_yticks(range(len(labels)),labels); ax.set_xticks(range(nlayer),range(1,nlayer+1)); ax.set_xlabel('WAM transformer layer'); ax.set_title(a.title,weight='bold'); cbar=fig.colorbar(im,ax=ax,pad=.02); cbar.set_label('Mutual-kNN alignment')
 for i,r in enumerate(rows):
  j=int(np.argmax(r)); ax.scatter(j,i,s=55,facecolors='none',edgecolors='white',linewidths=1.5)
 fig.tight_layout(); fig.savefig(a.modality_dir/'layer_heatmap.png',dpi=300,bbox_inches='tight'); fig.savefig(a.modality_dir/'layer_heatmap.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
