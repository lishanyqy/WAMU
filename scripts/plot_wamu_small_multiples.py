#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS=[('robotwin_dual_system_joint_cross_attention','Joint cross-attn','#1769aa'),('robotwin_dual_system_joint_cross_attention_detach','Joint cross-attn detach','#d95f02'),('robotwin_dual_system_joint_self_attention','Joint self-attn','#2ca25f'),('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated','#756bb1'),('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action','#e7298a')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--modality-dir',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args()
 data=[]
 for model,label,color in MODELS:
  f=a.modality_dir/model/'scores.csv'
  if f.exists():
   rows=sorted(list(csv.DictReader(open(f))),key=lambda r:int(r['video_layer'])); y=np.array([float(r['score']) for r in rows]); x=np.array([int(r['video_layer'])+1 for r in rows]); data.append((label,color,x,y))
 if not data: raise RuntimeError('no scores')
 ymin=min(z[3].min() for z in data); ymax=max(z[3].max() for z in data); pad=(ymax-ymin)*.12
 fig,axs=plt.subplots(2,3,figsize=(12,6.8),sharex=True,sharey=True,dpi=180); axs=axs.ravel()
 for ax,(label,color,x,y) in zip(axs,data):
  i=int(y.argmax()); ax.plot(x,y,color=color,lw=2.2); ax.scatter(x[i],y[i],s=48,color=color,edgecolor='white',zorder=3); ax.annotate(f'{y[i]:.3f}',(x[i],y[i]),xytext=(0,7),textcoords='offset points',ha='center',color=color,weight='bold',fontsize=9)
  ax.set_title(label,fontsize=10); ax.grid(True,alpha=.2); ax.spines[['top','right']].set_visible(False)
 for ax in axs[len(data):]: ax.axis('off')
 fig.supxlabel('WAM transformer layer'); fig.supylabel('Mutual-kNN alignment'); fig.suptitle(a.title,fontsize=14,weight='bold'); fig.subplots_adjust(top=.86,wspace=.18,hspace=.35); axs[0].set_ylim(ymin-pad,ymax+pad)
 fig.savefig(a.modality_dir/'small_multiples.png',dpi=300,bbox_inches='tight'); fig.savefig(a.modality_dir/'small_multiples.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
