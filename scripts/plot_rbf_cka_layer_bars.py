#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS=[('robotwin_dual_system_joint_cross_attention','Joint cross-attn','#1769aa'),('robotwin_dual_system_joint_cross_attention_detach','Cross-attn detach','#d95f02'),('robotwin_dual_system_joint_self_attention','Joint self-attn','#2ca25f'),('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated','#756bb1'),('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action','#e7298a'),('robotwin_dual_system_idm','IDM','#c44e52'),('robotwin_dual_system_joint_self_attention_mutual','Self-attn mutual','#7a5195')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--reference',required=True); p.add_argument('--title',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 rows=list(csv.DictReader((a.root/'all_layer_rbf_cka.csv').open())); data=[]
 for model,label,color in MODELS:
  z=[r for r in rows if r['model']==model and r['reference']==a.reference]; by={}
  for r in z:
   v=float(r['rbf_cka'])
   if np.isfinite(v): by.setdefault(int(r['video_layer']),[]).append(v)
  if by: data.append((label,color,np.array([max(by[i]) for i in sorted(by)])))
 if not data: raise RuntimeError('no RBF CKA rows')
 ymin=min(y.min() for _,_,y in data); ymax=max(y.max() for _,_,y in data); lo=np.floor(ymin*10)/10; hi=min(1.,np.ceil(ymax*10)/10+.1)
 fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True,dpi=180); axs=axs.ravel()
 for ax,(label,color,y) in zip(axs,data):
  x=np.arange(len(y))+1; i=int(y.argmax()); ax.bar(x,y,color=color,width=.78); ax.bar(x[i],y[i],color='white',edgecolor=color,linewidth=2,width=.78); ax.set_title(label,fontsize=10); ax.set_xticks([1,5,10,15,20,25,30]); ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False); ax.annotate(f'{y[i]:.3f}',(x[i],y[i]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8,color=color,weight='bold')
 for ax in axs[len(data):]: ax.axis('off')
 fig.supxlabel('WAM transformer layer'); fig.supylabel('Best RBF-kernel CKA'); fig.suptitle(a.title,fontsize=14,weight='bold'); fig.subplots_adjust(top=.88,wspace=.16,hspace=.34); axs[0].set_ylim(lo,hi); a.output_dir.mkdir(parents=True,exist_ok=True); fig.savefig(a.output_dir/'layer_bars.png',dpi=300,bbox_inches='tight'); fig.savefig(a.output_dir/'layer_bars.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
