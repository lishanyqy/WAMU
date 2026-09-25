#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

MODELS=[
 ('robotwin_dual_system_joint_cross_attention','Joint cross-attn','#1769aa'),
 ('robotwin_dual_system_joint_cross_attention_detach','Joint cross-attn detach','#d95f02'),
 ('robotwin_dual_system_joint_self_attention','Joint self-attn','#2ca25f'),
 ('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated','#756bb1'),
 ('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action','#e7298a'),
]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--modality-dir',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args()
 fig,ax=plt.subplots(figsize=(9,5.5),dpi=180); plotted=0
 for model,label,color in MODELS:
  f=a.modality_dir/model/'scores.csv'
  if not f.exists(): continue
  rows=list(csv.DictReader(open(f))); rows.sort(key=lambda r:int(r['video_layer']))
  if not rows: continue
  x=[int(r['video_layer'])+1 for r in rows]; y=[float(r['score']) for r in rows]
  i=max(range(len(y)),key=lambda j:y[j]); ax.plot(x,y,lw=2.2,color=color,label=label); ax.scatter(x[i],y[i],s=38,color=color,edgecolor='white',zorder=4)
  ax.annotate(f'{y[i]:.3f}',(x[i],y[i]),xytext=(0,7),textcoords='offset points',ha='center',color=color,fontsize=8)
  plotted+=1
 if not plotted: raise RuntimeError(f'no model scores in {a.modality_dir}')
 ax.set(xlabel='WAM transformer layer',ylabel='Mutual-kNN alignment',title=a.title); ax.grid(True,alpha=.22); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=9); fig.tight_layout()
 fig.savefig(a.modality_dir/'layer_comparison.png',dpi=300,bbox_inches='tight'); fig.savefig(a.modality_dir/'layer_comparison.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
