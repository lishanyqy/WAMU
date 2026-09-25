#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS=[('robotwin_dual_system_joint_cross_attention','Joint cross-attn'),('robotwin_dual_system_joint_cross_attention_detach','Cross-attn detach'),('robotwin_dual_system_joint_self_attention','Joint self-attn'),('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated'),('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action'),('robotwin_dual_system_idm','IDM'),('robotwin_dual_system_joint_self_attention_mutual','Self-attn mutual')]
REFS=[('dinov2','DINOv2'),('depthv3','Depthv3'),('d4rt','D4RT'),('videomaev2','VideoMAEv2')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args(); s=json.loads((a.root/'summary.json').read_text())['summary']; vals=np.full((len(MODELS),len(REFS)),np.nan)
 for i,(m,_) in enumerate(MODELS):
  for j,(r,_) in enumerate(REFS):
   if r in s.get(m,{}): vals[i,j]=s[m][r]['rbf_cka']
 labels=[x[1] for x in MODELS]; refs=[x[1] for x in REFS]
 fig,ax=plt.subplots(figsize=(10.5,7),dpi=180); im=ax.imshow(vals,cmap='YlGnBu',vmin=np.nanmin(vals)-.01,vmax=np.nanmax(vals)); ax.set_xticks(range(len(REFS)),refs); ax.set_yticks(range(len(labels)),labels); ax.set_title(a.title+' — best RBF CKA',weight='bold');
 for i in range(len(labels)):
  for j in range(len(REFS)):
   if np.isfinite(vals[i,j]): ax.text(j,i,f'{vals[i,j]:.3f}',ha='center',va='center',weight='bold',color='white' if vals[i,j]>.5 else '#16324f')
 fig.colorbar(im,ax=ax,label='Best RBF-kernel CKA'); fig.tight_layout(); fig.savefig(a.root/'rbf_cka_heatmap.png',dpi=300,bbox_inches='tight'); fig.savefig(a.root/'rbf_cka_heatmap.pdf',bbox_inches='tight'); plt.close(fig)
 fig,ax=plt.subplots(figsize=(14,6.5),dpi=180); x=np.arange(len(labels)); width=.19; colors=['#2878b5','#49a078','#e09f3e','#8c6bb1']
 for j,(_,lab) in enumerate(REFS): ax.bar(x+(j-1)*width,vals[:,j],width,label=lab,color=colors[j])
 ax.set_xticks(x,labels); ax.set_ylabel('Best RBF-kernel CKA'); ax.set_title(a.title+' — model comparison',weight='bold'); ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(a.root/'rbf_cka_grouped_bar.png',dpi=300,bbox_inches='tight'); fig.savefig(a.root/'rbf_cka_grouped_bar.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
