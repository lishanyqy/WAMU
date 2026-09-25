#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS=[('robotwin_dual_system_joint_cross_attention','Joint cross-attn'),('robotwin_dual_system_joint_cross_attention_detach','Joint cross-attn detach'),('robotwin_dual_system_joint_self_attention','Joint self-attn'),('robotwin_dual_system_joint_self_attention_isolated','Self-attn isolated'),('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attn video→action'),('robotwin_dual_system_idm','IDM'),('robotwin_dual_system_joint_self_attention_mutual','Self-attn mutual')]
REFS=[('dinov2','DINOv2'),('depthv3','Depthv3'),('d4rt','D4RT'),('videomaev2','VideoMAEv2')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args(); s=json.loads((a.root/'summary.json').read_text())['summary']; vals=np.full((len(MODELS),len(REFS)),np.nan); layers=np.full((len(MODELS),len(REFS)),-1)
 for i,(m,_) in enumerate(MODELS):
  for j,(r,_) in enumerate(REFS):
   z=s.get(m,{}).get(r)
   if z: vals[i,j]=z['cka']; layers[i,j]=int(z['video_layer'])+1
 labels=[x[1] for x in MODELS]; refs=[x[1] for x in REFS]
 # Heatmap of best layer-wise CKA.
 fig,ax=plt.subplots(figsize=(10.5,7.2),dpi=180); im=ax.imshow(vals,cmap='YlGnBu',vmin=np.nanmin(vals)-.01,vmax=np.nanmax(vals)); ax.set_xticks(range(len(REFS)),refs); ax.set_yticks(range(len(MODELS)),labels); ax.set_title(a.title+' — best linear CKA',weight='bold',pad=12); ax.set_xlabel('Reference representation'); ax.set_ylabel('OpenWAM model');
 for i in range(len(MODELS)):
  for j in range(len(REFS)):
   if np.isfinite(vals[i,j]): ax.text(j,i,f'{vals[i,j]:.3f}\nL{layers[i,j]}',ha='center',va='center',fontsize=9,weight='bold',color='white' if vals[i,j]>.5 else '#16324f')
 cbar=fig.colorbar(im,ax=ax,pad=.03); cbar.set_label('Best linear CKA'); fig.tight_layout(); fig.savefig(a.root/'cka_heatmap.png',dpi=300,bbox_inches='tight'); fig.savefig(a.root/'cka_heatmap.pdf',bbox_inches='tight'); plt.close(fig)
 # Grouped bars.
 fig,ax=plt.subplots(figsize=(14,6.4),dpi=180); x=np.arange(len(MODELS)); width=.19; colors=['#2878b5','#49a078','#e09f3e','#8c6bb1']
 for j,(r,label) in enumerate(REFS):
  b=ax.bar(x+(j-1)*width,vals[:,j],width,label=label,color=colors[j])
  for q,v,l in zip(b,vals[:,j],layers[:,j]):
   if np.isfinite(v): ax.text(q.get_x()+q.get_width()/2,v+.004,f'{v:.3f}',ha='center',va='bottom',fontsize=7,rotation=90)
 ax.set_xticks(x,labels); ax.set_ylabel('Best linear CKA'); ax.set_title(a.title+' — model comparison',weight='bold',pad=12); ax.set_ylim(0,max(.1,np.nanmax(vals)*1.16)); ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,ncol=3,loc='upper left'); fig.tight_layout(); fig.savefig(a.root/'cka_grouped_bar.png',dpi=300,bbox_inches='tight'); fig.savefig(a.root/'cka_grouped_bar.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
