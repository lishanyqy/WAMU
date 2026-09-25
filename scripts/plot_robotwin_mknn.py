#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out-dir',required=True); a=p.parse_args()
    rows=list(csv.DictReader(open(a.csv))); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    names={'joint_cross_attention':'Joint cross-attention','joint_cross_attention_detach':'Detached video branch'}
    colors={'joint_cross_attention':'#1769aa','joint_cross_attention_detach':'#d95f02'}
    fig,ax=plt.subplots(figsize=(8.2,5.2),dpi=180)
    for model in names:
        z=[r for r in rows if r['model']==model]; z.sort(key=lambda r:int(r['video_layer']))
        x=[int(r['video_layer'])+1 for r in z]; y=[float(r['score']) for r in z]
        i=max(range(len(y)),key=lambda j:y[j]); ax.plot(x,y,lw=2.6,color=colors[model],label=names[model])
        ax.scatter([x[i]],[y[i]],s=55,color=colors[model],edgecolor='white',linewidth=1.2,zorder=4)
        ax.annotate(f'{y[i]:.3f}',(x[i],y[i]),xytext=(0,10),textcoords='offset points',ha='center',color=colors[model],fontsize=10,weight='bold')
    ax.set(xlabel='WAM transformer layer',ylabel='Mutual-kNN alignment',title='Robotwin visual representation alignment')
    ax.grid(True,alpha=.22); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='best')
    fig.tight_layout(); fig.savefig(out/'dinov2_mknn.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'dinov2_mknn.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
