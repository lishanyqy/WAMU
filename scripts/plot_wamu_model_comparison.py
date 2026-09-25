#!/usr/bin/env python3
"""Plot model-level WAM alignment comparisons for one dataset/modality."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ('robotwin_dual_system_joint_cross_attention', 'Joint cross-attn'),
    ('robotwin_dual_system_joint_cross_attention_detach', 'Joint cross-attn\ndetach'),
    ('robotwin_dual_system_joint_self_attention', 'Joint self-attn'),
    ('robotwin_dual_system_joint_self_attention_isolated', 'Self-attn\nisolated'),
    ('robotwin_dual_system_joint_self_attention_video_sees_action', 'Self-attn\nvideo→action'),
    ('robotwin_dual_system_idm', 'IDM'),
    ('robotwin_dual_system_joint_self_attention_mutual', 'Self-attn\nmutual'),
]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--modality-dir',type=Path,required=True); p.add_argument('--title',required=True); a=p.parse_args()
    vals=[]; labels=[]; layers=[]
    for name,label in MODELS:
        f=a.modality_dir/name/'summary.json'
        if not f.exists(): continue
        data=json.loads(f.read_text())['summary']; item=next(iter(data.values()))
        vals.append(float(item['score'])); layers.append(int(item['video_layer'])+1); labels.append(label)
    if not vals: raise RuntimeError(f'no summaries under {a.modality_dir}')
    out=a.modality_dir; x=np.arange(len(vals)); fig,ax=plt.subplots(figsize=(10,5.8),dpi=180)
    colors=['#1769aa','#d95f02','#2ca25f','#756bb1','#e7298a','#c44e52','#7a5195'][:len(vals)]
    bars=ax.bar(x,vals,color=colors,width=.68)
    ax.set_xticks(x,labels); ax.set_ylabel('Best mutual-kNN alignment'); ax.set_title(a.title)
    ax.grid(axis='y',alpha=.22); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    for b,v,l in zip(bars,vals,layers): ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*.012,f'{v:.3f}\nL{l}',ha='center',va='bottom',fontsize=9,weight='bold')
    ax.set_ylim(0,max(vals)*1.18); fig.tight_layout(); fig.savefig(out/'model_comparison.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'model_comparison.pdf',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
