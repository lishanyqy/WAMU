#!/usr/bin/env python3
"""Layer-wise RBF-kernel CKA for Robotwin feature shards."""
import argparse, csv, json
from pathlib import Path
import numpy as np

MODELS=[
 ('robotwin_dual_system_joint_cross_attention','openwam_joint_cross_attention'),
 ('robotwin_dual_system_joint_cross_attention_detach','openwam_joint_cross_attention_detach'),
 ('robotwin_dual_system_joint_self_attention','robotwin_dual_system_joint_self_attention'),
 ('robotwin_dual_system_joint_self_attention_isolated','robotwin_dual_system_joint_self_attention_isolated'),
 ('robotwin_dual_system_joint_self_attention_video_sees_action','robotwin_dual_system_joint_self_attention_video_sees_action'),
 ('robotwin_dual_system_idm','robotwin_dual_system_idm'),
 ('robotwin_dual_system_joint_self_attention_mutual','robotwin_dual_system_joint_self_attention_mutual'),
]
REFS=[('dinov2','dinov2_large'),('depthv3','depth_anything3'),('d4rt','d4rt_encoder'),('videomaev2','videomaev2_large')]

def load(d):
    fs=sorted(Path(d).glob('*.npz'))
    return {f.name:f for f in fs}

def read(fs, keys):
    arr=np.stack([np.load(fs[k])['features'].astype(np.float32) for k in keys])
    if arr.ndim==2: arr=arr[:,None,:]
    else: arr=arr.reshape(arr.shape[0],arr.shape[1],-1)
    return arr

def standardize(x):
    x=x-x.mean(0,keepdims=True)
    s=x.std(0,keepdims=True); return x/np.maximum(s,1e-6)

def rbf_gram(x, max_points):
    # Median heuristic on pairwise squared distances; block-free for reproducibility.
    x=standardize(x)
    if len(x)>max_points:
        idx=np.linspace(0,len(x)-1,max_points).astype(int); z=x[idx]
    else: z=x
    zz=(z*z).sum(1,keepdims=True)
    d=np.maximum(zz+zz.T-2*z@z.T,0)
    med=np.median(d[np.triu_indices_from(d,1)])
    gamma=1.0/max(2.0*med,1e-6)
    # Compute the full kernel on the same sample used for the median.
    k=np.exp(-gamma*d).astype(np.float64)
    k-=k.mean(0,keepdims=True); k-=k.mean(1,keepdims=True); k+=k.mean()
    return k, float((k*k).sum())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--features-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-points',type=int,default=1000); p.add_argument('--only-reference'); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    refs=[x for x in REFS if not a.only_reference or x[0]==a.only_reference]; refmaps={n:load(a.features_root/d) for n,d in refs}; rows=[]
    for model,modeldir in MODELS:
        vm=load(a.features_root/modeldir)
        for ref,refdir in refs:
            common=sorted(set(vm)&set(refmaps[ref]))
            if not common: continue
            if len(common)>a.max_points:
                common=[common[i] for i in np.linspace(0,len(common)-1,a.max_points).astype(int)]
            vx=read(vm,common); ry=read(refmaps[ref],common)
            refgrams=[rbf_gram(ry[:,rl,:],a.max_points) for rl in range(ry.shape[1])]
            for vl in range(vx.shape[1]):
                kx,nx=rbf_gram(vx[:,vl,:],a.max_points)
                for rl,(ky,ny) in enumerate(refgrams):
                    if nx <= 1e-12 or ny <= 1e-12:
                        continue
                    score=float((kx*ky).sum()/np.sqrt(nx*ny))
                    rows.append({'dataset':a.dataset,'model':model,'reference':ref,'video_layer':vl,'reference_layer':rl,'rbf_cka':score})
            print(f'{model} {ref}: n={len(common)}',flush=True)
    fields=['dataset','model','reference','video_layer','reference_layer','rbf_cka']
    with (a.output/'all_layer_rbf_cka.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    best=[]; summary={}
    for model,_ in MODELS:
        summary[model]={}
        for ref,_ in refs:
            z=[r for r in rows if r['model']==model and r['reference']==ref]
            if not z: continue
            summary[model][ref]=max(z,key=lambda r:r['rbf_cka'])
            for vl in sorted({int(r['video_layer']) for r in z}):
                best.append(max((r for r in z if int(r['video_layer'])==vl),key=lambda r:r['rbf_cka']))
    with (a.output/'best_layer_rbf_cka.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(best)
    (a.output/'summary.json').write_text(json.dumps({'protocol':'RBF-kernel CKA','dataset':a.dataset,'max_points':a.max_points,'summary':summary},indent=2))

if __name__=='__main__': main()
