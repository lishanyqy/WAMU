#!/usr/bin/env python3
"""Layer-wise linear CKA between OpenWAM and reference representations."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch

MODELS=[('robotwin_dual_system_joint_cross_attention','openwam_joint_cross_attention'),('robotwin_dual_system_joint_cross_attention_detach','openwam_joint_cross_attention_detach'),('robotwin_dual_system_joint_self_attention','robotwin_dual_system_joint_self_attention'),('robotwin_dual_system_joint_self_attention_isolated','robotwin_dual_system_joint_self_attention_isolated'),('robotwin_dual_system_joint_self_attention_video_sees_action','robotwin_dual_system_joint_self_attention_video_sees_action'),('robotwin_dual_system_idm','robotwin_dual_system_idm'),('robotwin_dual_system_joint_self_attention_mutual','robotwin_dual_system_joint_self_attention_mutual')]
REFS=[('dinov2','dinov2_large'),('depthv3','depth_anything3'),('d4rt','d4rt_encoder'),('videomaev2','videomaev2_large')]
def load(d):
 fs=sorted(Path(d).glob('*.npz')); mp={f.name:f for f in fs}; return mp
def read(fs,keys):
 arr=[]
 for k in keys:
  x=np.load(fs[k])['features']; x=x.reshape(x.shape[0],-1); arr.append(x)
 return np.stack(arr).astype(np.float32)
def gram(x,device):
 x=torch.as_tensor(x,device=device); x=x-x.mean(0,keepdim=True); k=x@x.T
 k=k-k.mean(0,keepdim=True)-k.mean(1,keepdim=True)+k.mean()
 return k, k.square().sum().sqrt().clamp_min(1e-12)
def cka_gram(kx,nx,ky,ny): return float(((kx*ky).sum()/(nx*ny)).item())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--features-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--max-points',type=int,default=300); p.add_argument('--only-reference'); a=p.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device)
 refs=[x for x in REFS if not a.only_reference or x[0]==a.only_reference]; refmaps={n:load(a.features_root/d) for n,d in refs}; rows=[]
 for model,feature_dir in MODELS:
  vm=load(a.features_root/feature_dir)
  for ref,refdir in refs:
   common=sorted(set(vm)&set(refmaps[ref]));
   if not common: continue
   if len(common)>a.max_points: common=[common[i] for i in np.linspace(0,len(common)-1,a.max_points).astype(int)]
   vx=read(vm,common); ry=read(refmaps[ref],common)
   refgrams=[gram(ry[:,rl,:],device) for rl in range(ry.shape[1])]
   for vl in range(vx.shape[1]):
    kx,nx=gram(vx[:,vl,:],device)
    for rl,(ky,ny) in enumerate(refgrams): rows.append({'dataset':a.dataset,'model':model,'reference':ref,'video_layer':vl,'reference_layer':rl,'cka':cka_gram(kx,nx,ky,ny)})
   print(f'{model} {ref}: n={len(common)}',flush=True)
 with (out/'all_layer_cka.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 best=[]
 for model,model_key in MODELS:
  for ref,_ in refs:
   z=[r for r in rows if r['model']==model and r['reference']==ref]
   if z:
    for vl in sorted(set(int(r['video_layer']) for r in z)):
     q=max((r for r in z if int(r['video_layer'])==vl),key=lambda r:r['cka']); best.append(q)
 summary={}
 for model,_ in MODELS:
  summary[model]={}
  for ref,_ in refs:
   z=[r for r in rows if r['model']==model and r['reference']==ref]; summary[model][ref]=max(z,key=lambda r:r['cka']) if z else None
 with (out/'best_layer_cka.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=best[0]); w.writeheader(); w.writerows(best)
 (out/'summary.json').write_text(json.dumps({'protocol':'linear CKA','dataset':a.dataset,'summary':summary},indent=2))
if __name__=='__main__': main()
