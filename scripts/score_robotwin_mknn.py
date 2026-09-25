#!/usr/bin/env python3
"""Gen4U-style mutual-kNN alignment for the Robotwin feature shards."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch

def norm_clip(x, q=0.95):
    x = x.astype(np.float32, copy=True)
    x = np.clip(x, -np.quantile(np.abs(x), q), np.quantile(np.abs(x), q))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def torch_norm_clip(x, q, device):
    x = torch.as_tensor(x, device=device, dtype=torch.float32)
    lim = torch.quantile(x.abs(), q)
    x = x.clamp(-lim, lim)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

def knn(x, k):
    sim = x @ x.T
    sim.fill_diagonal_(-float('inf'))
    return sim.topk(k, dim=1).indices

def alignment(a, b, k):
    na, nb = knn(a, k), knn(b, k)
    # [N,k,k] membership test, kept on GPU.
    overlap = (na[:, :, None] == nb[:, None, :]).any(dim=2).float().sum(dim=1)
    return float((overlap / k).mean().item())

def load_dir(d, max_points=None):
    fs = sorted(Path(d).glob('*.npz'))
    if not fs: raise RuntimeError(f'no npz files under {d}')
    if max_points and len(fs)>max_points: fs=[fs[i] for i in np.linspace(0,len(fs)-1,max_points).astype(int)]
    rows = [np.load(f)['features'] for f in fs]
    arr=np.stack(rows)
    if arr.ndim == 2: arr=arr[:,None,:]
    return fs, arr

def list_dir(d):
    fs=sorted(Path(d).glob('*.npz'))
    if not fs: raise RuntimeError(f'no npz files under {d}')
    return fs

def load_selected(d, names):
    fs=[Path(d)/n for n in names]
    rows=[np.load(f)['features'] for f in fs]; arr=np.stack(rows)
    if arr.ndim==2: arr=arr[:,None,:]
    return arr

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video-a', required=True); p.add_argument('--video-b')
    p.add_argument('--reference', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--k', type=int, default=10); p.add_argument('--quantile', type=float, default=.95)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--max-points', type=int, default=300); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    fa=list_dir(a.video_a); fr=list_dir(a.reference); fb=(list_dir(a.video_b) if a.video_b else [])
    ma={f.name for f in fa}; mr={f.name for f in fr}; mb={f.name for f in fb}
    common=sorted(ma&mr&(mb if a.video_b else ma))
    if len(common)>a.max_points: common=[common[i] for i in np.linspace(0,len(common)-1,a.max_points).astype(int)]
    if len(common) <= a.k: raise RuntimeError('too few common feature shards')
    fa=[Path(a.video_a)/f for f in common]; xa=load_selected(a.video_a,common); xr=load_selected(a.reference,common)
    if a.video_b: xb=load_selected(a.video_b,common)
    print(f'using_common_shards={len(common)}', flush=True)
    if len(fa) <= a.k: raise RuntimeError('k must be smaller than number of videos')
    device=torch.device(a.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    results=[]; ref_knn=[]
    for rl in range(xr.shape[1]):
        ref_knn.append(knn(torch_norm_clip(xr[:,rl,:],a.quantile,device),a.k))
    model_inputs=[('joint_self_attention',xa)] if not a.video_b else [('joint_cross_attention',xa),('joint_cross_attention_detach',xb)]
    for model,x in model_inputs:
        for vl in range(x.shape[1]):
            av=torch_norm_clip(x[:,vl,:],a.quantile,device)
            for rl in range(xr.shape[1]):
                score=alignment(av, torch.empty(0), a.k) if False else float(((((knn(av,a.k)[:,:,None] == ref_knn[rl][:,None,:]).any(dim=2).float().sum(dim=1))/a.k).mean().item()))
                results.append({'model':model,'video_layer':vl,'reference_layer':rl,'score':score})
    with (out/'all_layer_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]); w.writeheader(); w.writerows(results)
    best=[]
    for model,_ in model_inputs:
        for vl in range(xa.shape[1]):
            z=max((r for r in results if r['model']==model and r['video_layer']==vl),key=lambda r:r['score'])
            best.append(z)
    with (out/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=best[0]); w.writeheader(); w.writerows(best)
    summary={}
    for model,_ in model_inputs:
        z=max((r for r in results if r['model']==model),key=lambda r:r['score']); summary[model]=z
    (out/'summary.json').write_text(json.dumps({'protocol':'Gen4U mutual-kNN','k':a.k,'quantile':a.quantile,'num_videos':len(fa),'summary':summary},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
