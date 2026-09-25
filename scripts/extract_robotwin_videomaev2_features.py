#!/usr/bin/env python3
"""Extract layer-wise VideoMAEv2-Large features for Robotwin videos."""
import argparse, json, sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np, torch
from decord import VideoReader, cpu as decord_cpu
from PIL import Image

def read_frames(path, n=33, stride=4):
    vr=VideoReader(str(path),ctx=decord_cpu(0)); total=len(vr); take=min(total,n*stride)
    xs=[vr[i].asnumpy() for i in range(take)]
    if not xs: raise RuntimeError(f'cannot decode {path}')
    ids=np.arange(n,dtype=np.int64)*stride
    if ids[-1]>=len(xs): ids=np.linspace(0,len(xs)-1,n,dtype=np.int64)
    return [Image.fromarray(xs[int(i)]).resize((224,224),Image.Resampling.BILINEAR) for i in ids]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--frames',type=int,default=16); p.add_argument('--source-frames',type=int,default=33); p.add_argument('--stride',type=int,default=4); p.add_argument('--limit',type=int,default=0); p.add_argument('--keep-frames',action='store_true'); a=p.parse_args()
    sys.path.insert(0,str(a.model)); from transformers import AutoConfig, AutoModel
    device=torch.device(a.device); dtype=torch.float16 if device.type=='cuda' else torch.float32
    config=AutoConfig.from_pretrained(str(a.model),trust_remote_code=True)
    model=AutoModel.from_pretrained(str(a.model),config=config,trust_remote_code=True).to(device).eval()
    blocks=model.model.blocks; a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]; rows=rows[:a.limit] if a.limit else rows
    print(f'device={device} layers={len(blocks)} dim={model.config.model_config["embed_dim"]} videos={len(rows)}',flush=True)
    for i,row in enumerate(rows,1):
        out=a.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        frames=read_frames(row['video'],a.source_frames,a.stride)
        idx=np.linspace(0,len(frames)-1,a.frames,dtype=np.int64); frames=[frames[int(j)] for j in idx]
        arr=np.stack([np.asarray(f).astype(np.float32)/255.0 for f in frames])
        arr=(arr-np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
        x=torch.from_numpy(arr).permute(3,0,1,2).unsqueeze(0).to(device)
        feats=[]
        amp_ctx=torch.cuda.amp.autocast(dtype=dtype) if device.type=='cuda' else nullcontext()
        with torch.inference_mode(), amp_ctx:
            x=model.model.patch_embed(x)
            x=x+model.model.pos_embed.to(x.dtype).to(device)
            x=model.model.pos_drop(x)
            for block in blocks:
                x=block(x)
                if a.keep_frames:
                    n=x.shape[1]//a.frames
                    feats.append(x.reshape(1,a.frames,n,x.shape[-1]).mean(dim=2).squeeze(0).float().cpu().numpy())
                else: feats.append(x.mean(dim=1).squeeze(0).float().cpu().numpy())
        if a.keep_frames: arr=np.stack(feats,axis=1)
        else: arr=np.stack(feats)
        np.savez_compressed(out,features=arr.astype(np.float16),task=row['task'],episode=row['episode'],layers=len(feats),dim=feats[0].shape[-1],frames=a.frames,source_frames=a.source_frames,stride=a.stride)
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
