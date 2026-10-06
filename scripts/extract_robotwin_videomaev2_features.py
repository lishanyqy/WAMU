#!/usr/bin/env python3
"""Extract layer-wise VideoMAEv2-Large features for Robotwin videos.

Aligned GT frames 1..32 are split into two consecutive 16-frame clips (the model's
native clip length).  With tubelet size 2 this yields 16 temporal positions, and
tubelets 2k-2 and 2k-1 cover exactly OpenWAM latent k (aligned frames 4k-3..4k).
"""
import argparse, json, sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np, torch
from PIL import Image
from robotwin_gt_frames import load_row_frames

FIRST, LAST = 1, 32  # aligned-frame range shared with D4RT; frame 0 is OpenWAM latent 0 only

def read_frames(row, n=33, stride=4):
    return [x.resize((224,224),Image.Resampling.BILINEAR) for x in load_row_frames(row,n,stride)[1]]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--limit',type=int,default=0); p.add_argument('--keep-frames',action='store_true',help='save (tubelets,layers,dim) instead of time-pooled (layers,dim)'); a=p.parse_args()
    sys.path.insert(0,str(a.model)); from transformers import AutoConfig, AutoModel
    device=torch.device(a.device); dtype=torch.float16 if device.type=='cuda' else torch.float32
    config=AutoConfig.from_pretrained(str(a.model),trust_remote_code=True)
    model=AutoModel.from_pretrained(str(a.model),config=config,trust_remote_code=True).to(device).eval()
    clip_len=model.config.model_config['num_frames']; tubelet=model.config.model_config['tubelet_size']
    frame_ids=np.arange(FIRST,LAST+1); assert len(frame_ids)%clip_len==0
    tubelet_frames=frame_ids.reshape(-1,tubelet)
    blocks=model.model.blocks; a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]; rows=rows[:a.limit] if a.limit else rows
    print(f'device={device} layers={len(blocks)} dim={model.config.model_config["embed_dim"]} clips={len(frame_ids)//clip_len}x{clip_len} tubelets={len(tubelet_frames)} videos={len(rows)}',flush=True)
    amp_ctx=(lambda: torch.cuda.amp.autocast(dtype=dtype)) if device.type=='cuda' else nullcontext
    for i,row in enumerate(rows,1):
        out=a.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        frames=read_frames(row)
        clip_feats=[]
        for c in range(0,len(frame_ids),clip_len):
            arr=np.stack([np.asarray(frames[int(j)]).astype(np.float32)/255.0 for j in frame_ids[c:c+clip_len]])
            arr=(arr-np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
            x=torch.from_numpy(arr).permute(3,0,1,2).unsqueeze(0).to(device)
            t=clip_len//tubelet; feats=[]
            with torch.inference_mode(), amp_ctx():
                x=model.model.patch_embed(x)
                x=x+model.model.pos_embed.to(x.dtype).to(device)
                x=model.model.pos_drop(x)
                for block in blocks:
                    x=block(x)
                    # Tokens are (t h w): pool space within each tubelet.
                    feats.append(x.reshape(1,t,x.shape[1]//t,x.shape[-1]).mean(dim=2).squeeze(0).float().cpu().numpy())
            clip_feats.append(np.stack(feats,axis=1))  # (t, layers, dim)
        arr=np.concatenate(clip_feats,axis=0)  # (tubelets, layers, dim)
        if not a.keep_frames: arr=arr.mean(0)
        np.savez_compressed(out,features=arr.astype(np.float16),task=row['task'],episode=row['episode'],layers=len(blocks),dim=arr.shape[-1],clip_len=clip_len,tubelet=tubelet,aligned_frames=frame_ids,tubelet_aligned_frames=tubelet_frames)
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
