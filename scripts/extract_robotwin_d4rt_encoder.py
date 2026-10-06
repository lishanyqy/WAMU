#!/usr/bin/env python
"""Extract OpenD4RT encoder memory (the tokens its decoder queries) for Robotwin GT frames.

Aligned GT frames 1..32 form the model's native 32-frame clip.  With tubelet size 2
this yields 16 temporal positions, and tubelets 2k-2 and 2k-1 cover exactly OpenWAM
latent k (aligned frames 4k-3..4k).  Output: (16, 16, 16, 1280) = (t, h, w, dim).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robotwin_gt_frames import load_row_frames

FIRST, LAST = 1, 32  # aligned-frame range shared with VideoMAEv2; frame 0 is OpenWAM latent 0 only

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--repo',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--limit',type=int,default=0); args=p.parse_args()
    sys.path.insert(0,str(args.repo)); from src.model import build_model
    from src.core.config import load_yaml_config
    cfg=load_yaml_config(args.config); model=build_model(cfg['model']).eval()
    payload=torch.load(args.checkpoint,map_location='cpu'); state=payload.get('state_dict',payload.get('model',payload)); model.load_state_dict(state,strict=False)
    device=torch.device(args.device); model.to(device).eval(); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mcfg=cfg['model']; clip_len=int(mcfg['input']['clip_frames']); size=tuple(mcfg['input']['image_size']); pt,ph,pw=mcfg['encoder']['patch_size_t_h_w']
    frame_ids=np.arange(FIRST,LAST+1); assert len(frame_ids)==clip_len, (len(frame_ids),clip_len)
    grid=(clip_len//pt,size[0]//ph,size[1]//pw)
    args.output.mkdir(parents=True,exist_ok=True); rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]; rows=rows[:args.limit] if args.limit else rows
    print(f'device={device} dtype={dtype} model={args.checkpoint} grid={grid}',flush=True)
    for i,row in enumerate(rows,1):
        out=args.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        pil=load_row_frames(row,33,4)[1]
        src_w,src_h=pil[0].size
        frames=[cv2.resize(np.asarray(pil[int(j)]),(size[1],size[0]),interpolation=cv2.INTER_AREA) for j in frame_ids]
        x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float().div_(255).unsqueeze(0).to(device)
        # Training feeds the native width/height ratio; the default (None) would claim a square source.
        aspect=torch.tensor([[src_w/src_h]],device=device)
        with torch.inference_mode(), torch.autocast('cuda',dtype=dtype):
            z=model.encode_video(x,aspect_ratio=aspect)  # (1, t*h*w [+1 aspect token], dim), tokens in (t h w) order
        n=grid[0]*grid[1]*grid[2]
        feat=z[0,:n].reshape(*grid,z.shape[-1]).float().cpu().numpy().astype(np.float16)
        np.savez_compressed(out,features=feat,task=row['task'],episode=row['episode'],grid=np.array(grid),aspect_ratio=src_w/src_h,
                            aligned_frames=frame_ids,tubelet_aligned_frames=frame_ids.reshape(-1,pt),dtype=str(dtype))
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
