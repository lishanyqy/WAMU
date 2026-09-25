#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch, yaml

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--repo',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--num-frames',type=int,default=32); p.add_argument('--limit',type=int,default=0); p.add_argument('--keep-frames',action='store_true'); args=p.parse_args()
    sys.path.insert(0,str(args.repo)); from src.model import build_model
    from src.core.config import load_yaml_config
    cfg=load_yaml_config(args.config); model=build_model(cfg['model']).eval()
    payload=torch.load(args.checkpoint,map_location='cpu'); state=payload.get('state_dict',payload.get('model',payload)); model.load_state_dict(state,strict=False)
    device=torch.device(args.device); model.to(device).eval(); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output.mkdir(parents=True,exist_ok=True); rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]; rows=rows[:args.limit] if args.limit else rows
    print(f'device={device} dtype={dtype} model={args.checkpoint}',flush=True)
    for i,row in enumerate(rows,1):
        out=args.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        cap=cv2.VideoCapture(row['video']); frames=[]
        while len(frames)<args.num_frames:
            ok,bgr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA); frames.append(rgb)
        cap.release()
        if not frames: raise RuntimeError(row['video'])
        while len(frames)<args.num_frames: frames.append(frames[-1])
        x=torch.from_numpy(np.stack(frames).copy()).permute(0,3,1,2).float().div_(255).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast('cuda',dtype=dtype):
            z=model.encode_video(x)
            if args.keep_frames and z.ndim >= 3 and z.shape[1] == args.num_frames:
                feat=z.squeeze(0).float().cpu().numpy().astype(np.float16)
            else:
                feat=z.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float16)
        np.savez_compressed(out,features=feat,task=row['task'],episode=row['episode'],num_frames=args.num_frames,dtype=str(dtype))
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
