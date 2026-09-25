#!/usr/bin/env python3
"""Extract temporally averaged Depth-Anything-3 maps as pooled depth features."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch,sys

def read_frames(path,n=33,stride=4):
    cap=cv2.VideoCapture(path); frames=[]
    while len(frames)<132:
        ok,b=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError(path)
    ids=np.arange(n,dtype=np.int64)*stride
    if ids[-1] >= len(frames): ids=np.linspace(0,len(frames)-1,n,dtype=np.int64)
    return [frames[int(i)] for i in ids]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--frames',type=int,default=33); p.add_argument('--keep-frames',action='store_true'); a=p.parse_args()
    sys.path.insert(0,'/home/a1905721/workspace/Depth-Anything-3/src')
    from depth_anything_3.api import DepthAnything3
    model=DepthAnything3.from_pretrained(str(a.model_dir)).to(a.device).eval(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    for i,row in enumerate(rows,1):
        out=a.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        with torch.inference_mode(): pred=model.inference(read_frames(row['video'],a.frames,row.get('frame_stride',4)),export_format='mini_npz')
        d=np.asarray(pred.depth,dtype=np.float32); d=np.nan_to_num(d,nan=0,posinf=0,neginf=0)
        # Preserve coarse spatial geometry: temporal mean of 16x16 normalized depth maps.
        x=np.stack([cv2.resize(z,(16,16),interpolation=cv2.INTER_AREA) for z in d])
        if a.keep_frames:
            x=(x-x.mean(axis=(1,2),keepdims=True))/(x.std(axis=(1,2),keepdims=True)+1e-6)
        else:
            x=x.mean(0); x=(x-x.mean())/(x.std()+1e-6); x=x.reshape(1,-1)
        np.savez_compressed(out,features=x.astype(np.float16),task=row['task'],episode=row['episode'],num_frames=a.frames,grid='16x16',model=str(a.model_dir))
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
