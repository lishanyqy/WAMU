#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from omegaconf import OmegaConf

def frames(path, n=33, stride=4):
    cap=cv2.VideoCapture(path); xs=[]
    while len(xs)<132:
        ok,b=cap.read()
        if not ok: break
        xs.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
    cap.release()
    if not xs: raise RuntimeError(path)
    ids=np.arange(n,dtype=np.int64)*stride
    if ids[-1] >= len(xs): ids=np.linspace(0,len(xs)-1,n,dtype=np.int64)
    return np.stack([cv2.resize(xs[int(i)],(320,384),interpolation=cv2.INTER_AREA) for i in ids])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt-dir',type=Path,required=True); p.add_argument('--wan-dir',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    sys.path.insert(0,'.'); from openwam.deploy.model_loader import load_from_checkpoint_dir
    with tempfile.TemporaryDirectory(prefix='openwam_extract_') as td:
        td=Path(td); cfg=OmegaConf.load(args.ckpt_dir/'config.yaml'); cfg.model.video_backbone.model_path=str(args.wan_dir); cfg.model.video_backbone.encoder.model_path=str(args.wan_dir); OmegaConf.save(cfg,td/'config.yaml'); (td/'checkpoint_step_1.safetensors').symlink_to(args.ckpt_dir/'checkpoint_step_9890.safetensors'); (td/'normalization_stats.npy').symlink_to(args.ckpt_dir/'normalization_stats.npy')
        cfg, arch=load_from_checkpoint_dir(str(td),device=args.device,ckpt_name='checkpoint_step_1.safetensors')
    arch.eval(); vb=arch.video_backbone; dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; args.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    print(f'device={args.device} dtype={dtype} layers={vb.num_layers}',flush=True)
    for i,row in enumerate(rows,1):
        out=args.output/f"{row['task']}__{row['episode']}.npz"
        if out.exists(): continue
        rgb=frames(row['video'], row.get('num_frames', 33), row.get('frame_stride', 4)); pil=[Image.fromarray(x) for x in rgb]
        with torch.inference_mode(), torch.autocast('cuda',dtype=dtype):
            inputs=vb.preprocess_input_for_train(frames=[pil],text=['A video of a robot manipulating objects'])
            clean=inputs.pop('input_latents'); noise=torch.randn_like(clean); t=torch.full((1,),500.0,device=clean.device,dtype=clean.dtype); inputs['latents']=0.5*clean+0.5*noise; inputs['timestep']=t
            state=vb.prepare(**inputs)
            feats=[]
            for layer in range(vb.num_layers):
                state=vb.run_block(layer,state); feats.append(state.hidden_states.mean(dim=1).squeeze(0).float().cpu().numpy())
        np.savez_compressed(out,features=np.stack(feats).astype(np.float16),task=row['task'],episode=row['episode'],layers=vb.num_layers,dtype=str(dtype))
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)
if __name__=='__main__': main()
