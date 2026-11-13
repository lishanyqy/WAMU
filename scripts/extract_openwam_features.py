#!/usr/bin/env python
"""Extract per-latent OpenWAM video-backbone features on Robotwin GT frames.

Input matches training: the 3-camera L-shape canvas with the replayed GT head camera
on top and the official left/right wrist cameras below (same raw frame indices; the
wrist cameras ride on the arm, whose replayed joints equal the official ones).

The causal Wan VAE maps the 33 aligned frames to 9 latents: latent 0 = aligned frame 0,
latent k = aligned frames 4k-3..4k.  Only head-camera tokens are kept, unpooled:
  features: (latents, layers, head_h, head_w, dim) = (9, 30, 8, 10, 3072), fp16, ~133 MB/episode
"""
from __future__ import annotations
import argparse, errno, json, os, sys, tempfile
from pathlib import Path
import h5py, numpy as np, torch
from omegaconf import OmegaConf

from robotwin_gt_frames import decode_rgb, load_row_frames
from PIL import Image

def official_hdf5(row):
    return row['video'].replace('/video/', '/data/').replace('.mp4', '.hdf5')

def canvas_frames(row, camera_layout, height, width, n=33, stride=4):
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    ids, head = load_row_frames(row, n, stride)
    views = {camera_layout[0]: head}
    with h5py.File(official_hdf5(row), 'r') as f:
        for cam in camera_layout[1:]:
            rgb = f[f'observation/{cam}/rgb']
            if len(rgb) <= ids[-1]:
                raise RuntimeError(f'{official_hdf5(row)} {cam} has {len(rgb)} frames, need {ids[-1] + 1}')
            views[cam] = [Image.fromarray(decode_rgb(rgb[int(i)])) for i in ids]
    return ids, [assemble_multiview_layout({c: views[c][t] for c in camera_layout}, camera_layout, height, width)
                 for t in range(len(ids))]

class OutputDirs:
    """Write to the first directory that has room; move to the next when a write hits a quota/space error."""
    FULL = (errno.EDQUOT, errno.ENOSPC)

    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.active = 0
        for d in self.dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
                probe = d / f'.write_probe_{os.getpid()}'
                probe.write_bytes(b'x'); probe.unlink()
                print(f'output dir ok: {d}', flush=True)
            except OSError as e:
                print(f'output dir NOT writable now: {d} ({e})', flush=True)

    def exists(self, name):
        return any((d / name).exists() for d in self.dirs)

    def save(self, name, **arrays):
        while True:
            d = self.dirs[self.active]
            tmp = d / f'.{name}.tmp'
            try:
                d.mkdir(parents=True, exist_ok=True)  # may itself fail with EDQUOT on a full filesystem
                # Write then rename, so a kill or a quota error never leaves a truncated .npz behind.
                with open(tmp, 'wb') as fh:
                    np.savez(fh, **arrays)
                os.replace(tmp, d / name)
                return d / name
            except OSError as e:
                tmp.unlink(missing_ok=True)
                if e.errno not in self.FULL or self.active + 1 >= len(self.dirs):
                    raise
                self.active += 1
                print(f'output dir full ({e.strerror}): {d} -> switching to {self.dirs[self.active]}', flush=True)

def latent_groups(n_frames, n_latents):
    """Aligned-frame [first, last] covered by each causal-VAE latent."""
    stride = (n_frames - 1) // (n_latents - 1)
    return np.array([[0, 0]] + [[stride * k - stride + 1, stride * k] for k in range(1, n_latents)])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt-dir',type=Path,required=True); p.add_argument('--wan-dir',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,nargs='+',required=True,help='output dirs in order of preference; later ones are used once earlier ones are full'); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--lean',action='store_true',help='load on CPU, keep the text encoder and action stream there, only the video backbone on the GPU (~11 GB; shared GPUs)')
    p.add_argument('--ckpt-name',default=None,help='checkpoint file in --ckpt-dir (default: the highest checkpoint_step_*.safetensors)'); args=p.parse_args()
    if args.ckpt_name is None:
        args.ckpt_name=max((q.name for q in args.ckpt_dir.glob('checkpoint_step_*.safetensors')),key=lambda n:int(n.split('_')[-1].split('.')[0]))
    print(f'checkpoint: {args.ckpt_dir/args.ckpt_name}',flush=True)
    sys.path.insert(0,'.'); from openwam.deploy.model_loader import load_from_checkpoint_dir
    with tempfile.TemporaryDirectory(prefix='openwam_extract_') as td:
        td=Path(td); cfg=OmegaConf.load(args.ckpt_dir/'config.yaml'); cfg.model.video_backbone.model_path=str(args.wan_dir); cfg.model.video_backbone.encoder.model_path=str(args.wan_dir)
        # action_film now defaults to on; checkpoints trained before it existed carry neither the key nor its weights.
        if OmegaConf.select(cfg,'model.architecture.action_film') is None:
            from safetensors import safe_open
            with safe_open(str(args.ckpt_dir/args.ckpt_name),'pt') as f:
                has_film=any('action_film' in k for k in f.keys())
            OmegaConf.update(cfg,'model.architecture.action_film',has_film,force_add=True)
            print(f'action_film not in config; checkpoint has film weights={has_film} -> action_film={has_film}',flush=True)
        OmegaConf.save(cfg,td/'config.yaml'); (td/'checkpoint_step_1.safetensors').symlink_to(args.ckpt_dir/args.ckpt_name); (td/'normalization_stats.npy').symlink_to(args.ckpt_dir/'normalization_stats.npy')
        _, arch=load_from_checkpoint_dir(str(td),device='cpu' if args.lean else args.device,ckpt_name='checkpoint_step_1.safetensors')
    dl=OmegaConf.load(args.ckpt_dir/'config.yaml').dataloader
    layout=list(dl.camera_layout); height,width=int(dl.height),int(dl.width)
    assert dl.multiview and len(layout)==3, (dl.multiview, layout)
    top_h=int(round(height*2/3))  # assemble_multiview_layout default: head camera fills the top 2/3
    arch.eval(); vb=arch.video_backbone; dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; outputs=OutputDirs(args.output)
    from attention_video_share import offload_text_encoder, oom_retry
    if args.lean:
        oom_retry(lambda: offload_text_encoder(vb,[PROMPT],args.device),'prompt encoding')
        dev=torch.device(args.device)
        def to_gpu():
            for n,m in vb.named_children():
                if n!='text_encoder': m.to(device=dev,dtype=torch.bfloat16)
            vb._device,vb._dtype=dev,torch.bfloat16; torch.cuda.empty_cache()
        oom_retry(to_gpu,'moving the video backbone to the GPU')
        print(f'lean: video backbone on GPU, {torch.cuda.memory_allocated()/2**30:.1f} GB allocated',flush=True)
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]; rows=rows[:args.limit] if args.limit else rows
    print(f'device={args.device} dtype={dtype} layers={vb.num_layers} layout={layout} canvas={height}x{width}',flush=True)
    for i,row in enumerate(rows,1):
        name=f"{row['task']}__{row['episode']}.npz"
        if outputs.exists(name): continue
        ids,pil=canvas_frames(row,layout,height,width,row.get('num_frames',33),row.get('frame_stride',4))
        feats=oom_retry(lambda: forward(vb,pil,args.seed,dtype,top_h,height),f"{row['task']}/{row['episode']}")
        # fp16 activations barely compress, so skip savez_compressed's CPU cost.
        outputs.save(name,features=feats,task=row['task'],episode=row['episode'],layers=vb.num_layers,dtype=str(dtype),seed=args.seed,
                            head_grid=np.array([feats.shape[2],feats.shape[3]]),raw_frame_indices=ids,latent_aligned_frames=latent_groups(len(ids),feats.shape[0]),camera_layout=np.array(layout))
        if i%10==0: print(f'progress={i}/{len(rows)}',flush=True)
    print(f'complete={len(rows)}',flush=True)


PROMPT='A video of a robot manipulating objects'


def forward(vb,pil,seed,dtype,top_h,height):
        with torch.inference_mode(), torch.autocast('cuda',dtype=dtype):
            inputs=vb.preprocess_input_for_train(frames=[pil],text=[PROMPT])
            clean=inputs.pop('input_latents'); noise=torch.randn(clean.shape,generator=torch.Generator(device=clean.device).manual_seed(seed),device=clean.device,dtype=clean.dtype); t=torch.full((1,),500.0,device=clean.device,dtype=clean.dtype); inputs['latents']=0.5*clean+0.5*noise; inputs['timestep']=t
            state=vb.prepare(**inputs)
            gf,gh,gw=state.grid_frames,state.grid_height,state.grid_width
            head_h=gh*top_h//height; assert head_h*height==gh*top_h, (gh,top_h,height)
            feats=[]
            for layer in range(vb.num_layers):
                state=vb.run_block(layer,state)
                # Tokens are (f h w); keep the head-camera rows of every latent frame.
                head=state.hidden_states[0].reshape(gf,gh,gw,-1)[:,:head_h]
                feats.append(head.to(torch.float16).cpu().numpy())
        return np.stack(feats,axis=1)


if __name__=='__main__': main()
