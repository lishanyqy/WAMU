#!/usr/bin/env python
"""Per-window Action Attention Dispersion (AAD) inputs and action->video token attention maps for a few episodes.

Same forward and the same hooks as scripts/attention_video_share.py (whose aloha-c2r run gives Table 5 of the paper):
video latents noised at sigma_v = 0.5, one action noise per action sigma (default 0.25 / 0.5 / 0.75), one torch
generator seeded once, windows at start / middle / end of each episode.  Unlike that script, nothing is averaged over
windows, so AAD can be computed per episode (and per window), and the attention of the action queries over the
360 video tokens is kept for heatmaps.

Per window w, action sigma s and layer l the output holds
    entropy[w, s, l]        attention entropy of the action queries over all keys they can see (nats; mean over heads
                            and action tokens) -- the quantity behind AAD in the paper (normalised by ln 360)
    entropy_video[w, s, l]  entropy of the attention renormalised over the 360 video keys
    mass_video[w, s, l]     softmax mass on video keys (1 for bridge cross-attention)
    attn[w, s, l, 360]      attention mass on each video token (mean over heads and action tokens), layout (f h w)
and frames.npz (shared by all models) the 9 RGB canvas frames of every window.

usage: python scripts/aad_episode_dump.py --ckpt sft_from_scratch_robotwin_clean \
           --tasks beat_block_hammer handover_block --episodes-per-task 3 --windows 3
Env: fastwam (CUDA); ~26 GB GPU.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import attention_video_share as avs  # noqa: E402

MAPS = {}
_orig_add = avs.add_video_entropies


def add_video_entropies(rec, p_video):
    """Called by both hook families once per layer with the attention over video keys (B, H, Sa, Sv); keep its mean."""
    _orig_add(rec, p_video)
    layer = next(i for i, r in enumerate(avs.STATE['rec']) if r is rec)
    MAPS[layer] = (p_video.float().mean((0, 1, 2)).cpu().numpy(), float(p_video.sum(-1).mean()))


avs.add_video_entropies = add_video_entropies


def seeded_sample(ds, ep_idx, start, key):
    """Instruction draw seeded by (task, episode, start), as read_subspace.seeded_sample: same prompt for every model."""
    random.seed(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    return ds._build_sample(ep_idx, start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='checkpoint folder name under ~/modelspace')
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--episodes-per-task', type=int, default=3); p.add_argument('--windows', type=int, default=3)
    p.add_argument('--variant', default='randomized_500'); p.add_argument('--embodiment', default='aloha-agilex')
    p.add_argument('--dataset-dir', type=Path, default=Path('/home/a1905721/dataspace/openwam-attn-aligned'))
    p.add_argument('--action-t', type=float, nargs='+', default=[0.25, 0.5, 0.75]); p.add_argument('--video-sigma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda')
    p.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B'))
    p.add_argument('--out-dir', type=Path, default=Path('artifacts/aad_episodes'))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path.home() / 'modelspace' / a.ckpt
    arch, cfg = avs.load_arch(ckpt, a.wan_dir, a.device)
    name = type(arch).__name__
    kind = 'self' if 'SelfAttn' in name else (('cross_joint' if 'Softmax' in name else 'cross') if 'CrossAttn' in name else None)
    if kind is None:
        raise SystemExit(f'{name}: not a self/cross-attention dual system')
    height = int(cfg.dataloader.height); top_h = int(round(height * 2 / 3))
    avs.install_grid_hook(arch, lambda gh: gh * top_h // height)
    L, H = (avs.install_self_attn_hooks if kind == 'self' else avs.install_cross_attn_hooks)(arch)
    vb, ab = arch.video_backbone, arch.action_backbone
    arch.init_training_schedulers(1000)
    vsched, asched = vb.scheduler, ab.scheduler
    vid_id = int((vsched.sigmas - a.video_sigma).abs().argmin())
    act_ids = [int((asched.sigmas - t).abs().argmin()) for t in a.action_t]
    print(f'{a.ckpt}: {kind}-attention ({name}), {L} layers, {H} heads; video sigma {float(vsched.sigmas[vid_id]):.3f}; '
          f'action sigmas {[round(float(asched.sigmas[i]), 3) for i in act_ids]}', flush=True)

    samples = []                                          # (task, ep_idx, episode file stem, start, dataset)
    for task in a.tasks:
        ds = avs.make_dataset(cfg, a.dataset_dir / task, ckpt, a.embodiment, a.variant)
        for ep in range(min(a.episodes_per_task, len(ds._episode_files))):
            max_start = max(0, ds._episode_lengths[ep] - ds._raw_window_len)
            for w in range(a.windows):
                start = int(round(max_start * w / max(a.windows - 1, 1)))
                samples.append((task, ep, Path(ds._episode_files[ep]).stem, start, ds))
    prompts = {(t, s, st): seeded_sample(ds, e, st, f'{t}/{s}/{st}')['prompt'] for t, e, s, st, ds in samples}
    avs.oom_retry(lambda: avs.offload_text_encoder(vb, sorted(set(prompts.values())), a.device), 'prompt encoding')
    avs.oom_retry(lambda: avs.place_on_gpu(arch, a.device), 'moving the model to the GPU')

    W, S = len(samples), len(act_ids)
    ent = np.zeros((W, S, L), np.float32); ent_v = np.zeros_like(ent); mass = np.zeros_like(ent); attn = None
    frames_path = a.out_dir / 'frames.npz'; frames = [] if not frames_path.exists() else None
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    t0 = time.time()
    for wi, (task, ep, stem, start, ds) in enumerate(samples):
        sample = seeded_sample(ds, ep, start, f'{task}/{stem}/{start}'); sample['prompt'] = prompts[(task, stem, start)]
        if frames is not None:
            frames.append(np.stack([np.asarray(im.convert('RGB')) for im in sample['video']]))

        def one_window():
            nonlocal attn
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = arch.prepare_inputs([sample])
                actions = inputs.pop('actions').to(arch.dtype); proprio = inputs.pop('proprio', None); inputs.pop('proprio_mask', None)
                for k in ('max_timestep_boundary', 'min_timestep_boundary', 'use_gradient_checkpointing',
                          'use_gradient_checkpointing_offload', 'action_is_pad', 'video_is_pad'):
                    inputs.pop(k, None)
                x0 = inputs['input_latents']; sig = vsched.sigmas[vid_id].to(x0)
                noise = torch.randn(x0.shape, generator=g, device=x0.device, dtype=x0.dtype)
                inputs['latents'] = (1 - sig) * x0 + sig * noise
                if inputs.get('first_frame_latents') is not None:
                    inputs['latents'][:, :, 0:1] = inputs['first_frame_latents']
                vt = vsched.timesteps[vid_id].to(x0).view(1)
                for si, aid in enumerate(act_ids):
                    asig = asched.sigmas[aid].to(actions); at = asched.timesteps[aid].to(actions).view(1)
                    anoise = torch.randn(actions.shape, generator=g, device=actions.device, dtype=actions.dtype)
                    noisy = asched.add_noise(actions, anoise, asig.view(1, 1, 1))
                    avs.STATE['ablate'] = False; avs.STATE['rec'] = avs.new_rec(L, H, 64, noisy.shape[1]); MAPS.clear()
                    arch(noisy, at, proprio=proprio, **inputs, timestep=vt)
                    if len(MAPS) != L:
                        raise RuntimeError(f'captured {len(MAPS)} of {L} layers')
                    if attn is None:
                        attn = np.zeros((W, S, L, MAPS[0][0].shape[0]), np.float16)
                    for l in range(L):
                        r = avs.STATE['rec'][l]
                        ent[wi, si, l] = r['entropy']; ent_v[wi, si, l] = r['entropy_video']
                        attn[wi, si, l] = MAPS[l][0]; mass[wi, si, l] = MAPS[l][1]

        avs.oom_retry(one_window, f'window {task}/{stem}/{start}')
        print(f'  {wi + 1}/{W} {task}/{stem}/{start}: H/ln360 by sigma '
              + ' '.join(f'{x:.3f}' for x in (ent[wi].mean(1) / np.log(360))) + f' ({(time.time() - t0) / (wi + 1):.1f} s/window)', flush=True)

    meta = np.array([(t, e, s, st) for t, e, s, st, _ in samples], dtype=object)
    prompt_list = np.array([prompts[(t, s, st)] for t, _, s, st, _ in samples], dtype=object)
    out = a.out_dir / f'{a.ckpt}.npz'
    np.savez(out, model=a.ckpt, kind=kind, arch=name, meta=meta, prompts=prompt_list, entropy=ent, entropy_video=ent_v, mass_video=mass, attn=attn,
             grid=np.array(avs.STATE['grid']), head_rows=avs.STATE['head_rows'], video_sigma=float(vsched.sigmas[vid_id]),
             action_sigmas=np.array([float(asched.sigmas[i]) for i in act_ids]))
    if frames is not None:
        np.savez_compressed(frames_path, meta=meta, frames=np.stack(frames))
        (a.out_dir / 'frames_meta.json').write_text(json.dumps([[str(x) for x in m] for m in meta]))
    h = ent.mean((0, 1)) / np.log(360)
    print(f'done {a.ckpt}: {W} windows -> {out}; pooled mu {h.mean():.4f} s2 {h.var(ddof=1):.5f} AAD {h.mean() - h.var(ddof=1):.3f}', flush=True)


if __name__ == '__main__':
    main()
