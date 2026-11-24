#!/usr/bin/env python
"""Step 3c input (attention enrichment) for the bridge cross-attention WAMs, on exactly the windows of
read_subspace.py --mode dump: same split episodes, gt window starts (0/33/66/99), seeded prompts, one torch generator
(seed 0) drawing the video noise and then one action noise per sigma in the same order.

Writes <dump-root>/<name>_<set>/{keys.json, meta.json, attn.npy}; attn (N, S, L, 360) = attention mass of the action
queries on each video token in the bridge cross-attention (mean over heads and action tokens), the same layout as the
self-attention dumps, so scripts/action_read/step3_bcd.py --part c --model <name> --set <set> runs unchanged.
For the joint-softmax variant only the video part of the softmax is stored (step 3c renormalises over video tokens).
The first window checks the recomputed attention against the module output (rebuilding o(p @ v)).

usage: python scripts/action_read/dump_attn_cross.py --name CA --ckpt cross_attn_clean --set OOD
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/action_read')
import attention_video_share as avs  # noqa: E402
from read_subspace import FARM, OUT, windows, window_starts, seeded_sample  # noqa: E402

CAP = {'att': {}, 'check': [], 'do_check': False}


def install(arch):
    ab = arch.action_backbone
    for li, block in enumerate(ab.blocks):
        mod = block.cross_attn
        direct = bool(getattr(block, 'direct_video_kv', False)); joint = bool(getattr(mod, 'joint_action_kv', False))

        def hook(m, args, kwargs, out, li=li, direct=direct, joint=joint):
            xa, xv = args[0], args[1]
            H = m.num_heads; Sa = xa.shape[1]
            q = m.norm_q(m.q(xa))
            k = xv if direct else m.norm_k(m.k(xv))
            if joint:
                k = torch.cat([m.norm_k_a(m.k_a(xa)), k], dim=1)
            q = rearrange(q, 'b s (n d) -> b n s d', n=H).float(); k = rearrange(k, 'b s (n d) -> b n s d', n=H).float()
            p = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).softmax(-1)          # (B, H, Sa, [Sa +] Sv)
            pv = p[..., Sa:] if joint else p
            CAP['att'][li] = pv.mean((0, 1, 2))                                      # (Sv,)
            if CAP['do_check']:
                v = args[2] if direct else m.v(xv)
                if joint:
                    v = torch.cat([m.v_a(xa), v], dim=1)
                v = rearrange(v, 'b s (n d) -> b n s d', n=H).float()
                rec = m.o(rearrange((p @ v).to(out.dtype), 'b n s d -> b s (n d)'))
                CAP['check'].append(float((rec.float() - out.float()).norm() / out.float().norm().clamp_min(1e-6)))
        mod.register_forward_hook(hook, with_kwargs=True)
    return len(ab.blocks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--set', choices=['ID', 'OOD'], default='OOD')
    p.add_argument('--action-t', type=float, nargs='+', default=[0.25, 0.5, 0.75]); p.add_argument('--video-sigma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--limit-windows', type=int, default=0)
    p.add_argument('--dump-root', type=Path, default=Path.home() / 'dataspace/action_read_features')
    p.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B')); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    split = json.loads((OUT / 'split.json').read_text())
    w1, variant = windows(split, a.set, 'fit'); w2, _ = windows(split, a.set, 'eval'); wins = sorted(w1 + w2)
    ckpt = Path.home() / 'modelspace' / a.ckpt
    arch, cfg = avs.load_arch(ckpt, a.wan_dir, a.device)
    if 'CrossAttn' not in type(arch).__name__:
        raise SystemExit(f'{type(arch).__name__}: not a cross-attention WAM')
    L = install(arch)
    vb, ab = arch.video_backbone, arch.action_backbone
    arch.init_training_schedulers(1000)
    vsched, asched = vb.scheduler, ab.scheduler
    vid_id = int((vsched.sigmas - a.video_sigma).abs().argmin()); act_ids = [int((asched.sigmas - t).abs().argmin()) for t in a.action_t]
    sig_a = [float(asched.sigmas[i]) for i in act_ids]
    print(f'{a.name} {a.ckpt} {type(arch).__name__} {a.set}: {L} layers; video sigma {float(vsched.sigmas[vid_id]):.3f}; action sigmas {sig_a}', flush=True)

    dsets, samples = {}, []
    for task, ep_idx, ep in wins:
        if task not in dsets:
            dsets[task] = avs.make_dataset(cfg, FARM / task, ckpt, 'aloha-agilex', variant)
        ds = dsets[task]
        for start in window_starts(ds._episode_lengths[ep_idx], 'gt'):
            samples.append((task, ep_idx, ep, start, seeded_sample(ds, ep_idx, start, f'{task}/{ep}/{start}')['prompt']))
    if a.limit_windows:
        samples = samples[:a.limit_windows]
    avs.oom_retry(lambda: avs.offload_text_encoder(vb, sorted({s[4] for s in samples}), a.device), 'prompt encoding')
    avs.oom_retry(lambda: avs.place_on_gpu(arch, a.device), 'moving the model to the GPU')

    droot = a.dump_root / f'{a.name}_{a.set}'; droot.mkdir(parents=True, exist_ok=True)
    (droot / 'keys.json').write_text(json.dumps([[s[0], s[2], s[3]] for s in samples]))
    att_mm = None
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    t0 = time.time()
    for wi, (task, ep_idx, ep, start, prompt) in enumerate(samples):
        sample = seeded_sample(dsets[task], ep_idx, start, f'{task}/{ep}/{start}'); sample['prompt'] = prompt
        CAP['do_check'] = wi == 0

        def one_window():
            nonlocal att_mm
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = arch.prepare_inputs([sample])
                actions = inputs.pop('actions').to(arch.dtype); proprio = inputs.pop('proprio', None); inputs.pop('proprio_mask', None)
                for k in ('max_timestep_boundary', 'min_timestep_boundary', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'action_is_pad', 'video_is_pad'):
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
                    CAP['att'].clear()
                    arch(noisy, at, proprio=proprio, **inputs, timestep=vt)
                    att = torch.stack([CAP['att'][l] for l in range(L)]).float().cpu().numpy()   # (L, Sv)
                    if att_mm is None:
                        from numpy.lib.format import open_memmap
                        att_mm = open_memmap(droot / 'attn.npy', 'w+', np.float16, (len(samples), len(act_ids), L, att.shape[1]))
                    att_mm[wi, si] = att.astype(np.float16)

        avs.oom_retry(one_window, f'window {task}/{ep}/{start}')
        if wi == 0:
            print(f'  check (first window, all layers and sigmas): max rel err of o(p @ v) vs module output {max(CAP["check"]):.2e}', flush=True)
            if max(CAP['check']) > 0.03:
                raise SystemExit('recomputed attention does not reproduce the module output')
        if (wi + 1) % 50 == 0 or wi + 1 == len(samples):
            print(f'  {wi + 1}/{len(samples)} windows ({(time.time() - t0) / (wi + 1):.2f} s/window)', flush=True)
    att_mm.flush()
    (droot / 'meta.json').write_text(json.dumps(dict(
        model=a.name, checkpoint=a.ckpt, architecture=type(arch).__name__, set=a.set, half='both', variant=variant, seed=a.seed,
        video_sigma=float(vsched.sigmas[vid_id]), action_sigmas=sig_a, n_windows=len(samples), split_seed=split['seed'],
        check_max_rel_err=max(CAP['check']), arrays={'attn': list(att_mm.shape)},
        note='attn: action->video token mass in the bridge cross-attention (mean over heads, action tokens) per action sigma and layer; '
             'joint softmax: video part only'), indent=1))
    print(f'done: {len(samples)} windows -> {droot}', flush=True)


if __name__ == '__main__':
    main()
