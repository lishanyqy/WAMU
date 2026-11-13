#!/usr/bin/env python
"""What does the action stream read from the video stream?  (wamu/action_read_subspace_plan.md, steps 0-2)

Extends scripts/attention_video_share.py (model loading, prompt caching, GPU placement, sample windows are reused from
it).  Joint self-attention (MoT) with attention_mask_mode=action_sees_video and no action FiLM: the video rows never
read action keys, so the video stream does not depend on anything the action stream does.  Per window and action
sigma the model is run ONCE as is (forward hooks capture every layer's video K/V, the action-row mask and the initial
action state); every readout is then computed by replaying only the action stream (ActionDiT pre/post-attention halves,
the joint softmax over [video keys ; action keys] written out explicitly), batched over intervention variants.  The
replay without intervention is checked against the model's own prediction on every forward (max relative error
logged).  Nothing in the model code or checkpoint is changed.

Read vector of action token i at layer l (before the action o-projection, concat over heads, d = H * Dh):
    m[i] = concat_h( sum_{j in video} alpha_h[i, j] V_h[j] )
Interventions edit m only (the action-key part of the attention output is untouched):
    keep-R(k)   m' = mu + P (m - mu)       remove-R(k)  m' = m - P (m - mu)       P = U_k U_k^T
    cut         action rows cannot attend any video key (softmax over action keys only), as attention_video_share.py

modes
  fit     stream mean / second moment of m per layer over the FIT windows (3 action sigmas pooled), eigendecompose,
          save artifacts/action_read/R/R_<model>_<set>.pt and results/step1_spectrum_<model>_<set>.csv
  causal  on the EVAL windows: NMSE for full, cut, keep/remove-R(k), random(k) controls (5 seeds), per-window rows
          -> artifacts/action_read/raw/causal_<model>_<set>.npz (step0 / step2 tables are built from these)
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import attention_video_share as avs  # noqa: E402

MODELS = {'M1': 'sft_from_scratch_robotwin_clean', 'M2': 'sft_robot_only_robotwin_clean',
          'M3': 'sft_ego_robot_cotrain_robotwin_clean_action_sees_video'}
FARM = Path.home() / 'dataspace/openwam-attn-aligned'
OUT = Path('artifacts/action_read')
CAP = {}


# ------------------------------------------------------------------ capture hooks (read-only)

def install_capture(arch):
    vb = arch.video_backbone
    drv = arch._mot_driver or arch.build_mot_driver()
    if drv.video_film is not None or drv.detach_video or drv.attention_mask_mode != 'action_sees_video':
        raise SystemExit(f'unsupported driver: film={drv.video_film is not None} detach_video={drv.detach_video} '
                         f'mask={drv.attention_mask_mode} (the replay assumes action_sees_video, no FiLM, no detach)')
    orig_step, orig_vpre = drv.step, vb.pre_attn_at_layer

    def step(layer_id, vstate, astate, **kw):
        if layer_id == 0:
            p = astate.payload
            CAP['payload0'] = dict(x_action=p.x_action.detach().clone(), t_mod=p.t_mod, t_embed=p.t_embed,
                                   action_freqs=p.action_freqs, context=p.context, context_mask=p.context_mask)
            CAP['mask'] = kw.get('attn_mask')
            CAP['kv'] = {}
        return orig_step(layer_id, vstate, astate, **kw)

    def vpre(layer_id, state):
        q, k, v, post = orig_vpre(layer_id, state)
        CAP['kv'][layer_id] = (k.detach(), v.detach())
        return q, k, v, post

    drv.step, vb.pre_attn_at_layer = step, vpre
    return vb.num_layers, vb.num_heads, vb.head_dim


# ------------------------------------------------------------------ action-stream replay

class Variant:
    """One readout: cut (no video keys), or an edit of m on a set of layers with per-layer bases U (d x k) and means."""

    def __init__(self, name, cut=False, kind=None, U=None, mu=None, layers=None):
        self.name, self.cut, self.kind, self.U, self.mu, self.layers = name, cut, kind, U, mu, layers


def replay(ab, P, kv, mask_a, H, Dh, variants, collect_m=False, collect_attn=None):
    """Run the action stream for len(variants) variants in one batch. Returns (pred (B, Sa, A), m per layer if asked).
    collect_attn: a list that receives, per layer, the attention mass on each video token (mean over heads and action
    tokens, variant 0) as a (Sv,) tensor."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState
    from openwam.model.architectures.base import ActionState
    B = len(variants)
    exp = lambda t: None if t is None else t.expand(B, *t.shape[1:])
    payload = ActionDiTState(x_action=P['x_action'].expand(B, *P['x_action'].shape[1:]).clone(), t_mod=P['t_mod'],
                             t_embed=P['t_embed'], action_freqs=P['action_freqs'], context=exp(P['context']),
                             context_mask=exp(P['context_mask']))
    astate = ActionState(action_latents=None, timestep=None, payload=payload)
    cut = torch.tensor([v.cut for v in variants], device=P['x_action'].device)
    ms = []
    for l in range(len(kv)):
        q_a, k_a, v_a, post = ab.pre_attn_at_layer(l, astate)
        k_v, v_v = kv[l]
        Sa, Sv = q_a.shape[1], k_v.shape[1]
        with torch.autocast('cuda', enabled=False):                        # attention math and edits in float32
            heads = lambda t: t.reshape(t.shape[0], t.shape[1], H, Dh).transpose(1, 2).float()
            q, kvh, vvh, kah, vah = heads(q_a), heads(k_v), heads(v_v), heads(k_a), heads(v_a)
            lv = q @ kvh.transpose(-1, -2) / math.sqrt(Dh)                     # (B, H, Sa, Sv)
            la = q @ kah.transpose(-1, -2) / math.sqrt(Dh)                     # (B, H, Sa, Sa)
            if mask_a is not None:
                lv = lv.masked_fill(~mask_a[:, :Sv], float('-inf')); la = la.masked_fill(~mask_a[:, Sv:], float('-inf'))
            lv = lv.masked_fill(cut.view(B, 1, 1, 1), float('-inf'))
            p = torch.cat([lv, la], -1).softmax(-1)
            m = (p[..., :Sv] @ vvh).transpose(1, 2).reshape(B, Sa, H * Dh)    # read vector (float32)
            a_part = (p[..., Sv:] @ vah).transpose(1, 2).reshape(B, Sa, H * Dh)
            if collect_m:
                ms.append(m)
            if collect_attn is not None:
                collect_attn.append(p[0, :, :, :Sv].mean((0, 1)))
            m2 = m.clone()
            for b, var in enumerate(variants):
                if var.kind is None or var.cut or (var.layers is not None and l not in var.layers):
                    continue
                if var.kind == 'zero':
                    m2[b] = 0.0
                    continue
                U, mu = var.U[l], var.mu[l]                                     # (d, k), (d,)
                proj = ((m[b] - mu) @ U) @ U.T
                m2[b] = mu + proj if var.kind == 'keep' else m[b] - proj
            attn = (m2 + a_part).to(q_a.dtype).contiguous()
        astate = ab.post_attn_at_layer(l, astate, attn, post)
    return ab.extract_prediction(astate), ms


# ------------------------------------------------------------------ windows

def windows(split, set_name, half):
    """(task, ds_index, start) for every episode of the split half: 3 windows (start / middle / end), as attention_video_share.py."""
    out = []
    var = split['sets'][set_name]['variant']
    for task, v in sorted(split['sets'][set_name]['tasks'].items()):
        files = sorted(str(p) for p in (FARM / task / f'aloha-agilex_{var}' / 'data').glob('episode*.hdf5'))
        nums = [int(Path(f).stem[len('episode'):]) for f in files]
        for ep in v[half]:
            out.append((task, nums.index(ep), ep))
    return out, var


def window_starts(ep_len, scheme, win=33):
    """causal: start / middle / end (as attention_video_share.py); gt: 0, 33, 66, 99 (all inside the 132 replayed GT
    frames, used for step 3), clipped to full windows."""
    ms = max(0, ep_len - win)
    if scheme == 'causal':
        return [int(round(ms * w / 2)) for w in range(3)]
    return sorted({s for s in (0, 33, 66, 99) if s <= ms}) or [0]


def seeded_sample(ds, ep_idx, start, key):
    """Build the clip with the instruction draw seeded by (task, episode, start): identical prompt for every model/run."""
    random.seed(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    return ds._build_sample(ep_idx, start)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(MODELS), required=True); p.add_argument('--set', choices=['ID', 'OOD'], required=True)
    p.add_argument('--mode', choices=['fit', 'causal', 'dump'], required=True)
    p.add_argument('--dump-layers', type=int, nargs='+', default=[4, 9, 14, 19, 24, 29])
    p.add_argument('--dump-k', type=int, default=256, help='dump: number of R directions for the m coordinates')
    p.add_argument('--dump-root', type=Path, default=Path.home() / 'dataspace/action_read_features')
    p.add_argument('--r-from', choices=['ID', 'OOD'], default=None, help='causal: which set the bases R come from (default: same set)')
    p.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument('--random-seeds', type=int, default=5)
    p.add_argument('--layer-windows', type=int, nargs='*', default=[], help='causal: also keep/remove R(k) on 5-layer windows for these k')
    p.add_argument('--mean-only', action='store_true', help="causal: only full, cut and mean-ablate (m' = mu on every layer; "
                   'same windows/noise as the main run, so its saved keep/remove NMSE can be re-referenced to the mean ablation)')
    p.add_argument('--limit-windows', type=int, default=0); p.add_argument('--chunk', type=int, default=24)
    p.add_argument('--action-t', type=float, nargs='+', default=[0.25, 0.5, 0.75]); p.add_argument('--video-sigma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--tag', default='')
    p.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B')); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    split = json.loads((OUT / 'split.json').read_text())
    half = {'fit': 'fit', 'causal': 'eval', 'dump': 'both'}[a.mode]
    if half == 'both':
        w1, variant = windows(split, a.set, 'fit'); w2, _ = windows(split, a.set, 'eval')
        wins = sorted(w1 + w2)
    else:
        wins, variant = windows(split, a.set, half)
    ckpt = Path.home() / 'modelspace' / MODELS[a.model]
    arch, cfg = avs.load_arch(ckpt, a.wan_dir, a.device)
    height = int(cfg.dataloader.height); top_h = int(round(height * 2 / 3))
    avs.install_grid_hook(arch, lambda gh: gh * top_h // height)
    L, H, Dh = install_capture(arch)
    vb, ab = arch.video_backbone, arch.action_backbone
    arch.init_training_schedulers(1000)
    vsched, asched = vb.scheduler, ab.scheduler
    vid_id = int((vsched.sigmas - a.video_sigma).abs().argmin()); act_ids = [int((asched.sigmas - t).abs().argmin()) for t in a.action_t]
    sig_a = [float(asched.sigmas[i]) for i in act_ids]
    print(f'{a.model} {MODELS[a.model]} {a.set}/{half} {a.mode}: {L} layers, {H} heads x {Dh}; video sigma {float(vsched.sigmas[vid_id]):.3f}; action sigmas {sig_a}', flush=True)

    # samples: 3 windows per episode, prompts seeded, built lazily
    dsets, samples = {}, []
    for task, ep_idx, ep in wins:
        if task not in dsets:
            dsets[task] = avs.make_dataset(cfg, FARM / task, ckpt, 'aloha-agilex', variant)
        ds = dsets[task]
        ep_len = ds._episode_lengths[ep_idx]; max_start = max(0, ep_len - ds._raw_window_len)
        for start in window_starts(ep_len, 'gt' if a.mode == 'dump' else 'causal'):
            prompt = seeded_sample(ds, ep_idx, start, f'{task}/{ep}/{start}')['prompt']
            samples.append((task, ep_idx, ep, start, prompt))
    if a.limit_windows:
        samples = samples[:a.limit_windows]
    avs.oom_retry(lambda: avs.offload_text_encoder(vb, sorted({s[4] for s in samples}), a.device), 'prompt encoding')
    avs.oom_retry(lambda: avs.place_on_gpu(arch, a.device), 'moving the model to the GPU')
    dev = torch.device(a.device)
    d = H * Dh

    # bases for causal mode
    variants_static = []
    if a.mode == 'causal':
        rset = a.r_from or a.set
        R = torch.load(OUT / 'R' / f'R_{a.model}_{rset}.pt', map_location='cpu')
        mu = [R['mu'][l].float().to(dev) for l in range(L)]
        evec = [R['evec'][l].float().to(dev) for l in range(L)]                 # (d, d) columns sorted by eigenvalue desc
        K99 = [int(x) for x in R['k99']]
        variants_static.append(Variant('full')); variants_static.append(Variant('cut', cut=True))
        if a.mean_only:
            # mean-ablate = keep-R with k = 0: m' = mu + 0 on every layer; attention weights and the video stream untouched
            variants_static.append(Variant('mean', kind='keep', U=[torch.zeros(d, 0, device=dev)] * L, mu=mu))
            a.ks, a.random_seeds, a.layer_windows = [], 0, []
        for k in a.ks:
            Uk = [evec[l][:, :k] for l in range(L)]
            variants_static += [Variant(f'keepR_{k}', kind='keep', U=Uk, mu=mu), Variant(f'removeR_{k}', kind='remove', U=Uk, mu=mu)]
        for s in range(a.random_seeds):
            g = torch.Generator().manual_seed(1000 + s)
            # Gaussian draws on the CPU generator (seeded), QR on the GPU (the CPU QR of up to 2300 x 2300 matrices is very slow on
            # a loaded login node).  M1 ID/OOD causal runs used the CPU QR: same seeds, different floating-point path.
            Qs = [torch.linalg.qr(torch.randn(K99[l], K99[l], generator=g, dtype=torch.float64).to(dev))[0].float() for l in range(L)]
            for k in a.ks:
                Uk = [evec[l][:, :K99[l]] @ Qs[l][:, :min(k, K99[l])] for l in range(L)]   # random k-dim subspace of the top-K99 PC span
                variants_static += [Variant(f'keepRand{s}_{k}', kind='keep', U=Uk, mu=mu), Variant(f'removeRand{s}_{k}', kind='remove', U=Uk, mu=mu)]
        for k in a.layer_windows:
            Uk = [evec[l][:, :k] for l in range(L)]
            for w0 in range(0, L, 5):
                lw = set(range(w0, min(L, w0 + 5)))
                variants_static += [Variant(f'win{w0}_keepR_{k}', kind='keep', U=Uk, mu=mu, layers=lw),
                                    Variant(f'win{w0}_removeR_{k}', kind='remove', U=Uk, mu=mu, layers=lw)]
            for w0 in range(0, L, 5):
                if f'win{w0}_zero' not in {v.name for v in variants_static}:
                    variants_static.append(Variant(f'win{w0}_zero', kind='zero', layers=set(range(w0, min(L, w0 + 5)))))   # m' = 0 on those layers
        print(f'{len(variants_static)} variants per (window, sigma); bases from {rset}; K99 per layer {K99}', flush=True)

    if a.mode == 'fit':
        s1 = torch.zeros(L, d, dtype=torch.float64, device=dev); s2 = torch.zeros(L, d, d, dtype=torch.float64, device=dev); n = 0
    n = 0
    if a.mode == 'dump':
        R = torch.load(OUT / 'R' / f'R_{a.model}_{a.set}.pt', map_location='cpu')
        Ud = [R['evec'][l][:, :a.dump_k].float().to(dev) for l in range(L)]; mud = [R['mu'][l].float().to(dev) for l in range(L)]
        droot = a.dump_root / f'{a.model}_{a.set}'; droot.mkdir(parents=True, exist_ok=True)
        DM = {}
        (droot / 'keys.json').write_text(json.dumps([[s[0], s[2], s[3]] for s in samples]))

    rows, names = [], [v.name for v in variants_static]
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    t0 = time.time(); max_rel = 0.0
    for wi, (task, ep_idx, ep, start, prompt) in enumerate(samples):
        sample = seeded_sample(dsets[task], ep_idx, start, f'{task}/{ep}/{start}'); sample['prompt'] = prompt

        def one_window():
            nonlocal n, max_rel
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
                am = sample.get('action_mask')
                for si, aid in enumerate(act_ids):
                    asig = asched.sigmas[aid].to(actions); at = asched.timesteps[aid].to(actions).view(1)
                    anoise = torch.randn(actions.shape, generator=g, device=actions.device, dtype=actions.dtype)
                    noisy = asched.add_noise(actions, anoise, asig.view(1, 1, 1)); target = asched.training_target(actions, anoise)
                    _, pred_model = arch(noisy, at, proprio=proprio, **inputs, timestep=vt)          # captures K/V, mask, action state
                    P, kv = CAP['payload0'], [CAP['kv'][l] for l in range(L)]
                    Sv = kv[0][1].shape[1]
                    mask_a = CAP['mask'][Sv:, :] if CAP['mask'] is not None and CAP['mask'].dim() == 2 else (None if CAP['mask'] is None else CAP['mask'][0, 0, Sv:, :])
                    mk = (torch.as_tensor(np.asarray(am)).to(pred_model.device).float().view(1, *np.asarray(am).shape) if am is not None else torch.ones_like(pred_model).float())
                    if mk.ndim == 2:
                        mk = mk.unsqueeze(-1).expand_as(pred_model)
                    tg = target.float() * mk; tg2 = float((tg ** 2).sum().clamp_min(1e-12))
                    if a.mode == 'fit':
                        pred, ms = replay(ab, P, kv, mask_a, H, Dh, [Variant('full')], collect_m=True)
                        for l, m in enumerate(ms):
                            x = m[0].double()                                                # (Sa, d)
                            s1[l] += x.sum(0); s2[l] += x.T @ x
                        n += ms[0].shape[1]
                    elif a.mode == 'dump':
                        att = []
                        pred, ms = replay(ab, P, kv, mask_a, H, Dh, [Variant('full')], collect_m=True, collect_attn=att)
                        Sa_ = ms[0].shape[1]
                        if not DM:
                            from numpy.lib.format import open_memmap
                            N = len(samples)
                            DM['V'] = open_memmap(droot / 'V.npy', 'w+', np.float16, (N, len(a.dump_layers), Sv, d))
                            DM['attn'] = open_memmap(droot / 'attn.npy', 'w+', np.float16, (N, len(act_ids), L, Sv))
                            DM['mcoord'] = open_memmap(droot / 'mcoord.npy', 'w+', np.float16, (N, len(act_ids), L, Sa_, a.dump_k))
                        if si == 0:
                            DM['V'][wi] = torch.stack([kv[l][1][0] for l in a.dump_layers]).float().cpu().numpy().astype(np.float16)
                        DM['attn'][wi, si] = torch.stack(att).float().cpu().numpy().astype(np.float16)
                        DM['mcoord'][wi, si] = torch.stack([(ms[l][0] - mud[l]) @ Ud[l] for l in range(L)]).float().cpu().numpy().astype(np.float16)
                    else:
                        preds = []
                        for c0 in range(0, len(variants_static), a.chunk):
                            pr, _ = replay(ab, P, kv, mask_a, H, Dh, variants_static[c0:c0 + a.chunk]); preds.append(pr)
                        pred = torch.cat(preds)
                        e = ((pred.float() * mk - tg) ** 2).sum((1, 2)) / tg2
                        rows.append((task, ep, start, sig_a[si], *e.cpu().tolist()))
                    rel = float((pred[0].float() - pred_model[0].float()).norm() / pred_model[0].float().norm().clamp_min(1e-6))
                    max_rel = max(max_rel, rel)

        avs.oom_retry(one_window, f'window {task}/{ep}/{start}')
        if (wi + 1) % 25 == 0 or wi + 1 == len(samples):
            print(f'  {wi + 1}/{len(samples)} windows ({(time.time() - t0) / (wi + 1):.2f} s/window); replay vs model max rel diff {max_rel:.2e}', flush=True)

    meta = dict(model=a.model, checkpoint=MODELS[a.model], set=a.set, half=half, variant=variant, seed=a.seed, video_sigma=float(vsched.sigmas[vid_id]),
                action_sigmas=sig_a, n_windows=len(samples), replay_max_rel_diff=max_rel, split_seed=split['seed'])
    if a.mode == 'fit':
        mu = s1 / n
        evals, evecs, rows_out, k50, k90, k99 = [], [], [], [], [], []
        for l in range(L):
            C = s2[l] / n - torch.outer(mu[l], mu[l])
            w, V = torch.linalg.eigh(C)
            w, V = w.flip(0).clamp_min(0), V.flip(1)
            cs = (w.cumsum(0) / w.sum()).cpu().numpy()
            kk = [int(np.searchsorted(cs, t) + 1) for t in (0.5, 0.9, 0.99)]
            eff = float(w.sum() ** 2 / (w ** 2).sum())
            evals.append(w.cpu()); evecs.append(V.float().cpu()); k50.append(kk[0]); k90.append(kk[1]); k99.append(kk[2])
            rows_out.append(dict(model=a.model, set=a.set, layer=l, n_vectors=n, total_var=float(w.sum()), eff_dim=eff, k50=kk[0], k90=kk[1], k99=kk[2],
                                 **{f'lambda_{i}': float(w[i]) for i in range(16)}, seed=a.seed))
        (OUT / 'R').mkdir(parents=True, exist_ok=True)
        torch.save(dict(mu=mu.float().cpu(), evec=evecs, eval=evals, k50=k50, k90=k90, k99=k99, meta=meta), OUT / 'R' / f'R_{a.model}_{a.set}.pt')
        with (OUT / 'results' / f'step1_spectrum_{a.model}_{a.set}.csv').open('w', newline='') as f:
            w_ = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w_.writeheader(); w_.writerows(rows_out)
        print(f'done fit: {n} read vectors per layer; eff dim by layer ' + ' '.join(f'{r["eff_dim"]:.1f}' for r in rows_out), flush=True)
        print('  k90 by layer ' + ' '.join(str(x) for x in k90) + ' | k99 ' + ' '.join(str(x) for x in k99), flush=True)
    elif a.mode == 'dump':
        for v in DM.values():
            v.flush()
        (droot / 'meta.json').write_text(json.dumps(dict(meta, dump_layers=a.dump_layers, dump_k=a.dump_k, R=f'R_{a.model}_{a.set}.pt',
                                                         arrays={k: list(v.shape) for k, v in DM.items()},
                                                         note='V: video value vectors (concat heads) at dump_layers; attn: action->video token mass '
                                                              '(mean over heads, action tokens) per action sigma and layer; mcoord: read vector m '
                                                              'minus mu, projected on the top dump_k R directions, per sigma, layer, action token'), indent=1))
        print(f'done dump: {len(samples)} windows -> {droot}; replay vs model max rel diff {max_rel:.2e}', flush=True)
    else:
        (OUT / 'raw').mkdir(parents=True, exist_ok=True)
        tag = f'_{a.tag}' if a.tag else ''
        rset = a.r_from or a.set
        np.savez(OUT / 'raw' / f'causal_{a.model}_{a.set}_R{rset}{tag}.npz', meta=json.dumps(meta), variants=np.array(names),
                 key=np.array([r[:3] for r in rows], dtype=object), action_sigma=np.array([r[3] for r in rows]),
                 nmse=np.array([r[4:] for r in rows], np.float64))
        print(f'done causal: {len(rows)} rows x {len(names)} variants; replay vs model max rel diff {max_rel:.2e}', flush=True)


if __name__ == '__main__':
    main()
