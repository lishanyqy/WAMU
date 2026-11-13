#!/usr/bin/env python
"""Step 5 of wamu/action_read_subspace_plan.md: model stitching (video stream from M_i, action stream from M_j).

For every window and action sigma, M_i runs as is (its video K/V per layer are captured by read_subspace.py's hooks;
with action_sees_video they do not depend on the action stream).  M_j's action stream is built from the same inputs
(ab_j.prepare_state with M_j's own proprio token, exactly as DualSystemSelfAttnArchitecture.forward) and replayed on
M_i's video K/V through per-layer alignment matrices  k' = k A_K[l],  v' = v A_V[l]  (d x d, identity at init).
Only the A's are trained; everything else is frozen.  Training budget (by instruction): 0 steps (zero-shot, identity)
and 1 step (one-shot: a single AdamW step on one batch of 16 ID-fit windows x 3 sigmas, lr chosen from {1e-5, 1e-4,
1e-3} on 32 other ID-fit windows).  The diagonal i == j gets the same treatment as the control.  Readout: NMSE on the
EVAL windows of ID and OOD, per action sigma, plus M_j's own NMSE_cut (no video) for reference.

usage: python scripts/action_read/stitch.py --video-model M1   (all three action models; one job per video model)
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '.'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/action_read')
import attention_video_share as avs  # noqa: E402
import read_subspace as rs  # noqa: E402

OUT = Path('artifacts/action_read')


def action_state(arch_j, inputs, proprio, noisy, at):
    pi = arch_j._append_proprio_context_token(dict(inputs), proprio)
    ctx, cm = pi.get('context'), pi.get('context_mask')
    if ctx is not None and cm is None and pi.get('seq_lens') is not None:
        sl = pi['seq_lens'].to(ctx.device); cm = torch.arange(ctx.shape[1], device=ctx.device)[None] < sl[:, None]
    st = arch_j.action_backbone.prepare_state(noisy, at, context=ctx, context_mask=cm)
    p = st.payload
    return dict(x_action=p.x_action, t_mod=p.t_mod, t_embed=p.t_embed, action_freqs=p.action_freqs, context=p.context, context_mask=p.context_mask)


def replay_aligned(ab, P, kv, mask_a, H, Dh, AK=None, AV=None, cut=False):
    """Differentiable action-stream replay (batch 1) on aligned video K/V; same math as read_subspace.replay."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState
    from openwam.model.architectures.base import ActionState
    payload = ActionDiTState(x_action=P['x_action'].clone(), t_mod=P['t_mod'], t_embed=P['t_embed'], action_freqs=P['action_freqs'],
                             context=P['context'], context_mask=P['context_mask'])
    astate = ActionState(action_latents=None, timestep=None, payload=payload)
    for l in range(len(kv)):
        q_a, k_a, v_a, post = ab.pre_attn_at_layer(l, astate)
        k_v, v_v = kv[l]
        with torch.autocast('cuda', enabled=False):
            k_v, v_v = k_v.float(), v_v.float()
            if AK is not None:
                k_v = k_v @ AK[l]; v_v = v_v @ AV[l]
            heads = lambda t: t.reshape(t.shape[0], t.shape[1], H, Dh).transpose(1, 2).float()
            q, kvh, vvh, kah, vah = heads(q_a), heads(k_v), heads(v_v), heads(k_a), heads(v_a)
            Sv = kvh.shape[2]
            lv = q @ kvh.transpose(-1, -2) / math.sqrt(Dh); la = q @ kah.transpose(-1, -2) / math.sqrt(Dh)
            if mask_a is not None:
                lv = lv.masked_fill(~mask_a[:, :Sv], float('-inf')); la = la.masked_fill(~mask_a[:, Sv:], float('-inf'))
            if cut:
                lv = torch.full_like(lv, float('-inf'))
            p = torch.cat([lv, la], -1).softmax(-1)
            o = torch.cat([p[..., :Sv], p[..., Sv:]], -1) @ torch.cat([vvh.expand(q.shape[0], -1, -1, -1), vah], 2)
            attn = o.transpose(1, 2).reshape(q.shape[0], q.shape[2], H * Dh).to(q_a.dtype)
        astate = ab.post_attn_at_layer(l, astate, attn, post)
    return ab.extract_prediction(astate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-model', choices=list(rs.MODELS), required=True)
    p.add_argument('--action-models', nargs='+', default=list(rs.MODELS))
    p.add_argument('--sets', nargs='+', default=['ID', 'OOD']); p.add_argument('--limit-windows', type=int, default=0)
    p.add_argument('--lrs', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3]); p.add_argument('--train-batch', type=int, default=16)
    p.add_argument('--val-windows', type=int, default=32); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--action-t', type=float, nargs='+', default=[0.25, 0.5, 0.75]); p.add_argument('--video-sigma', type=float, default=0.5)
    p.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B')); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    split = json.loads((OUT / 'split.json').read_text())
    archs, cfg = {}, None
    import gc
    for mk in [a.video_model] + [m for m in a.action_models if m != a.video_model]:
        ck = Path.home() / 'modelspace' / rs.MODELS[mk]
        arch, c = avs.load_arch(ck, a.wan_dir, a.device)
        if mk != a.video_model:
            # only the action stream (ActionDiT + proprio encoder) of the other models is used: drop their video stream
            # (DiT, text encoder, VAE) to keep host memory below the QOS limit
            arch.init_training_schedulers(1000)
            arch.video_backbone = None; gc.collect()
        archs[mk] = arch; cfg = cfg or c
    ai = archs[a.video_model]
    height = int(cfg.dataloader.height); top_h = int(round(height * 2 / 3))
    avs.install_grid_hook(ai, lambda gh: gh * top_h // height)
    L, H, Dh = rs.install_capture(ai)
    vb = ai.video_backbone; ai.init_training_schedulers(1000)       # the other models' schedulers were set before their video stream was dropped
    vsched, asched = vb.scheduler, ai.action_backbone.scheduler
    vid_id = int((vsched.sigmas - a.video_sigma).abs().argmin()); act_ids = [int((asched.sigmas - t).abs().argmin()) for t in a.action_t]
    sig_a = [float(asched.sigmas[i]) for i in act_ids]

    def build(set_name, half):
        wins, var = rs.windows(split, set_name, half)
        out = []
        for task, ep_idx, ep in wins:
            ds = dsets.setdefault((task, var), avs.make_dataset(cfg, rs.FARM / task, Path.home() / 'modelspace' / rs.MODELS[a.video_model], 'aloha-agilex', var))
            ms = max(0, ds._episode_lengths[ep_idx] - ds._raw_window_len)
            for w in range(3):
                s = int(round(ms * w / 2)); key = f'{task}/{ep}/{s}'
                out.append((task, ep_idx, ep, s, rs.seeded_sample(ds, ep_idx, s, key)['prompt'], ds, key))
        return out
    dsets = {}
    evals = {s: build(s, 'eval') for s in a.sets}
    fitw = build('ID', 'fit'); rng = np.random.default_rng(a.seed); perm = rng.permutation(len(fitw))
    train_w = [fitw[i] for i in perm[:a.train_batch]]; val_w = [fitw[i] for i in perm[a.train_batch:a.train_batch + a.val_windows]]
    if a.limit_windows:
        evals = {s: v[:a.limit_windows] for s, v in evals.items()}
    prompts = sorted({w[4] for v in list(evals.values()) + [train_w, val_w] for w in v})
    avs.oom_retry(lambda: avs.offload_text_encoder(vb, prompts, a.device), 'prompt encoding')
    avs.oom_retry(lambda: avs.place_on_gpu(ai, a.device), 'video model to GPU')
    for mk, ar in archs.items():                       # other models: only what the action stream needs
        if ar is not ai:
            ar.action_backbone.set_dtype_device(torch.bfloat16, dev) if hasattr(ar.action_backbone, 'set_dtype_device') else ar.action_backbone.to(dev, torch.bfloat16)
            if getattr(ar, 'proprio_encoder', None) is not None:
                ar.proprio_encoder.to(dev, torch.bfloat16)
            ar._device, ar._dtype = dev, torch.bfloat16
    for ar in archs.values():
        for prm in ar.parameters():
            prm.requires_grad_(False)

    g = torch.Generator(device=a.device).manual_seed(a.seed)

    def materialize(w):
        """Run M_i on window w for every sigma: returns list of (sigma, kv, mask_a, inputs, proprio, noisy, at, target, mk)."""
        task, ep_idx, ep, s, prompt, ds, key = w
        sample = rs.seeded_sample(ds, ep_idx, s, key); sample['prompt'] = prompt
        # noise seeded per window: identical for every action model j (all 9 combinations compare like for like) and
        # reproduced exactly when the window is retried after a CUDA OOM on the shared GPU
        g.manual_seed(a.seed * 1_000_003 + int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
        res = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = ai.prepare_inputs([sample])
            actions = inputs.pop('actions').to(ai.dtype); proprio = inputs.pop('proprio', None); inputs.pop('proprio_mask', None)
            for k in ('max_timestep_boundary', 'min_timestep_boundary', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'action_is_pad', 'video_is_pad'):
                inputs.pop(k, None)
            x0 = inputs['input_latents']; sig = vsched.sigmas[vid_id].to(x0)
            inputs['latents'] = (1 - sig) * x0 + sig * torch.randn(x0.shape, generator=g, device=x0.device, dtype=x0.dtype)
            if inputs.get('first_frame_latents') is not None:
                inputs['latents'][:, :, 0:1] = inputs['first_frame_latents']
            inputs['timestep'] = vsched.timesteps[vid_id].to(x0).view(1)
            am = sample.get('action_mask')
            for si, aid in enumerate(act_ids):
                asig = asched.sigmas[aid].to(actions); at = asched.timesteps[aid].to(actions).view(1)
                an = torch.randn(actions.shape, generator=g, device=actions.device, dtype=actions.dtype)
                noisy = asched.add_noise(actions, an, asig.view(1, 1, 1)); target = asched.training_target(actions, an)
                _, pred_i = ai(noisy, at, proprio=proprio, **inputs)
                kv = [tuple(t.clone() for t in rs.CAP['kv'][l]) for l in range(L)]
                Sv = kv[0][0].shape[1]; M = rs.CAP['mask']
                mask_a = None if M is None else (M[Sv:, :] if M.dim() == 2 else M[0, 0, Sv:, :])
                mk = (torch.as_tensor(np.asarray(am)).to(dev).float().view(1, *np.asarray(am).shape) if am is not None else torch.ones_like(pred_i).float())
                if mk.ndim == 2:
                    mk = mk.unsqueeze(-1).expand_as(pred_i)
                res.append(dict(si=si, kv=kv, mask_a=mask_a, inputs=inputs, proprio=proprio, noisy=noisy, at=at, target=target.float(), mk=mk, pred_i=pred_i.float()))
        return res

    def nmse(pred, r):
        tg = r['target'] * r['mk']
        return ((pred.float() * r['mk'] - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-12)

    def payload(aj, r):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return action_state(aj, r['inputs'], r['proprio'], r['noisy'], r['at'])

    rows = []
    for mj in a.action_models:
        aj = archs[mj]; abj = aj.action_backbone
        # ---- one-shot training of the alignment matrices
        t0 = time.time()
        mat = lambda w: avs.oom_retry(lambda: materialize(w), f'materialize {w[6]}')
        tr = [r for w in train_w for r in mat(w)]; va = [r for w in val_w for r in mat(w)]
        Ps_tr = [payload(aj, r) for r in tr]; Ps_va = [payload(aj, r) for r in va]
        if mj == a.video_model:   # diagonal sanity: rebuilt action state + replay == the model's own prediction
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pr = replay_aligned(abj, Ps_tr[0], tr[0]['kv'], tr[0]['mask_a'], H, Dh)
            rel = float((pr.float() - tr[0]['pred_i']).norm() / tr[0]['pred_i'].norm())
            print(f'diagonal check {mj}: replay vs model rel diff {rel:.2e}', flush=True)
        d = H * Dh
        def val_loss(AK, AV):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                return float(np.mean([float(nmse(replay_aligned(abj, P, r['kv'], r['mask_a'], H, Dh, AK, AV), r)) for P, r in zip(Ps_va, va)]))
        # One AdamW step from A = I with zero weight decay is, after bias correction, A - lr * g / (|g| + eps) = A - lr * sign(g)
        # (|g| >> eps = 1e-8).  The gradient at A = I is the same for every lr, so it is computed once and the update applied
        # per lr; no optimizer state is kept (saves ~4.5 GB on the shared GPU).
        def grad_sign():
            AK = torch.nn.Parameter(torch.eye(d, device=dev).repeat(L, 1, 1)); AV = torch.nn.Parameter(torch.eye(d, device=dev).repeat(L, 1, 1))
            for P, r in zip(Ps_tr, tr):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = nmse(replay_aligned(abj, P, r['kv'], r['mask_a'], H, Dh, AK, AV), r) / len(tr)
                loss.backward()
            return AK.grad.sign_(), AV.grad.sign_()
        sK, sV = avs.oom_retry(grad_sign, 'alignment gradient')
        torch.cuda.empty_cache()
        eye = torch.eye(d, device=dev)
        best = None
        for lr in a.lrs:
            AKl, AVl = eye - lr * sK, eye - lr * sV                      # broadcast over layers
            vl = val_loss(AKl, AVl)
            print(f'  {a.video_model}->{mj} one-shot lr {lr:g}: val NMSE {vl:.5f}', flush=True)
            if best is None or vl < best[0]:
                best = (vl, lr, AKl, AVl)
            else:
                del AKl, AVl
            torch.cuda.empty_cache()
        del sK, sV
        v0 = val_loss(None, None)
        print(f'  {a.video_model}->{mj}: val NMSE zero-shot {v0:.5f}, one-shot best lr {best[1]:g} -> {best[0]:.5f} ({time.time() - t0:.0f}s)', flush=True)
        del tr, va, Ps_tr, Ps_va
        # ---- evaluation
        for set_name, wl in evals.items():
            acc = {}
            def eval_window(w):
                out = []
                for r in materialize(w):
                    P = payload(aj, r)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        e0 = float(nmse(replay_aligned(abj, P, r['kv'], r['mask_a'], H, Dh), r))
                        e1 = float(nmse(replay_aligned(abj, P, r['kv'], r['mask_a'], H, Dh, best[2], best[3]), r))
                        ec = float(nmse(replay_aligned(abj, P, r['kv'], r['mask_a'], H, Dh, cut=True), r))
                    out.append((r['si'], (w[0], w[2], e0, e1, ec)))
                return out
            for w in wl:
                for si, row in avs.oom_retry(lambda: eval_window(w), f'eval {w[6]}'):
                    acc.setdefault(si, []).append(row)
            for si, lst in acc.items():
                eps = sorted({(x[0], x[1]) for x in lst}); pos = {e: i for i, e in enumerate(eps)}
                S = np.zeros((len(eps), 3)); n = np.zeros(len(eps))
                for x in lst:
                    S[pos[(x[0], x[1])]] += x[2:]; n[pos[(x[0], x[1])]] += 1
                bi = np.random.default_rng(0).integers(0, len(eps), size=(1000, len(eps)))
                bm = S[bi].sum(1) / n[bi].sum(1)[:, None]; pm = S.sum(0) / n.sum()
                for col, name, steps in ((0, 'zero_shot', 0), (1, 'one_shot', 1)):
                    rows.append(dict(video_model=a.video_model, action_model=mj, set=set_name, action_sigma=sig_a[si], train_steps=steps,
                                     lr=best[1] if steps else 0.0, nmse=pm[col], nmse_lo=float(np.percentile(bm[:, col], 2.5)), nmse_hi=float(np.percentile(bm[:, col], 97.5)),
                                     nmse_cut_action_model=pm[2], n_windows=int(n.sum()), n_episodes=len(eps), train_windows=len(train_w), seed=a.seed))
                print(f'  eval {a.video_model}->{mj} {set_name} sigma {sig_a[si]:.2f}: zero-shot {pm[0]:.5f} one-shot {pm[1]:.5f} (cut {pm[2]:.4f})', flush=True)
    out = OUT / 'results' / 'step5' / f'step5_stitch_video{a.video_model}.csv'; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
