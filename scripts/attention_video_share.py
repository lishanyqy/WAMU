#!/usr/bin/env python
"""How much does the action stream draw on the video stream?  Measured on training-faithful samples
(RoboTwin 33-step window: 9 video frames at stride 4, 32 action steps, proprio at step 0; video latents noised at a
fixed sigma, actions noised at fixed flow timesteps) for the two coupling families:

  joint self-attention (MoT, action_sees_video / mutual / video_sees_action / isolated / pretrained):
      per layer, from the actual mixed-attention logits of the action query rows:
        mass_video[f]      attention mass on video tokens of latent frame f (sums over frames = total video mass)
        mass_head / wrist  video mass split by camera region of the canvas (head rows vs wrist rows)
        mass_action        mass on the action tokens themselves
        head_video[h]      per-head video mass
        entropy            attention entropy (nats) over all keys
        r_video            ||gate * o(attention output sourced from video keys)|| / ||x_action||   (video-sourced residual update)
        r_attn             ||gate * o(full attention output)|| / ||x_action||
        frac_video         ||video-sourced update|| / ||full attention update||
  joint cross-attention (bridge, detach or not):
      per layer, from the bridge cross-attention of the ActionDiT block:
        mass_video[f], mass_head / wrist, head_video[h] (how the mass over video tokens is distributed; it sums to 1),
        entropy, r_video = ||gate_ca * cross_attn|| / ||x_action||, r_attn = ||gate_sa * self_attn|| / ||x_action||
  both: video ablation.  The same sample is run again with the action stream cut off from video
      (self-attn: action rows cannot attend any video key; cross-attn: the bridge term is dropped) and we record
        vel_change = ||v_abl - v|| / ||v||      (relative change of the predicted flow velocity)
        mse / mse_abl                            (flow-matching MSE against the training target, with / without video)
Output: artifacts/attention/<model>_<embodiment>.npz with per-layer means and per-sample ablation numbers.
Env: fastwam (CUDA); ~26 GB GPU.
"""
import argparse
import inspect
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, '.')

STATE = {}


def load_arch(ckpt_dir: Path, wan_dir: Path, device):
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    ckpts = sorted(ckpt_dir.glob('checkpoint_step_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
    if not ckpts:
        raise SystemExit(f'no checkpoint_step_*.safetensors in {ckpt_dir}')
    ckpt_file = ckpts[-1]
    print(f'checkpoint: {ckpt_file.name}', flush=True)
    with tempfile.TemporaryDirectory(prefix='openwam_attn_') as td:
        td = Path(td); cfg = OmegaConf.load(ckpt_dir / 'config.yaml')
        cfg.model.video_backbone.model_path = str(wan_dir); cfg.model.video_backbone.encoder.model_path = str(wan_dir)
        if OmegaConf.select(cfg, 'model.architecture.action_film') is None:
            from safetensors import safe_open
            with safe_open(str(ckpt_file), 'pt') as f:
                has_film = any('action_film' in k for k in f.keys())
            OmegaConf.update(cfg, 'model.architecture.action_film', has_film, force_add=True)
        OmegaConf.save(cfg, td / 'config.yaml')
        (td / 'checkpoint_step_1.safetensors').symlink_to(ckpt_file)
        (td / 'normalization_stats.npy').symlink_to(ckpt_dir / 'normalization_stats.npy')
        _, arch = load_from_checkpoint_dir(str(td), device='cpu', ckpt_name='checkpoint_step_1.safetensors')   # CPU first: the shared GPU cannot hold text encoder + DiT at once
    arch.eval()
    for attr in ('_compiled_mot_run_joint_loop', '_compiled_action_forward'):
        if hasattr(arch, attr):
            setattr(arch, attr, None)
    return arch, OmegaConf.load(ckpt_dir / 'config.yaml')


def make_dataset(cfg, task_root: Path, ckpt_dir: Path, embodiment: str, variant: str = 'clean_50'):
    from openwam.dataloader.robotwin import RoboTwinDataset
    dl = OmegaConf.to_container(cfg.dataloader, resolve=True)
    params = inspect.signature(RoboTwinDataset.__init__).parameters
    kw = {k: v for k, v in dl.items() if k in params and k not in ('data_root', 'split', 'embodiment', 'variant', 'color_jitter')}
    return RoboTwinDataset(str(task_root / f'{embodiment}_{variant}' / 'data'), split='train', embodiment=embodiment, variant=variant,
                           normalization_stats_path=str(ckpt_dir / 'normalization_stats.npy'), **kw)


# ------------------------------------------------------------------ hooks

def video_split(p_video, grid, head_rows):
    """p_video: (..., Sv) attention over video tokens laid out (f h w). Returns (by_frame (..., F), head_mass, wrist_mass)."""
    Fr, gh, gw = grid
    x = p_video.reshape(*p_video.shape[:-1], Fr, gh, gw)
    by_frame = x.sum((-2, -1))
    head = x[..., :head_rows, :].sum((-3, -2, -1)); wrist = x[..., head_rows:, :].sum((-3, -2, -1))
    return by_frame, head, wrist


def entropy(p):
    return -(p.clamp_min(1e-12).log() * p).sum(-1)


def video_entropies(p_video, grid):
    """Entropy of the attention over video keys only (renormalised), and per latent frame (renormalised within the frame).
    Same 360-key support for self- and cross-attention, so the two families are comparable."""
    Fr, gh, gw = grid
    pv = p_video / p_video.sum(-1, keepdim=True).clamp_min(1e-12)
    pf = p_video.reshape(*p_video.shape[:-1], Fr, gh * gw)
    pf = pf / pf.sum(-1, keepdim=True).clamp_min(1e-12)
    return float(entropy(pv).mean()), entropy(pf).mean(tuple(range(pf.dim() - 2))).cpu().numpy()   # scalar, (Fr,)


def add_video_entropies(rec, p_video):
    ev, ef = video_entropies(p_video, STATE['grid'])
    rec['entropy_video'] += ev; rec['entropy_frame'] = rec['entropy_frame'][:len(ef)] + ef


def install_self_attn_hooks(arch):
    vb, ab = arch.video_backbone, arch.action_backbone
    drv = arch._mot_driver or arch.build_mot_driver()
    H, Dh = vb.num_heads, vb.head_dim
    orig_step, orig_mixed, orig_mask, orig_post = drv.step, drv._mixed_attention, drv._build_attention_mask, ab.post_attn_at_layer

    def step(layer_id, vstate, astate, **kw):
        STATE['layer'] = layer_id
        if STATE.get('ablate') and layer_id >= STATE.get('ablate_from', 0) and kw.get('attn_mask') is not None:
            kw['attn_mask'] = STATE['cut_mask']                      # action rows cannot attend video from this layer on
        return orig_step(layer_id, vstate, astate, **kw)

    def build_mask(**kw):
        m = orig_mask(**kw)
        Sv = kw['s_video']
        cut = m.clone(); cut[Sv:Sv + kw['s_action'], :Sv] = False
        STATE['cut_mask'] = cut
        return m

    def mixed(q, k, v, attn_mask):
        out = orig_mixed(q, k, v, attn_mask)
        if STATE.get('ablate') or q.shape[1] != k.shape[1]:   # second, action-rows-only call of a detach_video model
            return out
        B, S, _ = q.shape; Fr, gh, gw = STATE['grid']; Sv = Fr * gh * gw; Sa = S - Sv
        qa = q[:, Sv:].reshape(B, Sa, H, Dh).transpose(1, 2).float()
        kk = k.reshape(B, S, H, Dh).transpose(1, 2).float()
        vv = v.reshape(B, S, H, Dh).transpose(1, 2).float()
        logits = qa @ kk.transpose(-1, -2) / math.sqrt(Dh)
        m = attn_mask if attn_mask.dim() == 2 else attn_mask[0, 0]
        logits = logits.masked_fill(~m[Sv:, :], float('-inf'))
        p = logits.softmax(-1)                                                   # (B, H, Sa, S)
        pv = p[..., :Sv]
        by_frame, head, wrist = video_split(pv, STATE['grid'], STATE['head_rows'])
        rec = STATE['rec'][STATE['layer']]
        bf = by_frame.mean((0, 1, 2)).cpu().numpy(); rec['mass_video_frame'] = rec['mass_video_frame'][:len(bf)] + bf
        rec['mass_head'] += float(head.mean()); rec['mass_wrist'] += float(wrist.mean())
        rec['mass_action'] += float(p[..., Sv:].sum(-1).mean())
        rec['head_video'] += pv.sum(-1).mean((0, 2)).cpu().numpy()
        rec['entropy'] += float(entropy(p).mean())
        add_video_entropies(rec, pv)
        rec['mass_video_query'] += pv.sum(-1).mean((0, 1)).cpu().numpy()          # per action step
        STATE['video_part'] = (pv @ vv[:, :, :Sv]).transpose(1, 2).reshape(B, Sa, H * Dh)
        return out

    def post(layer_id, astate, attn_out, post_state):
        if not STATE.get('ablate'):
            rx, gate, block = post_state['residual_x'], post_state['gate_msa'], post_state['block']
            with torch.autocast('cuda', enabled=False):
                upd = (gate * block.self_attn.o(attn_out)).float(); updv = (gate * block.self_attn.o(STATE['video_part'].to(attn_out.dtype))).float()
                xn = rx.float().norm(dim=-1).clamp_min(1e-6)
                rec = STATE['rec'][layer_id]
                rec['r_video'] += float((updv.norm(dim=-1) / xn).mean()); rec['r_attn'] += float((upd.norm(dim=-1) / xn).mean())
                rec['frac_video'] += float((updv.norm(dim=-1) / upd.norm(dim=-1).clamp_min(1e-6)).mean())
        astate = orig_post(layer_id, astate, attn_out, post_state)
        if layer_id in STATE.get('decode_layers', ()):
            STATE['xs'][layer_id] = astate.payload.x_action.detach()
        return astate

    drv.step, drv._mixed_attention, drv._build_attention_mask, ab.post_attn_at_layer = step, mixed, build_mask, post
    return vb.num_layers, H


def install_cross_attn_hooks(arch):
    from einops import rearrange
    ab = arch.action_backbone
    H = ab.num_heads
    for li, block in enumerate(ab.blocks):
        def fwd(x_action, x_video, t_mod, freqs=None, context=None, context_mask=None, x_video_v=None, *, self=block, li=li):
            # cross_attention_direct: x_video = the video DiT's own keys (normed + RoPE), x_video_v = its values
            (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff) = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(9, dim=1)
            h = self.self_attn_norm(x_action) * (1 + scale_sa) + shift_sa
            upd_sa = gate_sa * self.self_attn(h, freqs=freqs)
            xn = x_action.float().norm(dim=-1).clamp_min(1e-6)
            x_action = x_action + upd_sa
            h = self.bridge_attn_norm(x_action) * (1 + scale_ca) + shift_ca
            ca = self.cross_attn
            joint = getattr(ca, 'joint_action_kv', False)          # cross_attention_softmax
            direct = getattr(self, 'direct_video_kv', False)       # cross_attention_direct
            cut_here = STATE.get('ablate') and li >= STATE.get('ablate_from', 0)
            if joint and cut_here:
                # joint-softmax ablation = drop the video keys from the softmax (as the self-attn ablation masks them)
                qa = rearrange(ca.norm_q(ca.q(h)), 'b s (n d) -> b n s d', n=H); ka = rearrange(ca.norm_k_a(ca.k_a(h)), 'b s (n d) -> b n s d', n=H)
                va = rearrange(ca.v_a(h), 'b s (n d) -> b n s d', n=H)
                upd_ca = gate_ca * ca.o(rearrange(torch.nn.functional.scaled_dot_product_attention(qa, ka, va), 'b n s d -> b s (n d)'))
            else:
                upd_ca = gate_ca * (ca(h, x_video, x_video_v) if direct else ca(h, x_video))
            if not STATE.get('ablate'):
                Sa_ = h.shape[1]
                q = rearrange(ca.norm_q(ca.q(h)), 'b s (n d) -> b n s d', n=H).float(); k = rearrange(x_video if direct else ca.norm_k(ca.k(x_video)), 'b s (n d) -> b n s d', n=H).float()
                if joint:
                    k = torch.cat([rearrange(ca.norm_k_a(ca.k_a(h)), 'b s (n d) -> b n s d', n=H).float(), k], dim=2)
                p_all = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)   # (B, H, Sa, [Sa +] Sv)
                p = p_all[..., Sa_:] if joint else p_all                                 # weight on the video keys
                by_frame, head, wrist = video_split(p, STATE['grid'], STATE['head_rows'])
                rec = STATE['rec'][li]
                bf = by_frame.mean((0, 1, 2)).cpu().numpy(); rec['mass_video_frame'] = rec['mass_video_frame'][:len(bf)] + bf
                rec['mass_head'] += float(head.mean()); rec['mass_wrist'] += float(wrist.mean())
                rec['mass_action'] += float(p_all[..., :Sa_].sum(-1).mean()) if joint else 0.0
                rec['head_video'] += p.sum(-1).mean((0, 2)).cpu().numpy()
                rec['entropy'] += float(entropy(p_all).mean())
                add_video_entropies(rec, p)
                rec['mass_video_query'] += p.sum(-1).mean((0, 1)).cpu().numpy()
                if os.environ.get('ATTN_VERIFY') and not joint and STATE.setdefault('verified', set()).isdisjoint({li}):
                    # debug: rebuild the cross-attention output from p and compare with the module's own output
                    STATE['verified'].add(li)
                    vv = rearrange(x_video_v if direct else ca.v(x_video), 'b s (n d) -> b n s d', n=H).float()
                    rec_out = ca.o(rearrange((p @ vv).to(h.dtype), 'b n s d -> b s (n d)'))
                    ref = ca(h, x_video, x_video_v) if direct else ca(h, x_video)
                    lg = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                    print(f'VERIFY L{li:02d} rel_err {float((rec_out - ref).float().norm() / ref.float().norm()):.4f} '
                          f'|q| {float(q.norm(dim=-1).mean()):.2f} |k| {float(k.norm(dim=-1).mean()):.2f} logit_std {float(lg.std(-1).mean()):.2f} '
                          f'max_p {float(p.max(-1).values.mean()):.3f} H {float(entropy(p_all).mean()):.2f} eff_keys {float(entropy(p_all).exp().mean()):.0f}', flush=True)
                xn2 = x_action.float().norm(dim=-1).clamp_min(1e-6)
                if joint:
                    # video-key part of the bridge update (o is linear: subtract its bias once)
                    vv = rearrange(ca.v(x_video), 'b s (n d) -> b n s d', n=H).float()
                    upd_v = (gate_ca * (ca.o(rearrange((p @ vv).to(h.dtype), 'b n s d -> b s (n d)')) - ca.o.bias)).float()
                    upd_a = upd_ca.float() - upd_v
                    nv = upd_v.norm(dim=-1); na = upd_a.norm(dim=-1) + upd_sa.float().norm(dim=-1)
                    rec['r_video'] += float((nv / xn2).mean())
                    rec['frac_video'] += float((nv / (nv + na).clamp_min(1e-6)).mean())
                else:
                    rec['r_video'] += float((upd_ca.float().norm(dim=-1) / xn2).mean())
                    rec['frac_video'] += float((upd_ca.float().norm(dim=-1) / (upd_ca.float().norm(dim=-1) + upd_sa.float().norm(dim=-1)).clamp_min(1e-6)).mean())
                rec['r_attn'] += float((upd_sa.float().norm(dim=-1) / xn).mean())
                x_action = x_action + upd_ca
            elif joint or not cut_here:
                x_action = x_action + upd_ca
            if context is not None:
                x_action = x_action + self.context_attn(self.context_attn_norm(x_action), context, ctx_mask=context_mask)
            h = self.ffn_norm(x_action) * (1 + scale_ff) + shift_ff
            x_action = x_action + gate_ff * self.ffn(h)
            if li in STATE.get('decode_layers', ()):
                STATE['xs'][li] = x_action.detach()
            return x_action
        block.forward = fwd
    return ab.num_layers, H


def install_grid_hook(arch, head_rows_fn):
    vb = arch.video_backbone
    orig_prepare = vb.prepare

    def prepare(**kw):
        st = orig_prepare(**kw)
        STATE['grid'] = (int(st.grid_frames), int(st.grid_height), int(st.grid_width))
        STATE['head_rows'] = head_rows_fn(int(st.grid_height))
        return st
    vb.prepare = prepare


def offload_text_encoder(vb, prompts, device):
    """Encode every prompt once with the text encoder alone on the GPU, move it back to the CPU (it is the single largest
    tenant and is not needed afterwards) and serve wan_encode.encode_text from the cache."""
    from openwam.model.video_backbone.wan import encode as wan_encode
    cache = {}
    vb.text_encoder.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        for p in prompts:
            ctx, sl = wan_encode.encode_text([p], tokenizer=vb._tokenizer, text_encoder=vb.text_encoder, device=device)
            cache[p] = (ctx.cpu(), sl.cpu())
    vb.text_encoder.to('cpu'); torch.cuda.empty_cache()
    orig = wan_encode.encode_text

    def cached(prompts, *, tokenizer, text_encoder, device):
        if all(p in cache for p in prompts):
            ctx = torch.cat([cache[p][0] for p in prompts]).to(device); sl = torch.cat([cache[p][1] for p in prompts]).to(device)
            return ctx, sl
        return orig(prompts, tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    wan_encode.encode_text = cached
    print(f'cached {len(cache)} prompt embeddings; text encoder offloaded. GPU allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB', flush=True)


def oom_retry(fn, what, tries=90, wait=60):
    """The GPU is shared with bursty neighbours: on CUDA OOM wait and retry instead of dying."""
    for i in range(tries):
        try:
            return fn()
        except (torch.OutOfMemoryError, RuntimeError) as e:       # cuBLAS/cuDNN report OOM as a plain RuntimeError
            if not isinstance(e, torch.OutOfMemoryError) and 'out of memory' not in str(e):
                raise
            torch.cuda.empty_cache()
            if i == tries - 1:
                raise
            print(f'  CUDA OOM during {what}; retry {i + 1} in {wait}s', flush=True); time.sleep(wait)


def place_on_gpu(arch, device):
    """Everything except the (already offloaded) text encoder to the GPU in bf16."""
    dev, dt = torch.device(device), torch.bfloat16
    vb, ab = arch.video_backbone, arch.action_backbone
    for n, m in vb.named_children():
        if n != 'text_encoder':
            m.to(device=dev, dtype=dt)
    vb._device, vb._dtype = dev, dt
    if hasattr(ab, 'set_dtype_device'):
        ab.set_dtype_device(dt, dev)
    else:
        ab.to(device=dev, dtype=dt)
    for name, bb in getattr(arch, 'backbones', {}).items():
        if bb is not vb and bb is not ab:
            bb.set_dtype_device(dt, dev)
    pe = getattr(arch, 'proprio_encoder', None)
    if pe is not None:
        pe.to(device=dev, dtype=dt)
    arch._device, arch._dtype = dev, dt
    torch.cuda.empty_cache()
    print(f'model on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GB allocated', flush=True)


def new_rec(L, H, Fr, Sa):
    return [dict(mass_video_frame=np.zeros(Fr), mass_head=0.0, mass_wrist=0.0, mass_action=0.0, head_video=np.zeros(H), entropy=0.0, entropy_video=0.0, entropy_frame=np.zeros(Fr),
                 mass_video_query=np.zeros(Sa), r_video=0.0, r_attn=0.0, frac_video=0.0) for _ in range(L)]


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', type=Path, required=True); p.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B'))
    p.add_argument('--dataset-dir', type=Path, default=Path('/home/a1905721/dataspace/robotwin2.0/dataset'))
    p.add_argument('--embodiment', default='franka'); p.add_argument('--episodes-per-task', type=int, default=2); p.add_argument('--windows', type=int, default=3)
    p.add_argument('--variant', default='clean_50', help='RoboTwin data variant folder suffix: clean_50 | randomized_500')
    p.add_argument('--samples-per-task', type=int, default=0, help='>0: draw this many random (episode, window) pairs per task instead of episodes-per-task x windows')
    p.add_argument('--tag', default=None, help='label used in the output file name (default: embodiment)')
    p.add_argument('--limit-tasks', type=int, default=0); p.add_argument('--action-t', type=float, nargs='+', default=[0.25, 0.75])
    p.add_argument('--video-sigma', type=float, default=0.5); p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--cut-from', type=int, nargs='*', default=[], help='cumulative cut-off: extra forwards where the action stream sees video only in layers < k, for each k')
    p.add_argument('--decode-layers', type=int, nargs='*', default=[], help='logit lens: decode the action stream after these layers with the final action decoder')
    a = p.parse_args()
    name = a.ckpt_dir.name
    tag = a.tag or a.embodiment
    out = a.output or Path(f'artifacts/attention/{name}_{tag}.npz')
    arch, cfg = load_arch(a.ckpt_dir, a.wan_dir, a.device)
    kind = 'self' if 'SelfAttn' in type(arch).__name__ else (('cross_joint' if 'Softmax' in type(arch).__name__ else 'cross') if 'CrossAttn' in type(arch).__name__ else None)
    if kind is None:
        raise SystemExit(f'{type(arch).__name__}: not a self/cross-attention dual system')
    height = int(cfg.dataloader.height); top_h = int(round(height * 2 / 3))
    install_grid_hook(arch, lambda gh: gh * top_h // height)
    L, H = (install_self_attn_hooks if kind == 'self' else install_cross_attn_hooks)(arch)
    vb, ab = arch.video_backbone, arch.action_backbone
    arch.init_training_schedulers(1000)                      # training sigma grids (shift from the checkpoint config)
    vsched, asched = vb.scheduler, ab.scheduler
    vid_id = int((vsched.sigmas - a.video_sigma).abs().argmin())
    act_ids = [int((asched.sigmas - t).abs().argmin()) for t in a.action_t]
    print(f'{name}: {kind}-attention, {L} layers, {H} heads; video sigma {float(vsched.sigmas[vid_id]):.3f}; action sigmas {[round(float(asched.sigmas[i]), 3) for i in act_ids]}', flush=True)

    tasks = sorted(d for d in a.dataset_dir.iterdir() if (d / f'{a.embodiment}_{a.variant}' / 'data').is_dir())
    tasks = tasks[:a.limit_tasks] if a.limit_tasks else tasks
    samples = []
    for task in tasks:
        ds = make_dataset(cfg, task, a.ckpt_dir, a.embodiment, a.variant)
        if a.samples_per_task > 0:
            # random (episode, start) pairs among the full windows of this task, seeded per task so every model sees the same draws
            rng = np.random.default_rng(a.seed * 1000 + sum(map(ord, task.name)))
            pool = [(ep, st) for ep, st in ds._window_index if st <= ds._episode_lengths[ep] - ds._raw_window_len]
            for i in rng.choice(len(pool), size=min(a.samples_per_task, len(pool)), replace=False):
                ep, start = pool[int(i)]
                # lazy: only the prompt is kept now (RAM on a shared node); the clip is rebuilt when its window is processed
                samples.append((task.name, ep, start, {'_ds': ds, 'prompt': ds._build_sample(ep, start)['prompt']}))
            continue
        for ep in range(min(a.episodes_per_task, len(ds._episode_files))):
            ep_len = ds._episode_lengths[ep]; max_start = max(0, ep_len - ds._raw_window_len)
            for w in range(a.windows):
                start = int(round(max_start * w / max(a.windows - 1, 1)))
                samples.append((task.name, ep, start, {'_ds': ds, 'prompt': ds._build_sample(ep, start)['prompt']}))   # lazy, as above
    oom_retry(lambda: offload_text_encoder(vb, sorted({s[3]['prompt'] for s in samples}), a.device), 'prompt encoding')
    oom_retry(lambda: place_on_gpu(arch, a.device), 'moving the model to the GPU')
    rec_sum = None; n_rec = 0; abl = []; t0 = time.time(); n_samples = 0
    STATE['decode_layers'] = set(a.decode_layers); lens = []; cut = []
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    for task_name, ep, start, sample in samples:
        task = Path(task_name)
        if '_ds' in sample:
            prompt = sample['prompt']; sample = sample['_ds']._build_sample(ep, start); sample['prompt'] = prompt   # same (cached) prompt
        if True:
            if True:
                def one_window():
                  nonlocal rec_sum, n_rec
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
                    for aid in act_ids:
                        asig = asched.sigmas[aid].to(actions); at = asched.timesteps[aid].to(actions).view(1)
                        anoise = torch.randn(actions.shape, generator=g, device=actions.device, dtype=actions.dtype)
                        noisy = asched.add_noise(actions, anoise, asig.view(1, 1, 1)); target = asched.training_target(actions, anoise)
                        Sa = noisy.shape[1]
                        if rec_sum is None:
                            STATE['rec'] = new_rec(L, H, 64, Sa)       # frame count fixed after the first forward
                        STATE['ablate'] = False; STATE['rec'] = new_rec(L, H, STATE.get('Fr', 64), Sa); STATE['xs'] = {}
                        _, pred = arch(noisy, at, proprio=proprio, **inputs, timestep=vt)
                        xs_with = STATE['xs']
                        Fr = STATE['grid'][0]
                        if rec_sum is None:
                            rec_sum = new_rec(L, H, Fr, Sa); STATE['Fr'] = Fr
                        for l in range(L):
                            for k, v in STATE['rec'][l].items():
                                vv = v[:Fr] if k == 'mass_video_frame' else v
                                rec_sum[l][k] = rec_sum[l][k] + vv
                        n_rec += 1
                        STATE['ablate'] = True; STATE['ablate_from'] = 0; STATE['xs'] = {}
                        _, pred_abl = arch(noisy, at, proprio=proprio, **inputs, timestep=vt)
                        STATE['ablate'] = False; xs_wo = STATE['xs']
                        cut_preds = {}
                        for kcut in a.cut_from:
                            STATE['ablate'] = True; STATE['ablate_from'] = kcut; STATE['xs'] = {}
                            _, cut_preds[kcut] = arch(noisy, at, proprio=proprio, **inputs, timestep=vt)
                        STATE['ablate'] = False; STATE['ablate_from'] = 0
                        # valid (time, dim) cells only: unified 80-d action layouts pad the unused dims, and the training loss masks them too
                        am = sample.get('action_mask')
                        mk = (torch.as_tensor(np.asarray(am)).to(pred.device).view(1, *np.asarray(am).shape).float() if am is not None else torch.ones_like(pred)).float()
                        if mk.ndim == 2:
                            mk = mk.unsqueeze(-1).expand_as(pred).float()
                        pf, pa, tg = pred.float() * mk, pred_abl.float() * mk, target.float() * mk
                        n_valid = mk.sum().clamp_min(1.0)
                        e, d = pf - tg, pa - pf                                   # error with video; change caused by cutting video
                        tok_ok = mk.sum(-1) > 0
                        tok_r = (d.norm(dim=-1) / pf.norm(dim=-1).clamp_min(1e-6))[tok_ok]   # per action-token relative change
                        abl.append((task.name, ep, start, float(asig), float(d.norm() / pf.norm().clamp_min(1e-6)), float((e ** 2).sum() / n_valid), float(((pa - tg) ** 2).sum() / n_valid),
                                    float((e ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)), float(((pa - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)),
                                    float((e ** 2).sum()), float((d ** 2).sum()), float((e * d).sum()), float((tg ** 2).sum()),
                                    float(d.norm(dim=-1).sum() / pf.norm(dim=-1).sum().clamp_min(1e-6)), float(tok_r.mean()), float(tok_r.median())))
                        for kcut, pc in cut_preds.items():
                            pc = pc.float() * mk
                            cut.append((float(asig), kcut, float(((pc - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)), float(((pc - pf) ** 2).sum() / (pf ** 2).sum().clamp_min(1e-6)),
                                        float(((pf - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)), float(((pa - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6))))
                        for l in sorted(STATE['decode_layers']):
                            pw = ab.action_decoder(xs_with[l]).float() * mk; po = ab.action_decoder(xs_wo[l]).float() * mk
                            lens.append((float(asig), l, float(((pw - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)), float(((po - tg) ** 2).sum() / (tg ** 2).sum().clamp_min(1e-6)),
                                         float((po - pw).norm() / pw.norm().clamp_min(1e-6)), float(((pw - pf) ** 2).sum() / (pf ** 2).sum().clamp_min(1e-6))))
                oom_retry(one_window, f'window {task.name}/{ep}/{start}')
                n_samples += 1
                if n_samples % 10 == 0:
                    print(f'  {n_samples} windows ({(time.time() - t0) / n_samples:.1f} s/window) task {task.name}', flush=True)
    keys = list(rec_sum[0].keys())
    per_layer = {k: np.stack([np.asarray(rec_sum[l][k]) / n_rec for l in range(L)]) for k in keys}
    abl_arr = np.array([r[3:] for r in abl], np.float64)
    np.savez(out, kind=kind, model=name, embodiment=a.embodiment, variant=a.variant, samples_per_task=a.samples_per_task, n_forwards=n_rec, n_windows=n_samples, grid=np.array(STATE['grid']), head_rows=STATE['head_rows'],
             action_sigmas=np.array([float(asched.sigmas[i]) for i in act_ids]), video_sigma=float(vsched.sigmas[vid_id]),
             abl_meta=np.array([(r[0], r[1], r[2]) for r in abl], dtype=object), abl=abl_arr, lens=np.array(lens, np.float64).reshape(-1, 6),
             lens_cols=np.array(['action_sigma', 'layer', 'nmse_with', 'nmse_without', 'vel_change', 'dist_to_final']),
             cut=np.array(cut, np.float64).reshape(-1, 6), cut_cols=np.array(['action_sigma', 'cut_from', 'nmse_cut', 'dist_to_full', 'nmse_full', 'nmse_nocut']), abl_cols=np.array(['action_sigma', 'vel_change', 'mse', 'mse_abl', 'nmse', 'nmse_abl', 'e2', 'd2', 'ed', 'tg2', 'vel_sumratio', 'vel_tokmean', 'vel_tokmedian']), **per_layer)
    mv = per_layer['mass_video_frame'].sum(1)
    print(f'done {name}: {n_samples} windows, {n_rec} forwards. video mass by layer (mean {mv.mean():.3f}): ' + ' '.join(f'{x:.2f}' for x in mv), flush=True)
    print(f'  r_video by layer (mean {per_layer["r_video"].mean():.4f}): ' + ' '.join(f'{x:.3f}' for x in per_layer['r_video']), flush=True)
    print(f'  ablation: vel_change mean {abl_arr[:, 1].mean():.3f}; nmse {abl_arr[:, 4].mean():.4f} -> without video {abl_arr[:, 5].mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
