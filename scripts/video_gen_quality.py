#!/usr/bin/env python
"""Video generation quality of a checkpoint on RoboTwin Franka windows.

For each task, a few episodes with one random full window each (seeded per task, identical for every model): the model
generates the 33-step window (9 video frames on the 384x320 L-shape canvas: head camera on top, two wrist cameras below)
from the clean first frame, the prompt and the proprio state, with the deploy denoising schedule. Frames 1..8 are compared
with the ground truth canvas:
  PSNR, SSIM (gaussian, per channel), LPIPS (AlexNet), on the full canvas and on the head-camera region;
  per frame index, so drift over the clip is visible.
Two references on the same windows: "copy first frame" (predict frame 0 for every later frame) and the VAE round trip of
the ground truth (encode-decode ceiling).
Output: artifacts/videogen/<model>_<tag>.npz
"""
import argparse, inspect, math, sys, tempfile, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_video_share import load_arch, make_dataset, oom_retry, offload_text_encoder, place_on_gpu


def to_tensor(pils):
    return torch.stack([torch.from_numpy(np.asarray(p.convert('RGB'), np.float32) / 255.0).permute(2, 0, 1) for p in pils])  # (T,3,H,W) in [0,1]


def gaussian_ssim(x, y, win=11, sigma=1.5):
    """x, y: (N,3,H,W) in [0,1] -> (N,) mean SSIM over channels and pixels."""
    g = torch.exp(-(torch.arange(win, dtype=x.dtype, device=x.device) - win // 2) ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] @ g[None, :]).expand(3, 1, win, win)
    mu_x = F.conv2d(x, k, groups=3); mu_y = F.conv2d(y, k, groups=3)
    sxx = F.conv2d(x * x, k, groups=3) - mu_x ** 2; syy = F.conv2d(y * y, k, groups=3) - mu_y ** 2; sxy = F.conv2d(x * y, k, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean((1, 2, 3))


def metrics(pred, gt, lp, head_rows):
    """pred, gt: (T,3,H,W) in [0,1]; returns dict of (T,) arrays for the full canvas and the head-camera region."""
    out = {}
    for name, sl in (('full', slice(None)), ('head', slice(0, head_rows))):
        p, g = pred[:, :, sl], gt[:, :, sl]
        mse = ((p - g) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
        out[f'psnr_{name}'] = (10 * torch.log10(1.0 / mse)).cpu().numpy()
        out[f'ssim_{name}'] = gaussian_ssim(p, g).cpu().numpy()
        with torch.no_grad():
            out[f'lpips_{name}'] = lp(p * 2 - 1, g * 2 - 1).flatten().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', type=Path, required=True); ap.add_argument('--wan-dir', type=Path, default=Path('assets/video_backbone_ckpt/Wan2.2-TI2V-5B'))
    ap.add_argument('--dataset-dir', type=Path, default=Path('/home/a1905721/dataspace/robotwin2.0/dataset'))
    ap.add_argument('--embodiment', default='franka'); ap.add_argument('--variant', default='clean_50')
    ap.add_argument('--episodes-per-task', type=int, default=2); ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--limit-tasks', type=int, default=0)
    ap.add_argument('--tag', default='franka'); ap.add_argument('--device', default='cuda'); ap.add_argument('--output', type=Path, default=None)
    a = ap.parse_args()
    name = a.ckpt_dir.name
    out = a.output or Path(f'artifacts/videogen/{name}_{a.tag}.npz'); out.parent.mkdir(parents=True, exist_ok=True)
    arch, cfg = load_arch(a.ckpt_dir, a.wan_dir, a.device)
    arch.init_training_schedulers(1000)
    height, width = int(cfg.dataloader.height), int(cfg.dataloader.width); head_rows = int(round(height * 2 / 3))
    tasks = sorted(d for d in a.dataset_dir.iterdir() if (d / f'{a.embodiment}_{a.variant}' / 'data').is_dir())
    tasks = tasks[:a.limit_tasks] if a.limit_tasks else tasks
    samples = []
    for task in tasks:
        ds = make_dataset(cfg, task, a.ckpt_dir, a.embodiment, a.variant)
        rng = np.random.default_rng(a.seed * 1000 + sum(map(ord, task.name)))
        eps = rng.choice(len(ds._episode_files), size=min(a.episodes_per_task, len(ds._episode_files)), replace=False)
        for ep in eps:
            ep = int(ep); max_start = max(0, ds._episode_lengths[ep] - ds._raw_window_len)
            start = int(rng.integers(0, max_start + 1))
            samples.append((task.name, ep, start, ds._build_sample(ep, start)))
    vb = arch.video_backbone
    oom_retry(lambda: offload_text_encoder(vb, sorted({s[3]['prompt'] for s in samples}), a.device), 'prompt encoding')
    oom_retry(lambda: place_on_gpu(arch, a.device), 'moving the model to the GPU')
    import lpips
    lp = lpips.LPIPS(net='alex', verbose=False).to(a.device).eval()
    from openwam.deploy.denoise_schedule import make_schedule
    shift = getattr(arch.action_backbone, 'shift_action', None) or 5.0
    shift_video = getattr(vb, 'shift_video', None)
    gen_params = set(inspect.signature(arch.generate).parameters)
    rows = {k: [] for k in ('gen', 'copy', 'vae')}; meta = []; t0 = time.time()
    for i, (task_name, ep, start, s) in enumerate(samples):
        gt_frames = s['video']; T = len(gt_frames)
        proprio = s.get('proprio'); proprio = torch.as_tensor(np.asarray(proprio), dtype=torch.float32) if proprio is not None else None

        def run():
            schedule = make_schedule('sync', video_scheduler=arch.video_scheduler, action_scheduler=arch.action_scheduler, num_steps=a.steps, shift=shift, shift_video=shift_video)
            kw = dict(schedule=schedule, prompt=s['prompt'], first_frame_image=[gt_frames[0]], num_frames=T, action_num_frames=int(cfg.dataloader.num_frames),
                      height=height, width=width, seed=a.seed + i, tiled=True, num_inference_steps=a.steps, shift=shift, decode_video=True,
                      proprio=(proprio.to(a.device, arch.dtype) if proprio is not None else None))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                res = arch.generate(**{k: v for k, v in kw.items() if k in gen_params})
                lat = vb.preprocess_input_for_train(frames=[gt_frames], text=[s['prompt']])['input_latents']
                rec = vb.decode_video(lat, tiled=True)
            return res['video'], rec
        video, rec = oom_retry(run, f'window {task_name}/{ep}/{start}')
        gt = to_tensor(gt_frames).to(a.device); pr = to_tensor(video).to(a.device); vr = to_tensor(rec).to(a.device)
        if pr.shape[-2:] != gt.shape[-2:]:
            pr = F.interpolate(pr, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            vr = F.interpolate(vr, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        n = min(len(pr), len(gt))
        for key, pred in (('gen', pr[1:n]), ('copy', gt[0:1].expand(n - 1, -1, -1, -1)), ('vae', vr[1:n])):
            rows[key].append(metrics(pred.clamp(0, 1), gt[1:n], lp, head_rows))
        meta.append((task_name, ep, start))
        if (i + 1) % 10 == 0:
            g = rows['gen'][-1]; print(f'  {i + 1}/{len(samples)} ({(time.time() - t0) / (i + 1):.1f} s/clip) last: PSNR {g["psnr_full"].mean():.2f} LPIPS {g["lpips_full"].mean():.3f}', flush=True)
        if i == 0:
            vis = out.with_name(out.stem + '_example.npz')
            np.savez_compressed(vis, gt=(gt.cpu().numpy() * 255).astype(np.uint8), gen=(pr.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), task=task_name, episode=ep, start=start)
    save = {'model': name, 'tag': a.tag, 'steps': a.steps, 'head_rows': head_rows, 'meta': np.array(meta, dtype=object)}
    for key, lst in rows.items():
        for m in lst[0]:
            save[f'{key}_{m}'] = np.stack([r[m] for r in lst])          # (n_clips, T-1)
    np.savez(out, **save)
    msg = ' | '.join(f'{k}: PSNR {save[f"{k}_psnr_full"].mean():.2f} SSIM {save[f"{k}_ssim_full"].mean():.3f} LPIPS {save[f"{k}_lpips_full"].mean():.3f}' for k in rows)
    print(f'done {name}: {len(samples)} clips. {msg}', flush=True)


if __name__ == '__main__':
    main()
