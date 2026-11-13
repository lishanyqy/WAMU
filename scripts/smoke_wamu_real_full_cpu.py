#!/usr/bin/env python3
"""One-sample CPU pass of the FULL OpenWAM training graph on WAMU-real.

Uses the same Hydra config (train_robodojo_real.yaml), dataset registry,
OpenWAMTrainer, and Wan2.2-TI2V-5B weights as scripts/train.py, but skips
DeepSpeed/torchrun and runs on CPU: one real multiview sample -> prepare_inputs
-> joint video+action loss -> backward. Meant for a node without a usable GPU;
it is slow (minutes) and needs ~100GB RAM.

    python scripts/smoke_wamu_real_full_cpu.py [hydra overrides...]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir

from openwam.dataloader.registry import build_dataset
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.train.openwam_trainer import OpenWAMTrainer

CPU = torch.device("cpu")
_orig_set = BaseWAMArchitecture.set_dtype_device


def _cpu_set_dtype_device(self, dtype, device):  # force every backbone onto CPU
    return _orig_set(self, dtype, CPU)


BaseWAMArchitecture.set_dtype_device = _cpu_set_dtype_device


def main() -> None:
    overrides = [
        "model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B",
        "training.initialize_model_on_cpu=true",
        "training.batch_size=1",
        "dataloader.color_jitter.enabled=false",
        *sys.argv[1:],
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train_robodojo_real", overrides=overrides)

    torch.manual_seed(0)
    dataset = build_dataset(cfg.dataloader, split="train")
    sample = dataset[0]
    print(f"dataset_len={len(dataset)} video_frames={len(sample['video'])} "
          f"frame_size={sample['video'][0].size} action={tuple(sample['action'].shape)} prompt={sample['prompt']!r}")

    t0 = time.time()
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=dataset)
    arch = trainer.architecture
    arch.train()
    n_train = sum(p.numel() for p in arch.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in arch.parameters())
    print(f"built {type(arch).__name__} in {time.time() - t0:.0f}s: params={n_all/1e9:.2f}B trainable={n_train/1e9:.2f}B")

    t0 = time.time()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = trainer.compute_loss([sample])
    print("forward {:.0f}s: total={:.5f} video={:.5f} action={:.5f}".format(
        time.time() - t0, *(float(losses[k]) for k in ("total", "video", "action"))))

    t0 = time.time()
    losses["total"].backward()
    grads = [p.grad for p in arch.parameters() if p.requires_grad and p.grad is not None]
    gnorm = torch.norm(torch.stack([g.float().norm() for g in grads]))
    finite = all(torch.isfinite(g).all() for g in grads)
    print(f"backward {time.time() - t0:.0f}s: params_with_grad={len(grads)} grad_norm={float(gnorm):.4f} finite={finite}")
    assert torch.isfinite(losses["total"]) and finite and grads
    print("PASS full OpenWAM forward+backward on one WAMU-real sample (CPU)")


if __name__ == "__main__":
    main()
