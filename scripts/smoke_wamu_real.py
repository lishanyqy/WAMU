#!/usr/bin/env python3
"""Run a CPU-friendly WAMU-real data + ActionDiT lifecycle smoke test.

This intentionally exercises the real LeRobot samples and OpenWAM action
branch, while using a tiny ActionDiT and synthetic bridge features. Full
Wan2.2 video training remains a separate GPU/weights gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.dataloader.wamu_real import WamuRealDataset
from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture


def build_dataset(dataset_dir: str):
    return WamuRealDataset.from_config(
        {
            "dataset_dir": dataset_dir,
            "num_frames": 33,
            "video_stride": 4,
            "window_stride": 1,
            "height": 384,
            "width": 320,
            "multiview": True,
            "normalize_mode": None,
            "target_camera": "observation.images.top",
            "camera_layout": [
                "observation.images.top",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ],
            "unify_action": True,
            "unify_action_map": ["0-6", "34-40"],
            "unify_state_map": ["0-6", "34-40"],
            "color_jitter": False,
        },
        split="train",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/lishanyang_wamu_real")
    parser.add_argument("--output", default="outputs/wamu_real_debug/action_dit.safetensors")
    args = parser.parse_args()

    dataset = build_dataset(args.dataset_dir)
    sample = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=list)))[0]
    batch, horizon, action_dim = 1, int(sample["action"].shape[0]), int(sample["action"].shape[1])

    model_cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": action_dim,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": 64,
        "bridge_layers": (0,),
    }
    architecture = DualSystemCrossAttnArchitecture(cfg=model_cfg).train()
    action = sample["action"].unsqueeze(0)
    noise = torch.randn_like(action)
    noisy = 0.5 * action + 0.5 * noise
    bridges = {0: torch.randn(batch, horizon * 4, 64)}
    prediction = architecture.action_backbone(noisy, bridges, torch.tensor([500.0]))
    loss = F.mse_loss(prediction, noise - action)
    optimizer = torch.optim.Adam(architecture.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(architecture.action_backbone.state_dict(), str(output))
    restored = DualSystemCrossAttnArchitecture(cfg=model_cfg).eval()
    restored.action_backbone.load_state_dict(load_file(str(output)), strict=True)
    with torch.no_grad():
        inferred = restored.action_backbone(torch.randn_like(action), bridges, torch.tensor([300.0]))
    assert tuple(inferred.shape) == (1, horizon, action_dim)
    print(
        f"PASS dataset_len={len(dataset)} action={tuple(sample['action'].shape)} "
        f"video_frames={len(sample['video'])} loss={float(loss.detach()):.6f} "
        f"checkpoint={output} infer={tuple(inferred.shape)}"
    )


if __name__ == "__main__":
    main()
