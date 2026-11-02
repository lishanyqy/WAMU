"""Training-time corruption of the cross-attention bridge (``model.architecture.bridge_noise``).

Only the copy of the video features that the action stream reads through the bridge is perturbed; the video DiT and
its own loss are untouched. The point is to show the bridge read-out, during training, video it cannot trust, so it
learns not to rely on a few video tokens (cf. the sharp, head-camera-focused bridge attention seen OOD).

Per training sample:
  * with probability ``p_clean`` the sample is left clean;
  * otherwise a fraction ``rate ~ U(min_rate, max_rate)`` of the video tokens is selected as spatially contiguous
    ``block x block`` patches; future latent frames (index >= 1, generated at inference time) are picked
    ``future_weight`` times as often as frame 0 (the real observation);
  * the selected tokens are corrupted in one of two ways:
      - ``gauss``: x + sigma * rms(x_token) * eps, sigma ~ U(sigma_min, sigma_max)
      - ``swap``:  replaced by the same-position tokens of another sample in the batch (right statistics, wrong
        content; the swapped-in tokens are detached so the wrong action loss does not train the other sample's video).
    ``swap_prob`` picks between them (``gauss`` when the batch has a single sample).

The same token mask is used at every bridge layer of a sample (a region is unreliable, not one layer of it); the
Gaussian draw is fresh per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class BridgeNoiseConfig:
    enabled: bool = False
    p_clean: float = 0.3
    min_rate: float = 0.1
    max_rate: float = 0.3
    block: int = 2
    future_weight: float = 2.5
    swap_prob: float = 0.5
    sigma_min: float = 0.5
    sigma_max: float = 1.5

    @classmethod
    def from_cfg(cls, cfg: Optional[Mapping[str, Any]]) -> "BridgeNoiseConfig":
        if cfg is None:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg.keys()) - known
        if unknown:
            raise ValueError(f"bridge_noise: unknown keys {sorted(unknown)}; allowed {sorted(known)}")
        out = cls(**{k: cfg[k] for k in cfg.keys()})
        if not 0.0 <= out.min_rate <= out.max_rate <= 1.0:
            raise ValueError(f"bridge_noise: need 0 <= min_rate <= max_rate <= 1, got {out.min_rate}, {out.max_rate}")
        if out.block < 1:
            raise ValueError(f"bridge_noise: block must be >= 1, got {out.block}")
        return out


@dataclass
class BridgeNoisePlan:
    mask: Tensor    # (B, T) bool: tokens to corrupt
    swap: Tensor    # (B,) bool: swap (True) or gauss (False)
    sigma: Tensor   # (B,) float: gauss strength


def make_plan(batch: int, grid: Tuple[int, int, int], cfg: BridgeNoiseConfig, device, generator=None) -> BridgeNoisePlan:
    """Draw which tokens of each sample are corrupted and how. ``grid`` = (frames, height, width) of the bridge tokens,
    laid out (f h w) along the token axis."""
    frames, gh, gw = (int(g) for g in grid)
    b = cfg.block
    lh, lw = -(-gh // b), -(-gw // b)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    active = rand(batch) >= cfg.p_clean
    rate = cfg.min_rate + (cfg.max_rate - cfg.min_rate) * rand(batch)
    w = torch.ones(frames, device=device)
    w[1:] = cfg.future_weight
    w = w / w.mean()                                                   # keeps the mean rate at `rate`
    p = (rate[:, None] * w[None, :]).clamp(max=0.95)                   # (B, F)
    low = (rand(batch, frames, lh, lw) < p[:, :, None, None]).to(torch.uint8)
    mask = low.repeat_interleave(b, dim=2).repeat_interleave(b, dim=3)[:, :, :gh, :gw].bool()
    mask = mask.reshape(batch, frames * gh * gw) & active[:, None]
    swap = (rand(batch) < cfg.swap_prob) & torch.tensor(batch > 1, device=device)
    sigma = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * rand(batch)
    return BridgeNoisePlan(mask=mask, swap=swap, sigma=sigma)


def apply_plan(bridge: Tensor, plan: BridgeNoisePlan, generator=None) -> Tensor:
    """Corrupt ``bridge`` (B, T, C) according to ``plan``. Gradients still flow to the untouched tokens and through the
    additive Gaussian noise; swapped-in tokens are detached."""
    if bridge.shape[:2] != plan.mask.shape:
        raise ValueError(f"bridge tokens {tuple(bridge.shape[:2])} do not match the noise plan {tuple(plan.mask.shape)}")
    x = bridge.float()
    rms = x.detach().pow(2).mean(-1, keepdim=True).sqrt()
    eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    gauss = x + plan.sigma[:, None, None] * rms * eps
    other = x.detach().roll(1, dims=0)
    corrupted = torch.where(plan.swap[:, None, None], other, gauss)
    out = torch.where(plan.mask[:, :, None], corrupted, x)
    return out.to(bridge.dtype)


__all__ = ["BridgeNoiseConfig", "BridgeNoisePlan", "make_plan", "apply_plan"]
