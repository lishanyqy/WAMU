"""Training-time corruption of the video features that the ACTION stream reads (wamu/bridge_noise_plan.md).

Shared by joint_cross_attn (the bridge features) and joint_self_attn (the video K/V that the action queries see),
so both architectures corrupt the same tensor space with the same plan:

- per sample: clean with probability ``p_clean``; otherwise corrupt a fraction r ~ U(ratio_min, ratio_max) of the video
  tokens, chosen as contiguous ``block`` x ``block`` spatial patches, with frames 1.. ``future_weight`` times as likely
  as frame 0 while the overall fraction stays r;
- the corrupted tokens are the SAME in every layer (a region that is unreliable everywhere);
- per corrupted sample, one noise type: Gaussian ``x + sigma * rms(x) * eps`` (rms per token, sigma ~ U(sigma_min,
  sigma_max)) or token swap (the same position from another sample in the batch, detached so a wrong action loss never
  trains the other sample's video);
- the plan is a pure function of (seed, sample id, call index), so two architectures that see the same sample at the
  same training step get exactly the same corruption, and gradient-checkpoint recomputation reproduces it bit for bit;
- active only while the module is training; eval and deploy are untouched.

Config (``model.architecture.bridge_noise``)::

    bridge_noise:
      enabled: false
      p_clean: 0.3
      ratio_min: 0.10
      ratio_max: 0.20
      block: 2
      future_weight: 2.5
      gauss_prob: 0.5
      sigma_min: 0.5
      sigma_max: 1.5
      seed: 0
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import torch
from torch import Tensor

CLEAN, GAUSS, SWAP = 0, 1, 2


@dataclass
class BridgeNoiseConfig:
    enabled: bool = False
    p_clean: float = 0.3
    ratio_min: float = 0.10
    ratio_max: float = 0.20
    block: int = 2
    future_weight: float = 2.5
    gauss_prob: float = 0.5
    sigma_min: float = 0.5
    sigma_max: float = 1.5
    seed: int = 0

    @classmethod
    def from_cfg(cls, cfg) -> "BridgeNoiseConfig":
        if cfg is None:
            return cls()
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        out = cls()
        for f in fields(cls):
            v = get(f.name, None)
            if v is not None:
                default = getattr(out, f.name)
                setattr(out, f.name, bool(v) if isinstance(default, bool) else type(default)(v))
        if not 0.0 <= out.p_clean <= 1.0 or not 0.0 <= out.gauss_prob <= 1.0:
            raise ValueError("bridge_noise: p_clean and gauss_prob must lie in [0, 1]")
        if not 0.0 <= out.ratio_min <= out.ratio_max <= 1.0:
            raise ValueError("bridge_noise: need 0 <= ratio_min <= ratio_max <= 1")
        if out.block < 1 or out.future_weight <= 0:
            raise ValueError("bridge_noise: block must be >= 1 and future_weight > 0")
        return out


def _seed_of(*parts) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass
class BridgeNoisePlan:
    """Per-forward corruption plan for a batch, on the full (frames, height, width) video token grid."""

    mask: Tensor          # (B, F*H*W) bool, token order (f h w)
    kind: Tensor          # (B,) int: CLEAN / GAUSS / SWAP
    sigma: Tensor         # (B,) float, used by GAUSS samples
    partner: Tensor       # (B,) long, source sample for SWAP (== self when no partner exists)
    seeds: Sequence[int]  # per-sample seeds for the per-layer Gaussian draws

    def active(self) -> bool:
        return bool((self.kind != CLEAN).any())

    def apply(self, x: Tensor, layer: int) -> Tensor:
        """Corrupt ``x`` (B, S, C) with S <= F*H*W (a prefix of the grid, e.g. first_frame mode). Same tokens every layer."""
        if not self.active():
            return x
        B, S, C = x.shape
        m = self.mask[:, :S].to(x.device)
        out = x
        g = (self.kind == GAUSS).to(x.device)
        if bool(g.any()):
            rms = x.detach().float().pow(2).mean(-1, keepdim=True).sqrt()            # (B, S, 1), per token
            eps = torch.empty(B, S, C, device=x.device, dtype=torch.float32)
            for b in range(B):
                if bool(g[b]):
                    gen = torch.Generator(device=x.device).manual_seed(_seed_of(self.seeds[b], "eps", layer))
                    eps[b] = torch.randn(S, C, generator=gen, device=x.device, dtype=torch.float32)
            add = self.sigma.to(x.device).view(B, 1, 1) * rms * eps
            sel = (m & g.view(B, 1)).unsqueeze(-1)
            out = out + torch.where(sel, add, torch.zeros((), device=x.device)).to(x.dtype)
        s = (self.kind == SWAP).to(x.device)
        if bool(s.any()):
            src = x.detach()[self.partner.to(x.device)]                               # other sample, same position
            sel = (m & s.view(B, 1)).unsqueeze(-1)
            out = torch.where(sel, src, out)
        return out


class BridgeNoise:
    """Builds plans; one instance per architecture."""

    def __init__(self, cfg: BridgeNoiseConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self._calls = 0

    @classmethod
    def from_arch_cfg(cls, arch_cfg) -> Optional["BridgeNoise"]:
        raw = arch_cfg.get("bridge_noise") if hasattr(arch_cfg, "get") else None
        c = BridgeNoiseConfig.from_cfg(raw)
        return cls(c) if c.enabled else None

    def plan(self, batch: int, grid: tuple, sample_ids: Optional[Sequence[str]], device) -> BridgeNoisePlan:
        """``grid`` = (frames, height, width) of the video token grid. ``sample_ids``: stable per-sample identities
        (episode path + window start); falls back to the batch position when absent."""
        c = self.cfg
        call = self._calls
        self._calls += 1
        Fr, Hh, Ww = (int(v) for v in grid)
        ids = list(sample_ids) if sample_ids is not None else [f"pos{b}" for b in range(batch)]
        seeds = [_seed_of(c.seed, ids[b], call) for b in range(batch)]
        bh, bw = math.ceil(Hh / c.block), math.ceil(Ww / c.block)                     # block grid per frame
        w = torch.tensor([1.0] + [c.future_weight] * (Fr - 1))
        w = w / w.mean()                                                              # mean frame ratio == r
        mask = torch.zeros(batch, Fr, Hh, Ww, dtype=torch.bool)
        kind = torch.zeros(batch, dtype=torch.long)
        sigma = torch.zeros(batch)
        for b in range(batch):
            gen = torch.Generator().manual_seed(seeds[b])
            u = torch.rand(4, generator=gen)
            if u[0] < c.p_clean:
                continue
            kind[b] = GAUSS if u[1] < c.gauss_prob else SWAP
            sigma[b] = c.sigma_min + (c.sigma_max - c.sigma_min) * u[2]
            r = c.ratio_min + (c.ratio_max - c.ratio_min) * u[3]
            for f in range(Fr):
                n = int(round(min(1.0, float(r * w[f])) * bh * bw))
                if n == 0:
                    continue
                for idx in torch.randperm(bh * bw, generator=gen)[:n].tolist():
                    i, j = divmod(idx, bw)
                    mask[b, f, i * c.block:(i + 1) * c.block, j * c.block:(j + 1) * c.block] = True
        # swap partners: another sample of the batch; with a batch of one there is nobody to swap with -> Gaussian
        partner = torch.arange(batch)
        if batch > 1:
            gen = torch.Generator().manual_seed(_seed_of(c.seed, "partner", *seeds))
            for b in range(batch):
                k = int(torch.randint(batch - 1, (1,), generator=gen))
                partner[b] = k if k < b else k + 1
        else:
            kind[kind == SWAP] = GAUSS            # sigma was drawn for every corrupted sample, so it is already set
        return BridgeNoisePlan(mask=mask.reshape(batch, -1).to(device), kind=kind, sigma=sigma, partner=partner, seeds=seeds)


__all__ = ["BridgeNoise", "BridgeNoiseConfig", "BridgeNoisePlan", "CLEAN", "GAUSS", "SWAP"]
