import pytest
import torch

from openwam.model.architectures.utils.bridge_noise import BridgeNoiseConfig, apply_plan, make_plan

GRID = (3, 12, 10)
T = 3 * 12 * 10


def test_disabled_by_default():
    assert not BridgeNoiseConfig.from_cfg(None).enabled


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        BridgeNoiseConfig.from_cfg({"enabled": True, "rate": 0.2})


def test_rate_and_future_weighting():
    cfg = BridgeNoiseConfig(enabled=True, p_clean=0.0, min_rate=0.2, max_rate=0.2, block=2, future_weight=2.5)
    g = torch.Generator().manual_seed(0)
    plan = make_plan(4000, GRID, cfg, device="cpu", generator=g)
    m = plan.mask.float().reshape(4000, 3, 12, 10)
    assert abs(m.mean().item() - 0.2) < 0.01                       # mean rate preserved
    f0, fut = m[:, 0].mean().item(), m[:, 1:].mean().item()
    assert abs(fut / f0 - 2.5) < 0.15                              # future frames picked 2.5x as often


def test_blocks_are_contiguous():
    cfg = BridgeNoiseConfig(enabled=True, p_clean=0.0, min_rate=0.3, max_rate=0.3, block=2)
    plan = make_plan(8, GRID, cfg, device="cpu", generator=torch.Generator().manual_seed(1))
    m = plan.mask.reshape(8, 3, 6, 2, 5, 2)                        # 12x10 grid in 2x2 blocks
    assert torch.equal(m.all(dim=(3, 5)), m.any(dim=(3, 5)))       # every 2x2 block is all-or-nothing


def test_p_clean_leaves_samples_untouched():
    cfg = BridgeNoiseConfig(enabled=True, p_clean=1.0)
    plan = make_plan(16, GRID, cfg, device="cpu")
    x = torch.randn(16, T, 8)
    assert torch.equal(apply_plan(x, plan), x)


def test_swap_and_gauss_only_touch_masked_tokens_and_keep_grad():
    cfg = BridgeNoiseConfig(enabled=True, p_clean=0.0, min_rate=0.3, max_rate=0.3, swap_prob=0.5)
    g = torch.Generator().manual_seed(2)
    plan = make_plan(6, GRID, cfg, device="cpu", generator=g)
    x = torch.randn(6, T, 8, requires_grad=True)
    y = apply_plan(x, plan, generator=g)
    keep = ~plan.mask
    assert torch.equal(y[keep], x[keep])
    other = x.detach().roll(1, dims=0)
    for i in range(6):
        if plan.swap[i]:
            assert torch.equal(y[i][plan.mask[i]], other[i][plan.mask[i]])
        else:
            assert not torch.equal(y[i][plan.mask[i]], x[i][plan.mask[i]])
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_single_sample_never_swaps():
    cfg = BridgeNoiseConfig(enabled=True, p_clean=0.0, swap_prob=1.0)
    plan = make_plan(1, GRID, cfg, device="cpu")
    assert not plan.swap.any()
