"""Trace and intervene in offline WAMU predictions on recorded egg-packing views."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import linear

OUT = Path(__file__).resolve().parents[5] / "output/first-frame-attention/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = OUT.parents[2]
os.chdir(ROOT / "WAMU")
sys.path.insert(0, os.getcwd())

from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

ROLLOUTS = ROOT / "output/aloha-ceremony-test/rollouts"
ROLES = ("head", "left_wrist", "right_wrist")
REFERENCE = "20261124-125952-187637-eggpack"
REFERENCE_STAGES = json.loads((OUT.parent / "stages/results.json").read_text())


def read_events(rollout):
    return [json.loads(line) for line in (ROLLOUTS / rollout / "events.jsonl").read_text().splitlines()]


def opening_views(rollout):
    events = read_events(rollout)
    target_ns = round(next(e["time"] for e in events if e["event"] == "ceremony_settled") * 1e9)
    return {
        role: min((ROLLOUTS / rollout / role).glob("*.jpg"),
                  key=lambda file: abs(int(file.stem) - target_ns)).name
        for role in ROLES
    }


def policies_cases():
    for i, label in [(0, "opening"), (2, "early"), (3, "mid"), (4, "late")]:
        stage = REFERENCE_STAGES[i]
        yield label, REFERENCE, stage["files"], stage["state"]
    # Same robot input across the real opening views. Visual scenes also differ
    # in illumination, wrist viewpoints and carton contents.
    state = REFERENCE_STAGES[0]["state"]
    for label, rollout in [
        ("green-left", "20261124-131231-115913-eggpack"),
        ("green-right", "20261123-172601-933674-eggpack"),
        ("pink-right", "20261124-132312-188537-eggpack"),
    ]:
        yield label, rollout, opening_views(rollout), state


class Probe:
    def __init__(self, arch):
        self.arch = arch
        self.driver = arch._mot_driver
        assert self.driver is not None and not self.driver.detach_video
        self.mixed = self.driver._mixed_attention
        self.prepare = arch.video_backbone.prepare
        self.post = arch.action_backbone.post_attn_at_layer
        self.driver._mixed_attention = self.attention
        arch.video_backbone.prepare = self.new_pass
        arch.action_backbone.post_attn_at_layer = self.post_attention

    def reset(self, trace=False, blocked=()):
        self.trace = trace
        self.blocked = blocked
        self.weights = []
        self.near_weights = []
        self.updates = []
        self.grid = None
        self.passes = 0

    def new_pass(self, **kwargs):
        state = self.prepare(**kwargs)
        self.grid = tuple(int(getattr(state, "grid_" + axis)) for axis in ("frames", "height", "width"))
        self.passes += 1
        return state

    def attention(self, q, k, v, mask):
        frames, height, width = self.grid
        patches = height * width
        num_video = frames * patches
        assert mask is not None and mask.dtype == torch.bool and q.shape == k.shape
        if self.blocked:
            mask = mask.clone()
            mask[..., num_video:, list(self.blocked)] = False
        result = self.mixed(q, k, v, mask)
        if self.trace:
            heads = self.driver.num_heads

            def split(tensor):
                return tensor.unflatten(-1, (heads, -1)).transpose(1, 2).float()

            with torch.autocast(q.device.type, enabled=False):
                query = split(q[:, num_video:])
                key = split(k)
                logits = (query @ key.transpose(-1, -2)) / math.sqrt(key.shape[-1])
                logits.masked_fill_(~mask[..., num_video:, :], -torch.inf)
                probs = logits.softmax(-1)
                first = probs[..., :patches].reshape(1, heads, -1, height, width)
                self.weights.append(first.mean((0, 2)).cpu().numpy())
                self.near_weights.append(first[:, :, :8].mean((0, 2)).cpu().numpy())
                # Measure the projected, gated term for the corner and a
                # same-row background patch, and the local effect of deleting
                # each connection with softmax renormalization.
                token_ids = [0, width - 1]
                p = probs[..., token_ids].unsqueeze(-1)
                selected_v = split(v)[..., token_ids, :].unsqueeze(2)
                full = split(result[:, num_video:]).unsqueeze(-2)
                direct = p * selected_v
                delta = p * (full - selected_v) / (1 - p).clamp_min(1e-7)
                self.terms = torch.cat((direct, delta), dim=-2).permute(3, 0, 2, 1, 4).flatten(-2)
        return result

    def post_attention(self, layer, state, output, post_state):
        if self.trace:
            projection = post_state["block"].self_attn.o
            with torch.autocast(output.device.type, enabled=False):
                weight = projection.weight.float()
                bias = projection.bias
                if bias is not None:
                    bias = bias.float()
                gate = post_state["gate_msa"].float()
                terms = linear(self.terms, weight) * gate
                full = linear(output.float(), weight, bias) * gate
                residual = post_state["residual_x"].float()
                # Store squared norms to support pooled RMS ratios.
                self.updates.append(torch.cat((
                    terms.square().sum((1, 2, 3)),
                    full.square().sum().reshape(1),
                    residual.square().sum().reshape(1),
                )).cpu().numpy())
        return self.post(layer, state, output, post_state)


def main():
    torch.set_num_threads(12)
    device = torch.device(os.environ.get("WAMU_ANALYSIS_DEVICE", "cpu"))
    run = read_events(REFERENCE)[0]
    checkpoint = Path(os.environ.get("WAMU_ANALYSIS_CHECKPOINT", run["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = _load_deploy_yaml()
    cfg.optimization.compile.enabled = False
    cfg.optimization.dit_cache.enabled = False
    server = build_server_from_config(cfg, str(checkpoint.parent), str(device), checkpoint.name)
    server._init_policy()
    arch = server.engine.architecture
    probe = Probe(arch)
    print("Checkpoint loaded", flush=True)
    metadata = {"checkpoint": str(checkpoint), "prompt": run["prompt"],
                "denoise_steps": cfg.inference.denoise_steps, "device": str(device), "cases": {}}
    for name, rollout, files, state in policies_cases():
        images = {
            role + "_camera": Image.open(ROLLOUTS / rollout / role / file).convert("RGB")
            for role, file in files.items()
        }
        obs = server._obs_preprocessor.preprocess(dict(images=images, prompt=run["prompt"], state=state))
        obs["image"].save(OUT / f"{name}.png")
        conditions = server._policy._build_conditions(obs)
        case = {"rollout": rollout, "files": files, "state": state, "variants": {}}
        variants = [("baseline", (), 42)]
        if name in {"opening", "early"}:
            variants += [("corner-blocked", (0,), 42), ("background-blocked", (9,), 42)]
        if name in {"opening", "green-left", "green-right", "pink-right"}:
            variants.append(("second-seed", (), 137))
        for variant, blocked, seed in variants:
            stem = f"{name}-{variant}"
            trace = variant == "baseline"
            probe.reset(trace, blocked)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                result = server.engine.generate(dict(conditions, seed=seed))
            actions = result["actions"]
            assert probe.grid == (3, 12, 10)
            assert probe.passes == cfg.inference.denoise_steps
            assert actions.shape[1] == 14 and np.isfinite(actions).all()
            arrays = {"actions": actions}
            if trace:
                layers = arch.video_backbone.num_layers
                assert len(probe.weights) == len(probe.updates) == probe.passes * layers
                arrays["weights"] = np.stack(probe.weights).reshape(probe.passes, layers, *probe.weights[0].shape)
                arrays["first_eight_weights"] = np.stack(probe.near_weights).reshape(arrays["weights"].shape)
                arrays["update_squared_norms"] = np.stack(probe.updates).reshape(probe.passes, layers, 6)
                assert all(np.isfinite(a).all() for a in arrays.values())
            np.savez_compressed(OUT / f"{stem}.npz", **arrays)
            case["variants"][variant] = dict(seed=seed, blocked=list(blocked), result=f"{stem}.npz")
            metadata["cases"][name] = case
            (OUT / "runs.json").write_text(json.dumps(metadata, indent=2))
            print(f"Completed {stem}", flush=True)


if __name__ == "__main__":
    main()
