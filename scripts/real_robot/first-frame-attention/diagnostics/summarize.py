"""Summarize WAMU traces and convert predicted joint targets to gripper paths."""
import json
from pathlib import Path

import mujoco
import numpy as np

OUT = Path(__file__).resolve().parents[5] / "output/first-frame-attention/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
QUEST = Path("/home/King/Desktop/QuestController")
SIDES = ("left", "right")
JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
ARM_JOINTS = [*range(6), *range(7, 13)]
# Observed 32 px cell containing the egg center, checked against the head image.
EGG_PATCH = {"opening": (3, 3), "early": (3, 3), "green-left": (3, 3),
             "green-right": (4, 6), "pink-right": (4, 6)}


def trace_summary(data, name):
    weights = data["weights"].astype(np.float64)
    flat = weights.reshape(*weights.shape[:3], -1)
    patch_mass = flat.sum(axis=-1)
    assert np.all((patch_mass > 0) & (patch_mass <= 1))
    pooled = flat.mean((1, 2))
    near = data["first_eight_weights"].mean((1, 2))
    norms = data["update_squared_norms"].astype(np.float64)
    norm_ratios = np.sqrt(norms[..., :4].sum((0, 1)) / norms[..., 4].sum())
    residual_ratios = np.sqrt(norms[..., :4].sum((0, 1)) / norms[..., 5].sum())
    summary = {
        "corner_percent_by_step": (100 * pooled[:, 0] / pooled.sum(axis=-1)).tolist(),
        "corner_percent_first_eight_by_step": (100 * near[:, 0, 0] / near.sum((-2, -1))).tolist(),
        "observed_frame_percent_by_step": (100 * pooled.sum(axis=-1)).tolist(),
        "corner_is_peak_at_steps": int((pooled.argmax(axis=-1) == 0).sum()),
        "layer_head_corner_peak_rate_by_step": (flat.argmax(axis=-1) == 0).mean((1, 2)).tolist(),
        "head_corner_peak_rates": (flat.argmax(axis=-1) == 0).mean((0, 1)).tolist(),
        "layer_corner_share": (flat[..., 0].sum((0, 2)) / patch_mass.sum((0, 2))).tolist(),
        "gated_update_rms_ratios": dict(zip(("corner_term", "background_term", "corner_deletion",
                                            "background_deletion"), norm_ratios.tolist())),
        "residual_rms_ratios": dict(zip(("corner_term", "background_term", "corner_deletion",
                                        "background_deletion"), residual_ratios.tolist())),
    }
    if name in EGG_PATCH:
        y, x = EGG_PATCH[name]
        index = y * weights.shape[-1] + x
        egg = pooled[:, index]
        summary["egg_patch"] = [y, x]
        summary["egg_percent_by_step"] = (100 * egg / pooled.sum(axis=-1)).tolist()
        summary["egg_patch_rank_by_step"] = (1 + (pooled > egg[:, None]).sum(axis=-1)).tolist()
        summary["egg_is_peak_fraction_layer_heads_by_step"] = (flat.argmax(-1) == index).mean((1, 2)).tolist()
    return summary


class Grippers:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(QUEST / "scenes/dual_viperx_scene.xml"))
        rig = json.loads((QUEST / "webxr_teleop/hardware/rig.json").read_text())
        for side in SIDES:
            base = self.model.body(f"{side}/base_link")
            base.pos[:] = rig["arms"][side]["base_position"]
            base.quat[:] = rig["arms"][side]["base_quaternion_wxyz"]
        self.data = mujoco.MjData(self.model)
        self.ids = [int(self.model.joint(f"{side}/{j}").qposadr[0]) for side in SIDES for j in JOINTS]

    def path(self, actions):
        pts = []
        for row in actions:
            self.data.qpos[self.ids] = row[ARM_JOINTS]
            mujoco.mj_kinematics(self.model, self.data)
            pts.append([self.data.site(f"{side}/gripper").xpos.copy() for side in SIDES])
        return np.asarray(pts) * 100


def analyze():
    manifest = json.loads((OUT / "runs.json").read_text())
    summary = {"checkpoint": manifest["checkpoint"], "denoise_steps": manifest["denoise_steps"],
               "trace": {}, "ablation": {}, "reach": {}}
    grippers = Grippers()
    for name, case in manifest["cases"].items():
        state = np.asarray(case["state"])
        start = grippers.path(state[None])[0]
        with np.load(OUT / case["variants"]["baseline"]["result"]) as baseline:
            summary["trace"][name] = trace_summary(baseline, name)
            original = baseline["actions"]
        original_path = grippers.path(original)
        summary["reach"][name] = {}
        for variant, item in case["variants"].items():
            with np.load(OUT / item["result"]) as data:
                actions = data["actions"]
            path = grippers.path(actions)
            motion = path - start
            summary["reach"][name][variant] = {
                "seed": item["seed"], "start_cm": start.tolist(), "path_cm": path.tolist(),
                "delta_8_cm": motion[7].tolist(), "delta_32_cm": motion[-1].tolist(),
                "max_displacement_cm": np.linalg.norm(motion, axis=-1).max(axis=0).tolist(),
            }
            if item["blocked"]:
                dq = np.rad2deg(actions[:, ARM_JOINTS] - original[:, ARM_JOINTS])
                dp = np.linalg.norm(path - original_path, axis=-1)
                result = {
                    "joint_rmse_deg_8": float(np.sqrt(np.mean(dq[:8] ** 2))),
                    "joint_rmse_deg_32": float(np.sqrt(np.mean(dq ** 2))),
                    "gripper_delta_cm_at_8": dp[7].tolist(),
                    "gripper_delta_cm_at_32": dp[-1].tolist(),
                    "gripper_peak_delta_cm": dp.max(axis=0).tolist(),
                }
                summary["ablation"].setdefault(name, {})[variant] = result
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    analyze()


if __name__ == "__main__":
    main()
