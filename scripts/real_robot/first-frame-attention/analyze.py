"""Offline first-denoising-pass attention on the earliest recorded camera triplet."""
from pathlib import Path
import sys
import json
import math
import os
import torch
from PIL import Image

root = Path(__file__).resolve().parents[4]
out = root / "output/first-frame-attention"
out.mkdir(parents=True, exist_ok=True)
os.chdir(root / "WAMU")
sys.path.insert(0, os.getcwd())
from openwam.deploy.server import build_server_from_config, _load_deploy_yaml
from scripts.attention_video_share import install_grid_hook, STATE

roll = root / "output/aloha-ceremony-test/rollouts/20261124-125952-187637-eggpack"
with (roll / "events.jsonl").open() as f:
    run = json.loads(next(f))
ck = Path(run["checkpoint"])
cfg = _load_deploy_yaml()
cfg.optimization.compile.enabled = False
cfg.inference.inference_horizon = run["horizon"]
torch.set_num_threads(12)
srv = build_server_from_config(cfg, str(ck.parent), "cpu", ck.name)
srv._init_policy()
print("Checkpoint and policy loaded", flush=True)
arch = srv.engine.architecture
install_grid_hook(arch, lambda gh: 2 * gh // 3)
drv = arch._mot_driver or arch.build_mot_driver()
mixed = drv._mixed_attention
maps = []
class Captured(Exception):
    pass

def capture(q, k, v, mask):
    frames, rows, cols = STATE["grid"]
    cells = rows * cols
    offset = cells * frames
    heads = drv.num_heads
    def split(x):
        return x.unflatten(-1, (heads, -1)).transpose(1, 2).float()
    logits = split(q[:, offset:]) @ split(k).transpose(-1, -2)
    logits /= math.sqrt(q.shape[-1] // heads)
    if mask is not None:
        logits.masked_fill_(~mask[..., offset:, :], -torch.inf)
    weights = logits.float().softmax(-1)[..., :cells]
    maps.append(weights.mean((0, 1, 2)).reshape(rows, cols).cpu())
    if len(maps) == len(arch.video_backbone.dit.blocks):
        raise Captured
    return mixed(q, k, v, mask)

drv._mixed_attention = capture
roles = ("head", "left_wrist", "right_wrist")
available = {r: sorted((roll/r).glob("*.jpg"), key=lambda p: int(p.stem)) for r in roles}
anchor = max(int(paths[0].stem) for paths in available.values())
files = {r:min(paths, key=lambda p:abs(int(p.stem)-anchor)) for r,paths in available.items()}
# Convert the nearest recorded feedback from ALOHA calibration to training units.
stamp = sum(int(p.stem) for p in files.values()) / len(files) / 1e9
events = [json.loads(line) for line in (roll / "events.jsonl").read_text().splitlines()]
near = min((e for e in events if e["event"] == "feedback"), key=lambda e: abs(e["time"] - stamp))
measured = dict(zip(run["motor_order"], near["measured"], strict=True))
rig = json.loads(Path("/home/King/Desktop/QuestController/webxr_teleop/hardware/rig.json").read_text())["arms"]
state = []
for side, arm in rig.items():
    local = json.loads((Path("/home/King/alohaX_lerobot_fork/.cache/calibration/aloha_solo") / f"{side}_follower.json").read_text())
    train = json.loads(Path(arm["calibration"]).read_text())
    for j, name in enumerate(("waist","shoulder","elbow","forearm_roll","wrist_angle","wrist_rotate","gripper")):
        i = local["motor_names"].index(name)
        t = train["motor_names"].index(name)
        val = measured[f"{side}.{name}"]
        if name == "gripper":
            tick = round(local["start_pos"][i] + val / 100 * (local["end_pos"][i] - local["start_pos"][i]))
            state.append(min(1.,max(0.,(tick-train["start_pos"][t])/(train["end_pos"][t]-train["start_pos"][t]))))
        else:
            tick = round((val * 4096 / 360 - local["homing_offset"][i]) * (-1 if local["drive_mode"][i] else 1))
            deg = (tick * (-1 if train["drive_mode"][t] else 1) + train["homing_offset"][t]) * 360 / 4096
            state.append(math.radians((deg-arm["offset_deg"][j])/arm["sign"][j]))
obs = dict(images={r + "_camera": Image.open(p).convert("RGB") for r,p in files.items()},
           prompt=run["prompt"], state=state)
canvas = srv._obs_preprocessor.preprocess(obs.copy())["image"]
try:
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        srv.predict(obs)
except Captured:
    pass
assert len(maps) == len(arch.video_backbone.dit.blocks)
import numpy as np
from PIL import ImageDraw, ImageFont
raw = torch.stack(maps).float().mean(0)
assert raw.isfinite().all() and raw.sum() > 0
density = (raw / raw.mean()).numpy()
h,w = density.shape
assert h % 3 == 0 and w % 2 == 0
parts = [density[:2*h//3], density[2*h//3:,:w//2], density[2*h//3:,w//2:]]
summary = dict(rollout=roll.name, checkpoint=str(ck), files={r:p.name for r,p in files.items()},
               feedback_time=near["time"], state=state, grid=STATE["grid"], layers=len(maps), observed_frame_mass=float(raw.sum()),
               share=[float(p.sum()/density.sum()) for p in parts],density=[float(p.mean()) for p in parts])
(out/"results.json").write_text(json.dumps(summary, indent=2))
np.save(out/"attention-density.npy",density)
colors = np.array([[34,27,76],[116,49,124],[213,87,76],[251,172,65],[255,246,167]])
def colorize(values):
    t = np.clip(values/density.max(),0,1)*(len(colors)-1)
    i = np.minimum(t.astype(int),len(colors)-2)
    frac = (t-i)[...,None]
    return ((1-frac)*colors[i]+frac*colors[i+1]).astype("uint8")
heat = Image.fromarray(colorize(density)).resize(canvas.size,Image.Resampling.NEAREST)
overlay = Image.blend(canvas.convert("RGB"),heat,.62)
w,h = canvas.size
figure = Image.new("RGB",(2*w+60,h+108),"white")
draw = ImageDraw.Draw(figure)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
for x,title,im in [(20,"Camera input",canvas),(w+40,"First-frame attention",overlay)]:
    draw.text((x,12),title,font=font,fill="black")
    figure.paste(im,(x,40))
bar = np.linspace(0,density.max(),w)[None,:].repeat(10,0)
figure.paste(Image.fromarray(colorize(bar)),(w+40,h+52))
draw.text((w+40,h+67),f"0 to {density.max():.2f} × mean patch attention",font=font,fill="black")
figure.save(out/"attention.png")
print(summary)
