"""Render selected denoising maps and the per-head corner-peak frequency."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[5] / "output/first-frame-attention/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
STAGES = ("opening", "early", "mid", "late")
STEPS = (0, 4, 9, 14, 19)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = np.array([[36, 30, 81], [107, 46, 125], [187, 64, 103], [241, 136, 63], [255, 245, 173]])


def colorize(values):
    t = np.clip(values, 0, 1) * (len(COLORS) - 1)
    index = np.minimum(t.astype(int), len(COLORS) - 2)
    f = (t - index)[..., None]
    rgb = COLORS[index] * (1 - f) + COLORS[index + 1] * f
    return Image.fromarray(rgb.astype(np.uint8))


def label(draw, xy, text, size=16, bold=False, color="#282b35"):
    draw.text(xy, text, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color)


def plot():
    figure = Image.new("RGB", (1236, 1080), "#fcfcfd")
    draw = ImageDraw.Draw(figure)
    label(draw, (26, 18), "Egg packing · WAMU step 20,000", 28, True)
    label(draw, (26, 60), "Action attention to observed camera patches through the denoising schedule", 17)
    x0, y0, panel_w, panel_h = 27, 126, 145, 174
    for i, title in enumerate(("Camera input", *(f"Denoising step {step + 1}" for step in STEPS), "Peak by head")):
        label(draw, (x0 + i * (panel_w + 14), 99), title, 14, True)
    for row, stage in enumerate(STAGES):
        canvas = Image.open(OUT / f"{stage}.png").convert("RGB").resize((panel_w, panel_h))
        data = np.load(OUT / f"{stage}-baseline.npz")["weights"]
        y = y0 + row * 215
        label(draw, (x0, y - 2), stage.title(), 15, True)
        figure.paste(canvas, (x0, y + 20))
        average = data.mean((1, 2))
        density = average / average.mean((-1, -2), keepdims=True)
        for i, step in enumerate(STEPS, 1):
            x = x0 + i * (panel_w + 14)
            colors = colorize(density[step] / 4).resize(canvas.size, Image.Resampling.NEAREST)
            figure.paste(Image.blend(canvas, colors, .60), (x, y + 20))
            percent = average[step, 0, 0] / average[step].sum() * 100
            label(draw, (x, y + 2), f"Corner {percent:.1f}%", 12)
        peak = (data.reshape(*data.shape[:3], -1).argmax(-1) == 0).mean(axis=1).T
        x = x0 + 6 * (panel_w + 14)
        figure.paste(colorize(peak).resize((220, panel_h), Image.Resampling.NEAREST), (x, y + 20))
        label(draw, (x, y + 2), "24 heads × 20 steps", 12)
    ramp = np.linspace(0, 1, 215)[None, :].repeat(12, axis=0)
    for x in (27, 645):
        figure.paste(colorize(ramp), (x, 1013))
    label(draw, (27, 1032), "Map scale  0–4× mean patch weight, clipped", 15)
    label(draw, (645, 1032), "Head scale  0–100% of layers with the corner as peak", 15)
    figure.save(OUT / "denoising-attention.png")


def plot_reaches():
    summary = json.loads((OUT / "summary.json").read_text())
    scenes = ("opening", "green-left", "green-right", "pink-right")
    variants = ("baseline", "second-seed")
    arm_colors = ("#2166ac", "#b74116")
    figure = Image.new("RGB", (1156, 660), "#fcfcfd")
    draw = ImageDraw.Draw(figure)
    label(draw, (22, 16), "Predicted gripper paths from recorded egg scenes", 27, True)
    label(draw, (22, 57), "Same input robot state and prompt. Paths use the rig’s forward kinematics.", 16)
    all_paths = [
        np.concatenate((np.asarray(item["start_cm"])[None], np.asarray(item["path_cm"])))
        for name in scenes for variant in variants
        for item in [summary["reach"][name][variant]]
    ]
    all_points = np.concatenate(all_paths).reshape(-1, 3)
    limits = np.stack((all_points.min(axis=0), all_points.max(axis=0)))[:, :2]
    limits = np.stack((np.floor(limits[0] / 10) * 10 - 5, np.ceil(limits[1] / 10) * 10 + 5))
    for col, name in enumerate(scenes):
        x = 22 + 283 * col
        label(draw, (x, 97), ("Opening · green on left" if name == "opening" else name.replace("-", " ").title()), 16, True)
        canvas = Image.open(OUT / f"{name}.png").convert("RGB")
        figure.paste(canvas.crop((0, 0, 320, 256)).resize((260, 208)), (x, 127))
        left, top, w, h = x + 28, 371, 228, 201
        draw.rectangle((left, top, left + w, top + h), outline="#ccd1db", fill="white")

        def screen(point):
            u, v = (point[:2] - limits[0]) / (limits[1] - limits[0])
            return left + u * w, top + (1 - v) * h

        for tick in np.arange(np.ceil(limits[0, 1] / 10) * 10, limits[1, 1], 10):
            _, y = screen((0, tick))
            draw.line((left, y, left + w, y), fill="#eceef4")
            label(draw, (x + 3, y - 7), f"{tick:.0f}", 11)
        for tick in np.arange(np.ceil(limits[0, 0] / 20) * 20, limits[1, 0], 20):
            u, _ = screen((tick, 0))
            draw.line((u, top, u, top + h), fill="#eceef4")
            label(draw, (u - 8, top + h + 3), f"{tick:.0f}", 11)

        for j, variant in enumerate(variants):
            path = all_paths[col * len(variants) + j]
            for arm, color in enumerate(arm_colors):
                points = [screen(p) for p in path[:, arm]]
                if j == 0:
                    draw.line(points, fill=color, width=3)
                else:
                    for k in range(0, len(points) - 2, 4):
                        draw.line(points[k:k + 3], fill=color, width=2)
                for node, radius, filled in ((0, 4, False), (8, 3, True), (32, 5, j == 0)):
                    u, v = points[node]
                    box = (u - radius, v - radius, u + radius, v + radius)
                    draw.ellipse(box, fill=color if filled else "white", outline=color)
        label(draw, (left, 599), "Lateral position (cm)", 14)
    label(draw, (22, 360), "Forward position (cm)", 13)
    label(draw, (22, 631), "Left arm", 14, True, arm_colors[0])
    label(draw, (103, 631), "Right arm", 14, True, arm_colors[1])
    label(draw, (202, 631), "Solid  seed 42    Dashed  seed 137    Markers  start, action 8, action 32", 14)
    figure.save(OUT / "predicted-reaches.png")

if __name__ == "__main__":
    plot()
    plot_reaches()
