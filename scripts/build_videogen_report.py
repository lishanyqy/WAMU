#!/usr/bin/env python
"""Report for scripts/video_gen_quality.py: reports/openwam_videogen_<tag>.html (tag = franka | aloha-random)."""
import base64, html, io, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else 'franka'
SRC = ROOT / 'artifacts/videogen'; OUT = ROOT / f'reports/openwam_videogen_{TAG.split("-")[0] if TAG != "aloha-random" else "aloha_random"}.html'
ORDER = ['sft_from_scratch_robotwin_clean', 'cross_attn_clean', 'cross_attn_detach_robotwin_clean', 'cross_attn_clean_fix3072']
NAME = {'sft_from_scratch_robotwin_clean': 'self_attention_30k', 'cross_attn_clean': 'cross_attention_30k', 'cross_attn_detach_robotwin_clean': 'video_generation_30k', 'cross_attn_clean_fix3072': 'cross_attention_kv3072_30k'}
SUCCESS = {'sft_from_scratch_robotwin_clean': 14.0, 'cross_attn_clean': 4.8, 'cross_attn_detach_robotwin_clean': 4.0, 'cross_attn_clean_fix3072': 3.0}
DATA = {'franka': 'RoboTwin Franka clean_50 (another robot: out of distribution)', 'aloha-random': 'RoboTwin Aloha-Agilex randomized_500 (same robot, unseen scenes)'}[TAG]
esc = html.escape
METRICS = [('psnr', 'PSNR (dB) ↑', 2), ('ssim', 'SSIM ↑', 3), ('lpips', 'LPIPS ↓', 3)]


def load():
    D = {}
    for m in ORDER:
        f = SRC / f'{m}_{TAG}.npz'
        if f.exists():
            D[m] = dict(np.load(f, allow_pickle=True))
    return D


def summary(D, region):
    heads = ['Model', 'Success %'] + [lab for _, lab, _ in METRICS] + ['LPIPS vs copy-first-frame']
    rows = []
    ref = next(iter(D.values())) if D else None
    for m in ORDER:
        if m not in D:
            rows.append(f'<tr><td class="mono">{NAME[m]}</td><td class="num">{SUCCESS[m]:.1f}</td><td class="num partial" colspan="4">pending</td></tr>'); continue
        z = D[m]
        cells = [f'<td class="mono">{NAME[m]}</td>', f'<td class="num">{SUCCESS[m]:.1f}</td>']
        for k, _, d in METRICS:
            cells.append(f'<td class="num">{z[f"gen_{k}_{region}"].mean():.{d}f}</td>')
        d_lp = z[f'gen_lpips_{region}'].mean() - z[f'copy_lpips_{region}'].mean()
        cells.append(f'<td class="num">{d_lp:+.3f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    if ref is not None:
        for key, lab in (('copy', 'copy first frame (baseline)'), ('vae', 'VAE round trip of the ground truth (ceiling)')):
            cells = [f'<td><em>{lab}</em></td>', '<td class="num">–</td>'] + [f'<td class="num">{ref[f"{key}_{k}_{region}"].mean():.{d}f}</td>' for k, _, d in METRICS] + ['<td class="num">–</td>']
            rows.append('<tr class="ref">' + ''.join(cells) + '</tr>')
    return '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def per_frame(D, metric, label, d):
    if not D:
        return ''
    T = next(iter(D.values()))[f'gen_{metric}_full'].shape[1]
    heads = ['Model'] + [f'frame {t + 1}' for t in range(T)]
    rows = []
    for m in ORDER:
        if m in D:
            v = D[m][f'gen_{metric}_full'].mean(0)
            rows.append(f'<tr><td class="mono">{NAME[m]}</td>' + ''.join(f'<td class="num">{x:.{d}f}</td>' for x in v) + '</tr>')
    ref = next(iter(D.values()))
    rows.append('<tr class="ref"><td><em>copy first frame</em></td>' + ''.join(f'<td class="num">{x:.{d}f}</td>' for x in ref[f'copy_{metric}_full'].mean(0)) + '</tr>')
    return f'<h3>{esc(label)}, full canvas, by frame</h3><div class="tbl-wrap"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def examples(D):
    from PIL import Image
    parts = []
    for m in ORDER:
        f = SRC / f'{m}_{TAG}_example.npz'
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True); gt, gen = z['gt'], z['gen']; idx = [0, 2, 4, 6, 8][:min(len(gt), len(gen))]
        grid = np.concatenate([np.concatenate([gt[t].transpose(1, 2, 0) for t in idx], 1), np.concatenate([gen[t].transpose(1, 2, 0) for t in idx], 1)], 0)
        im = Image.fromarray(grid.astype(np.uint8)); im.thumbnail((1200, 800)); buf = io.BytesIO(); im.save(buf, format='JPEG', quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode()
        parts.append(f'<figure><img alt="{NAME[m]}: ground truth (top) and generated (bottom), frames {idx}" src="data:image/jpeg;base64,{b64}"><figcaption><span class="mono">{NAME[m]}</span> · {esc(str(z["task"]))} episode {int(z["episode"])}, start {int(z["start"])} · top: ground truth, bottom: generated · frames {", ".join(map(str, idx))}</figcaption></figure>')
    return ''.join(parts)


def main():
    D = load()
    n = len(next(iter(D.values()))['meta']) if D else 0
    steps = int(next(iter(D.values()))['steps']) if D else 0
    body = f'''<section><h2>Summary</h2>
<p>{n} clips per model, the same clips for every model ({DATA}; per task 2 random episodes, one random full window each). Each model generates the 9-frame clip on the 384 × 320 canvas (head camera on top, two wrist cameras below) from the clean first frame, the prompt and the proprio state, with {steps} denoising steps; frames 1–8 are compared with the ground truth. "Copy first frame" predicts frame 0 for every later frame and is the trivial baseline; the VAE round trip is the reconstruction ceiling. The last column is the model's LPIPS minus the baseline's: negative means better than doing nothing.</p>
<h3>Full canvas</h3><div class="tbl-wrap">{summary(D, 'full')}</div>
<h3>Head camera only (top 2/3 of the canvas)</h3><div class="tbl-wrap">{summary(D, 'head')}</div></section>
<section><h2>Drift over the clip</h2>{''.join(per_frame(D, k, lab, d) for k, lab, d in METRICS)}</section>
<section><h2>Example clips</h2>{examples(D)}</section>'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video generation quality</title>
<style>
:root {{ --bg: #f7f8fa; --surface: #fff; --fg: #1b1f2a; --muted: #5d6577; --line: #d9dde5; --soft: #e3e9f8; --partial: #9a6700; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --soft: #232c45; --partial: #e2b53e; }} }}
:root[data-theme="dark"] {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --soft: #232c45; --partial: #e2b53e; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width: 1240px; margin: 0 auto; }} h1 {{ font-size: 1.6rem; margin: 0 0 4px; }} h2 {{ font-size: 1.2rem; margin: 32px 0 8px; border-top: 1px solid var(--line); padding-top: 18px; }} h3 {{ font-size: 0.95rem; margin: 18px 0 6px; color: var(--muted); }}
p {{ max-width: 90ch; }} .meta {{ color: var(--muted); font-size: 0.85rem; }}
.tbl-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; margin: 6px 0 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }} th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ background: var(--soft); font-weight: 600; }} td.num {{ text-align: right; font-variant-numeric: tabular-nums; }} td.partial {{ color: var(--partial); }} tr.ref td {{ color: var(--muted); }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }}
figure {{ margin: 12px 0 20px; }} figure img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; }} figcaption {{ font-size: 0.82rem; color: var(--muted); margin-top: 4px; }}
</style></head><body><main>
<h1>Video generation quality vs ground truth</h1>
<p class="meta">Updated {time.strftime('%Y-%m-%d %H:%M')} · four 30k-step checkpoints trained on Aloha-Agilex clean_50 · {esc(DATA)}</p>
{body}
</main></body></html>'''
    OUT.write_text(page); print('wrote', OUT, len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
