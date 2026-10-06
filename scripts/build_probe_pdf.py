#!/usr/bin/env python
"""Printable PDF of the downstream probe results (A4 landscape), from the same CSVs as the HTML report.

Per metric: one page of per-layer curves (Agilex, Franka) and one per-layer table per embodiment
(rows = models, columns = layers, row-wise shading, best layer boxed, plus the spread across the 7
OpenWAM architectures at each layer). Safe to run while probes are still going; unfinished layers are blank.

  python scripts/build_probe_pdf.py            -> reports/openwam_downstream_probes.pdf
"""
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_probe_report import N_LAYERS, ROOT, WAM, collect

OUT = ROOT / 'reports/openwam_downstream_probes.pdf'
A4 = (11.69, 8.27)
SHORT = {'idm': 'idm', 'joint_cross_attention': 'cross_attn', 'joint_cross_attention_detach': 'cross_attn_detach',
         'joint_self_attention': 'self_attn', 'joint_self_attention_isolated': 'self_attn_isolated',
         'joint_self_attention_mutual': 'self_attn_mutual', 'joint_self_attention_video_sees_action': 'self_attn_video_sees_action',
         'videomaev2': 'VideoMAEv2-L', 'dinov2': 'DINOv2-L', 'd4rt': 'D4RT'}
REFS = ['videomaev2', 'dinov2', 'd4rt']
EMB = {'agilex': 'Agilex (50 tasks)', 'franka': 'Franka (46 tasks)'}
# (task, metric, title, lower_is_better, decimals)
METRICS = [
    ('cls', 'acc', 'Task classification accuracy', False, 3),
    ('depth', 'absrel', 'Depth AbsRel', True, 4),
    ('depth', 'pairwise', 'Depth pairwise ordinal accuracy', False, 3),
    ('seg', 'iou', 'Target-object segmentation IoU', False, 3),
]
DEFS = {
    'acc': 'Share of test episodes whose task the probe names correctly from one layer (chance 0.020 Agilex, 0.022 Franka). '
           'One-block attention read-out over all tokens, 10 epochs, 5 training episodes per task.',
    'absrel': 'Mean |predicted - true| / true depth per pixel; 0.05 = depth off by 5% on average. Lower is better. '
              'Linear head per token upsampled to 120 x 160, 60 epochs, aligned frames 4..32.',
    'pairwise': 'For 200 random pixel pairs per frame whose true depths differ by >= 10%, the share where the probe gets '
                '"which is closer" right (DA-2K style). 0.5 is chance.',
    'iou': 'Overlap / union between the predicted and true mask of the manipulated object (covers ~1-2% of the image), '
           'averaged over frames where it is visible. Linear head per token, class-balanced loss, 60 epochs.',
}
WAM_COLORS = ['#1f77b4', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
REF_STYLE = {'videomaev2': ('#ff7f0e', '-'), 'dinov2': ('#2ca02c', '-'), 'd4rt': ('#000000', '--')}


def series(data, task, emb, model, metric):
    c = data[task].get(emb, {}).get(model, {}).get(metric)
    return c if c and any(v is not None for v in c) else None


def best(c, lower):
    idx = [i for i, v in enumerate(c) if v is not None]
    return min(idx, key=lambda i: c[i]) if lower else max(idx, key=lambda i: c[i])


def lmean(c, task, model):
    v = [x for i, x in enumerate(c) if x is not None and not (task == 'cls' and model == 'dinov2' and i == 0)]
    return sum(v) / len(v)


def cover(pdf, data):
    fig = plt.figure(figsize=A4)
    fig.text(0.06, 0.92, 'Downstream probes on OpenWAM representations', fontsize=22, weight='bold')
    p = data['progress']
    fig.text(0.06, 0.875, f'Generated {time.strftime("%Y-%m-%d %H:%M")}  ·  RoboTwin replayed GT, Agilex 2,500 / Franka 2,300 episodes  ·  '
             f'dense probe runs {p["done"]}/{p["total"]}' + ('' if p['done'] >= p['total'] else ' (in progress)'), fontsize=10, color='#555')
    lines = ['Primary metric: mean over all layers (robust to seed noise, no best-layer selection); sum30 = sum over the 30 OpenWAM layers; best layer in brackets.',
             'OpenWAM entry = the architecture with the best all-layer mean; "spread" = range of all-layer means across the 7.', '']
    for task, metric, title, lower, dec in METRICS:
        lines.append(title + (' (lower is better)' if lower else ''))
        for emb in ('agilex', 'franka'):
            parts = []
            wam = [(m, series(data, task, emb, m, metric)) for m in WAM]
            wam = [(m, c) for m, c in wam if c]
            if wam:
                m, c = (min if lower else max)(wam, key=lambda t: lmean(t[1], task, t[0]))
                b = best(c, lower)
                means = [lmean(c2, task, m2) for m2, c2 in wam]
                parts.append(f'OpenWAM {lmean(c, task, m):.{dec}f} (sum30 {lmean(c, task, m) * 30:.{dec - 1}f}) [{c[b]:.{dec}f} L{b}] ({SHORT[m]}; spread {max(means) - min(means):.{dec}f})')
            for r in REFS:
                c = series(data, task, emb, r, metric)
                if c:
                    b = best(c, lower)
                    parts.append(f'{SHORT[r]} {lmean(c, task, r):.{dec}f}' + (f' [{c[b]:.{dec}f} L{b}]' if N_LAYERS[r] > 1 else ''))
            lines.append(f'    {emb.capitalize():7s} ' + ('  |  '.join(parts) if parts else 'not run yet'))
        lines.append('')
    lines.append('Scale: absolute values are low for every model, specialist encoders included (5 training episodes per task, coarse token grids). '
                 'Read architecture gaps against the span the three references occupy: at the best layer the 7 architectures spread over '
                 '1-12% of that span (largest on classification and segmentation, smallest on depth pairwise, which saturates above 0.9).')
    lines.append('')
    lines += ['Protocol: every probe reads the full, unpooled token grid of one layer; trained on the 10% split (5 episodes per task), '
              'scored on the remaining 90%; one seed.',
              'Tokens per layer: OpenWAM 9 latents x 8 x 10 head tokens x 3072; VideoMAEv2 16 tubelets x 14 x 14 x 1024; '
              'D4RT 16 x 16 x 16 x 1280 (final layer); DINOv2 patch tokens (dense) / per-frame CLS (classification).',
              'Single seed: architecture orderings are not stable across robots and should not be ranked without bootstrap intervals.']
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 150, subsequent_indent='      ') or ['']
    fig.text(0.06, 0.82, '\n'.join(wrapped), fontsize=8.6, va='top', family='DejaVu Sans Mono', linespacing=1.35)
    pdf.savefig(fig); plt.close(fig)


def curves(pdf, data, task, metric, title, lower, dec):
    fig, axes = plt.subplots(1, 2, figsize=A4)
    fig.suptitle(title + (' (lower is better)' if lower else '') + ' by layer', fontsize=15, weight='bold', x=0.06, ha='left')
    fig.text(0.06, 0.905, DEFS[metric], fontsize=8.5, color='#444', wrap=True)
    for ax, emb in zip(axes, ('agilex', 'franka')):
        for m, col in zip(WAM, WAM_COLORS):
            c = series(data, task, emb, m, metric)
            if c:
                xs = [i / 29 * 100 for i, v in enumerate(c) if v is not None]
                ax.plot(xs, [v for v in c if v is not None], color=col, lw=1.2, marker='o', ms=2, label=SHORT[m])
        for r in REFS:
            c = series(data, task, emb, r, metric)
            if not c:
                continue
            col, ls = REF_STYLE[r]
            if N_LAYERS[r] == 1:
                ax.axhline(c[0], color=col, ls=ls, lw=1.4, label=f'{SHORT[r]} (final layer)')
            else:
                pts = [(i / (N_LAYERS[r] - 1) * 100, v) for i, v in enumerate(c) if v is not None and not (task == 'cls' and r == 'dinov2' and i == 0)]
                ax.plot(*zip(*pts), color=col, ls=ls, lw=2.2, label=SHORT[r])
        ax.set_title(EMB[emb], fontsize=11)
        ax.set_xlabel('relative layer depth (%)'); ax.set_ylabel(metric)
        ax.grid(alpha=0.3); ax.set_xlim(-2, 102)
        if lower:
            ax.invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.07, right=0.98, wspace=0.18)
    if lower:
        fig.text(0.07, 0.125, 'y-axis inverted: higher on the chart = better.', fontsize=8, color='#666')
    pdf.savefig(fig); plt.close(fig)


def table(pdf, data, task, metric, title, lower, dec, emb):
    models = WAM + REFS
    rows, labels = [], []
    for m in models:
        c = series(data, task, emb, m, metric)
        rows.append(c or [None] * N_LAYERS[m]); labels.append(SHORT[m])
    wam_rows = [r for r, m in zip(rows, models) if m in WAM]
    spread = [None] * 30
    for l in range(30):
        v = [r[l] for r in wam_rows if r[l] is not None]
        spread[l] = max(v) - min(v) if len(v) == len(WAM) else None
    n_cols = 30
    fig = plt.figure(figsize=A4)
    fig.text(0.03, 0.95, f'{title} per layer: {EMB[emb]}', fontsize=14, weight='bold')
    note = ('Columns are layer indices (OpenWAM 30, VideoMAEv2 24, DINOv2 25, D4RT final only). Shading is per row '
            f'({"darker = lower = better" if lower else "darker = higher = better"}); the boxed cell is that row\'s best layer. '
            'Last row: spread (max - min) across the 7 OpenWAM architectures at each layer, i.e. how far apart the architectures are.')
    fig.text(0.03, 0.925, '\n'.join(textwrap.wrap(note, 190)), fontsize=7.5, color='#444', va='top')
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.80]); ax.axis('off')
    fmt = (lambda v: f'{v * 1000:.1f}') if dec == 4 else (lambda v: f'{v:.3f}'[1:] if v < 1 else f'{v:.2f}')
    head = ['model'] + [f'L{i}' for i in range(n_cols)] + ['best', 'mean']
    cells, colors = [], []
    cmap = plt.get_cmap('Blues')
    for lab, r in zip(labels + ['OpenWAM spread'], rows + [spread]):
        vals = [v for v in r if v is not None]
        line = [lab] + [fmt(r[i]) if i < len(r) and r[i] is not None else '' for i in range(n_cols)]
        cl = ['white'] * (n_cols + 3)
        if vals and lab != 'OpenWAM spread':
            b = best(r, lower)
            line += [f'{fmt(r[b])} L{b}' if len(r) > 1 else fmt(r[b]), fmt(sum(vals) / len(vals))]
            lo, hi = min(vals), max(vals)
            for i in range(len(r)):
                if r[i] is not None and hi > lo:
                    t = (r[i] - lo) / (hi - lo)
                    cl[1 + i] = cmap(0.08 + 0.55 * ((1 - t) if lower else t))
        elif vals:
            line += [f'max {fmt(max(vals))}', fmt(sum(vals) / len(vals))]
        else:
            line += ['', '']
        cells.append(line); colors.append(cl)
    tb = ax.table(cellText=cells, colLabels=head, cellColours=colors, loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(5.8); tb.scale(1, 2.1)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_linewidth(0.2)
        if ci == 0:
            cell.set_width(0.11); cell._loc = 'left'
        elif ci > n_cols:
            cell.set_width(0.047)
        else:
            cell.set_width(0.0255)
        if ri == 0:
            cell.set_text_props(weight='bold')
    for ri, r in enumerate(rows, start=1):
        if any(v is not None for v in r):
            b = best(r, lower)
            cell = tb[ri, 1 + b]; cell.set_linewidth(1.2); cell.set_edgecolor('#c0392b'); cell.set_text_props(weight='bold')
    if dec == 4:
        fig.text(0.03, 0.035, 'AbsRel values shown x1000 (45.2 = 0.0452).', fontsize=7.5, color='#444')
    pdf.savefig(fig); plt.close(fig)


def main():
    data = collect()
    with PdfPages(OUT) as pdf:
        cover(pdf, data)
        for task, metric, title, lower, dec in METRICS:
            curves(pdf, data, task, metric, title, lower, dec)
            for emb in ('agilex', 'franka'):
                table(pdf, data, task, metric, title, lower, dec, emb)
        d = pdf.infodict(); d['Title'] = 'Downstream probes on OpenWAM representations'
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
