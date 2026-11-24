#!/usr/bin/env python
"""Per-episode Action Attention Dispersion (AAD) and action->video token heatmaps from scripts/aad_episode_dump.py.

AAD as in the paper (Sec. 6, Table 5): H_l = attention entropy of the action queries (mean over heads, action tokens,
windows and action sigmas), Hhat_l = H_l / ln 360, mu = mean_l Hhat_l, s2 = var_l Hhat_l (ddof 1), AAD = mu - s2.
Here the windows are those of one episode (or one window) instead of the whole randomized split.

Heatmaps: per layer the attention over the 360 video keys is renormalised to sum to 1, averaged over layers and action
sigmas, and shown as enrichment (mass / uniform share 1/360; 1 = uniform) on the 3 latent frames of each window
(latent f0 = clean current frame, f1 = video frames 1-4, f2 = frames 5-8; the image shown is the last frame of each).

Outputs in <dir>/figs: <task>_<episode>.png (heatmaps for every window + Hhat_l curves),
<task>_<episode>_layers.png (middle window, maps per band of 5 layers), summary.png, and <dir>/aad_episodes.csv.

usage: python scripts/plot_aad_episodes.py [--dir artifacts/aad_episodes]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [  # Table 5 order (OOD success); display name, OOD %
    ('cross_attn_detach_robotwin_clean', 'Cross-Attn (VG)', 4.56),
    ('cross_attn_clean', 'Cross-Attn', 4.84),
    ('cross_attn_direct', 'Cross-Attn (Direct)', 5.33),
    ('sft_from_scratch_robotwin_clean', 'Self-Attn', 14.1),
    ('sft_robot_only_robotwin_clean', 'Robot-Pretrained', 23.0),
    ('sft_ego_robot_cotrain_robotwin_clean_action_sees_video', 'Robot+Human-Pretrained', 26.0),
]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # categorical slots 1-6, fixed order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LN_K = np.log(360)
FRAME_OF_LATENT = [0, 4, 8]
CMAP = 'RdBu_r'          # diverging: blue (depleted) - white (uniform) - red (enriched)
VMAX = 2.0               # colour saturates at 4x / 0.25x of the uniform share


def aad(h):
    """h: (L,) normalised entropy per layer -> (mu, s2, AAD)."""
    mu, s2 = float(h.mean()), float(h.var(ddof=1))
    return mu, s2, mu - s2


def video_maps(attn):
    """attn (..., L, Sv) raw mass -> per-layer distribution over video keys."""
    a = attn.astype(np.float64)
    return a / a.sum(-1, keepdims=True).clip(1e-12)


def overlay(ax, img, enr, vmax, grid):
    """Dimmed grayscale frame + token-level log2 enrichment (nearest upsampling, no smoothing): red = attended more than
    uniform, blue = less, near-uniform tokens stay transparent.  vmax = |log2| at which the colour saturates."""
    gray = img.astype(np.float32).mean(-1, keepdims=True) / 255.0
    ax.imshow(np.repeat(0.2 + 0.6 * gray, 3, -1), interpolation='bilinear')
    H, W = img.shape[:2]
    x = np.clip(np.log2(np.clip(enr, 1e-6, None)) / vmax, -1, 1)
    rgba = plt.get_cmap(CMAP)((x + 1) / 2)
    rgba[..., 3] = 0.9 * np.abs(x) ** 0.8
    ax.imshow(rgba, extent=(-0.5, W - 0.5, H - 0.5, -0.5), interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def colorbar(fig, cax, vmax):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax)
    ticks = np.arange(-int(vmax), int(vmax) + 1)
    cb.set_ticks(ticks); cb.set_ticklabels([f'{2.0 ** t:g}x' for t in ticks]); cb.ax.tick_params(labelsize=7)
    cb.set_label('attention / uniform (log)', fontsize=8, color=INK2)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=Path, default=Path('artifacts/aad_episodes'))
    a = p.parse_args()
    figs = a.dir / 'figs'; figs.mkdir(parents=True, exist_ok=True)
    fr = np.load(a.dir / 'frames.npz', allow_pickle=True)
    frames, meta = fr['frames'], [tuple(m) for m in fr['meta']]
    D = {}
    for key, name, ood in MODELS:
        f = a.dir / f'{key}.npz'
        if not f.exists():
            print(f'missing {f}; skipped'); continue
        d = np.load(f, allow_pickle=True)
        if [tuple(m) for m in d['meta']] != meta:
            raise SystemExit(f'{key}: windows differ from frames.npz')
        D[key] = dict(name=name, ood=ood, ent=d['entropy'] / LN_K, maps=video_maps(d['attn']), grid=tuple(d['grid']),
                      head_rows=int(d['head_rows']), kind=str(d['kind']))
    keys = [k for k, _, _ in MODELS if k in D]
    color = {k: COLORS[i] for i, (k, _, _) in enumerate(MODELS)}
    Fr, gh, gw = D[keys[0]]['grid']
    episodes = list(dict.fromkeys((m[0], m[2]) for m in meta))

    rows = []
    for k in keys:
        mu, s2, v = aad(D[k]['ent'].mean((0, 1)))
        rows.append(dict(model=D[k]['name'], ood=D[k]['ood'], task='ALL', episode='ALL', window='ALL', mu=mu, s2=s2, aad=v))
    for task, ep in episodes:
        idx = [i for i, m in enumerate(meta) if (m[0], m[2]) == (task, ep)]
        ep_aad = {}
        for k in keys:
            mu, s2, v = aad(D[k]['ent'][idx].mean((0, 1))); ep_aad[k] = (mu, s2, v)
            rows.append(dict(model=D[k]['name'], ood=D[k]['ood'], task=task, episode=ep, window='ALL', mu=mu, s2=s2, aad=v))
            for i in idx:
                mu_w, s2_w, v_w = aad(D[k]['ent'][i].mean(0))
                rows.append(dict(model=D[k]['name'], ood=D[k]['ood'], task=task, episode=ep, window=int(meta[i][3]), mu=mu_w, s2=s2_w, aad=v_w))

        # ---- figure 1: heatmaps for every window x latent frame, then Hhat_l per layer
        enr = {k: D[k]['maps'][idx].mean((1, 2)).reshape(len(idx), Fr, gh, gw) * (Fr * gh * gw) for k in keys}   # (nw, F, gh, gw)
        nw, nm = len(idx), len(keys)
        img_h, img_w = frames.shape[2:4]
        cell = 1.25
        fig_h = cell * nm + 3.6
        fig = plt.figure(figsize=(1.9 + cell * nw * Fr * img_w / img_h + 0.6, fig_h))
        gs = fig.add_gridspec(nm + 1, nw * Fr + 1, height_ratios=[1] * nm + [2.0], width_ratios=[1] * (nw * Fr) + [0.08],
                              left=0.13, right=0.93, top=1 - 0.95 / fig_h, bottom=0.11, hspace=0.12, wspace=0.04)
        for r, k in enumerate(keys):
            for wj, i in enumerate(idx):
                for f in range(Fr):
                    ax = fig.add_subplot(gs[r, wj * Fr + f])
                    overlay(ax, frames[i, FRAME_OF_LATENT[f]], np.kron(enr[k][wj, f], np.ones((img_h // gh, img_w // gw))), VMAX, (gh, gw))
                    if r == 0:
                        ax.set_title((f'window @{meta[i][3]}\n' if f == 1 else '\n') + f'f{f}' + (' (current)' if f == 0 else ''),
                                     fontsize=8, color=INK)
                    if f == 0:
                        ax.text(0.03, 0.03, f"{aad(D[k]['ent'][i].mean(0))[2]:.3f}", transform=ax.transAxes, ha='left', va='bottom',
                                fontsize=6.5, color=INK, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
                    if wj == 0 and f == 0:
                        mu, s2, v = ep_aad[k]
                        ax.text(-0.08, 0.5, f"{D[k]['name']}\nAAD {v:.3f}\n(OOD {D[k]['ood']:.1f}%)", transform=ax.transAxes,
                                ha='right', va='center', fontsize=8, color=INK)
        colorbar(fig, fig.add_subplot(gs[:nm, -1]), VMAX)
        ax = fig.add_subplot(gs[nm, :nw * Fr])
        for k in keys:
            h = D[k]['ent'][idx].mean((0, 1))
            ax.plot(np.arange(len(h)), h, color=color[k], lw=2, marker='o', ms=3.5, label=f"{D[k]['name']} (AAD {ep_aad[k][2]:.3f})")
        style(ax); ax.set_xlabel('layer', fontsize=9, color=INK2); ax.set_ylabel(r'$\hat{\mathcal{H}}_\ell = \mathcal{H}_\ell / \ln 360$', fontsize=9, color=INK2)
        ax.legend(fontsize=7.5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22))
        fig.suptitle(f'{task} / {ep} (randomized): action-to-video attention, mean over layers and action sigmas\n'
                     'row label: episode AAD; corner of each f0: that window\'s AAD', fontsize=10, color=INK)
        fig.savefig(figs / f'{task}_{ep}.png', dpi=150); plt.close(fig)

        # ---- figure 2: middle window, maps per band of 5 layers (3 latent frames side by side in each cell)
        i = idx[len(idx) // 2]; wj = len(idx) // 2
        L = D[keys[0]]['ent'].shape[-1]; bands = [(b, min(b + 5, L)) for b in range(0, L, 5)]
        band_enr = {k: np.stack([D[k]['maps'][i, :, b0:b1].mean((0, 1)).reshape(Fr, gh, gw) * (Fr * gh * gw) for b0, b1 in bands]) for k in keys}
        strip = np.concatenate([frames[i, FRAME_OF_LATENT[f]] for f in range(Fr)], axis=1)
        fig_h = 1.05 * nm + 0.9
        fig, axes = plt.subplots(nm, len(bands), figsize=(1.9 + 2.3 * len(bands), fig_h), squeeze=False)
        fig.subplots_adjust(left=0.11, right=0.93, top=1 - 0.6 / fig_h, bottom=0.03, hspace=0.08, wspace=0.05)
        for r, k in enumerate(keys):
            for c, (b0, b1) in enumerate(bands):
                e = np.concatenate([band_enr[k][c, f] for f in range(Fr)], axis=1)
                hb = D[k]['ent'][i, :, b0:b1].mean()
                overlay(axes[r, c], strip, np.kron(e, np.ones((img_h // gh, img_w // gw))), VMAX, (gh, Fr * gw))
                axes[r, c].text(0.98, 0.03, f'Ĥ {hb:.2f}', transform=axes[r, c].transAxes, ha='right', va='bottom', fontsize=7, color=INK,
                                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
                if r == 0:
                    axes[r, c].set_title(f'layers {b0}-{b1 - 1}', fontsize=8, color=INK)
            axes[r, 0].text(-0.06, 0.5, D[k]['name'], transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=8, color=INK)
        cax = fig.add_axes([0.945, 0.1, 0.008, 0.7])
        colorbar(fig, cax, VMAX)
        fig.suptitle(f'{task} / {ep}, window @{meta[i][3]}: maps per layer band (f0 | f1 | f2 in each cell; Ĥ = band mean normalised entropy)',
                     fontsize=10, color=INK)
        fig.savefig(figs / f'{task}_{ep}_layers.png', dpi=120); plt.close(fig)

    with open(a.dir / 'aad_episodes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- summary: per-episode AAD per model (dots) and the pooled value (bar end marker)
    ep_rows = [r for r in rows if r['episode'] != 'ALL' and r['window'] == 'ALL']
    tasks = list(dict.fromkeys(t for t, _ in episodes))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for xi, k in enumerate(keys):
        name = D[k]['name']
        for ti, t in enumerate(tasks):
            ys = [r['aad'] for r in ep_rows if r['model'] == name and r['task'] == t]
            xs = xi + (ti - (len(tasks) - 1) / 2) * 0.12 + np.zeros(len(ys))
            ax.scatter(xs, ys, s=34, marker=markers[ti % len(markers)], color=color[k], edgecolor='white', linewidth=1.2, zorder=3,
                       label=t if xi == 0 else None)
        pooled = next(r['aad'] for r in rows if r['model'] == name and r['task'] == 'ALL')
        ax.hlines(pooled, xi - 0.3, xi + 0.3, color=INK, lw=2, zorder=4)
    style(ax)
    ax.set_xticks(range(len(keys)), [f"{D[k]['name']}\nOOD {D[k]['ood']:.1f}%" for k in keys], fontsize=8)
    ax.set_ylabel('AAD (per episode)', fontsize=9, color=INK2)
    leg = ax.legend(title='task (marker); black bar = all selected windows', fontsize=7.5, title_fontsize=7.5, frameon=False, loc='upper left')
    for h in getattr(leg, 'legend_handles', None) or leg.legendHandles:
        h.set_color(INK2)
    fig.tight_layout(); fig.savefig(figs / 'summary.png', dpi=140); plt.close(fig)

    # index.html next to the figures: episode x model AAD table, then every figure
    def cell(name, task, ep):
        return next(r['aad'] for r in rows if r['model'] == name and r['task'] == task and r['episode'] == ep and r['window'] == 'ALL')
    head = ''.join(f"<th>{D[k]['name']}<br><small>OOD {D[k]['ood']:.1f}%</small></th>" for k in keys)
    body = ''.join(f'<tr><td><a href="#{t}_{e}">{t} / {e}</a></td>' + ''.join(f"<td>{cell(D[k]['name'], t, e):.3f}</td>" for k in keys) + '</tr>'
                   for t, e in episodes)
    body += '<tr class="all"><td>all selected windows</td>' + ''.join(f"<td>{cell(D[k]['name'], 'ALL', 'ALL'):.3f}</td>" for k in keys) + '</tr>'
    sections = ''.join(f'<h2 id="{t}_{e}">{t} / {e}</h2><img src="figs/{t}_{e}.png"><img src="figs/{t}_{e}_layers.png">' for t, e in episodes)
    (a.dir / 'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>AAD per episode</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#0b0b0b;background:#fcfcfb;max-width:1500px}} img{{max-width:100%;display:block;margin:8px 0 24px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{padding:4px 10px;border-bottom:1px solid #e4e3df;text-align:right}} td:first-child{{text-align:left}}
tr.all td{{font-weight:600}} p{{color:#52514e;max-width:900px}}</style></head><body>
<h1>Action Attention Dispersion per episode (RoboTwin randomized)</h1>
<p>AAD = mu - s2 of the per-layer attention entropy of the action queries normalised by ln 360 (paper Eq. 7-9), averaged over the
{len(meta) // len(episodes)} windows of each episode (start / middle / end) and action sigmas {{0.25, 0.5, 0.75}}, video sigma 0.5.
Heatmaps: attention over the 360 video tokens (3 latent frames x 12 x 10), renormalised per layer and averaged over layers and sigmas,
shown as enrichment over the uniform share (red = more, blue = less).</p>
<img src="figs/summary.png" style="max-width:820px"><table><tr><th>task / episode</th>{head}</tr>{body}</table>{sections}</body></html>''')

    # console table: episode x model
    print(f"{'task/episode':38s}" + ''.join(f"{D[k]['name'][:14]:>16s}" for k in keys))
    for task, ep in episodes:
        print(f'{task + "/" + ep:38s}' + ''.join(f"{next(r['aad'] for r in ep_rows if r['model'] == D[k]['name'] and r['task'] == task and r['episode'] == ep):16.3f}" for k in keys))
    print(f"{'ALL selected windows':38s}" + ''.join(f"{next(r['aad'] for r in rows if r['model'] == D[k]['name'] and r['task'] == 'ALL'):16.3f}" for k in keys))
    print(f'figures -> {figs}; table -> {a.dir / "aad_episodes.csv"}')


if __name__ == '__main__':
    main()
