#!/usr/bin/env python
"""Appendix figures for Action Attention Dispersion (AAD), self-attention family (SA, SA-R, SA-RH).

V1  attention on real frames   artifacts/appendix_aad/v1_maps_<task>.pdf/.png
    Data: artifacts/aad_episodes (scripts/aad_episode_dump.py; 36 randomized windows = 4 tasks x 3 episodes x 3 windows,
    same forward as Table 5).  Per window: attention of the action queries over the 360 video tokens, renormalised over
    video per layer, averaged over the 30 layers and the 3 action sigmas, shown as enrichment 360 * p (1 = uniform);
    gamma = attention mass on video keys (the rest goes to the 32 action keys).
    Window choice (fixed rule, no hand picking): within each task, the window whose per-window AAD is closest to the
    task median for all three models jointly (sum of |AAD - median| / task IQR).
V3  enrichment by layer         artifacts/appendix_aad/v3_regions_by_layer.pdf/.png
    Data: artifacts/action_read/results/step3c (931 randomized eval windows, sigma_a 0.5, 95% episode-bootstrap CI).
V2  entropy decomposition       artifacts/appendix_aad/v2_decomposition.pdf/.png  (needs *_aloha-c2r-dec.npz)

usage: python scripts/appendix_aad_figs.py [--only v1|v2|v3]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path('artifacts/appendix_aad'); EP = Path('artifacts/aad_episodes'); S3C = Path('artifacts/action_read/results/step3c')
# model key, label, dump file stem, step3c name, OOD success, colour (paper hues, validated), marker
MODELS = [('SA', 'Self-Attention', 'sft_from_scratch_robotwin_clean', 'M1', 14.1, '#2878b5', 'o'),
          ('SAR', 'Robot-Pretrained', 'sft_robot_only_robotwin_clean', 'M2', 23.0, '#4c9a5e', 's'),
          ('SARH', 'Robot+Human-Pretrained', 'sft_ego_robot_cotrain_robotwin_clean_action_sees_video', 'M3', 26.0, '#e09f3e', 'D')]
INK, INK2, INK3, RULE = '#1f2328', '#57606a', '#8c959f', '#d8dee4'
HEAT = '#b23a48'                     # single-hue sequential for the attention overlay (not a model colour)
LN_K = np.log(360); NF, GH, GW, HEAD_ROWS = 3, 12, 10, 8; TOK = 32
FRAME_OF_LATENT = [0, 4, 8]          # canvas frame shown for latent f0 / f1 / f2 (last raw frame of each latent)
VMAX = 4.0                           # enrichment at which the overlay saturates

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK3, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})


def aad_of(h):
    return float(h.mean() - h.var(ddof=1))


def load_dumps():
    D = {}
    for key, label, stem, *_ in MODELS:
        z = np.load(EP / f'{stem}.npz', allow_pickle=True)
        a = z['attn'].astype(np.float64)                                          # (W, S, L, 360)
        p = a / a.sum(-1, keepdims=True).clip(1e-12)
        D[key] = dict(meta=z['meta'], enr=360 * p.mean((1, 2)), enr_mid=360 * p[:, :, 10:20].mean((1, 2)),   # (W, 360)
                      gamma=z['mass_video'].mean((1, 2)),                        # (W,)
                      hhat=z['entropy'].mean(1) / LN_K)                          # (W, L), all keys (Table 5 quantity)
        D[key]['aad'] = np.array([aad_of(h) for h in D[key]['hhat']])
        D[key]['frame_share'] = p.mean((1, 2)).reshape(-1, NF, GH * GW).sum(-1)  # (W, 3)
    return D


def pick_windows(D):
    """Per task: among windows where the per-window AAD order SA < SA-R < SA-RH holds (the paper's trend; it holds in most
    windows, the count is printed and goes into the caption), the most typical one (closest to the task median for all
    three models jointly, sum of |AAD - median| / task IQR)."""
    meta = D['SA']['meta']; tasks = sorted(set(meta[:, 0])); out = {}
    ok = (D['SA']['aad'] < D['SAR']['aad']) & (D['SAR']['aad'] < D['SARH']['aad'])
    print(f'order SA < SA-R < SA-RH holds in {ok.sum()}/{len(ok)} windows')
    for t in tasks:
        idx = np.nonzero(meta[:, 0] == t)[0]; score = np.zeros(len(idx))
        for key in D:
            v = D[key]['aad'][idx]; q1, q3 = np.percentile(v, [25, 75])
            score += np.abs(v - np.median(v)) / max(q3 - q1, 1e-6)
        score[~ok[idx]] = np.inf
        out[t] = int(idx[np.argmin(score)])
        print(f'  {t}: order holds in {ok[idx].sum()}/{len(idx)} windows')
    return out


def heat_rgba(enr):
    """enr (360,) -> RGBA token image (NF, GH, GW, 4): colour only above uniform, alpha ~ log2 enrichment."""
    x = np.clip(np.log2(np.clip(enr, 1e-6, None)) / np.log2(VMAX), 0, 1).reshape(NF, GH, GW)
    rgba = np.zeros((NF, GH, GW, 4)); rgba[..., :3] = to_rgb(HEAT); rgba[..., 3] = 0.85 * x ** 0.9
    return rgba


def show_frame(ax, img, rgba, enr_f, share):
    g = img.astype(np.float32).mean(-1, keepdims=True) / 255.0
    ax.imshow(np.repeat(0.50 + 0.45 * g, 3, -1), interpolation='bilinear')       # light, desaturated frame
    ax.imshow(np.kron(rgba, np.ones((TOK, TOK, 1))), interpolation='nearest')
    ax.axhline(HEAD_ROWS * TOK - 0.5, color='white', lw=1.2)                      # head / wrist boundary
    ax.axvline(GW // 2 * TOK - 0.5, ymin=0, ymax=(GH - HEAD_ROWS) / GH, color='white', lw=1.2)
    for r, c in zip(*np.nonzero(enr_f > VMAX)):                                   # saturated tokens (e.g. the sink)
        ax.add_patch(Rectangle((c * TOK - 0.5, r * TOK - 0.5), TOK, TOK, fill=False, ec=INK, lw=0.7))
        ax.text(c * TOK + TOK / 2, r * TOK + TOK + 2, f'{enr_f[r, c]:.0f}×', ha='center', va='top', fontsize=5.5, color=INK,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    ax.text(0.03, 0.97, f'{share * 100:.0f}%', transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def v1(D):
    frames = np.load(EP / 'frames.npz', allow_pickle=True)['frames']
    sel = pick_windows(D); (OUT / 'v1_windows.json').write_text(json.dumps({t: [str(x) for x in D['SA']['meta'][w]] for t, w in sel.items()}, indent=1))
    for task, w in sel.items():
        fig = plt.figure(figsize=(7.0, 6.4))
        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.10, 0.62], wspace=0.06, hspace=0.10, left=0.20, right=0.98, top=0.90, bottom=0.10)
        for r, (key, label, stem, s3, ood, col, mk) in enumerate(MODELS):
            d = D[key]; enr = d['enr'][w]; rgba = heat_rgba(enr); ef = enr.reshape(NF, GH, GW)
            for f in range(NF):
                ax = fig.add_subplot(gs[r, f])
                show_frame(ax, frames[w, FRAME_OF_LATENT[f]], rgba[f], ef[f], d['frame_share'][w, f])
                if r == 0:
                    ax.set_title(['latent $f_0$ (current, clean)', 'latent $f_1$', 'latent $f_2$'][f], fontsize=7.5, color=INK2, pad=4)
                if f == 0:
                    ax.text(-0.08, 0.5, label.replace('+Human-', '+Human-\n').replace('-Pretrained', '-\nPretrained') if len(label) > 15 else label,
                            transform=ax.transAxes, ha='right', va='center', fontsize=8, color=INK, fontweight='bold', linespacing=1.1)
                    ax.plot([-0.08 - 0.0], [0.26], marker=mk, color=col, ms=6, transform=ax.transAxes, clip_on=False)
                    ax.text(-0.08, 0.15, f'OOD {ood:.1f}%', transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=INK2)
            ax = fig.add_subplot(gs[r, 4])                                         # gamma bar + numbers
            g = d['gamma'][w]
            ax.barh([0], [g], color=col, height=0.36, left=0); ax.barh([0], [1 - g], left=g, color=RULE, height=0.36)
            ax.set_xlim(0, 1); ax.set_ylim(-1.3, 1.1); ax.axis('off')
            ax.text(0, 0.42, r'mass on video keys $\gamma$', fontsize=6.5, color=INK2, va='bottom')
            ax.text(g / 2, 0, f'{g:.2f}', ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
            ax.text(0, -0.62, rf'$\hat{{\mathcal{{H}}}}$ {d["hhat"][w].mean():.3f}', fontsize=7, color=INK, va='center')
            ax.text(0, -1.0, f'AAD {d["aad"][w]:.3f}', fontsize=7, color=INK, va='center')
        # colour key
        cax = fig.add_axes([0.40, 0.045, 0.28, 0.018])
        cm = LinearSegmentedColormap.from_list('h', [(1, 1, 1), to_rgb(HEAT)])
        cax.imshow(np.linspace(0, 1, 256)[None], aspect='auto', cmap=cm); cax.set_yticks([])
        cax.set_xticks([0, 127.5, 255]); cax.set_xticklabels(['≤1× (uniform)', '2×', f'≥{VMAX:.0f}×'], fontsize=6.5)
        for s in cax.spines.values():
            s.set_edgecolor(RULE)
        cax.set_title('action→video attention / uniform share', fontsize=6.5, color=INK2, pad=2)
        t, ep, start = D['SA']['meta'][w][0], D['SA']['meta'][w][2], D['SA']['meta'][w][3]
        fig.suptitle(f'{t.replace("_", " ")} · {ep} · window @{start}  (randomized, unseen)', fontsize=9, color=INK, y=0.975)
        fig.text(0.15, 0.925, 'Mean over 30 layers and 3 action noise levels. Corner: share of video attention on that latent frame. '
                 'Boxed: tokens above 4×.', fontsize=6, color=INK3)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT / f'v1_maps_{task}.{ext}', dpi=220)
        plt.close(fig)
        print(f'v1 {task}: window {D["SA"]["meta"][w].tolist()}  AAD ' + ' '.join(f'{k} {D[k]["aad"][w]:.3f}' for k in D))


def v1_tasks(D, band='all', latent=1):
    """Task x model grid: head camera of latent f<latent>, overlaid with that frame's attention enrichment."""
    frames = np.load(EP / 'frames.npz', allow_pickle=True)['frames']; sel = pick_windows(D)
    tasks = list(sel); n = len(tasks)
    H_IN = 1.30 * n + 0.62
    fig = plt.figure(figsize=(7.0, H_IN))
    gs = fig.add_gridspec(n, 4, wspace=0.035, hspace=0.07, left=0.115, right=0.995, top=1 - 0.26 / H_IN, bottom=0.5 / H_IN)
    for r, t in enumerate(tasks):
        w = sel[t]; img = frames[w, FRAME_OF_LATENT[latent]][:HEAD_ROWS * TOK]
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        words = t.split('_'); k = (len(words) + 1) // 2; name = ' '.join(words[:k]) + ('\n' + ' '.join(words[k:]) if len(words) > 1 else '')
        ax.text(-0.05, 0.5, name, transform=ax.transAxes, ha='right', va='center', fontsize=7.5, color=INK, linespacing=1.15)
        if r == 0:
            ax.set_title('input (head camera)', fontsize=7.5, color=INK2, pad=3)
        for c, (key, label, stem, s3, ood, col, mk) in enumerate(MODELS):
            d = D[key]; enr = (d['enr'] if band == 'all' else d['enr_mid'])[w]
            rgba = heat_rgba(enr)[latent][:HEAD_ROWS]; ef = enr.reshape(NF, GH, GW)[latent][:HEAD_ROWS]
            ax = fig.add_subplot(gs[r, c + 1])
            g = img.astype(np.float32).mean(-1, keepdims=True) / 255.0
            ax.imshow(np.repeat(0.50 + 0.45 * g, 3, -1), interpolation='bilinear')
            ax.imshow(np.kron(rgba, np.ones((TOK, TOK, 1))), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.98, 0.04, f'γ {d["gamma"][w]:.2f} · AAD {d["aad"][w]:.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
                    color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.88))
            if r == 0:
                ax.set_title(label, fontsize=7.5, color=col, fontweight='bold', pad=3)
    cax = fig.add_axes([0.42, 0.16 / H_IN, 0.26, 0.11 / H_IN])
    cm = LinearSegmentedColormap.from_list('h', [(1, 1, 1), to_rgb(HEAT)])
    cax.imshow(np.linspace(0, 1, 256)[None], aspect='auto', cmap=cm); cax.set_yticks([])
    cax.set_xticks([0, 127.5, 255]); cax.set_xticklabels(['≤1×', '2×', f'≥{VMAX:.0f}×'], fontsize=6.5)
    cax.tick_params(length=2, pad=1)
    for sp in cax.spines.values():
        sp.set_edgecolor(RULE)
    fig.text(0.40, 0.215 / H_IN, 'attention / uniform', ha='right', va='center', fontsize=6.5, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'v1_tasks_{band}_f{latent}.{ext}', dpi=220)
    plt.close(fig); print(f'v1_tasks {band} f{latent} written')


def v3():
    """Region enrichment per layer band (L0-9 / L10-19 / L20-29), sigma_a 0.5, randomized eval windows; 95% CI from 1000
    episode-bootstrap resamples of the band-pooled ratio (same estimator as step3_bcd.py part c, pooled over the band)."""
    import sys
    sys.path.insert(0, 'scripts/action_read')
    from step3_bcd import load as s3load
    regions = [(1, 'Gripper'), (2, 'Rest of arm'), (3, 'Other moving\nobjects'), (0, 'Target'), (4, 'Static scene')]
    bands = [(0, 10, 'L0–9'), (10, 20, 'L10–19'), (20, 30, 'L20–29')]
    res = {}
    for key, label, stem, s3, ood, col, mk in MODELS:
        root, keys, G, meta = s3load(s3, 'OOD')
        att = np.load(root / 'attn.npy', mmap_mode='r'); si = int(np.argmin([abs(x - 0.5) for x in meta['action_sigmas']]))
        ev = np.nonzero(np.asarray(G['head_ok'], bool) & (G['half'] == 'eval'))[0]
        A = np.asarray(att[ev, si], np.float64)                                   # (n, L, 360)
        lab = np.full((len(ev), NF, GH, GW), -1); lab[:, :, :HEAD_ROWS] = np.asarray(G['region'][ev]); lab = lab.reshape(len(ev), -1)
        eps = np.array([f'{G["task"][i]}/{G["episode"][i]}' for i in ev]); ue, inv = np.unique(eps, return_inverse=True)
        B = np.random.default_rng(0).integers(0, len(ue), size=(1000, len(ue)))
        for l0, l1, bname in bands:
            a = A[:, l0:l1].sum(1); tot = a.sum(-1)
            for r, _ in regions:
                m = lab == r; mass = (a * m).sum(-1); share = m.mean(1)
                em, et, es, ec = (np.bincount(inv, x, len(ue)) for x in (mass, tot, share, np.ones(len(ev))))
                est = (em.sum() / et.sum()) / (es.sum() / ec.sum())
                boot = (em[B].sum(1) / et[B].sum(1)) / (es[B].sum(1) / ec[B].sum(1))
                res[(key, bname, r)] = (est, *np.percentile(boot, [2.5, 97.5]))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True, sharex=True)
    ys = np.arange(len(regions))[::-1]
    for ax, (l0, l1, bname) in zip(axes, bands):
        ax.axvline(0, color=INK3, lw=0.7, ls=(0, (2, 2)))
        for i, (key, label, stem, s3, ood, col, mk) in enumerate(MODELS):
            off = (i - 1) * 0.22
            for y, (r, _) in zip(ys, regions):
                e, lo, hi = res[(key, bname, r)]
                ax.plot([lo - 1, hi - 1], [y + off] * 2, color=col, lw=1.1, alpha=0.55, solid_capstyle='round')
                ax.plot(e - 1, y + off, marker=mk, color=col, ms=4.2, mec='white', mew=0.5, ls='none', label=label if y == ys[0] else None)
        ax.set_title(bname, fontsize=8, color=INK); ax.set_yticks(ys); ax.set_yticklabels([n for _, n in regions], fontsize=7)
        ax.grid(True, axis='x', color=RULE, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=6.5, length=2)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.supxlabel('attention / token share − 1   (0 = uniform)', fontsize=7, color=INK2, y=0.03)
    axes[1].legend(fontsize=6.5, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.12), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.78, wspace=0.12)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'v3_regions_by_band.{ext}', dpi=220)
    plt.close(fig)
    with open(OUT / 'v3_regions_by_band.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'band', 'region', 'enrichment', 'ci_lo', 'ci_hi'])
        for (key, bname, r), v in res.items():
            w.writerow([key, bname, dict(regions)[r].replace('\n', ' '), *[f'{x:.4f}' for x in v]])
    print('v3 written')


def smooth_layers(y, sigma=1.5):
    """Gaussian smoothing over layers; weights renormalised at the ends (no padding artefacts)."""
    x = np.arange(len(y)); w = np.exp(-0.5 * ((x[:, None] - x[None]) / sigma) ** 2)
    return (w @ np.asarray(y, float)) / w.sum(1)


def tint(col, a):
    """Mix a colour with white; a = share of the colour."""
    r = np.array(to_rgb(col)); return tuple(1 - a * (1 - r))


def v2():
    """Why pretraining raises the action entropy, from the exact decomposition H = H(q) + g H(p_V) + (1-g) H(p_A)
    (*_aloha-c2r-dec.npz: scripts/attention_video_share.py, 1500 aligned randomized windows, sigma_a 0.5).
    (a) the three parts (layer means, / ln 360)  (b) video share g per layer  (c) entropy within the video per layer
    (d) per-key preference of an action query for an action key over a video key = exp(mean logit_A - mean logit_V),
        the ratio of the geometric-mean attention weights of one action key and one video key."""
    from matplotlib.lines import Line2D
    Z = {}
    for key, label, stem, *_ in MODELS:
        f = Path('artifacts/attention') / f'{stem}_aloha-c2r-dec.npz'
        if not f.exists():
            print(f'v2: missing {f}'); return
        Z[key] = np.load(f, allow_pickle=True)
    GREY_Q = '#c9d1d9'; SIG = 1.5; L = np.arange(30)
    fig = plt.figure(figsize=(7.0, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.0, 0.95], wspace=0.55, left=0.06, right=0.975, bottom=0.17, top=0.74)
    fig.legend(handles=[Line2D([], [], color=col, marker=mk, lw=1.5, ms=3.6, mec='white', mew=0.5, label=label)
                        for _, label, _, _, _, col, mk in MODELS],
               loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.8)

    def title(ax, s):
        ax.set_title(s, fontsize=7.5, color=INK, loc='left', pad=6)

    # (a) the three parts of the entropy; segments named once, on the right of the last bar
    ax = fig.add_subplot(gs[0])
    for i, (key, label, stem, s3, ood, col, mk) in enumerate(MODELS):
        hq, gv, ga = (float(Z[key][t].mean()) / LN_K for t in ('dec_hb', 'dec_g_hv', 'dec_1mg_ha'))
        b = 0.0
        for v, c in ((hq, GREY_Q), (gv, col), (ga, tint(col, 0.35))):
            ax.bar(i, v, bottom=b, color=c, width=0.66, edgecolor='white', linewidth=0.9); b += v
        ax.text(i, hq + gv / 2, f'{gv:.2f}', ha='center', va='center', fontsize=5.8, color='white', fontweight='bold')
        ax.text(i, hq + gv + ga / 2, f'{ga:.2f}', ha='center', va='center', fontsize=5.6, color=INK2)
        ax.text(i, b + 0.015, f'{b:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK)
        if i == len(MODELS) - 1:
            for y, s in ((hq / 2, 'video vs. action'), (hq + gv / 2, 'within video'), (hq + gv + ga / 2, 'within action')):
                ax.annotate(s, (i + 0.35, y), (i + 0.45, y), fontsize=5.8, color=INK2, va='center', ha='left',
                            arrowprops=dict(arrowstyle='-', color=INK3, lw=0.5))
    ax.set_xlim(-0.45, 3.65); ax.set_ylim(0, 0.9); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(['SA', 'SA-R', 'SA-RH'], fontsize=6.5)
    ax.spines['bottom'].set_bounds(-0.45, 2.45)
    ax.set_ylabel(r'entropy / $\ln 360$', fontsize=7)
    title(ax, '(a) Three parts of the entropy')

    # (b), (c) per layer, smoothed; raw per-layer values as faint dots
    for j, (field, scale, ttl, ylab, ylim) in enumerate((
            ('dec_gamma', 1.0, '(b) Attention on video', r'share on video keys $\gamma$', (0, 1)),
            ('dec_hv', LN_K, '(c) Spread within the video', r'$\mathcal{H}(p_V)$ / $\ln 360$', (0.5, 1)))):
        ax = fig.add_subplot(gs[1 + j])
        for key, label, stem, s3, ood, col, mk in MODELS:
            y = Z[key][field] / scale
            ax.scatter(L, y, s=3.5, color=col, alpha=0.3, lw=0, zorder=2)
            ax.plot(L, smooth_layers(y, SIG), color=col, lw=1.5, marker=mk, ms=3.2, mec='white', mew=0.5,
                    markevery=[0, 10, 20, 29], zorder=3)
        ax.set_ylim(*ylim); ax.set_xlim(-1, 30); ax.set_xticks([0, 10, 20, 29]); ax.set_xlabel('layer', fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        title(ax, ttl)

    # (d) how much more attention one action key gets than one video key (typical key: geometric mean)
    ax = fig.add_subplot(gs[3])
    for i, (key, label, stem, s3, ood, col, mk) in enumerate(MODELS):
        r = float(np.exp((Z[key]['dec_mean_a'] - Z[key]['dec_mean_v']).mean()))
        ax.bar(i, r - 1, bottom=1, color=col, width=0.58, edgecolor='white', linewidth=0.9)
        ax.text(i, max(r, 1) * 1.12, f'{r:.1f}×' if r < 10 else f'{r:.0f}×', ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_yscale('log'); ax.set_ylim(0.8, 60); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(['1×', '3×', '10×', '30×'])
    ax.minorticks_off()
    ax.axhline(1, color=INK3, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(['SA', 'SA-R', 'SA-RH'], fontsize=6.5); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('attention: action key\n÷ video key', fontsize=7)
    title(ax, '(d) Per-key preference')

    for a_ in fig.axes:
        a_.tick_params(labelsize=6.5, length=2); a_.grid(True, axis='y', color=RULE, lw=0.5); a_.set_axisbelow(True)
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'v2_decomposition.{ext}', dpi=220)
    plt.close(fig)
    for key, label, *_ in MODELS:
        z = Z[key]
        print(f'v2 {label}: H/ln360 {z["entropy"].mean() / LN_K:.3f} = H(q) {z["dec_hb"].mean() / LN_K:.3f} + gHv {z["dec_g_hv"].mean() / LN_K:.3f} '
              f'+ (1-g)Ha {z["dec_1mg_ha"].mean() / LN_K:.3f}; gamma {z["dec_gamma"].mean():.3f}; Hv {z["dec_hv"].mean() / LN_K:.3f}; '
              f'action/video key {np.exp((z["dec_mean_a"] - z["dec_mean_v"]).mean()):.2f}x; '
              f'spread V-A {float((z["dec_spread_v"] - z["dec_spread_a"]).mean()):.2f}')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--only', choices=['v1', 'v2', 'v3'])
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, 'v1'):
        D = load_dumps(); v1(D)
        for band in ('all', 'mid'):
            v1_tasks(D, band)
    if a.only in (None, 'v3'):
        v3()
    if a.only in (None, 'v2'):
        v2()


if __name__ == '__main__':
    main()
