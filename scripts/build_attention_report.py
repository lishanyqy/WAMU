#!/usr/bin/env python
"""Report page for scripts/attention_video_share.py results: reports/openwam_attention.html"""
import html
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_probe_report import SUCCESS, disp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'artifacts/attention'
OUT = ROOT / 'reports/openwam_attention.html'
ORDER = ['joint_self_attention', 'joint_self_attention_mutual', 'joint_self_attention_video_sees_action', 'joint_self_attention_isolated',
         'sft_ego_robot_cotrain_robotwin_full_action_sees_video', 'joint_cross_attention', 'joint_cross_attention_detach']
MODE = {'joint_self_attention': 'action sees all video', 'joint_self_attention_mutual': 'mutual: action sees all video, video sees action',
        'joint_self_attention_video_sees_action': 'action sees first frame only; video sees action', 'joint_self_attention_isolated': 'action sees first frame only',
        'sft_ego_robot_cotrain_robotwin_full_action_sees_video': 'action sees all video (co-trained)', 'joint_cross_attention': 'bridge cross-attention, every layer',
        'joint_cross_attention_detach': 'bridge cross-attention, gradient detached'}
esc = html.escape


def load():
    out = {}
    for f in sorted(SRC.glob('*_franka.npz')):
        z = np.load(f, allow_pickle=True)
        m = str(z['model']).replace('robotwin_dual_system_', '')
        out[m] = {k: z[k] for k in z.files}
    return out


def fmt(x, d=3):
    return f'{x:.{d}f}'


def summary_table(D):
    """Headline: the difference. Per model, how the action tokens' attention splits between video and action."""
    rows = []
    for m in ORDER:
        if m not in D:
            rows.append(f'<tr><td><span class="mono">{esc(disp(m))}</span></td><td>{esc(MODE[m])}</td><td class="num partial" colspan="9">pending</td></tr>'); continue
        z = D[m]; kind = str(z['kind']); mv = z['mass_video_frame'].sum(1); ab = z['abl']; fv = z['frac_video']
        if kind == 'self':
            w_video, w_action = f'{mv.mean():.1%}', f'{z["mass_action"].mean():.1%}'
        else:
            w_video, w_action = 'separate softmax', 'separate softmax'
        cells = [f'<td><span class="mono">{esc(disp(m))}</span></td>', f'<td>{esc(MODE[m])}</td>', f'<td class="num">{SUCCESS.get(m, float("nan")):.2f}</td>',
                 f'<td class="num">{w_video}</td>', f'<td class="num">{w_action}</td>',
                 f'<td class="num"><b>{fv.mean():.1%}</b></td>', f'<td class="num">{1 - fv.mean():.1%}</td>',
                 f'<td class="num">{ab[:, 1].mean():.1%}</td>', f'<td class="num">×{ab[:, 5].mean() / max(ab[:, 4].mean(), 1e-9):.2f}</td>']
        half = ab[np.round(ab[:, 0], 2) == 0.5]
        cells += [f'<td class="num">{half[:, 1].mean():.1%}</td>', f'<td class="num">×{half[:, 5].mean() / max(half[:, 4].mean(), 1e-9):.2f}</td>'] if len(half) else ['<td class="num">–</td>'] * 2
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    heads = ['Model', 'Coupling', 'Success %', 'Attention weight on video', 'Attention weight on action', 'Video share of the attention update (norm)',
             'Action share of the attention update (norm)', 'Velocity change when video is cut (all σ)', 'Flow-matching error without ÷ with video (all σ)',
             'Velocity change, σ = 0.5', 'Error ratio, σ = 0.5']
    return '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def layer_table(D, key, label, d=3, transform=None):
    L = max(len(D[m][key]) for m in D)
    heads = ['Model'] + [f'L{i}' for i in range(L)]
    rows = []
    for m in ORDER:
        if m not in D:
            continue
        v = D[m][key]; v = transform(D[m]) if transform else v
        lo, hi = float(np.min(v)), float(np.max(v))
        cells = [f'<td><span class="mono">{esc(disp(m))}</span></td>']
        for x in v:
            t = (x - lo) / (hi - lo) if hi > lo else 0.5
            cells.append(f'<td class="num" style="background: color-mix(in srgb, var(--accent) {int(8 + 55 * t)}%, transparent)">{x:.{d}f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<h3>{esc(label)}</h3><div class="tbl-wrap"><table class="lt"><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def frame_table(D):
    parts = []
    for m in ORDER:
        if m not in D:
            continue
        z = D[m]; mf = z['mass_video_frame']; Fr = mf.shape[1]
        share = mf / np.maximum(mf.sum(1, keepdims=True), 1e-9)
        heads = ['Layer'] + [f'frame {f}' for f in range(Fr)] + ['video mass']
        rows = []
        for l in range(mf.shape[0]):
            cells = [f'<td>L{l}</td>'] + [f'<td class="num" style="background: color-mix(in srgb, var(--accent) {int(8 + 55 * share[l, f] / max(share[l].max(), 1e-9))}%, transparent)">{share[l, f]:.2f}</td>' for f in range(Fr)] + [f'<td class="num">{mf[l].sum():.3f}</td>']
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        parts.append(f'<details><summary><span class="mono">{esc(disp(m))}</span>: share of video attention per latent frame, by layer</summary><div class="tbl-wrap"><table class="lt"><thead><tr>'
                     + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>')
    return ''.join(parts)


def abl_table(D):
    """Ablation decomposition per action noise level. nmse_abl / nmse = 1 + 2<e,d>/|e|^2 + |d|^2/|e|^2 with e = v - target (with video),
    d = v_ablated - v. Older result files without the decomposition columns show '–' there."""
    parts = []
    sig_levels = sorted({round(float(x), 2) for m in D for x in D[m]['abl'][:, 0]})
    for sig in sig_levels:
        heads = ['Model', 'NMSE with video', 'NMSE without', 'ratio', '2⟨e,d⟩/‖e‖²', '‖d‖²/‖e‖²', 'velocity change: Σ‖Δv‖/Σ‖v‖', 'mean of per-token ratios', 'median of per-token ratios', 'window-level (previous column 8)']
        rows = []
        for m in ORDER:
            if m not in D:
                continue
            z = D[m]; cols = {str(c): i for i, c in enumerate(z['abl_cols'])}; ab = z['abl']; x = ab[np.round(ab[:, 0], 2) == sig]
            if len(x) == 0:
                continue
            nm, na = x[:, cols['nmse']].mean(), x[:, cols['nmse_abl']].mean()
            cells = [f'<td><span class="mono">{esc(disp(m))}</span></td>', f'<td class="num">{nm:.4f}</td>', f'<td class="num">{na:.4f}</td>', f'<td class="num">×{na / max(nm, 1e-9):.2f}</td>']
            if 'ed' in cols:
                e2, d2, ed = x[:, cols['e2']].sum(), x[:, cols['d2']].sum(), x[:, cols['ed']].sum()          # pooled over windows
                cells += [f'<td class="num">{2 * ed / e2:+.3f}</td>', f'<td class="num">{d2 / e2:+.3f}</td>', f'<td class="num">{x[:, cols["vel_sumratio"]].mean():.3f}</td>',
                          f'<td class="num">{x[:, cols["vel_tokmean"]].mean():.3f}</td>', f'<td class="num">{x[:, cols["vel_tokmedian"]].mean():.3f}</td>']
            else:
                cells += ['<td class="num">–</td>'] * 5
            cells.append(f'<td class="num">{x[:, cols["vel_change"]].mean():.3f}</td>')
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        parts.append(f'<h3>Action noise σ = {sig:.2f} ({"low noise: the model mostly refines the GT actions" if sig < 0.5 else "high noise: the model mostly has to guess the actions"})</h3><div class="tbl-wrap"><table><thead><tr>'
                     + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    return ''.join(parts)


def main():
    D = load()
    any_ = next(iter(D.values())) if D else None
    meta = (f'{int(any_["n_windows"])} windows per model (Franka, 2 episodes × 3 windows per task), {int(any_["n_forwards"])} forwards; video sigma {float(any_["video_sigma"]):.2f}; '
            f'action sigmas {", ".join(f"{s:.2f}" for s in any_["action_sigmas"])}; token grid {tuple(int(x) for x in any_["grid"])} (frames × rows × cols), head rows {int(any_["head_rows"])}') if any_ else 'no results yet'
    body = f'''
<section><h2>The difference</h2>
<p>Each row is one checkpoint; every number is a mean over the 32 action tokens, all heads, all 30 layers, all windows and both action noise levels. <b>Attention weight</b> columns read the softmax of the joint self-attention directly: what fraction of an action token's attention lands on video tokens versus on the other action tokens (they sum to 1; the proprio/text token lives in a separate cross-attention and is not part of this split). The cross-attention family runs two separate softmaxes (self-attention over action tokens, bridge attention over video tokens), each summing to 1, so the weight columns do not exist for it; the <b>norm share</b> columns put both families on the same footing: of the residual update the action stream receives from attention in a layer, the part that came from video versus from action tokens (self-attention: video-key part of the mixed attention output after o-projection and gate; cross-attention: bridge update versus self-attention update). The last two columns are causal: the same sample re-run with the action stream cut off from video.</p>
<div class="tbl-wrap">{summary_table(D)}</div>
</section>
<section><h2>Definitions</h2>
<ul>
<li><b>Attention weight on video / action</b> (self-attention family): Σ of the softmax weights an action query places on video keys / on action keys.</li>
<li><b>Video share of the attention update</b>: self-attention ‖gate · o(attention output restricted to video keys)‖ / ‖gate · o(full attention output)‖; cross-attention ‖gate_ca · bridge‖ / (‖gate_ca · bridge‖ + ‖gate_sa · self-attention‖). The action share is one minus this.</li>
<li><b>Velocity change when video is cut</b>: ‖v_ablated − v‖ / ‖v‖ of the predicted flow velocity when action rows cannot attend any video key (self-attention) or the bridge term is dropped (cross-attention).</li>
<li><b>Flow-matching error ratio</b>: NMSE against the training target without video ÷ with video; 1.0 would mean the video stream contributes nothing to the action prediction at these noise levels.</li>
<li>Per-layer values, the split by latent frame and camera region, per-head values and entropies are below.</li>
</ul></section>
<section><h2>Ablation decomposition (why does cutting video change the prediction but not always the error?)</h2>
<p>Write the error with video as e = v − target and the change caused by cutting video as d = v_ablated − v. Then NMSE_without / NMSE_with = 1 + 2⟨e,d⟩/‖e‖² + ‖d‖²/‖e‖². The two terms separate two situations: ⟨e,d⟩ ≈ 0 means the video-driven part of the prediction is orthogonal to the residual error (video changes the prediction along directions the target does not reward or penalise); a negative ⟨e,d⟩ means cutting video partly cancels the error. The absolute NMSE with video is listed so a high baseline is visible. The velocity change is given three ways: norms summed over the 32 action tokens before dividing, the mean of per-token ratios, and the median of per-token ratios, next to the window-level ratio used in the headline table. Everything is split by action noise level.</p>
{abl_table(D)}
</section>
<section><h2>Per layer</h2>
{layer_table(D, 'mass_video_frame', 'Attention mass on video (self-attention: share; cross-attention: 1)', transform=lambda z: z['mass_video_frame'].sum(1))}
{layer_table(D, 'r_video', 'r_video: video-sourced residual update / ‖x_action‖', 4)}
{layer_table(D, 'frac_video', 'Video share of the attention update')}
{layer_table(D, 'entropy', 'Attention entropy over keys (nats)', 2)}
{layer_table(D, 'mass_head', 'Head-camera share of the video mass', transform=lambda z: z['mass_head'] / np.maximum(z['mass_head'] + z['mass_wrist'], 1e-9))}
{layer_table(D, 'mass_video_frame', 'First-frame share of the video mass', transform=lambda z: z['mass_video_frame'][:, 0] / np.maximum(z['mass_video_frame'].sum(1), 1e-9))}
</section>
<section><h2>Which latent frames the action stream looks at</h2>{frame_table(D)}</section>'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action-to-video attention</title>
<style>
:root {{ --bg: #f7f8fa; --surface: #fff; --fg: #1b1f2a; --muted: #5d6577; --line: #d9dde5; --accent: #2c4fa3; --soft: #e3e9f8; --partial: #9a6700; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --accent: #8ea8f0; --soft: #232c45; --partial: #e2b53e; }} }}
:root[data-theme="dark"] {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --accent: #8ea8f0; --soft: #232c45; --partial: #e2b53e; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width: 1240px; margin: 0 auto; }} h1 {{ font-size: 1.6rem; margin: 0 0 4px; }} h2 {{ font-size: 1.2rem; margin: 32px 0 8px; border-top: 1px solid var(--line); padding-top: 18px; }} h3 {{ font-size: 0.95rem; margin: 18px 0 6px; color: var(--muted); }}
p, li {{ max-width: 90ch; }} .meta {{ color: var(--muted); font-size: 0.85rem; }}
.tbl-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; margin: 6px 0 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }} th, td {{ padding: 5px 9px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ background: var(--soft); font-weight: 600; }} td.num {{ text-align: right; font-variant-numeric: tabular-nums; }} td.partial {{ color: var(--partial); }}
table.lt td, table.lt th {{ padding: 3px 6px; font-size: 0.78rem; }} .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }}
details {{ margin: 8px 0; }} summary {{ cursor: pointer; }}
</style></head><body><main>
<h1>How much does the action stream attend to video?</h1>
<p class="meta">Updated {time.strftime('%Y-%m-%d %H:%M')} · {esc(meta)}</p>
<p>Training-faithful forwards on RoboTwin Franka clips (33-step window, 9 video frames at stride 4 → latent frames, 32 action steps, proprio token) through each checkpoint, with the attention of the action stream instrumented. The question: in the joint self-attention family, how much of the action tokens' attention goes to video, and how does the resulting video influence compare with the bridge cross-attention family?</p>
{body}
</main></body></html>'''
    OUT.write_text(page); print('wrote', OUT, len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
