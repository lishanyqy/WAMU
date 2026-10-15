#!/usr/bin/env python
"""Seed-robustness page: the seed-777 re-run (new 10% split + probe seed, each model's top-10 seed-0 layers) against seed 0
on exactly those layers. Per task and robot: Σ10 (metric summed over the 10 planned layers) and best planned layer under
both seeds, ranks among the OpenWAM models, Spearman / Kendall agreement of the two rankings, and the success-rate
correlation under seed 777. Output: reports/openwam_seed777.html
"""
import csv
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_probe_report import EMB_NAME, EMBS, N_LAYERS, REF_NAME, SUCCESS, TASKS, WAM, disp

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/downstream'
OUT = ROOT / 'reports/openwam_seed777.html'
PLAN = {e: {t: {m.replace('robotwin_dual_system_', ''): ls for m, ls in d.items()} for t, d in td.items()}
        for e, td in json.loads((D / 'layer_plan_top10.json').read_text()).items()}      # model names as in build_probe_report (prefix stripped)
DIRS = {'cls': 'task_cls_layers_ep10', 'depth': 'dense_layers_ep60', 'seg': 'dense_layers_ep60', 'flow': 'flow_layers_ep60', 'track': 'flow_layers_ep60'}
PLAN_TASK = {'cls': 'cls', 'depth': 'depth', 'seg': 'seg', 'flow': 'flow', 'track': 'flow'}
TASK_FILTER = {'depth': 'depth', 'seg': 'segmentation'}
REFS_BY_TASK = {'cls': ['videomaev2', 'dinov2', 'd4rt'], 'depth': ['videomaev2', 'dinov2', 'd4rt', 'da3'], 'seg': ['videomaev2', 'dinov2', 'd4rt', 'sam'],
                'flow': ['videomaev2', 'd4rt'], 'track': ['videomaev2', 'd4rt']}
esc = html.escape


def load(task, emb, seed):
    """{model: {layer: value}} of the task's primary metric."""
    d = D / (DIRS[task] + ('' if seed == 0 else f'_seed{seed}')) / emb
    met = TASKS[task][0]
    out = {}
    for f in sorted(d.glob('layers*.csv')) if d.exists() else []:
        if 'zeroshot' in f.name:
            continue
        for r in csv.DictReader(open(f)):
            if task in TASK_FILTER and r.get('task') != TASK_FILTER[task]:
                continue
            if r.get(met, '') == '':
                continue
            out.setdefault(r['model'].replace('robotwin_dual_system_', ''), {})[int(r['layer'])] = float(r[met])
    return out


def agg(vals, layers, lower):
    """(Σ over planned layers, best planned value, best layer) or None until every planned layer exists."""
    if vals is None or any(l not in vals for l in layers):
        return None
    v = [vals[l] for l in layers]
    b = min(range(len(v)), key=lambda i: v[i]) if lower else max(range(len(v)), key=lambda i: v[i])
    return sum(v), v[b], layers[b]


def ranks(d, lower):
    order = sorted(d, key=lambda m: d[m], reverse=not lower)
    return {m: i + 1 for i, m in enumerate(order)}


def td(x, cls='num'):
    return f'<td class="{cls}">{x}</td>'


def task_block(task):
    met, label, lower, dec, _ = TASKS[task]
    parts = [f'<section><h2>{esc(label)} {"↓" if lower else "↑"}</h2>']
    stats = []
    for emb in EMBS:
        s0, s7 = load(task, emb, 0), load(task, emb, 777)
        plan = PLAN[emb][PLAN_TASK[task]]
        rows = []
        a0 = {m: agg(s0.get(m), plan.get(m, []), lower) for m in WAM if m in plan}
        a7 = {m: agg(s7.get(m), plan.get(m, []), lower) for m in WAM if m in plan}
        r0 = ranks({m: a[0] for m, a in a0.items() if a}, lower)
        r7 = ranks({m: a[0] for m, a in a7.items() if a}, lower)
        both = [m for m in WAM if a0.get(m) and a7.get(m)]
        for m in sorted(WAM, key=lambda m: -(SUCCESS.get(m) or -1)):
            if m not in plan:
                continue
            x0, x7 = a0.get(m), a7.get(m)
            n7 = sum(l in s7.get(m, {}) for l in plan[m])
            cells = [f'<td><span class="mono">{esc(disp(m))}</span></td>', td(f'{SUCCESS[m]:.2f}' if m in SUCCESS else 'n/a'),
                     td(f'{x0[0]:.{dec - 1}f} ({r0[m]})' if x0 else '–'),
                     td(f'{x7[0]:.{dec - 1}f} ({r7[m]})' if x7 else f'{n7}/{len(plan[m])} layers', 'num' if x7 else 'num partial'),
                     td(f'{x0[1]:.{dec}f} L{x0[2]}' if x0 else '–'), td(f'{x7[1]:.{dec}f} L{x7[2]}' if x7 else '–'),
                     td(f'{(x7[0] - x0[0]) / x0[0] * 100:+.1f}%' if x0 and x7 else '–')]
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        for r in REFS_BY_TASK[task]:
            if r not in plan:
                continue
            x0, x7 = agg(s0.get(r), plan[r], lower), agg(s7.get(r), plan[r], lower)
            cells = [f'<td><em>{esc(REF_NAME[r])} (reference)</em></td>', td('–'),
                     td(f'{x0[0]:.{dec - 1}f}' if x0 else '–'), td(f'{x7[0]:.{dec - 1}f}' if x7 else '–'),
                     td(f'{x0[1]:.{dec}f} L{x0[2]}' if x0 else '–'), td(f'{x7[1]:.{dec}f} L{x7[2]}' if x7 else '–'),
                     td(f'{(x7[0] - x0[0]) / x0[0] * 100:+.1f}%' if x0 and x7 else '–')]
            rows.append('<tr class="ref">' + ''.join(cells) + '</tr>')
        heads = ['Model', 'Success %', f'Σ10 seed 0 (rank)', f'Σ10 seed 777 (rank)', 'best seed 0', 'best seed 777', 'Σ10 change']
        parts.append(f'<h3>{EMB_NAME[emb]}</h3><div class="tbl-wrap"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
        if len(both) >= 4:
            x = np.array([a0[m][0] for m in both]); y = np.array([a7[m][0] for m in both])
            sp = spearmanr(x, y); kt = kendalltau(x, y)
            succ = [m for m in both if m in SUCCESS]
            sc = spearmanr([a7[m][0] * (-1 if lower else 1) for m in succ], [SUCCESS[m] for m in succ]) if len(succ) >= 4 else None
            spread0 = (x.max() - x.min()) / x.mean() * 100; spread7 = (y.max() - y.min()) / y.mean() * 100
            stats.append(f'<li><b>{EMB_NAME[emb]}</b> ({len(both)} models): rank agreement seed 0 vs 777 Spearman ρ = {sp.correlation:+.2f} (p {sp.pvalue:.2f}), Kendall τ = {kt.correlation:+.2f}; '
                         f'architecture spread of Σ10 = {spread0:.1f}% of mean under seed 0, {spread7:.1f}% under seed 777; '
                         + (f'Σ10 (seed 777) vs success ρ = {sc.correlation:+.2f} (p {sc.pvalue:.2f}).' if sc else '') + '</li>')
        else:
            stats.append(f'<li><b>{EMB_NAME[emb]}</b>: {len(both)} models complete under both seeds; agreement needs 4.</li>')
    parts.append('<h3>Agreement between seeds</h3><ul>' + ''.join(stats) + '</ul></section>')
    return ''.join(parts)


def main():
    body = ''.join(task_block(t) for t in TASKS)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seed robustness check</title>
<style>
:root {{ --bg: #f7f8fa; --surface: #fff; --fg: #1b1f2a; --muted: #5d6577; --line: #d9dde5; --accent: #2c4fa3; --soft: #e3e9f8; --partial: #9a6700; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --accent: #8ea8f0; --soft: #232c45; --partial: #e2b53e; }} }}
:root[data-theme="dark"] {{ --bg: #12151c; --surface: #1a1e27; --fg: #e4e7ee; --muted: #9aa1b0; --line: #2d3340; --accent: #8ea8f0; --soft: #232c45; --partial: #e2b53e; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; }} h1 {{ font-size: 1.6rem; margin: 0 0 4px; }} h2 {{ font-size: 1.2rem; margin: 32px 0 8px; border-top: 1px solid var(--line); padding-top: 18px; }} h3 {{ font-size: 0.95rem; margin: 16px 0 6px; color: var(--muted); }}
.lede, li, p {{ max-width: 80ch; }} .meta {{ color: var(--muted); font-size: 0.85rem; }}
.tbl-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }} th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ background: var(--soft); font-weight: 600; }} td.num {{ text-align: right; font-variant-numeric: tabular-nums; }} td.partial {{ color: var(--partial); }} tr.ref td {{ color: var(--muted); }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }}
</style></head><body><main>
<h1>Seed robustness of the downstream probes</h1>
<p class="meta">Updated {time.strftime('%Y-%m-%d %H:%M')} · seed 777 = new 10% few-shot split (seed 777) + probe seed 777 · layers = each model's 10 best seed-0 layers per task (<code>artifacts/downstream/layer_plan_top10.json</code>)</p>
<p class="lede">Every number is computed on the same 10 layers under both seeds, so "Σ10" (the metric summed over those layers) and "best" are directly comparable between seeds. Ranks are among the OpenWAM models (1 = best). "Σ10 change" is the relative change from seed 0 to seed 777; the per-task "Agreement" list gives the rank correlation between the two seeds, the architecture spread under each seed, and the success-rate correlation under seed 777. Rows marked "n/10 layers" are still running.</p>
{body}
</main></body></html>'''
    OUT.write_text(page)
    print('wrote', OUT, len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
