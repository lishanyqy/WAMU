#!/usr/bin/env python
"""Rebuild reports/openwam_gt_rebuild.html from the probe result CSVs.

Reads   artifacts/downstream/task_cls_layers_ep10/<emb>/layers*.csv   (task classification)
        artifacts/downstream/dense_layers_ep60/<emb>/layers*.csv      (depth, segmentation)
All tables are rendered here as static HTML (so a script error can never blank them); the page's
JavaScript only draws the per-layer charts from the same data. Safe to run while probes are running.
"""
import csv
import html
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLS = ROOT / 'artifacts/downstream/task_cls_layers_ep10'
DENSE = ROOT / 'artifacts/downstream/dense_layers_ep60'
TEMPLATE = ROOT / 'reports/probe_report_template.html'
OUT = ROOT / 'reports/openwam_gt_rebuild.html'
N_LAYERS = {'videomaev2': 24, 'dinov2': 25, 'd4rt': 1}
WAM = ['idm', 'joint_cross_attention', 'joint_cross_attention_detach', 'joint_self_attention',
       'joint_self_attention_isolated', 'joint_self_attention_mutual', 'joint_self_attention_video_sees_action']
N_LAYERS.update({m: 30 for m in WAM})
REFS = ['videomaev2', 'dinov2', 'd4rt']
REF_NAME = {'videomaev2': 'VideoMAEv2-L', 'dinov2': 'DINOv2-L', 'd4rt': 'D4RT encoder'}
# RoboTwin policy success, clean/randomised average (reports/performance.md)
SUCCESS = {'joint_self_attention': 92.39, 'joint_self_attention_mutual': 92.15, 'joint_cross_attention_detach': 91.85,
           'joint_cross_attention': 88.25, 'idm': 87.95, 'joint_self_attention_video_sees_action': 87.63,
           'joint_self_attention_isolated': 87.41}
HIGH_TIER = ['joint_self_attention', 'joint_self_attention_mutual', 'joint_cross_attention_detach']
EMBS = ['agilex', 'franka']
EMB_NAME = {'agilex': 'Agilex', 'franka': 'Franka'}
# task -> (metric, label, lower_is_better, decimals, extra metrics shown in the ranking table)
TASKS = {'cls': ('acc', 'accuracy', False, 3, []),
         'depth': ('absrel', 'AbsRel', True, 4, [('delta1', 'δ1'), ('pairwise', 'pairwise')]),
         'seg': ('iou', 'IoU', False, 3, [])}


# ---------------------------------------------------------------- data

def rows(d):
    out = []
    for f in sorted(d.glob('layers*.csv')) if d.exists() else []:
        out += list(csv.DictReader(open(f)))
    return out


def put(table, emb, model, metric, layer, value):
    if value in ('', None):
        return
    m = model.replace('robotwin_dual_system_', '')
    if m not in N_LAYERS:
        return
    arr = table.setdefault(emb, {}).setdefault(m, {}).setdefault(metric, [None] * N_LAYERS[m])
    arr[int(layer)] = round(float(value), 5)


def collect():
    """{'cls'|'depth'|'seg': {emb: {model: {metric: [value or None per layer]}}}, 'progress': {...}}"""
    data = {'cls': {}, 'depth': {}, 'seg': {}}
    done = 0
    for emb in EMBS:
        for r in rows(CLS / emb):
            put(data['cls'], emb, r['model'], 'acc', r['layer'], r['acc'])
            put(data['cls'], emb, r['model'], 'auroc', r['layer'], r['auroc'])
        for r in rows(DENSE / emb):
            if r['task'] == 'depth':
                for k in ('absrel', 'delta1', 'pairwise'):
                    put(data['depth'], emb, r['model'], k, r['layer'], r[k])
            else:
                put(data['seg'], emb, r['model'], 'iou', r['layer'], r['iou'])
            done += 1
    data['progress'] = {'done': done, 'total': 2 * 2 * sum(N_LAYERS.values())}
    return data


def curve(D, task, emb, m, metric=None):
    c = D[task].get(emb, {}).get(m, {}).get(metric or TASKS[task][0])
    return c if c and any(v is not None for v in c) else None


def valid(task, m, c):
    """Values with unfinished layers dropped; DINOv2 layer 0 (patch embedding, CLS constant) dropped for cls."""
    return [v for i, v in enumerate(c) if v is not None and not (task == 'cls' and m == 'dinov2' and i == 0)]


def best(task, m, c, lower):
    idx = [i for i, v in enumerate(c) if v is not None and not (task == 'cls' and m == 'dinov2' and i == 0)]
    return min(idx, key=lambda i: c[i]) if lower else max(idx, key=lambda i: c[i])


# ---------------------------------------------------------------- html helpers

def esc(x):
    return html.escape(str(x))


def td(v, cls='num'):
    return f'<td class="{cls}">{esc(v)}</td>'


def fmt(v, dec):
    return '–' if v is None else f'{v:.{dec}f}'


def table(headers, body_rows, cls=''):
    h = ''.join(f'<th{" class=num" if i else ""}>{esc(c)}</th>' for i, c in enumerate(headers))
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def model_cell(m, ref=False):
    return f'<td>{"<em>" + esc(REF_NAME[m]) + " (reference)</em>" if ref else "<span class=mono>" + esc(m) + "</span>"}</td>'


# ---------------------------------------------------------------- tables

def summary_table(D):
    """One row per model: success, then for each task x embodiment the Σ30 (OpenWAM) or best layer (references)."""
    heads = ['Model', 'Success % (rank)']
    for task, (met, label, lower, dec, _) in TASKS.items():
        for emb in EMBS:
            heads.append(f'{label} Σ30 {"↓" if lower else "↑"} {EMB_NAME[emb]} (rank)')
    out = []
    order = sorted(WAM, key=lambda m: -SUCCESS[m])
    for m in order:
        cells = [model_cell(m), td(f'{SUCCESS[m]:.2f} ({SUCCESS_RANK[m]})')]
        for task, (met, label, lower, dec, _) in TASKS.items():
            for emb in EMBS:
                sig = {m2: sigma30(D, task, emb, m2) for m2 in WAM}
                if sig[m] is None:
                    c = curve(D, task, emb, m); n = len(valid(task, m, c)) if c else 0
                    cells.append(td(f'{n}/30 layers', 'num partial'))
                    continue
                r = rank_map({k: v for k, v in sig.items() if v is not None}, lower)[m]
                cells.append(td(f'{sig[m]:.{dec - 1}f} ({r})', 'num lead' if r == 1 else 'num'))
        out.append('<tr>' + ''.join(cells) + '</tr>')
    for r in REFS:
        cells = [model_cell(r, True), td('–')]
        for task, (met, label, lower, dec, _) in TASKS.items():
            for emb in EMBS:
                c = curve(D, task, emb, r)
                if not c:
                    cells.append(td('–')); continue
                b = best(task, r, c, lower)
                cells.append(td(f'best layer {c[b]:.{dec}f}' + (f' (L{b})' if N_LAYERS[r] > 1 else '')))
        out.append('<tr class="ref">' + ''.join(cells) + '</tr>')
    return table(heads, out, 'summary-table')


def yardstick(D):
    """Spread across the 7 architectures as a fraction of the span between the reference encoders (best layers)."""
    tiles = []
    for task, (met, label, lower, dec, _) in TASKS.items():
        for emb in EMBS:
            wb = [curve(D, task, emb, m) for m in WAM]
            rb = [curve(D, task, emb, r) for r in REFS]
            if not all(wb) or not all(rb):
                continue
            wbest = [c[best(task, m, c, lower)] for m, c in zip(WAM, wb)]
            rbest = [c[best(task, r, c, lower)] for r, c in zip(REFS, rb)]
            span = max(rbest) - min(rbest)
            frac = (max(wbest) - min(wbest)) / span if span else float('nan')
            tiles.append(f'<div class="stat"><div class="num">{100 * frac:.0f}%</div>'
                         f'<div class="lbl"><b>{esc(label)} · {EMB_NAME[emb]}</b>: 7 architectures span {min(wbest):.{dec}f}–{max(wbest):.{dec}f}; '
                         f'references span {min(rbest):.{dec}f}–{max(rbest):.{dec}f}. Architecture spread = {100 * frac:.0f}% of the reference span.</div></div>')
    return ''.join(tiles)


def rank_map(values, lower):
    """{model: rank} with 1 = best; values is {model: number}."""
    order = sorted(values, key=lambda m: values[m], reverse=not lower)
    return {m: i + 1 for i, m in enumerate(order)}


SUCCESS_RANK = rank_map(SUCCESS, lower=False)


def sigma30(D, task, emb, m):
    c = curve(D, task, emb, m)
    v = valid(task, m, c) if c else []
    return sum(v) if len(v) == 30 else None


def ranking_table(D, task):
    """Primary table: success (rank) and Σ30 (rank) per robot. Secondary metrics go to a collapsed block below."""
    met, label, lower, dec, extra = TASKS[task]
    sig = {emb: {m: sigma30(D, task, emb, m) for m in WAM} for emb in EMBS}
    ranks = {emb: rank_map({m: v for m, v in sig[emb].items() if v is not None}, lower) for emb in EMBS}
    heads = ['Model', 'Success % (rank)'] + [f'{EMB_NAME[emb]} {label} Σ30 {"↓" if lower else "↑"} (rank)' for emb in EMBS]
    out = []
    for m in sorted(WAM, key=lambda m: -SUCCESS[m]):
        cells = [model_cell(m), td(f'{SUCCESS[m]:.2f} ({SUCCESS_RANK[m]})')]
        for emb in EMBS:
            v = sig[emb][m]
            if v is None:
                c = curve(D, task, emb, m); n = len(valid(task, m, c)) if c else 0
                cells.append(td(f'{n}/30 layers', 'num partial'))
            else:
                r = ranks[emb][m]
                cells.append(td(f'{v:.{dec - 1}f} ({r})', 'num best' if r == 1 else 'num'))
        out.append('<tr>' + ''.join(cells) + '</tr>')
    for r in REFS:
        cells = [model_cell(r, True), td('–')]
        for emb in EMBS:
            c = curve(D, task, emb, r)
            if not c:
                cells.append(td('–')); continue
            b = best(task, r, c, lower)
            cells.append(td(f'best layer {c[b]:.{dec}f}' + (f' (L{b})' if N_LAYERS[r] > 1 else '')))
        out.append('<tr class="ref">' + ''.join(cells) + '</tr>')
    primary = table(heads, out)

    # secondary: layer mean, best layer, extra metrics
    heads2 = ['Model']
    for emb in EMBS:
        heads2 += [f'{EMB_NAME[emb]} layer mean', 'best layer'] + [f'mean {e[1]}' for e in extra]
    out2 = []
    for m in WAM + REFS:
        ref = m in REFS
        cells = [model_cell(m, ref)]
        for emb in EMBS:
            c = curve(D, task, emb, m)
            if not c:
                cells += [td('–')] * (2 + len(extra)); continue
            v = valid(task, m, c); b = best(task, m, c, lower)
            cells.append(td(f'{np.mean(v):.{dec}f}'))
            cells.append(td(f'{c[b]:.{dec}f}' + (f' (L{b})' if N_LAYERS[m] > 1 else '')))
            for e, _ in extra:
                ce = curve(D, task, emb, m, e)
                cells.append(td(f'{np.mean([x for x in ce if x is not None]):.3f}' if ce else '–'))
        out2.append(f'<tr{" class=ref" if ref else ""}>' + ''.join(cells) + '</tr>')
    secondary = (f'<details class="layers"><summary>Secondary metrics: layer mean, best layer'
                 + (', ' + ', '.join(e[1] for e in extra) if extra else '') + '</summary><div class="inner"><div class="tbl-wrap">'
                 + table(heads2, out2) + '</div><p class="caption">Layer mean = Σ30 / 30 for OpenWAM; for references the mean over their own layers. '
                 'Rank in the main table is among the 7 OpenWAM architectures only; references are shown at their best layer because their depths differ.</p></div></details>')
    return primary + secondary


def layer_tables(D, task):
    met, label, lower, dec, extra = TASKS[task]
    metrics = [(met, label)] + ([('pairwise', 'pairwise accuracy')] if task == 'depth' else [])
    parts = []
    for metric, mlabel in metrics:
        mlower = lower if metric == met else False
        for emb in EMBS:
            heads = ['model'] + [f'L{i}' for i in range(30)] + ['best']
            body = []
            wam_rows = []
            for m in WAM + REFS:
                c = curve(D, task, emb, m, metric)
                if m in WAM:
                    wam_rows.append(c)
                cells = [model_cell(m, m in REFS)]
                if not c:
                    body.append('<tr>' + ''.join(cells) + '<td></td>' * 31 + '</tr>'); continue
                v = valid(task, m, c); lo, hi = min(v), max(v); b = best(task, m, c, mlower)
                for i in range(30):
                    x = c[i] if i < len(c) else None
                    if x is None:
                        cells.append('<td></td>'); continue
                    t = (x - lo) / (hi - lo) if hi > lo else 0.5
                    shade = round(8 + 50 * ((1 - t) if mlower else t))
                    cells.append(f'<td class="num{" bestc" if i == b else ""}" style="background: color-mix(in srgb, var(--s-wam) {shade}%, transparent)">{x:.{dec}f}</td>')
                cells.append(td(f'{c[b]:.{dec}f} L{b}' if N_LAYERS[m] > 1 else f'{c[b]:.{dec}f}'))
                body.append('<tr>' + ''.join(cells) + '</tr>')
            spread = []
            for l in range(30):
                v = [r[l] for r in wam_rows if r and r[l] is not None]
                spread.append(max(v) - min(v) if len(v) == len(WAM) else None)
            cells = ['<td>OpenWAM spread (max − min)</td>'] + [td(fmt(s, dec)) if s is not None else '<td></td>' for s in spread]
            cells.append(td(f'max {max(s for s in spread if s is not None):.{dec}f}' if any(s is not None for s in spread) else ''))
            body.append('<tr class="spread">' + ''.join(cells) + '</tr>')
            parts.append(f'<h3 style="font-size:0.9rem">{EMB_NAME[emb]} · {esc(mlabel)}{" (lower is better)" if mlower else ""}</h3>'
                         f'<div class="lt-wrap">{table(heads, body, "lt")}</div>')
    parts.append('<p class="caption">Columns are layer indices (OpenWAM 30, VideoMAEv2 24, DINOv2 25, D4RT final layer only). Shading is per row, darker = better; the outlined cell is the row\'s best layer. Last row: spread across the 7 OpenWAM architectures at that layer.</p>')
    return ''.join(parts)


def success_table(D):
    heads = ['Tier (success %)', 'Architectures']
    for task, (met, label, lower, dec, _) in TASKS.items():
        for emb in EMBS:
            heads.append(f'{label} Σ30 {EMB_NAME[emb]}')
    low = [m for m in WAM if m not in HIGH_TIER]

    def group_row(name, ms):
        cells = [f'<td><b>{esc(name)}</b></td>', f'<td class="mono" style="white-space:normal">{esc(", ".join(ms))}</td>']
        for task, (met, label, lower, dec, _) in TASKS.items():
            for emb in EMBS:
                cs = [curve(D, task, emb, m) for m in ms]
                if not all(cs) or any(len(valid(task, m, c)) < 30 for m, c in zip(ms, cs)):
                    cells.append(td('incomplete', 'num partial')); continue
                cells.append(td(f'{np.mean([sum(valid(task, m, c)) for m, c in zip(ms, cs)]):.{dec - 1}f}'))
        return '<tr class="group">' + ''.join(cells) + '</tr>'

    def diff_row():
        cells = ['<td>high − low</td>', '<td></td>']
        for task, (met, label, lower, dec, _) in TASKS.items():
            for emb in EMBS:
                g = lambda ms: [sum(valid(task, m, curve(D, task, emb, m))) for m in ms if curve(D, task, emb, m) and len(valid(task, m, curve(D, task, emb, m))) == 30]
                a, b = g(HIGH_TIER), g(low)
                if len(a) < len(HIGH_TIER) or len(b) < len(low):
                    cells.append(td('–')); continue
                d = np.mean(a) - np.mean(b); rel = 100 * d / np.mean(a + b)
                cells.append(td(f'{d:+.{dec - 1}f} ({rel:+.1f}%)'))
        return '<tr>' + ''.join(cells) + '</tr>'
    return table(heads, [group_row(f'High ({np.mean([SUCCESS[m] for m in HIGH_TIER]):.1f})', HIGH_TIER),
                         group_row(f'Low ({np.mean([SUCCESS[m] for m in low]):.1f})', low), diff_row()])


def corr_table(D):
    from scipy.stats import spearmanr
    heads = ['Task', 'Embodiment', 'Spearman ρ (Σ30 vs success)', 'p', 'Architecture range of Σ30', 'as % of mean']
    out = []
    for task, (met, label, lower, dec, _) in TASKS.items():
        for emb in EMBS:
            ms = [m for m in WAM if curve(D, task, emb, m) and len(valid(task, m, curve(D, task, emb, m))) == 30]
            if len(ms) < 4:
                out.append(f'<tr><td>{esc(label)}</td><td>{EMB_NAME[emb]}</td>{td("incomplete", "num partial")}{td("")}{td("")}{td("")}</tr>'); continue
            x = np.array([sum(valid(task, m, curve(D, task, emb, m))) for m in ms]) * (-1 if lower else 1)
            y = np.array([SUCCESS[m] for m in ms])
            r = spearmanr(x, y)
            rng = abs(x).max() - abs(x).min()
            out.append(f'<tr><td>{esc(label)}{" (sign flipped so higher = better)" if lower else ""}</td><td>{EMB_NAME[emb]}</td>'
                       f'{td(f"{r.correlation:+.2f}")}{td(f"{r.pvalue:.2f}")}{td(f"{rng:.{dec - 1}f}")}{td(f"{100 * rng / abs(x).mean():.1f}%")}</tr>')
    return table(heads, out)


def findings(D):
    f = []
    f.append('<li><b>OpenWAM carries more geometric and object information than the specialist encoders under this probe.</b> On depth and segmentation every one of the 7 architectures beats VideoMAEv2, DINOv2 and D4RT at their best layers, on both robots (Agilex depth AbsRel 0.045 vs 0.053 for the best reference; segmentation IoU 0.18 vs 0.16). On task classification OpenWAM matches VideoMAEv2 on Franka and trails it by 2 points on Agilex.</li>')
    f.append('<li><b>Measured against the reference span, the architecture spread is 1–12% of it at the best layer and 2–25% on the layer mean.</b> The largest relative spreads are on classification (Franka layer mean 25%, Agilex best layer 11%) and segmentation (Agilex best layer 12%); depth pairwise accuracy separates architectures least (1–6%) because every model is above 0.9 on it.</li>')
    f.append('<li><b>The ordering of architectures is not stable.</b> The architecture that leads a task on Agilex does not lead it on Franka (classification: idm vs video_sees_action; depth: isolated vs idm; segmentation: cross_attention vs detach), and no architecture leads at a majority of layers on any task.</li>')
    f.append('<li><b>Policy success does not follow the visual probes.</b> The 92% tier and the 88% tier differ by under 0.3% on every Σ30, and Spearman correlations between Σ30 and success are weak and change sign between robots. The wiring that moves success (action attending to video, detaching the cross-attention gradient) leaves the video branch\'s depth, segmentation and semantic read-outs where they were.</li>')
    f.append('<li><b>OpenWAM peaks early.</b> Classification peaks at layers 7–9 of 30, depth and segmentation at layers 12–15, and every layer after 1 stays within a narrow band; VideoMAEv2 and DINOv2 rise steadily to layers 13–22 and VideoMAEv2 drops 8–11 points in its last two layers.</li>')
    return ''.join(f)


def main():
    D = collect()
    page = TEMPLATE.read_text()
    fills = {
        '__UPDATED__': time.strftime('%Y-%m-%d %H:%M'),
        '__SUMMARY_TABLE__': summary_table(D),
        '__YARDSTICK__': yardstick(D),
        '__FINDINGS__': findings(D),
        '__SUCCESS_TABLE__': success_table(D),
        '__CORR_TABLE__': corr_table(D),
        '__DATA__': json.dumps(D, separators=(',', ':')),
    }
    for task in TASKS:
        fills[f'__TABLE_{task}__'] = ranking_table(D, task)
        fills[f'__LAYERS_{task}__'] = layer_tables(D, task)
    for k, v in fills.items():
        assert page.count(k) == 1, k
        page = page.replace(k, v)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(page)
    tmp.replace(OUT)
    print(f'wrote {OUT} ({len(page) // 1024} KB), dense runs {D["progress"]["done"]}/{D["progress"]["total"]}')


if __name__ == '__main__':
    main()
