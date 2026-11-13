#!/usr/bin/env python
"""HTML report for the clean2rand probes (wamu/clean2rand_analysis.md).

Probes are trained on the same 250 clean Agilex episodes and tested on 500 clean test episodes (agilex_c2c) and on the
500 replayed randomized episodes (agilex_c2r). Reads artifacts/downstream/{c2c,c2r}/*/agilex_*/layers.csv and writes
reports/openwam_clean2rand.html (self-contained: data embedded as JSON, charts drawn in the page).
"""
import csv
import json
from pathlib import Path

MODELS = [  # fixed order = fixed colour slot
    ('sft_from_scratch_robotwin_clean', 'self-attn', 'MoT self-attention, action sees video'),
    ('cross_attn_clean', 'cross-attn', 'action DiT reads video through bridge cross-attention'),
    ('cross_attn_detach_robotwin_clean', 'cross-attn-detach', 'cross-attn, bridge detached (no action gradient into video)'),
    ('self_attn_detach_robotwin_clean', 'self-attn-detach', 'self-attn, video K/V detached for the action rows'),
    ('sft_ego_robot_cotrain_robotwin_clean_action_sees_video', 'pretrained self-attn', 'self-attn from the ego/robot co-trained checkpoint'),
    ('sft_robot_only_robotwin_clean', 'pretrained-robot-only', 'self-attn from a robot-only pretrained checkpoint (rand test only)'),
]
# (key, label, file, task filter, csv column, lower is better, decimals, unit, reference line {domain: value} or None)
METRICS = [
    ('depth_pairwise', 'Depth · pairwise order', 'dense_layers_ep60', 'depth', 'pairwise', False, 3, '', None),
    ('depth_absrel', 'Depth · AbsRel', 'dense_layers_ep60', 'depth', 'absrel', True, 3, '', None),
    ('seg_iou', 'Segmentation · target IoU', 'dense_layers_ep60', 'segmentation', 'iou', False, 3, '', None),
    ('cls_acc', 'Task classification · top-1', 'task_cls_layers_ep10', None, 'acc', False, 3, '', {'chance': 0.02}),
    ('flow_epe3d', '3D scene flow · EPE', 'flow_layers_ep60', None, 'epe3d', True, 1, 'mm', 'zero3d'),
    ('track_aj', '2D tracking · AJ', 'flow_layers_ep60', None, 'aj', False, 3, '', None),
]
EXTRA = [  # shown in the full table only
    ('depth_delta1', 'Depth · δ1', 'dense_layers_ep60', 'depth', 'delta1', False, 3),
    ('cls_auroc', 'Classification · AUROC', 'task_cls_layers_ep10', None, 'auroc', False, 3),
    ('flow_moving', 'Flow · EPE moving objects (mm)', 'flow_layers_ep60', None, 'epe3d_moving', True, 1),
    ('track_epe2d', 'Tracking · 2D endpoint error (px)', 'flow_layers_ep60', None, 'epe2d', True, 1),
    ('track_delta', 'Tracking · δ_avg', 'flow_layers_ep60', None, 'delta_avg', False, 3),
]
ROOT = Path('artifacts/downstream')


def load(domain, f, task):
    out = {}
    for r in csv.DictReader(open(ROOT / domain / f / f'agilex_{domain}' / 'layers.csv')):
        if task and r.get('task') != task:
            continue
        out.setdefault(r['model'], {})[int(r['layer'])] = r
    return out


def series(domain, f, task, col):
    D = load(domain, f, task)
    return {m: ([float(D[m][l][col]) for l in range(30)] if m in D else None) for m, *_ in MODELS}   # None: model not run in this domain


def main():
    data = {'models': [{'id': m, 'name': n, 'desc': d} for m, n, d in MODELS], 'metrics': []}
    for key, label, f, task, col, lower, dec, unit, ref in METRICS + [x + ('', None) for x in EXTRA]:
        entry = {'key': key, 'label': label, 'lower': lower, 'dec': dec, 'unit': unit, 'chart': key in {m[0] for m in METRICS},
                 'clean': series('c2c', f, task, col), 'rand': series('c2r', f, task, col)}
        if ref == 'zero3d':
            entry['ref'] = {'clean': float(next(iter(load('c2c', f, task).values()))[0]['zero3d']),
                            'rand': float(next(iter(load('c2r', f, task).values()))[0]['zero3d']), 'label': 'predict no motion'}
        elif ref:
            entry['ref'] = {'clean': ref['chance'], 'rand': ref['chance'], 'label': 'chance (50 tasks)'}
        data['metrics'].append(entry)
    html = TEMPLATE.replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
    out = Path('reports/openwam_clean2rand.html')
    out.write_text(html)
    print(f'wrote {out} ({out.stat().st_size / 1024:.0f} KB)')


TEMPLATE = r'''<title>Clean2Rand Probes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; summary table first, retention chart, then per-layer small multiples (2-up, 1-up on phones). */
:root {
  --bg: #fbfbfa; --surface: #ffffff; --ink: #121417; --ink-2: #4e5258; --ink-3: #80848a; --rule: #e3e4e2; --grid: #eeefed;
  --accent: #2a78d6; --good: #0a7a3b; --bad: #b3261e; --chip: #f1f2f0;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --sans: "Figtree", "Segoe UI", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #161716; --surface: #1d1e1d; --ink: #eceeea; --ink-2: #b9bcb6; --ink-3: #8b8e88; --rule: #31332f; --grid: #282a27;
  --accent: #3987e5; --good: #4cc27c; --bad: #f08a83; --chip: #262825;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #161716; --surface: #1d1e1d; --ink: #eceeea; --ink-2: #b9bcb6; --ink-3: #8b8e88; --rule: #31332f; --grid: #282a27;
  --accent: #3987e5; --good: #4cc27c; --bad: #f08a83; --chip: #262825;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--sans); }
.wrap { max-width: 1080px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 40px; }
header { display: grid; gap: 10px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }
h1 { font-size: clamp(28px, 4vw, 38px); line-height: 1.15; margin: 0; font-weight: 700; text-wrap: balance; }
h2 { font-size: 20px; margin: 0; font-weight: 650; text-wrap: balance; }
p { margin: 0; max-width: 68ch; color: var(--ink-2); }
section { display: grid; gap: 14px; min-width: 0; }
.setup { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.setup div { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; display: grid; gap: 4px; }
.setup b { font: 500 12px/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); }
.setup span { color: var(--ink); }
ul.findings { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 72ch; }
ul.findings li { color: var(--ink-2); } ul.findings strong { color: var(--ink); font-weight: 600; }
.tablebox { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font: 500 11.5px/1.3 var(--mono); letter-spacing: .04em; color: var(--ink-3); text-transform: uppercase; background: var(--surface); vertical-align: bottom; }
tbody tr:last-child td { border-bottom: 0; }
td.num { font-family: var(--mono); font-size: 12.5px; }
td .lay { color: var(--ink-3); font-size: 11px; margin-left: 4px; }
.best { font-weight: 600; color: var(--ink); background: var(--chip); }
.ret { font-family: var(--mono); font-size: 12px; }
.ret.hi { color: var(--good); } .ret.lo { color: var(--bad); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; vertical-align: -1px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; font-size: 13px; color: var(--ink-2); }
.legend .key { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { overflow: visible; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 14px 8px; display: grid; gap: 6px; min-width: 0; position: relative; }
.panel h3 { margin: 0; font-size: 14px; font-weight: 600; }
.panel .sub { font-size: 12px; color: var(--ink-3); }
svg.chart { width: 100%; height: auto; display: block; }
svg text { fill: var(--ink-3); font: 10.5px var(--mono); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px;
  font: 12px/1.45 var(--mono); color: var(--ink); box-shadow: 0 4px 16px rgb(0 0 0 / .12); min-width: 220px; z-index: 5; }
.tip .h { color: var(--ink-3); margin-bottom: 4px; }
.tip .r { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; }
.tip .r span:nth-child(n+2) { text-align: right; }
.toggle { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.toggle button { font: 500 12.5px var(--sans); color: var(--ink-2); background: var(--surface); border: 0; padding: 6px 12px; cursor: pointer; }
.toggle button[aria-pressed="true"] { background: var(--chip); color: var(--ink); }
.toggle button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
.notes { display: grid; gap: 8px; max-width: 76ch; }
.notes p { font-size: 14px; }
code { font: 12.5px var(--mono); background: var(--chip); padding: 1px 5px; border-radius: 4px; }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">OpenWAM · downstream probes · Aloha-Agilex</div>
  <h1>Clean2Rand probes: what survives domain randomization</h1>
  <p>Six clean-trained OpenWAM models (pretrained-robot-only was tested on rand only, so it has no clean score or retention). Each probe is trained once on clean scenes, then tested on held-out clean scenes and on the same
     tasks replayed with randomized backgrounds, lighting, table height and cluttered tables. All numbers are frozen video-backbone tokens
     (head camera, unpooled, σ = 0.5), probed layer by layer.</p>
</header>

<section>
  <div class="setup">
    <div><b>Probe training</b><span>250 clean episodes (the existing 10% split, 5 per task)</span></div>
    <div><b>Clean test</b><span>500 held-out clean episodes, 10 per task (seed 0)</span></div>
    <div><b>Rand test</b><span>500 randomized episodes, 10 per task, replayed GT</span></div>
    <div><b>Probes</b><span>depth &amp; segmentation 60 ep · flow/tracking 60 ep · classification 10 ep · all 30 layers</span></div>
  </div>
</section>

<section>
  <h2>Findings</h2>
  <ul class="findings" id="findings"></ul>
</section>

<section>
  <div class="row"><h2>Clean → rand by model</h2>
    <div class="toggle" role="group" aria-label="Layer choice">
      <button type="button" id="mode-best" aria-pressed="true">Best layer per domain</button>
      <button type="button" id="mode-fixed" aria-pressed="false">Rand at the clean-best layer</button>
      <button type="button" id="mode-top10" aria-pressed="false">Top-10 layer mean</button></div></div>
  <p id="mode-note"></p>
  <div class="tablebox"><table id="summary"></table></div>
</section>

<section>
  <h2>Retention on randomized scenes</h2>
  <p>Rand score as a share of the clean score, oriented so that 100% means nothing was lost (for error metrics: clean ÷ rand).
     <span id="ret-note">Best layer per domain.</span> The view follows the switch above the summary table.</p>
  <div class="legend" id="legend-ret"></div>
  <div class="panel"><svg class="chart" id="retention" role="img" aria-label="Retention of each metric per model"></svg></div>
</section>

<section>
  <h2>Layer by layer</h2>
  <p>Solid lines are the clean test, dashed lines the randomized test. Hover a panel for every model at that layer.</p>
  <div class="legend" id="legend-layers"></div>
  <div class="grid2" id="panels"></div>
</section>

<section>
  <h2>All metrics</h2>
  <div class="tablebox"><table id="full"></table></div>
</section>

<section class="notes">
  <h2>Notes</h2>
  <p><strong>Segmentation target on randomized scenes.</strong> The clean rule picks the frame-0 object whose mask centroid moves furthest.
     With clutter on the table, a pressed bell or clock barely moves and a partly occluded distractor won in 19% of episodes, so distractors
     (from each episode's official clutter list) are excluded; 497/500 rand targets are task objects afterwards. Clean targets are unchanged.</p>
  <p><strong>Layer choice is optimistic.</strong> “Best layer” is picked on the test set itself, as in the clean report. The toggle above shows
     the stricter reading, where the rand score is taken at the layer that was best on clean.</p>
  <p><strong>Restored rows.</strong> 40 rand flow rows (self-attn-detach layers 20–29, pretrained self-attn all layers) were lost to two probe
     processes writing the same file and were restored from the run log at log precision (0.1 mm, 3 decimals).</p>
  <p><strong>Single seed.</strong> One probe seed and one noise draw per episode; differences of about 0.01 in pairwise order or 0.5 mm in EPE are within run-to-run noise seen in the clean report.</p>
</section>
</div>

<script>
const DATA = /*DATA*/;
const M = DATA.models, MET = DATA.metrics, L = [...Array(30).keys()];
const col = i => `var(--s${i + 1})`;
const fmt = (v, d) => (v == null || isNaN(v)) ? '–' : v.toFixed(d);
const better = (m, a, b) => m.lower ? a < b : a > b;
function bestLayer(m, arr) { let b = 0; arr.forEach((v, i) => { if (better(m, v, arr[b])) b = i; }); return b; }
function top10(m, arr) { return [...arr].sort((a, b) => m.lower ? a - b : b - a).slice(0, 10).reduce((x, y) => x + y, 0) / 10; }
const hasClean = id => MET[0].clean[id] != null;
function stats(m, id, mode) {
  const c = m.clean[id], r = m.rand[id];
  if (!c) { const lr = mode === 'top10' ? null : bestLayer(m, r); return { lc: null, lr, vc: null, vr: mode === 'top10' ? top10(m, r) : r[lr], ret: NaN }; }
  if (mode === 'top10') { const vc = top10(m, c), vr = top10(m, r); return { lc: null, lr: null, vc, vr, ret: m.lower ? vc / vr : vr / vc }; }
  const lc = bestLayer(m, c), lr = mode === 'fixed' ? lc : bestLayer(m, r);
  const vc = c[lc], vr = r[lr], ret = m.lower ? vc / vr : vr / vc;
  return { lc, lr, vc, vr, ret };
}
const esc = s => String(s).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

// ---------- findings (computed, so they always match the tables)
(function findings() {
  const charted = MET.filter(m => m.chart);
  const wins = M.map(() => 0), last = M.map(() => 0), retAvg = M.map(() => 0);
  charted.forEach(m => {
    const s = M.map(x => stats(m, x.id, 'best'));
    const order = [...s.keys()].sort((a, b) => better(m, s[a].vr, s[b].vr) ? -1 : 1);
    wins[order[0]]++; last[order.at(-1)]++;
    s.forEach((x, i) => retAvg[i] += isNaN(x.ret) ? NaN : x.ret / charted.length);
  });
  const top = wins.indexOf(Math.max(...wins)), bot = last.indexOf(Math.max(...last));
  const pct = v => isNaN(v) ? 'n/a (rand only)' : `${(v * 100).toFixed(0)}%`;
  const MC = M.filter(x => hasClean(x.id));
  const spread = key => { const m = MET.find(x => x.key === key); const v = MC.map(x => stats(m, x.id, 'best').vc); return [Math.min(...v), Math.max(...v)]; };
  const dp = spread('depth_pairwise'), ca = spread('cls_acc'), fe = spread('flow_epe3d');
  const cls = MET.find(x => x.key === 'cls_acc'), clsR = M.map(x => stats(cls, x.id, 'best').vr);
  const geo = ['depth_pairwise', 'flow_epe3d', 'track_aj'].map(k => MET.find(x => x.key === k));
  const geoRet = geo.flatMap(m => MC.map(x => stats(m, x.id, 'best').ret));
  const items = [
    `<strong>On clean scenes the ${MC.length} models with a clean test are indistinguishable.</strong> Depth pairwise order ${fmt(dp[0], 3)}–${fmt(dp[1], 3)}, task accuracy ${fmt(ca[0], 3)}–${fmt(ca[1], 3)}, flow EPE ${fmt(fe[0], 1)}–${fmt(fe[1], 1)} mm. The architectures only separate once the scene changes.`,
    `<strong>${esc(M[top].name)} transfers best</strong>: best rand score on ${wins[top]} of ${charted.length} headline metrics, mean retention ${pct(retAvg[top])}.`,
    `<strong>${esc(M[bot].name)} transfers worst</strong>: last on ${last[bot]} of ${charted.length} metrics, mean retention ${pct(retAvg[bot])}.`,
    `<strong>Geometry survives, identity does not.</strong> Depth, flow and tracking keep ${(Math.min(...geoRet) * 100).toFixed(0)}–${(Math.max(...geoRet) * 100).toFixed(0)}% of their clean score; task classification falls to ${fmt(Math.min(...clsR), 3)}–${fmt(Math.max(...clsR), 3)} (chance 0.020) and target segmentation to roughly a quarter.`,
  ];
  document.getElementById('findings').innerHTML = items.map(x => `<li>${x}</li>`).join('');
})();

// ---------- summary table
function summary(mode) {
  const charted = MET.filter(m => m.chart);
  let h = '<thead><tr><th>Model</th>' + charted.map(m => `<th>${esc(m.label)}${m.unit ? ` (${m.unit})` : ''}<br>${m.lower ? '↓' : '↑'} clean → rand</th>`).join('') + '</tr></thead><tbody>';
  const bestRand = charted.map(m => { const v = M.map(x => stats(m, x.id, mode).vr); return m.lower ? Math.min(...v) : Math.max(...v); });
  M.forEach((x, i) => {
    h += `<tr><td><span class="sw" style="background:${col(i)}"></span>${esc(x.name)}</td>`;
    charted.forEach((m, j) => {
      const s = stats(m, x.id, mode), cls = isNaN(s.ret) ? '' : s.ret >= 0.85 ? 'hi' : s.ret < 0.5 ? 'lo' : '';
      const lay = l => l == null ? '' : `<span class="lay">L${l}</span>`;
      h += `<td class="num${s.vr === bestRand[j] ? ' best' : ''}">${fmt(s.vc, m.dec)}${lay(s.lc)} → ${fmt(s.vr, m.dec)}${lay(s.lr)}
             <br><span class="ret ${cls}">${isNaN(s.ret) ? 'rand only' : (s.ret * 100).toFixed(0) + '%'}</span></td>`;
    });
    h += '</tr>';
  });
  document.getElementById('summary').innerHTML = h + '</tbody>';
  document.getElementById('mode-note').textContent = {
    best: 'Each domain at its own best layer (Lnn). Shaded cell: best rand score in the column. Percentage: retention.',
    fixed: 'Rand scored at the layer that was best on clean, the setting you get when the layer is chosen before seeing randomized data.',
    top10: 'Mean over each domain\'s 10 best layers (chosen separately for clean and rand), less sensitive to a single lucky layer. Percentage: retention of the mean.',
  }[mode];
  document.getElementById('ret-note').textContent = { best: 'Best layer per domain.', fixed: 'Rand at the clean-best layer.', top10: 'Top-10 layer mean per domain.' }[mode];
  retention(mode);
}
['best', 'fixed', 'top10'].forEach(mode => document.getElementById(`mode-${mode}`).addEventListener('click', () => {
  document.querySelectorAll('.toggle button').forEach(b => b.setAttribute('aria-pressed', String(b.id === `mode-${mode}`)));
  summary(mode);
}));

// ---------- legends
const keyLine = (dash, color) => `<svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ''}/></svg>`;
document.getElementById('legend-ret').innerHTML = M.map((x, i) => `<span class="key"><span class="sw" style="background:${col(i)}"></span>${esc(x.name)}</span>`).join('');
document.getElementById('legend-layers').innerHTML = M.map((x, i) => `<span class="key"><span class="sw" style="background:${col(i)}"></span>${esc(x.name)}</span>`).join('')
  + `<span class="key">${keyLine(false, 'var(--ink-2)')} clean</span><span class="key">${keyLine(true, 'var(--ink-2)')} rand</span>`;

// ---------- retention grouped bars
function retention(mode) {
  const charted = MET.filter(m => m.chart), W = 960, H = 300, pl = 44, pr = 10, pt = 14, pb = 46;
  const gw = (W - pl - pr) / charted.length, bw = Math.min(22, (gw - 26) / M.length - 2);
  const y = v => pt + (H - pt - pb) * (1 - v);
  let s = `<rect width="${W}" height="${H}" fill="none"/>`;
  [0, .25, .5, .75, 1].forEach(t => s += `<line x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)" stroke-width="1"/><text x="${pl - 6}" y="${y(t) + 3.5}" text-anchor="end">${t * 100}%</text>`);
  charted.forEach((m, g) => {
    const x0 = pl + g * gw + (gw - M.length * (bw + 2)) / 2;
    M.forEach((x, i) => {
      if (!hasClean(x.id)) return;   // no retention without a clean score; the slot stays empty
      const r = Math.min(1, stats(m, x.id, mode).ret), bx = x0 + i * (bw + 2), by = y(r), bh = y(0) - by;
      s += `<path d="M${bx},${y(0)} V${by + 4} q0,-4 4,-4 h${bw - 8} q4,0 4,4 V${y(0)} Z" fill="${col(i)}" data-tip="${esc(x.name)} · ${esc(m.label)}: ${(stats(m, x.id, mode).ret * 100).toFixed(1)}%"/>`;
    });
    const lab = m.label.split(' · ');
    s += `<text x="${pl + g * gw + gw / 2}" y="${H - pb + 18}" text-anchor="middle">${esc(lab[0])}</text><text x="${pl + g * gw + gw / 2}" y="${H - pb + 32}" text-anchor="middle">${esc(lab[1] || '')}</text>`;
  });
  s += `<line x1="${pl}" x2="${W - pr}" y1="${y(0)}" y2="${y(0)}" stroke="var(--ink-3)" stroke-width="1"/>`;
  const svg = document.getElementById('retention');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = s;
  const panel = svg.parentElement;
  if (panel.dataset.wired) return; panel.dataset.wired = '1';
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; panel.appendChild(tip);
  svg.addEventListener('mousemove', e => {
    const t = e.target.getAttribute && e.target.getAttribute('data-tip');
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t; tip.hidden = false;
    const r = panel.getBoundingClientRect(); tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 240) + 'px'; tip.style.top = (e.clientY - r.top + 12) + 'px';
  });
  svg.addEventListener('mouseleave', () => tip.hidden = true);
}

// ---------- per-layer small multiples
(function panels() {
  const host = document.getElementById('panels');
  MET.filter(m => m.chart).forEach(m => {
    const W = 480, H = 250, pl = 46, pr = 12, pt = 10, pb = 28;
    const all = M.flatMap(x => [...(m.clean[x.id] || []), ...m.rand[x.id]]).concat(m.ref ? [m.ref.clean, m.ref.rand] : []);
    let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * .06; lo -= pad; hi += pad;
    const X = l => pl + (W - pl - pr) * l / 29, Y = v => pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo));
    const ticks = []; const step = niceStep((hi - lo) / 4); for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
    let s = '';
    ticks.forEach(t => s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${pl - 6}" y="${Y(t) + 3.5}" text-anchor="end">${+t.toFixed(4)}</text>`);
    [0, 5, 10, 15, 20, 25, 29].forEach(l => s += `<text x="${X(l)}" y="${H - pb + 16}" text-anchor="middle">${l}</text>`);
    if (m.ref) s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(m.ref.rand)}" y2="${Y(m.ref.rand)}" stroke="var(--ink-3)" stroke-dasharray="1 3"/><text x="${W - pr}" y="${Y(m.ref.rand) - 4}" text-anchor="end">${esc(m.ref.label)}</text>`;
    const path = a => a.map((v, l) => `${l ? 'L' : 'M'}${X(l).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    M.forEach((x, i) => {
      if (m.clean[x.id]) s += `<path d="${path(m.clean[x.id])}" fill="none" stroke="${col(i)}" stroke-width="2" stroke-linejoin="round"/>`;
      s += `<path d="${path(m.rand[x.id])}" fill="none" stroke="${col(i)}" stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round"/>`;
    });
    s += `<line class="xh" x1="0" x2="0" y1="${pt}" y2="${H - pb}" stroke="var(--ink-3)" visibility="hidden"/>`;
    s += `<rect class="hit" x="${pl}" y="${pt}" width="${W - pl - pr}" height="${H - pt - pb}" fill="transparent"/>`;
    const div = document.createElement('div'); div.className = 'panel';
    div.innerHTML = `<h3>${esc(m.label)}${m.unit ? ` (${m.unit})` : ''}</h3><div class="sub">${m.lower ? 'lower is better' : 'higher is better'} · x: layer 0–29</div>
      <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(m.label)} per layer, clean and randomized">${s}</svg>`;
    host.appendChild(div);
    const svg = div.querySelector('svg'), xh = svg.querySelector('.xh'), tip = document.createElement('div');
    tip.className = 'tip'; tip.hidden = true; div.appendChild(tip);
    svg.querySelector('.hit').addEventListener('mousemove', e => {
      const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse());
      const l = Math.max(0, Math.min(29, Math.round((q.x - pl) / (W - pl - pr) * 29)));
      xh.setAttribute('x1', X(l)); xh.setAttribute('x2', X(l)); xh.setAttribute('visibility', 'visible');
      tip.innerHTML = `<div class="h">layer ${l} · clean / rand</div>` + M.map((x, i) => `<div class="r"><span><span class="sw" style="background:${col(i)}"></span>${esc(x.name)}</span><span>${fmt(m.clean[x.id] ? m.clean[x.id][l] : null, m.dec)}</span><span>${fmt(m.rand[x.id][l], m.dec)}</span></div>`).join('');
      tip.hidden = false;
      const r = div.getBoundingClientRect(); tip.style.left = Math.max(8, Math.min(e.clientX - r.left + 14, r.width - 250)) + 'px'; tip.style.top = (e.clientY - r.top + 14) + 'px';
    });
    svg.querySelector('.hit').addEventListener('mouseleave', () => { tip.hidden = true; xh.setAttribute('visibility', 'hidden'); });
  });
  function niceStep(raw) { const e = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / e; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * e; }
})();

summary('best');

// ---------- full table
(function full() {
  let h = '<thead><tr><th>Metric</th><th>Domain</th>' + M.map((x, i) => `<th><span class="sw" style="background:${col(i)}"></span>${esc(x.name)}</th>`).join('') + '</tr></thead><tbody>';
  MET.forEach(m => ['clean', 'rand'].forEach(dom => {
    const vals = M.map(x => { const a = m[dom][x.id]; if (!a) return [null, null]; const l = bestLayer(m, a); return [a[l], l]; });
    const got = vals.map(v => v[0]).filter(v => v != null), b = m.lower ? Math.min(...got) : Math.max(...got);
    h += `<tr><td>${dom === 'clean' ? esc(m.label) + (m.lower ? ' ↓' : ' ↑') : ''}</td><td>${dom}</td>` +
      vals.map(([v, l], i) => v == null ? '<td class="num">–<br><span class="lay">not run</span></td>' : `<td class="num${v === b ? ' best' : ''}">${fmt(v, m.dec)}<span class="lay">L${l}</span><br><span class="lay">top-10 ${fmt(top10(m, m[dom][M[i].id]), m.dec)}</span></td>`).join('') + '</tr>';
  }));
  document.getElementById('full').innerHTML = h + '</tbody>';
})();
</script>
'''

if __name__ == '__main__':
    main()
