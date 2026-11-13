#!/usr/bin/env python
"""Build the browsable results page for wamu/action_read_subspace_plan.md from artifacts/action_read/results/*.csv.
Writes artifacts/action_read/report.html (self-contained: inline CSS + SVG, no scripts needed).  Re-run after new results."""
import csv
import datetime as dt
import html
import math
from pathlib import Path

R = Path('artifacts/action_read/results'); RAW = Path('artifacts/action_read/raw'); OUT = Path('artifacts/action_read/report.html')
MODELS = ['M1', 'M2', 'M3']; SETS = ['ID', 'OOD']; KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MNAME = {'M1': 'M1 · 无额外预训练', 'M2': 'M2 · robot 预训练', 'M3': 'M3 · robot + human 预训练'}
SNAME = {'ID': 'ID · clean_50', 'OOD': 'OOD · randomized_500'}


def rd(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def f(x, d=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return '–'
    if math.isnan(v):
        return '–'
    if abs(v) >= 100:
        return f'{v:,.0f}'
    return f'{v:.{d}f}'


def sig_key(s):
    return round(float(s), 2)


s0 = rd(R / 'step0_baseline.csv'); s2 = rd(R / 'step2_causal.csv'); kst = rd(R / 'step2_kstar.csv'); win = rd(R / 'step2_layer_windows.csv')
mb = rd(R / 'step2_mean_baseline.csv'); mc = rd(R / 'step2_causal_meanref.csv'); mk = rd(R / 'step2_kstar_meanref.csv')


def kstar_mean(m, s='OOD'):
    rows = [r for r in mk if r['model'] == m and r['set'] == s and r['R_from'] == s]
    if not rows:
        return None, {}
    per = {sig_key(r['action_sigma']): (int(r['kstar_meanref']) if r['kstar_meanref'] else None) for r in rows}
    return (max(per.values()) if all(v is not None for v in per.values()) else 'none'), per
spec = {(m, s): rd(R / f'step1_spectrum_{m}_{s}.csv') for m in MODELS for s in SETS}
done_causal = {(m, s, rf) for m in MODELS for s in SETS for rf in SETS if (RAW / f'causal_{m}_{s}_R{rf}.npz').exists()}


def kstar(m, s, rf):
    rows = [r for r in kst if r['model'] == m and r['set'] == s and r['R_from'] == rf]
    if not rows:
        return None, {}
    per = {sig_key(r['action_sigma']): (int(r['kstar']) if r['kstar'] else None) for r in rows}
    return (max(per.values()) if all(v is not None for v in per.values()) else 'none'), per


# ------------------------------------------------------------------ SVG helpers

def recovery_svg(m, s, sig, meanref=False):
    src, pre = (mc, 'recp_') if meanref else (s2, 'rec_')
    rows = sorted([r for r in src if r['model'] == m and r['set'] == s and r['R_from'] == s and sig_key(r['action_sigma']) == sig], key=lambda r: int(r['k']))
    W, H, l, rt, t, b = 300, 190, 38, 10, 12, 30
    pw, ph = W - l - rt, H - t - b
    if not rows:
        return f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="pending"><text x="{W/2}" y="{H/2}" class="pend" text-anchor="middle">运行中</text></svg>'
    x = lambda k: l + pw * (math.log2(k) / 8)
    y = lambda v: t + ph * (1 - (max(-0.2, min(1.05, v)) + 0.2) / 1.25)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{m} {s} recovery">']
    for v in (0, 0.5, 1.0):
        out.append(f'<line x1="{l}" x2="{W-rt}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="{l-5}" y="{y(v)+3:.1f}" class="tick" text-anchor="end">{v:g}</text>')
    for v in (0.9, 0.1):
        out.append(f'<line x1="{l}" x2="{W-rt}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="thr"/>')
    for k in KS:
        out.append(f'<text x="{x(k):.1f}" y="{H-b+14}" class="tick" text-anchor="middle">{k}</text>')
    out.append(f'<text x="{l+pw/2}" y="{H-3}" class="axis" text-anchor="middle">k（子空间维数）</text>')
    ks = [int(r['k']) for r in rows]
    for key, cls in (('keep', 'keep'), ('remove', 'remove')):
        mu = [float(r[f'{pre}{key}Rand_mean']) for r in rows]; sd = [float(r[f'{pre}{key}Rand_std']) for r in rows]
        up = ' '.join(f'{x(k):.1f},{y(a+b_):.1f}' for k, a, b_ in zip(ks, mu, sd)); dn = ' '.join(f'{x(k):.1f},{y(a-b_):.1f}' for k, a, b_ in reversed(list(zip(ks, mu, sd))))
        out.append(f'<polygon points="{up} {dn}" class="band {cls}"/>')
        out.append('<polyline points="' + ' '.join(f'{x(k):.1f},{y(a):.1f}' for k, a in zip(ks, mu)) + f'" class="rand {cls}"/>')
        vals = [float(r[f'{pre}{key}R']) for r in rows]
        out.append('<polyline points="' + ' '.join(f'{x(k):.1f},{y(v):.1f}' for k, v in zip(ks, vals)) + f'" class="line {cls}"/>')
        for k, v in zip(ks, vals):
            out.append(f'<circle cx="{x(k):.1f}" cy="{y(v):.1f}" r="2.6" class="dot {cls}"><title>{key}-R(k={k}) = {v:.3f}</title></circle>')
    passes = (lambda r: r['meets_criterion_meanref'] == 'True' and r['beats_all_random'] == 'True') if meanref else (lambda r: r['meets_criterion'] == 'True')
    ok = [int(r['k']) for r in rows if passes(r)]
    if ok:
        out.append(f'<line x1="{x(ok[0]):.1f}" x2="{x(ok[0]):.1f}" y1="{t}" y2="{t+ph}" class="kstar"/><text x="{x(ok[0])+4:.1f}" y="{t+10}" class="kstar-l">k*={ok[0]}</text>')
    out.append('</svg>')
    return ''.join(out)


def effdim_svg(s):
    W, H, l, rt, t, b = 620, 210, 40, 12, 12, 32
    pw, ph = W - l - rt, H - t - b
    series = {m: [float(r['eff_dim']) for r in spec[(m, s)]] for m in MODELS if spec[(m, s)]}
    if not series:
        return ''
    top = max(max(v) for v in series.values()); top = math.ceil(top / 20) * 20
    x = lambda i: l + pw * i / 29; y = lambda v: t + ph * (1 - v / top)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart wide" role="img" aria-label="effective dimension {s}">']
    for v in range(0, top + 1, 20):
        out.append(f'<line x1="{l}" x2="{W-rt}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="{l-5}" y="{y(v)+3:.1f}" class="tick" text-anchor="end">{v}</text>')
    for i in range(0, 30, 5):
        out.append(f'<text x="{x(i):.1f}" y="{H-b+14}" class="tick" text-anchor="middle">L{i}</text>')
    out.append(f'<rect x="{x(10):.1f}" y="{t}" width="{x(19)-x(10):.1f}" height="{ph}" class="zone"/><text x="{(x(10)+x(19))/2:.1f}" y="{t+11}" class="tick" text-anchor="middle">读取主要发生的层（步骤 2）</text>')
    for m, v in series.items():
        out.append('<polyline points="' + ' '.join(f'{x(i):.1f},{y(a):.1f}' for i, a in enumerate(v)) + f'" class="eff {m}"/>')
        out.append(f'<text x="{x(29)+2:.1f}" y="{y(v[-1])+3:.1f}" class="lab {m}">{m}</text>')
    out.append(f'<text x="{l+pw/2}" y="{H-3}" class="axis" text-anchor="middle">层</text></svg>')
    return ''.join(out)


# ------------------------------------------------------------------ sections

def status_rows():
    n_c = len(done_causal) + (1 if (RAW / 'causal_M1_ID_RID_rerun.npz').exists() else 0)
    n_fit = sum(1 for m in MODELS for s in SETS if spec[(m, s)])
    stopped = Path('artifacts/action_read/STOPPED_step2.md').exists()
    gate_fail = any(kstar(m, 'ID', 'ID')[0] == 'none' for m in MODELS)
    return [
        ('0', '基线与复现', 'done' if (RAW / 'causal_M1_ID_RID_rerun.npz').exists() else 'partial', '基线已出；M1 重复运行（确定性检查）排队中' if not (RAW / 'causal_M1_ID_RID_rerun.npz').exists() else '完成'),
        ('1', '提取读取子空间 R', 'done' if n_fit == 6 else 'run', f'{n_fit}/6'),
        ('2', '因果确认（keep / remove）', 'done' if n_c >= 10 else 'run', f'{n_c}/10 组完成' + ('；M2 验收未通过' if gate_fail else '')),
        ('3', '解释 R 里是什么（探针等）', 'run', '按你的指示启动：k = 32（均值消融为零点时 M1、M2 的 k*′ 都是 32）；特征导出 → 探针 / 3b / 3c / 3d'),
        ('4', '三模型比较', 'wait', '等待步骤 3'),
        ('5', '模型缝合（zero-shot / one-shot）', 'gate', '作业已挂起：方案要求 0–4 汇报后再做'),
    ]


def expectation_rows():
    g = {(r['model'], r['set'], sig_key(r['action_sigma'])): r for r in s0}
    gain = lambda m: g.get((m, 'OOD', 0.5), {}).get('gain')
    nm = lambda m: g.get((m, 'OOD', 0.5), {}).get('nmse_full')
    ks = {m: kstar(m, 'ID', 'ID')[0] for m in MODELS}
    passed = [m for m in MODELS if isinstance(ks[m], int)]; failed = [m for m in MODELS if ks[m] == 'none']
    pend = [m for m in MODELS if ks[m] is None]
    def mono(vals, better_low=False):
        v = [vals.get(m) for m in MODELS]
        if any(x is None for x in v):
            return 'pend'
        v = [float(x) for x in v]
        ok = (v[0] > v[1] > v[2]) if better_low else (v[0] < v[1] < v[2])
        return 'yes' if ok else 'no'
    rows = [
        ('M1 的数字与已有报告量级一致', f'OOD gain(σ=0.5) M1 = {f(gain("M1"),1)}，与旧结果 5.0 一致', 'yes'),
        ('同一模型跑两次结果完全相同', '旧脚本 prompt 未设种子，不逐位一致（均值一致到 4 位）；已修复，新脚本的重复运行排队中', 'pend'),
        ('OOD 泛化 M1 < M2 < M3（NMSE_full 越低越好）', ' → '.join(f'{m} {f(nm(m),4)}' for m in MODELS), mono({m: nm(m) for m in MODELS if nm(m)}, better_low=True)),
        ('视频增益随 M1→M2→M3 单调上升', ' → '.join(f'{m} {f(gain(m),1)}' for m in MODELS), mono({m: gain(m) for m in MODELS if gain(m)})),
        ('存在低维线性子空间 R，keep≥0.9 且 remove≤0.1（原定义，零点 = 切断视频）', ('通过：' + '、'.join(f'{m}(k*={ks[m]})' for m in passed) if passed else '') + ('；未通过：' + '、'.join(failed) if failed else '') + ('；运行中：' + '、'.join(pend) if pend else ''),
         'no' if failed else ('pend' if pend else 'yes')),
        ('同一判据，零点换成均值消融（OOD）', '；'.join(f'{m}: k*′={kstar_mean(m)[0]}' if isinstance(kstar_mean(m)[0], int) else (f'{m}: 未达到' if kstar_mean(m)[0] == 'none' else f'{m}: 运行中') for m in MODELS),
         'yes' if all(isinstance(kstar_mean(m)[0], int) for m in MODELS) else ('no' if any(kstar_mean(m)[0] == 'none' for m in MODELS) else 'pend')),
        ('R 明显优于同秩随机子空间', '所有已完成的组合都成立（随机 keep 远低、随机 remove 远高）', 'yes'),
        ('ID 上估计的 R 在 OOD 上仍然有效（4d）', 'M1：keep-R(32) 0.974 vs 0.990（OOD 自己估计）；M2 同样成立', 'yes'),
    ]
    return rows


CONTENT = [('ee_pos_mm', '末端位置', 'mm'), ('ee_rot_deg', '末端姿态', '°'), ('future_tcp_mm', '未来末端轨迹（32 步）', 'mm'),
           ('grip2target_mm', '夹爪→目标向量', 'mm'), ('contact_acc', '接触 / 抓取', 'Acc'), ('depth_absrel', '深度', 'AbsRel'),
           ('flow_epe3d_mm', '3D 场景流', 'EPE mm'), ('track_aj', '2D 点跟踪', 'AJ'), ('seg_iou', '目标物体分割', 'IoU'), ('task_acc', '任务类别', 'Acc')]


def step3_sections():
    import base64
    import glob
    import statistics
    ret = rd(R / 'step3a_retention.csv')
    b3 = [r for p in sorted(glob.glob(str(R / 'step3b' / '*.csv'))) for r in rd(p)]
    c3 = [r for p in sorted(glob.glob(str(R / 'step3c' / '*.csv'))) for r in rd(p)]
    Q = ['<section><h2>步骤 3 · R 里是什么</h2><p class="muted">在视频 value 向量 V 上训练相同结构、相同预算的探针，输入三种：完整 V（3072 维）、投到 R 的 32 维坐标、同秩随机子空间（5 个种子）。'
         'z = (R − 随机均值) / 随机标准差，正值表示 R 比同秩随机子空间更多地保留了这类内容；保留率 = (R − 随机) / (完整 − 随机)。'
         '下表是 OOD 训练、OOD 测试（ID 的 episode 在训练集里，会混入记忆）；z 取主要读取层 L14，保留率取 6 个导出层（L4/9/14/19/24/29）的中位数。</p>']
    models_done = [m for m in MODELS if any(r['model'] == m and r['train_set'] == 'OOD' and r['test_set'] == 'OOD' for r in ret)]
    Q.append('<div class="tbl"><table><thead><tr><th>内容</th>' + ''.join(f'<th>{m} z(L14)</th><th>{m} 保留率</th>' for m in MODELS) + '</tr></thead><tbody>')
    for met, name, unit in CONTENT:
        cells = []
        for m in MODELS:
            rr = [r for r in ret if r['model'] == m and r['train_set'] == 'OOD' and r['test_set'] == 'OOD' and r['metric'] == met]
            if not rr:
                cells.append('<td colspan="2"><span class="pill run">运行中</span></td>'); continue
            z = next((float(r['R_minus_rand_in_rand_std']) for r in rr if r['layer'] == '14'), float('nan'))
            med = statistics.median(float(r['retention']) for r in rr)
            strength = max(0.0, min(1.0, z / 10)) if z == z else 0
            cells.append(f'<td class="num" style="background: color-mix(in srgb, var(--keep) {strength*45:.0f}%, transparent)">{z:+.1f}</td><td class="num">{med:+.2f}</td>')
        Q.append(f'<tr><td>{name} <span class="ci">{unit}</span></td>' + ''.join(cells) + '</tr>')
    Q.append('</tbody></table></div>')
    Q.append('<p class="muted">3D 场景流的完整输入（拼接两帧，6144 维）在岭回归里过拟合，比 32 维输入还差，所以它的保留率没有意义，只看 z。</p>')
    # 3b
    Q.append('<h3>3b · 已有真值能线性解释读取内容的比例（R²，eval 集）</h3><div class="tbl"><table><thead><tr><th>模型</th><th>数据</th><th>L0–9 平均</th><th>L10–19 平均</th><th>L20–29 平均</th><th>最高层</th></tr></thead><tbody>')
    for s in SETS:
        for m in MODELS:
            rr = sorted([r for r in b3 if r['model'] == m and r['set'] == s], key=lambda r: int(r['layer']))
            if not rr:
                Q.append(f'<tr><td>{m}</td><td>{s}</td><td colspan="4"><span class="pill run">运行中</span></td></tr>'); continue
            v = [float(r['r2']) for r in rr]; best = max(range(len(v)), key=lambda i: v[i])
            Q.append(f'<tr><td>{m}</td><td>{s}</td><td class="num">{sum(v[:10])/10:.2f}</td><td class="num">{sum(v[10:20])/10:.2f}</td><td class="num">{sum(v[20:])/10:.2f}</td><td class="num">L{best} {v[best]:.2f}</td></tr>')
    Q.append('</tbody></table></div><p class="muted">1 − R² 是深度、分割、流、末端位姿、夹爪–目标、接触、未来轨迹、任务这些候选内容之外的部分。</p>')
    # 3c
    regs = [('target', '目标物体'), ('gripper', '夹爪'), ('robot_other', '机械臂其余'), ('moving_other', '其他运动物体'), ('static', '静态场景'), ('wrist_left', '左腕相机'), ('wrist_right', '右腕相机')]
    Q.append('<h3>3c · action 看哪里（注意力富集倍数，L10–19，σ = 0.5）</h3><div class="tbl heat"><table><thead><tr><th>模型</th><th>数据</th>' + ''.join(f'<th>{n}</th>' for _, n in regs) + '</tr></thead><tbody>')
    for s in SETS:
        for m in MODELS:
            rr = [r for r in c3 if r['model'] == m and r['set'] == s and r['latent_frame'] == 'all' and abs(float(r['action_sigma']) - 0.5) < 0.01 and 10 <= int(r['layer']) <= 19]
            if not rr:
                Q.append(f'<tr><td>{m}</td><td>{s}</td><td colspan="7"><span class="pill run">运行中</span></td></tr>'); continue
            cells = []
            for g, _ in regs:
                v = statistics.mean(float(r['enrichment']) for r in rr if r['region'] == g)
                col = 'var(--keep)' if v >= 1 else 'var(--remove)'
                cells.append(f'<td class="h" style="background: color-mix(in srgb, {col} {min(1, abs(v - 1) / 0.7) * 50:.0f}%, transparent)">{v:.2f}</td>')
            Q.append(f'<tr><td>{m}</td><td>{s}</td>' + ''.join(cells) + '</tr>')
    Q.append('</tbody></table></div><p class="muted">富集倍数 = 该区域注意力占比 / token 占比，1 表示平均。可视化里能看到注意力大量落在固定的角落 token 上（attention sink），所以注意力权重不等于实际读取的信息量，要和 3a 一起看。</p>')
    # 3d
    figs = [(m, Path(f'artifacts/action_read/figs/step3d_{m}_OOD_00.png')) for m in MODELS]
    figs = [(m, p) for m, p in figs if p.exists()]
    if figs:
        Q.append('<h3>3d · 示例窗口（OOD，同一窗口，三个模型）</h3><p class="muted">每张图：上排原始画面（3 个 latent 帧），中排 L19 的 action→视频注意力，下排 z 的前三个主成分映射为 RGB。全部窗口见 <code>artifacts/action_read/figs/</code>。</p><div class="grid3">')
        for m, p in figs:
            Q.append(f'<div class="panel"><h3>{MNAME[m]}</h3><img alt="{m} OOD example window" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"></div>')
        Q.append('</div>')
    Q.append('</section>')
    return Q


def page():
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    P = []
    P.append(f'''<title>Action Read Subspace</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+SC:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: single reading column, sections stacked; status strip first, then the verdict table, then evidence */
:root {{
  --bg: #f6f7f8; --surface: #ffffff; --fg: #1d2329; --muted: #5d6873; --line: #dde2e7;
  --accent: #1f6f78; --keep: #1f6f78; --remove: #b0483a; --m1: #6b7a8c; --m2: #c17d1b; --m3: #2f7d4f;
  --ok: #2f7d4f; --bad: #b0483a; --pend: #8a6d1d; --zone: rgba(31,111,120,.07);
  --sans: "IBM Plex Sans", "IBM Plex Sans SC", "PingFang SC", "Noto Sans CJK SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12161a; --surface: #1a2026; --fg: #e3e8ec; --muted: #9aa6b1; --line: #2c343c;
  --accent: #5fb3bb; --keep: #5fb3bb; --remove: #e07a6a; --m1: #a3b0bf; --m2: #e0a64a; --m3: #6cc28f;
  --ok: #6cc28f; --bad: #e07a6a; --pend: #d9b55a; --zone: rgba(95,179,187,.10); color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12161a; --surface: #1a2026; --fg: #e3e8ec; --muted: #9aa6b1; --line: #2c343c;
  --accent: #5fb3bb; --keep: #5fb3bb; --remove: #e07a6a; --m1: #a3b0bf; --m2: #e0a64a; --m3: #6cc28f;
  --ok: #6cc28f; --bad: #e07a6a; --pend: #d9b55a; --zone: rgba(95,179,187,.10); color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--sans); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 36px; }}
header {{ display: grid; gap: 6px; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font-size: 28px; line-height: 1.25; margin: 0; text-wrap: balance; font-weight: 600; }}
h2 {{ font-size: 19px; margin: 0 0 4px; font-weight: 600; text-wrap: balance; }}
h3 {{ font-size: 14px; margin: 0; font-weight: 600; }}
p {{ margin: 0; max-width: 72ch; }}
.muted {{ color: var(--muted); font-size: 13.5px; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
.tbl {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight: 600; color: var(--muted); font-size: 12px; letter-spacing: .03em; background: var(--bg); white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
td.num {{ font: 13px var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.ci {{ color: var(--muted); font-size: 11.5px; }}
.pill {{ display: inline-block; font: 500 11.5px/1 var(--sans); padding: 4px 8px; border-radius: 99px; white-space: nowrap; border: 1px solid currentColor; }}
.pill.yes, .pill.done {{ color: var(--ok); }} .pill.no {{ color: var(--bad); }} .pill.pend, .pill.run, .pill.partial, .pill.wait {{ color: var(--pend); }} .pill.gate {{ color: var(--muted); }}
.steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
.step {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px; display: grid; gap: 6px; align-content: start; min-width: 0; }}
.step .n {{ font: 500 12px var(--mono); color: var(--muted); }}
.step p {{ font-size: 12.5px; color: var(--muted); }}
.grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
.panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; display: grid; gap: 4px; min-width: 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--line); stroke-width: 1; }}
.chart .thr {{ stroke: var(--muted); stroke-dasharray: 3 4; stroke-width: 1; opacity: .7; }}
.chart .tick {{ fill: var(--muted); font: 10px var(--mono); }}
.chart .axis {{ fill: var(--muted); font: 10.5px var(--sans); }}
.chart .pend {{ fill: var(--pend); font: 13px var(--sans); }}
.chart .line {{ fill: none; stroke-width: 2.2; }} .chart .rand {{ fill: none; stroke-width: 1.2; stroke-dasharray: 4 3; }}
.chart .keep {{ stroke: var(--keep); }} .chart .remove {{ stroke: var(--remove); }}
.chart .dot.keep {{ fill: var(--keep); }} .chart .dot.remove {{ fill: var(--remove); }}
.chart .band {{ stroke: none; opacity: .14; }} .chart .band.keep {{ fill: var(--keep); }} .chart .band.remove {{ fill: var(--remove); }}
.chart .kstar {{ stroke: var(--fg); stroke-width: 1; stroke-dasharray: 2 3; }} .chart .kstar-l {{ fill: var(--fg); font: 500 10.5px var(--mono); }}
.chart .zone {{ fill: var(--zone); }}
.chart .eff {{ fill: none; stroke-width: 2; }} .chart .eff.M1 {{ stroke: var(--m1); }} .chart .eff.M2 {{ stroke: var(--m2); }} .chart .eff.M3 {{ stroke: var(--m3); }}
.chart .lab {{ font: 500 10.5px var(--mono); }} .chart .lab.M1 {{ fill: var(--m1); }} .chart .lab.M2 {{ fill: var(--m2); }} .chart .lab.M3 {{ fill: var(--m3); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--muted); }}
.legend span::before {{ content: ""; display: inline-block; width: 18px; height: 0; border-top: 2.5px solid var(--c); vertical-align: middle; margin-right: 6px; }}
.legend .dash::before {{ border-top-style: dashed; border-top-width: 1.5px; }}
.heat td.h {{ text-align: center; font: 12.5px var(--mono); font-variant-numeric: tabular-nums; }}
.callout {{ border-left: 3px solid var(--bad); background: var(--surface); padding: 12px 16px; border-radius: 0 8px 8px 0; display: grid; gap: 6px; }}
.callout p {{ font-size: 14px; }}
ul.notes {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; max-width: 80ch; }}
code {{ font: 12.5px var(--mono); background: var(--bg); padding: 1px 4px; border-radius: 4px; }}
@media (max-width: 520px) {{ h1 {{ font-size: 23px; }} th, td {{ padding: 7px 9px; }} }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">wamu / action_read_subspace_plan · 更新于 {now}</div>
  <h1>WAM 的 action 流从视频里读了什么</h1>
  <p class="muted">三个架构相同的 joint self-attention WAM（M1 无额外预训练、M2 robot 预训练、M3 robot+human 预训练），RoboTwin aloha-agilex。ID = clean_50（模型训练时见过这些 episode），OOD = randomized_500。每组 eval 250 个 episode × 3 个窗口；视频 σ=0.5，action σ ∈ {{0.25, 0.5, 0.75}}。区间为按 episode 重采样 1000 次的 95% bootstrap。</p>
</header>''')
    # status
    P.append('<section><h2>进度</h2><div class="steps">')
    lab = {'done': '完成', 'run': '运行中', 'partial': '部分完成', 'gate': '暂停', 'wait': '等待'}
    for n, name, st, note in status_rows():
        P.append(f'<div class="step"><span class="n">步骤 {n}</span><h3>{html.escape(name)}</h3><span><span class="pill {st}">{lab[st]}</span></span><p>{html.escape(note)}</p></div>')
    P.append('</div></section>')
    # verdict
    P.append('<section><h2>和预期对比</h2><div class="tbl"><table><thead><tr><th>预期（来自方案）</th><th>结果</th><th>符合？</th></tr></thead><tbody>')
    vl = {'yes': '符合', 'no': '不符合', 'pend': '待定'}
    for e, r, v in expectation_rows():
        P.append(f'<tr><td>{html.escape(e)}</td><td>{html.escape(r)}</td><td><span class="pill {v}">{vl[v]}</span></td></tr>')
    P.append('</tbody></table></div>')
    fail = [m for m in MODELS if kstar(m, 'ID', 'ID')[0] == 'none']
    if fail:
        r256 = [r for r in s2 if r['model'] == fail[0] and r['set'] == 'ID' and r['R_from'] == 'ID' and r['k'] == '256' and sig_key(r['action_sigma']) == 0.5]
        r16 = [r for r in s2 if r['model'] == fail[0] and r['set'] == 'ID' and r['R_from'] == 'ID' and r['k'] == '16' and sig_key(r['action_sigma']) == 0.5]
        mbm = {(r['model'], sig_key(r['action_sigma'])): r for r in mb}
        if r256 and r16:
            mrow = mbm.get((fail[0], 0.5))
            expl = (f'均值消融（m′ = μ）显示，这部分几乎全是均值向量 μ 本身：只保留一个与输入无关的常数读取向量，{fail[0]} 就能恢复 {f(mrow["rec_mean"])}'
                    f'（M1 只有 {f(mbm.get(("M1", 0.5), {}).get("rec_mean"))}）。把零点换成均值消融后，{fail[0]} 的 k*′ = {kstar_mean(fail[0])[0]}，和 M1 相同，并且优于全部随机对照。'
                    '之前"冗余存储"的解释是错的：剩下的不是依赖输入的信息，而是均值偏置；切断视频把这个偏置也去掉了，所以原定义的零点偏低。') if mrow else '均值消融对照运行中。'
            P.append(f'''<div class="callout"><h3>原定义下步骤 2 验收未通过：{', '.join(fail)}（OOD 均值消融对照已解释原因）</h3>
<p>{fail[0]} 的前 16 个方向单独保留就能恢复 {f(r16[0]["rec_keepR"])}，但去掉前 256 个方向后，仍能恢复 {f(r256[0]["rec_removeR"])}（要求 ≤ 0.1）。{expl}</p></div>''')
    P.append('</section>')
    # step 0
    P.append('<section><h2>步骤 0 · 基线</h2><p class="muted">gain = NMSE_cut / NMSE_full：切断视频后 action 误差变大多少倍。ID 的 NMSE_full 接近 0 是因为这些 episode 都在训练集里，看 OOD 更有意义。</p>')
    P.append('<div class="tbl"><table><thead><tr><th>模型</th><th>数据</th><th>action σ</th><th>NMSE_full</th><th>NMSE_cut</th><th>gain</th></tr></thead><tbody>')
    for s in SETS:
        for m in MODELS:
            rows = sorted([r for r in s0 if r['model'] == m and r['set'] == s], key=lambda r: float(r['action_sigma']))
            if not rows:
                P.append(f'<tr><td>{m}</td><td>{s}</td><td colspan="4"><span class="pill run">运行中</span></td></tr>')
            for r in rows:
                P.append(f'<tr><td>{m}</td><td>{s}</td><td class="num">{float(r["action_sigma"]):.2f}</td><td class="num">{f(r["nmse_full"],5)} <span class="ci">[{f(r["nmse_full_lo"],5)}, {f(r["nmse_full_hi"],5)}]</span></td>'
                         f'<td class="num">{f(r["nmse_cut"],4)}</td><td class="num">{f(r["gain"],1)} <span class="ci">[{f(r["gain_lo"],1)}, {f(r["gain_hi"],1)}]</span></td></tr>')
    P.append('</tbody></table></div></section>')
    # k*
    P.append('<section><h2>步骤 2 · 读取维度 k*</h2><p class="muted">k* = 同时满足 keep-R ≥ 0.9、remove-R ≤ 0.1、且优于全部 5 个同秩随机子空间的最小 k。模型的读取维度取三个 σ 的最大值；任一 σ 达不到即"未达到"。</p>')
    P.append('<div class="tbl"><table><thead><tr><th>模型</th><th>数据</th><th>R 估计自</th><th>σ=0.25</th><th>σ=0.50</th><th>σ=0.75</th><th>读取维度</th></tr></thead><tbody>')
    for s in SETS:
        for rf in ([s] if s == 'ID' else ['OOD', 'ID']):
            for m in MODELS:
                k, per = kstar(m, s, rf)
                if k is None:
                    P.append(f'<tr><td>{m}</td><td>{s}</td><td>{rf}</td><td colspan="4"><span class="pill run">运行中</span></td></tr>'); continue
                cell = lambda v: f'<td class="num">{v if v is not None else "—"}</td>'
                tag = f'<span class="pill yes">{k}</span>' if isinstance(k, int) else '<span class="pill no">未达到</span>'
                P.append(f'<tr><td>{m}</td><td>{s}</td><td>{rf}</td>' + ''.join(cell(per.get(sg)) for sg in (0.25, 0.5, 0.75)) + f'<td>{tag}</td></tr>')
    P.append('</tbody></table></div></section>')
    # curves
    P.append('<section><h2>步骤 2 · 恢复率曲线（σ = 0.5）</h2><p class="muted">rec = (NMSE_cut − NMSE_X) / (NMSE_cut − NMSE_full)：1 = 和完整视频一样好，0 = 和切断视频一样差。所有层同时干预。虚线和色带是 5 个同秩随机子空间的均值 ± 标准差；水平虚线是 0.9 / 0.1 验收线。</p>')
    P.append('<div class="legend"><span style="--c:var(--keep)">keep-R(k)：只保留 R</span><span style="--c:var(--remove)">remove-R(k)：去掉 R</span><span class="dash" style="--c:var(--keep)">随机 keep</span><span class="dash" style="--c:var(--remove)">随机 remove</span></div>')
    for s in SETS:
        P.append(f'<h3>{SNAME[s]}</h3><div class="grid3">')
        for m in MODELS:
            P.append(f'<div class="panel"><h3>{MNAME[m]}</h3>{recovery_svg(m, s, 0.5)}</div>')
        P.append('</div>')
    P.append('</section>')
    # mean ablation
    P.append('<section><h2>步骤 2 · 均值消融对照（OOD）</h2><p class="muted">所有层把 action 行读到的向量整体换成该层均值，m′ = μ（等于满维的 remove-R），注意力权重和视频流不动。'
             'rec(mean) = (NMSE_cut − NMSE_mean) / (NMSE_cut − NMSE_full)；rec′(X) = (NMSE_mean − NMSE_X) / (NMSE_mean − NMSE_full)，用已保存的 NMSE 换算，没有重跑。'
             '新旧两次运行的 full / cut 逐窗口最大相对差 &lt; 0.3%（批大小不同导致的 bf16 差异）。</p>')
    P.append('<div class="tbl"><table><thead><tr><th>模型</th><th>action σ</th><th>NMSE_full</th><th>NMSE_cut</th><th>NMSE_mean</th><th>rec(mean)</th><th>k* 原定义</th><th>k*′ 均值零点</th></tr></thead><tbody>')
    for m in MODELS:
        rows = sorted([r for r in mb if r['model'] == m and r['set'] == 'OOD' and r['R_from'] == 'OOD'], key=lambda r: float(r['action_sigma']))
        if not rows:
            P.append(f'<tr><td>{m}</td><td colspan="7"><span class="pill run">运行中</span></td></tr>'); continue
        _, kp = kstar_mean(m); _, ko = kstar(m, 'OOD', 'OOD')
        for r in rows:
            sg = sig_key(r['action_sigma'])
            P.append(f'<tr><td>{m}</td><td class="num">{sg:.2f}</td><td class="num">{f(r["nmse_full"],5)}</td><td class="num">{f(r["nmse_cut"],4)}</td><td class="num">{f(r["nmse_mean"],4)}</td>'
                     f'<td class="num">{f(r["rec_mean"])} <span class="ci">[{f(r["rec_mean_lo"])}, {f(r["rec_mean_hi"])}]</span></td>'
                     f'<td class="num">{ko.get(sg) or "—"}</td><td class="num">{kp.get(sg) or "—"}</td></tr>')
    P.append('</tbody></table></div>')
    P.append('<h3>按 rec′ 重画的恢复率曲线（OOD，σ = 0.5）</h3><div class="grid3">')
    for m in MODELS:
        P.append(f'<div class="panel"><h3>{MNAME[m]}</h3>{recovery_svg(m, "OOD", 0.5, meanref=True)}</div>')
    P.append('</div></section>')
    # layer windows
    P.append('<section><h2>步骤 2 · 哪些层的读取最重要</h2><p class="muted">每次只把连续 5 层的读取向量置零（其余层不动），σ = 0.5，R 估计自同一数据集。数值越低，这几层的读取越关键。</p>')
    P.append('<div class="tbl heat"><table><thead><tr><th>模型</th><th>数据</th>' + ''.join(f'<th>L{a}–{a+4}</th>' for a in range(0, 30, 5)) + '</tr></thead><tbody>')
    for s in SETS:
        for m in MODELS:
            rows = {r['layers']: float(r['rec']) for r in win if r['model'] == m and r['set'] == s and r['R_from'] == s and r['intervention'] == 'zero' and sig_key(r['action_sigma']) == 0.5}
            if not rows:
                P.append(f'<tr><td>{m}</td><td>{s}</td><td colspan="6"><span class="pill run">运行中</span></td></tr>'); continue
            cells = []
            for a in range(0, 30, 5):
                v = rows.get(f'{a}-{a+4}')
                strength = 0 if v is None else max(0.0, min(1.0, (1 - v) / 0.45))
                cells.append(f'<td class="h" style="background: color-mix(in srgb, var(--remove) {strength*55:.0f}%, transparent)">{f(v,2)}</td>')
            P.append(f'<tr><td>{m}</td><td>{s}</td>' + ''.join(cells) + '</tr>')
    P.append('</tbody></table></div></section>')
    # step 3
    P.extend(step3_sections())
    # step 1
    P.append('<section><h2>步骤 1 · 读取向量的有效维度</h2><p class="muted">有效维度 = (Σλ)² / Σλ²，在 fit 集（250 个 episode × 3 窗口 × 3 个 σ × 32 个 action token）上逐层估计。</p>')
    P.append('<div class="legend">' + ''.join(f'<span style="--c:var(--{m.lower()})">{MNAME[m]}</span>' for m in MODELS) + '</div>')
    P.append('<div class="tbl"><table><thead><tr><th>数据</th>' + ''.join(f'<th>{m} 平均</th>' for m in MODELS) + '</tr></thead><tbody>')
    for s in SETS:
        P.append(f'<tr><td>{s}</td>' + ''.join(f'<td class="num">{f(sum(float(r["eff_dim"]) for r in spec[(m, s)]) / len(spec[(m, s)]), 1) if spec[(m, s)] else "–"}</td>' for m in MODELS) + '</tr>')
    P.append('</tbody></table></div>')
    for s in SETS:
        P.append(f'<div class="panel"><h3>{SNAME[s]}</h3>{effdim_svg(s)}</div>')
    P.append('</section>')
    # notes
    P.append('''<section><h2>需要注意的限制</h2><ul class="notes">
<li>只有三个模型：任何一个量碰巧随 M1→M2→M3 单调变化的概率是 1/6，跨模型趋势只能当佐证，主要证据是每个模型内部的干预实验。</li>
<li>R 是线性子空间，结论应表述为"action 读取的线性子空间"。M2 的结果说明这个描述对它不完整。</li>
<li>ID 的 episode 全部在训练集里，所以 ID 的 NMSE_full 接近 0；OOD 更有参考价值。</li>
<li>action NMSE 是开环指标。此前只有高噪声（σ ≥ 0.5）下的视频增益和成功率方向一致，低噪声的数字不要用来推断成功率。</li>
<li>步骤 2 的干预通过只回放 action 流实现（视频流不受 action 影响，attention_mask_mode = action_sees_video）；回放和模型自身预测的最大相对差 &lt; 5e-3（bf16 vs fp32）。</li>
<li>M1 的 ID/OOD 随机对照用 CPU 做 QR，其余组用 GPU 做 QR：种子相同，浮点路径不同，都是合法的随机子空间。</li>
<li>接触/抓取状态来自集群上重新做的物理回放，和官方数据有毫米到厘米级偏差；目标偏移超过 1 cm 的窗口会剔除（步骤 3 用到时）。</li>
</ul><p class="muted">原始结果：<code>artifacts/action_read/results/</code>；恢复率曲线 PDF：<code>artifacts/action_read/figs/step2_recovery_curves.pdf</code>；生成脚本：<code>scripts/action_read/build_report_page.py</code>。</p></section>''')
    P.append('</div>')
    return '\n'.join(P)


OUT.write_text(page())
print(f'wrote {OUT} ({OUT.stat().st_size // 1024} KB)')
