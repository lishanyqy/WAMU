#!/usr/bin/env python3
"""Build a compact, self-contained HTML report from OpenWAM artifacts."""
import csv, html, json, math
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'openwam_experiment_report.html'
MODELS=[
 ('robotwin_dual_system_joint_cross_attention','Joint cross-attention','action sees video; video does not see action'),
 ('robotwin_dual_system_joint_cross_attention_detach','Joint cross-attention + detach','same mask; bridge detached from action loss'),
 ('robotwin_dual_system_joint_self_attention','Joint self-attention','video/action jointly attend; action sees video'),
 ('robotwin_dual_system_joint_self_attention_isolated','Self-attention isolated','video branch isolated; action sees first video frame'),
 ('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attention video→action','video sees action; action sees first video frame'),
 ('robotwin_dual_system_idm','IDM','video representation conditions the action branch'),
 ('robotwin_dual_system_joint_self_attention_mutual','Self-attention mutual','both modalities mutually visible'),
]
REFS=[('dinov2','DINOv2'),('depthv3','Depth Anything'),('d4rt','D4RT'),('videomaev2','VideoMAEv2')]
PERF={'Joint self-attention':(92.98,91.80,92.39),'Self-attention isolated':(88.08,86.74,87.41),'Self-attention video→action':(87.92,87.34,87.63),'Self-attention mutual':(92.50,91.80,92.15),'Joint cross-attention':(87.86,88.64,88.25),'Joint cross-attention + detach':(92.06,91.64,91.85),'IDM':(87.76,88.14,87.95)}
DATASETS=['agilex','franka']

def read_json(path):
    try: return json.loads(path.read_text())
    except Exception: return {}

def fmt(v):
    if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return '—'
    return f'{float(v):.3f}'

def cka_table(dataset, rbf=False):
    root=ROOT/'artifacts'/('cka_rbf' if rbf else 'cka')/dataset/'summary.json'
    s=read_json(root).get('summary',{})
    rows=[]
    for key,label,_ in MODELS:
        vals=[]
        for ref,ref_label in REFS:
            z=s.get(key,{}).get(ref)
            vals.append(fmt(z.get('rbf_cka') if rbf else z.get('cka')) if z else '—')
        rows.append((label,vals))
    title='RBF CKA' if rbf else 'Linear CKA'
    th=''.join(f'<th>{html.escape(x[1])}</th>' for x in REFS)
    body=''.join('<tr><td>'+html.escape(label)+'</td>'+''.join(f'<td class="num">{v}</td>' for v in vals)+'</tr>' for label,vals in rows)
    return f'<h4>{title}</h4><table><thead><tr><th>Model</th>{th}</tr></thead><tbody>{body}</tbody></table>'

def mknn_table(dataset):
    rows=[]
    for key,label,_ in MODELS:
        vals=[]
        for ref,ref_label in REFS:
            p=ROOT/'artifacts'/f'{dataset}_mknn'/ref/key/'summary.json'
            s=read_json(p).get('summary',{})
            z=next(iter(s.values()),None)
            vals.append(fmt(z.get('score')) if z else '—')
        rows.append((label,vals))
    th=''.join(f'<th>{html.escape(x[1])}</th>' for x in REFS)
    body=''.join('<tr><td>'+html.escape(label)+'</td>'+''.join(f'<td class="num">{v}</td>' for v in vals)+'</tr>' for label,vals in rows)
    return f'<h4>Mutual-kNN alignment</h4><table><thead><tr><th>Model</th>{th}</tr></thead><tbody>{body}</tbody></table>'

def img(path, caption, cls='figure'):
    rel='../'+str(path.relative_to(ROOT)).replace('\\','/')
    return f'<figure class="{cls}"><img src="{html.escape(rel)}" loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>'

def architecture_svg():
    cards=[]
    for i,(key,label,desc) in enumerate(MODELS):
        y=20+i*78
        if key.endswith('idm'):
            arrows='<path d="M220 28 H305"/><path d="M220 52 H305" stroke-dasharray="4 4"/>'
            boxes='<rect x="20" y="12" width="200" height="56"/><text x="120" y="35">Video Transformer</text><text x="120" y="54">final video states</text><rect x="305" y="12" width="150" height="56"/><text x="380" y="35">Action Transformer</text><text x="380" y="54">conditioned on video</text>'
        elif 'cross_attention' in key:
            arrows='<path d="M220 28 H305"/><path d="M220 52 H305" stroke-dasharray="4 4"/>'
            boxes='<rect x="20" y="12" width="200" height="56"/><text x="120" y="35">Video stream</text><text x="120" y="54">independent blocks</text><rect x="305" y="12" width="150" height="56"/><text x="380" y="35">Action stream</text><text x="380" y="54">cross-attn bridge</text>'
        else:
            arrows='<path d="M220 28 H305"/><path d="M305 52 H220"/>'
            boxes='<rect x="20" y="12" width="200" height="56"/><text x="120" y="35">Video tokens</text><text x="120" y="54">joint self-attention</text><rect x="305" y="12" width="150" height="56"/><text x="380" y="35">Action tokens</text><text x="380" y="54">mask-controlled</text>'
        cards.append(f'<g transform="translate(0,{y})">{boxes}{arrows}<text class="name" x="485" y="34">{html.escape(label)}</text><text class="desc" x="485" y="55">{html.escape(desc)}</text></g>')
    return '<svg class="arch" viewBox="0 0 1000 570" role="img" aria-label="OpenWAM architecture comparison"><style>rect{fill:#f7fafc;stroke:#9aa9b8;stroke-width:1.5}text{text-anchor:middle;font:13px sans-serif;fill:#25364a}.name{text-anchor:start;font-weight:700;font-size:14px}.desc{text-anchor:start;font-size:12px;fill:#607080}path{fill:none;stroke:#3b82a0;stroke-width:2}path:not([stroke-dasharray]){marker-end:url(#a)}</style><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#3b82a0"/></marker></defs>'+''.join(cards)+'</svg>'

def dataset_section(ds):
    title='Agilex' if ds=='agilex' else 'Franka'
    cka=ROOT/'artifacts'/'cka'/ds
    rbf=ROOT/'artifacts'/'cka_rbf'/ds
    mknn=ROOT/'artifacts'/f'{ds}_mknn'
    figs=''
    for ref,label in REFS:
        p=mknn/ref/'model_comparison.png'
        if p.exists(): figs+=img(p,f'{title} — {label} mutual-kNN model comparison')
    for p,cap in [(cka/'cka_grouped_bar.png',f'{title} — linear CKA model comparison'),(rbf/'rbf_cka_grouped_bar.png',f'{title} — RBF CKA model comparison')]:
        if p.exists(): figs+=img(p,cap)
    bars=''
    for root,prefix in [(cka,'Linear CKA'),(rbf,'RBF CKA')]:
        for ref,label in REFS:
            p=root/ref/'layer_bars.png'
            if p.exists(): bars+=img(p,f'{title} — {prefix}, {label}, layer-wise', 'figure small')
    return f'''<section><h2>{title}</h2><p class="muted">数据源：<code>artifacts/{ds}_features</code>；当前结果按共同文件名匹配视频。</p><div class="tables">{mknn_table(ds)}{cka_table(ds)}{cka_table(ds,True)}</div><div class="figures">{figs}</div><details><summary>逐层 layer bars</summary><div class="figures">{bars}</div></details></section>'''

def performance_section():
    ranks={j:{k:1+sum(PERF[k][j]<PERF[x][j] for x in PERF) for k in PERF} for j in range(3)}
    rows=[]
    for _,label,_ in MODELS:
        cells=''.join(f'<td class="num">{PERF[label][j]:.2f} (rank {ranks[j][label]})</td>' for j in range(3))
        rows.append(f'<tr><td>{html.escape(label)}</td>{cells}</tr>')
    return '<h2>性能结果（reports/performance.md）</h2><p class="muted">Clean、Rand.、Avg 均按降序排名，rank 1 为最高。</p><table><thead><tr><th>架构</th><th>Clean</th><th>Rand.</th><th>Avg</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'

def main():
    OUT.parent.mkdir(exist_ok=True)
    html_doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM 表征对齐实验报告</title><style>
body{{margin:0;background:#f4f7f9;color:#243447;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;line-height:1.65}}main{{max-width:1420px;margin:auto;padding:34px 44px 80px}}header{{background:linear-gradient(135deg,#103c54,#247b83);color:white;padding:34px 42px;border-radius:20px;box-shadow:0 10px 30px #173b4a22}}h1{{font-size:32px;margin:0 0 6px}}h2{{font-size:24px;margin:42px 0 14px;border-left:5px solid #3b9c9b;padding-left:12px}}h3{{color:#1d6d76;margin-top:28px}}h4{{margin:18px 0 8px;color:#46616f}}p{{margin:8px 0}}.muted{{color:#667784}}.lead{{font-size:17px;max-width:950px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:22px 0}}.card{{background:white;border:1px solid #e0e8ec;border-radius:14px;padding:16px 18px;box-shadow:0 4px 15px #24414b0b}}.card b{{font-size:25px;color:#19777d;display:block}}table{{border-collapse:collapse;background:white;width:100%;margin-bottom:18px;border-radius:10px;overflow:hidden;box-shadow:0 3px 12px #21424a0b}}th,td{{padding:8px 10px;border-bottom:1px solid #e7eef1;text-align:left;font-size:13px}}th{{background:#e6f1f1;color:#29525d}}td.num{{font-variant-numeric:tabular-nums;text-align:center;font-weight:600}}.tables{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;align-items:start}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;align-items:start}}figure{{background:white;margin:0;padding:12px;border-radius:12px;border:1px solid #e0e8ec;box-shadow:0 3px 12px #21424a0b}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:12px;color:#62717a;margin-top:8px}}figure.small{{min-width:0}}details{{margin-top:20px;background:#edf4f4;padding:12px 15px;border-radius:12px}}summary{{cursor:pointer;font-weight:700;color:#276e73}}.arch{{width:100%;height:auto;background:white;border:1px solid #dce7ea;border-radius:15px;padding:8px;box-sizing:border-box}}code{{background:#e8eff2;padding:2px 5px;border-radius:4px}}.note{{border-left:4px solid #e2a93b;background:#fff8e8;padding:12px 16px;border-radius:8px}}.ok{{border-left:4px solid #3a9b7a;background:#edf8f3;padding:12px 16px;border-radius:8px}}footer{{margin-top:44px;color:#71808a;font-size:12px;border-top:1px solid #d8e2e6;padding-top:16px}}@media(max-width:700px){{main{{padding:20px 14px}}h1{{font-size:25px}}.figures{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>OpenWAM 表征对齐实验报告</h1><div>Robotwin 2.0 · Agilex / Franka · 生成日期：{date.today().isoformat()}</div><p class="lead">比较不同 OpenWAM 双流架构是否保留视频中的语义、几何、时序与运动相关表征。报告中的数值均从当前 artifacts 自动读取。</p></header>
<div class="grid"><div class="card"><b>7</b>OpenWAM 架构</div><div class="card"><b>2</b>Robot 平台</div><div class="card"><b>4</b>参考表征</div><div class="card"><b>3</b>相似度指标</div></div>
<h2>1. 实验流程</h2><p>每个 Robotwin 视频按照统一 manifest 处理：截取可用视频片段，最多读取 10 秒范围，并按每 4 帧抽取 1 帧。OpenWAM 在固定噪声状态下提取所有 video Transformer 层的 pooled feature，然后分别与外部视觉/几何参考表征对齐。</p><div class="grid"><div class="card"><b>50%</b>OpenWAM latent noise：<code>t=500/1000</code></div><div class="card"><b>33 → 16</b>VideoMAEv2：33 帧候选后均匀取 16 帧</div><div class="card"><b>k=10</b>mutual-kNN，q=0.95 clipping</div><div class="card"><b>24 / 30</b>VideoMAEv2 / OpenWAM 层数</div></div><div class="note"><b>解释：</b>当前 OpenWAM 架构比较不是 noise sweep；所有架构都在 50% 噪声下提取。WAMU 原始 diffusion noise 实验另行扫描 10%、30%、60%、90%。</div>
<h2>2. OpenWAM 架构</h2><p>下图用信息流和 mask/gradient 约束概括当前比较的 7 个模型。箭头表示主要条件或可见关系；虚线表示 action loss 不通过 bridge 回传到 video 分支。</p>{architecture_svg()}<table><thead><tr><th>模型</th><th>核心机制</th><th>关注点</th></tr></thead><tbody>{''.join(f'<tr><td>{html.escape(label)}</td><td>{html.escape(desc)}</td><td>{("video/action 交互" if "self" in key or "cross" in key else "video→action 条件路径")}</td></tr>' for key,label,desc in MODELS)}</tbody></table>
<h2>3. Agilex 与 Franka 结果</h2>{dataset_section('agilex')}{dataset_section('franka')}
<h2>4. 实验讨论</h2><div class="ok"><b>初步观察</b><ul><li>线性 CKA 在 DINOv2 上整体高于 Depth/D4RT，说明 OpenWAM 的 video 表征与外观/语义视觉表征更接近，而几何和点跟踪表征更难保留。</li><li>Agilex 与 Franka 的相对趋势不完全一致，因此架构差异需要结合机器人平台、动作分布和任务组成解释。</li><li>IDM 与 joint self-attention mutual 并没有在所有参考模态上稳定领先；“让 video 看 action”并不自动带来更强的视觉表征。</li><li>RBF CKA 通常高于线性 CKA，说明部分表征关系可能是非线性的；但 RBF 的数值不能直接与线性 CKA 当作同一尺度比较。</li></ul></div><div class="note"><b>解读限制</b><ul><li>当前 CKA/RBF CKA 使用统一采样子集计算；RBF 结果使用 300 个均匀采样视频，线性 CKA/m-kNN 的补算结果使用 100 个均匀采样视频，比较时需记录这一点。</li><li>Franka 的 RBF-DINOv2 结果存在 NaN，报告以“—”显示，不能用于结论。</li><li>相似度是表征对齐，不等于下游任务成功率；最好再结合 action success、motion prediction 或 point tracking accuracy。</li><li>VideoMAEv2 特征已准备为独立参考表征，后续可直接加入同样的 CKA/m-kNN pipeline，重点观察其 motion/temporal alignment。</li></ul></div>
<h2>5. 复现实验位置</h2><p>原始 noise 实验记录见 <code>wamu/EXPERIMENT_LOG.md</code>；OpenWAM 实验记录见 <code>wamu/openwam_exp_log.md</code>。主要结果目录为 <code>artifacts/cka</code>、<code>artifacts/cka_rbf</code>、<code>artifacts/agilex_mknn</code> 和 <code>artifacts/franka_mknn</code>。</p><footer>本报告为静态 HTML；图像使用相对于报告文件的本地路径加载。若移动 HTML，请同时保留 artifacts 目录结构。</footer></main></body></html>'''
    html_doc=html_doc.replace('<h2>4. 实验讨论</h2>',performance_section()+'<h2>4. 实验讨论</h2>')
    OUT.write_text(html_doc,encoding='utf-8')
    print(OUT)
if __name__=='__main__': main()
