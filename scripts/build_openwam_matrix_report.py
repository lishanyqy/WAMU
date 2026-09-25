#!/usr/bin/env python3
"""Build the requested model-by-modality matrix report with layer mini-bars."""
import csv, html, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'openwam_matrix_report.html'
MODELS=[
 ('robotwin_dual_system_joint_self_attention','Joint self-attention','bidirectional video/action self-attention','#2ca25f'),
 ('robotwin_dual_system_joint_cross_attention','Joint cross-attention','action attends to video; video is independent','#1769aa'),
 ('robotwin_dual_system_joint_cross_attention_detach','Cross-attention + detach','action-to-video gradient is detached','#d95f02'),
 ('robotwin_dual_system_joint_self_attention_isolated','Self-attention isolated','action sees only the first video frame','#756bb1'),
 ('robotwin_dual_system_joint_self_attention_video_sees_action','Self-attention video→action','video sees action; action sees first frame','#e7298a'),
 ('robotwin_dual_system_idm','IDM','video final states condition action transformer','#c44e52'),
 ('robotwin_dual_system_joint_self_attention_mutual','Self-attention mutual','both modalities mutually visible','#7a5195'),
]
REFS=[('dinov2','DINOv2'),('depthv3','Depthv3'),('d4rt','D4RT'),('videomaev2','VideoMAEv2')]
METRICS=[('mknn','m-kNN','#2878b5'),('cka','Linear CKA','#2a9d8f'),('rbf','RBF CKA','#e09f3e')]
PERF={'robotwin_dual_system_joint_self_attention_isolated':(88.08,86.74,87.41),'robotwin_dual_system_joint_self_attention_video_sees_action':(87.92,87.34,87.63),'robotwin_dual_system_joint_self_attention':(92.98,91.80,92.39),'robotwin_dual_system_joint_self_attention_mutual':(92.50,91.80,92.15),'robotwin_dual_system_joint_cross_attention':(87.86,88.64,88.25),'robotwin_dual_system_joint_cross_attention_detach':(92.06,91.64,91.85),'robotwin_dual_system_idm':(87.76,88.14,87.95)}

def rows_csv(path):
    if not path.exists(): return []
    with path.open() as f: return list(csv.DictReader(f))

def vals_for(ds,model,ref,metric):
    if metric=='mknn':
        p=ROOT/'artifacts'/f'{ds}_mknn'/ref/model/'scores.csv'
        rows=rows_csv(p)
        # The m-kNN script uses a short internal model label; the file is already model-specific.
        return [(int(r['video_layer']),float(r['score'])) for r in rows if math.isfinite(float(r['score']))]
    if metric=='cka':
        rows=rows_csv(ROOT/'artifacts'/'cka'/ds/'best_layer_cka.csv')
        return [(int(r['video_layer']),float(r['cka'])) for r in rows if r['model']==model and r['reference']==ref and math.isfinite(float(r['cka']))]
    rows=rows_csv(ROOT/'artifacts'/'cka_rbf'/ds/'best_layer_rbf_cka.csv')
    return [(int(r['video_layer']),float(r['rbf_cka'])) for r in rows if r['model']==model and r['reference']==ref and math.isfinite(float(r['rbf_cka']))]

def all_best(ds,ref,metric):
    out={}
    for model,_,_,_ in MODELS:
        z=vals_for(ds,model,ref,metric)
        if z: out[model]=max(v for _,v in z)
    return out

def rank(value, scores):
    return 1+sum(v>value+1e-12 for v in scores.values())

def mini_bar(ds,model,ref,metric):
    z=vals_for(ds,model,ref,metric)
    if not z: return '<div class="metric missing">no result</div>'
    color=dict((x[0],x[2]) for x in METRICS)[metric]; label=dict((x[0],x[1]) for x in METRICS)[metric]
    best_layer,best=max(z,key=lambda x:x[1]); scores=all_best(ds,ref,metric); rk=rank(best,scores)
    lo=min(v for _,v in z); hi=max(v for _,v in z); span=max(hi-lo,1e-6); w=190; h=42
    bars=[]
    for i,v in z:
        x=3+i*(w-6)/max(len(z)-1,1); bh=5+(v-lo)/span*27
        bars.append(f'<rect x="{x-1.5:.1f}" y="{h-bh-5:.1f}" width="3" height="{bh:.1f}" fill="{color}" opacity="{1.0 if i==best_layer else .60}"/>')
    return f'<div class="metric"><div class="metric-head"><b>{label}</b><span>max {best:.3f} · L{best_layer+1} · <strong>rank {rk}/{len(scores)}</strong></span></div><svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-label="{label} layer bar">{bars}<line x1="3" y1="{h-5}" x2="{w-3}" y2="{h-5}" stroke="#cbd5d8"/></svg></div>'

def arch_card(label,desc,color):
    ranks={j:1+sum(PERF[key][j]<PERF[x][j] for x in PERF) for j in range(3)} if (key:=next((k for k,v,_,_ in MODELS if v==label),None)) else {}
    perf=''.join(f'<span class="perf">{name} {PERF[key][j]:.2f} <b>R{ranks[j]}</b></span>' for j,name in enumerate(['Clean','Rand','Avg'])) if key in PERF else ''
    return f'<div class="arch-card" style="--accent:{color}"><div class="arch-title">{html.escape(label)}</div><svg viewBox="0 0 220 64"><rect x="8" y="13" width="82" height="30"/><rect x="130" y="13" width="82" height="30"/><path d="M90 23 H130 M130 35 H90"/><text x="49" y="32">video</text><text x="171" y="32">action</text></svg><div class="arch-desc">{html.escape(desc)}</div><div class="perf-row">{perf}</div></div>'

def matrix(ds):
    title='Agilex' if ds=='agilex' else 'Franka'
    heads=''.join(f'<th>{html.escape(label)}</th>' for _,label in REFS)
    body=[]
    for model,label,desc,color in MODELS:
        cells=[]
        for ref,_ in REFS:
            cells.append('<td>'+''.join(mini_bar(ds,model,ref,metric) for metric,_,_ in METRICS)+'</td>')
        body.append(f'<tr><th class="model-col">{arch_card(label,desc,color)}</th>'+''.join(cells)+'</tr>')
    return f'<section><h2>{title} · 模型 × 模态矩阵</h2><p class="muted">每个单元格依次显示 m-kNN、linear CKA、RBF CKA；柱状图横轴为层号，标注为该模型在该模态下的最佳层、最大值和跨模型排名。</p><div class="matrix-wrap"><table class="matrix"><thead><tr><th class="model-col">架构 / attention</th>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>'

def performance_section():
    ranks={j:{k:1+sum(PERF[k][j]<PERF[x][j] for x in PERF) for k in PERF} for j in range(3)}
    body=[]
    for key,label,_,_ in MODELS:
        cells=''.join(f'<td class="num">{PERF[key][j]:.2f} <span class="rank">rank {ranks[j][key]}</span></td>' for j in range(3))
        body.append(f'<tr><td>{html.escape(label)}</td>{cells}</tr>')
    return '<section><h2>性能结果（reports/performance.md）</h2><p class="muted">Clean、Rand.、Avg 均按降序排名，rank 1 为最高。</p><table><thead><tr><th>架构</th><th>Clean</th><th>Rand.</th><th>Avg</th></tr></thead><tbody>'+''.join(body)+'</tbody></table></section>'

def main():
    OUT.parent.mkdir(exist_ok=True)
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM 模型×模态相似度矩阵</title><style>
body{margin:0;background:#f3f7f8;color:#263640;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}main{max-width:1800px;margin:auto;padding:28px 30px 70px}header{background:linear-gradient(135deg,#123b52,#287b7a);color:#fff;border-radius:18px;padding:25px 30px;box-shadow:0 8px 25px #17434c22}h1{margin:0 0 5px;font-size:29px}h2{margin:35px 0 10px;border-left:5px solid #2a9d8f;padding-left:11px}p{line-height:1.6}.muted{color:#667881;font-size:13px}.matrix-wrap{overflow-x:auto;border:1px solid #dbe6e8;border-radius:14px;background:#fff;box-shadow:0 4px 18px #244a550c}.matrix{border-collapse:separate;border-spacing:0;width:100%;min-width:1240px}.matrix th,.matrix td{border-right:1px solid #e1eaec;border-bottom:1px solid #e1eaec;vertical-align:top}.matrix thead th{position:sticky;top:0;background:#e3f0f0;color:#2d5b61;padding:11px;text-align:center;z-index:2}.matrix .model-col{width:280px;min-width:280px;position:sticky;left:0;z-index:3;background:#f8fbfb}.matrix tbody .model-col{padding:8px}.matrix td{padding:7px;min-width:290px}.arch-card{border-left:4px solid var(--accent);padding:6px 8px;background:#fff}.arch-title{font-weight:750;color:#294856;font-size:14px}.arch-card svg{width:100%;height:52px}.arch-card rect{fill:#f4f8f9;stroke:var(--accent);stroke-width:1.5}.arch-card path{fill:none;stroke:var(--accent);stroke-width:2;marker-end:url(#arrow)}.arch-card text{text-anchor:middle;font-size:11px;fill:#566b73}.arch-desc{font-size:11px;color:#718087;line-height:1.35}.perf-row{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}.perf{font-size:10px;color:#526a70;background:#edf5f4;border-radius:4px;padding:2px 4px;white-space:nowrap}.perf b{color:#237477}.metric{margin:2px 0 7px;padding:5px 6px;border-radius:7px;background:#f8fafb;border:1px solid #e6edef}.metric-head{display:flex;justify-content:space-between;gap:5px;align-items:baseline;font-size:11px;color:#405b63}.metric-head span{font-size:10px;color:#64767d;white-space:nowrap}.metric-head strong{color:#246f73}.metric svg{width:100%;height:37px;display:block;margin-top:2px}.missing{color:#a46630;font-size:11px;padding:10px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:17px 0;color:#526870;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.note{background:#fff7e6;border-left:4px solid #dda23c;border-radius:8px;padding:11px 14px;color:#5d4b2c;font-size:13px}.rank{color:#237477;font-weight:700}code{background:#e7eff1;padding:2px 4px;border-radius:4px}@media(max-width:800px){main{padding:15px 9px}h1{font-size:23px}}
</style></head><body><main><header><h1>OpenWAM 模型 × 模态相似度矩阵</h1><div>每个单元格 = 逐层 layer bar + 最佳层 + 最大值 + 跨模型排名</div><p>用于直接比较 7 个 OpenWAM 架构在 DINOv2、Depthv3 和 D4RT 三种视觉/几何表征上的对齐表现。</p></header><div class="legend"><span><i class="dot" style="background:#2878b5"></i>m-kNN</span><span><i class="dot" style="background:#2a9d8f"></i>linear CKA</span><span><i class="dot" style="background:#e09f3e"></i>RBF CKA</span><span>rank 1 = 当前模态与指标下跨模型最高</span></div><div class="note"><b>阅读方式：</b>每行一个架构，每列一个参考模态；单元格内三条小柱状图从上到下依次为 m-kNN、linear CKA、RBF CKA。横向越高表示该层相似度越高，粗体 rank 表示该模型在当前列/指标下的排名。</div>'''+matrix('agilex')+matrix('franka')+'''<footer><p>数据来自 <code>artifacts/agilex_mknn</code>、<code>artifacts/franka_mknn</code>、<code>artifacts/cka</code> 和 <code>artifacts/cka_rbf</code>。由于部分结果使用不同的抽样规模，排名适合做当前实验内的相对比较，不应跨计算协议直接比较绝对值。</p></footer></main></body></html>'''
    doc=doc.replace('</body>', performance_section()+'</body>')
    # Add an arrow marker to the inline architecture SVGs through a harmless shared definition.
    doc=doc.replace('</style></head>','</style><svg width="0" height="0"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#467"/></marker></defs></svg></head>')
    OUT.write_text(doc,encoding='utf-8'); print(OUT)
if __name__=='__main__': main()
