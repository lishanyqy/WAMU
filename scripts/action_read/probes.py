#!/usr/bin/env python
"""Step 3a / 4d of wamu/action_read_subspace_plan.md: what is in the read subspace R?

For one model and one dumped layer, probes are trained on three inputs built from the video value vectors V[j]
(read_subspace.py --mode dump, concat over heads, d = 3072):
  full     V[j]
  R        z[j] = U_k^T (V[j] - mu_V)          U_k = top-k* eigenvectors of the read-vector covariance (step 1)
  rand<s>  same rank, random k-dim subspace of the top-K99 span of that covariance (seeds 0..4, as step 2's control)
(linear probes on U^T x are equivalent to probes on the projection U U^T x, at a fraction of the cost.)

Same probe structure and budget for every input:
  token level (head-camera tokens, ridge regression, closed form, lambda chosen on a 20% episode hold-out of FIT):
    depth  (log metres; AbsRel)          target segmentation (target pixel fraction; IoU of frac >= 0.5)
    flow3d (concat tokens at t, t+1 -> camera-frame 3D flow t->t+1, mm; EPE3D)
    track  (concat tokens at latent 0,1,2 -> displacement + visibility of the token centre; AJ over 1,2,4,8,16 px)
  window level (attention-pooling probe over all video tokens: Linear(in,256) keys/values, 4 learned queries, linear
  head; AdamW 1e-3, 40 epochs, batch 32; 3 seeds):
    task (50-way; Acc)  ee_pose (TCP position mm / link6 rotation deg, both arms, latent frames 0-2)
    grip2target (mm)    contact (grasp + finger-on-target, both arms; Acc)    future_tcp (32 steps, both arms; mm)
Training on the FIT half of --train-set, testing on the EVAL half of --test-set (ID->OOD for step 4d).
Output: artifacts/action_read/results/step3a_probe_<model>_L<layer>_<train>2<test>.csv
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path('artifacts/action_read')
FEAT = Path.home() / 'dataspace/action_read_features'
GH, GW, GRID_H, NF = 8, 10, 12, 3          # head token rows, cols, canvas token rows, latent frames
SV = NF * GRID_H * GW
AJ_THR = (1, 2, 4, 8, 16)


def head_token_index():
    return np.array([[[f * GRID_H * GW + r * GW + c for c in range(GW)] for r in range(GH)] for f in range(NF)])   # (3, 8, 10)


def wait_for_gt(set_name, keys, timeout_h=12):
    """The token GT is built by a separate process; wait until its file exists and covers every dumped window."""
    p = OUT / 'gt' / f'{set_name}.npz'; t0 = time.time()
    while True:
        if p.exists():
            try:
                gt = np.load(p, allow_pickle=True)
                have = {(str(t), int(e), int(s)) for t, e, s in zip(gt['task'], gt['episode'], gt['start'])}
                if all((t, int(e), int(s)) in have for t, e, s in keys):
                    return gt
            except Exception:
                pass
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit(f'GT {p} missing or incomplete after {timeout_h} h')
        print(f'waiting for complete GT {p} ...', flush=True); time.sleep(300)


def load_set(model, set_name, layer_pos):
    root = FEAT / f'{model}_{set_name}'
    keys = [tuple(k) for k in json.loads((root / 'keys.json').read_text())]
    V = np.load(root / 'V.npy', mmap_mode='r')[:, layer_pos]                                  # (N, Sv, d) fp16
    gt = wait_for_gt(set_name, keys)
    gidx = {(str(t), int(e), int(s)): i for i, (t, e, s) in enumerate(zip(gt['task'], gt['episode'], gt['start']))}
    order = np.array([gidx[(t, int(e), int(s))] for t, e, s in keys])
    G = {k: gt[k][order] for k in gt.files}
    return V, G


def bases(model, rset, layer, k, dev, novel=None):
    R = torch.load(OUT / 'R' / f'R_{model}_{rset}.pt', map_location='cpu')
    ev = R['evec'][layer].float(); K99 = int(R['k99'][layer])
    U = {'R': ev[:, :k]}
    for s in range(5):
        g = torch.Generator().manual_seed(1000 + s)                       # same random rotations as read_subspace.py's step-2 control
        Q = torch.linalg.qr(torch.randn(K99, K99, generator=g, dtype=torch.float64).to(dev))[0].float().cpu()   # same draws as read_subspace.py (QR on GPU)
        U[f'rand{s}'] = ev[:, :K99] @ Q[:, :min(k, K99)]
        if novel is not None:                                             # 4b: same-rank random control for the novel subspace
            U[f'nrand{s}'] = ev[:, :K99] @ Q[:, :novel[layer].shape[1]]
    if novel is not None:
        U['novel'] = novel[layer].float()
    return {n: u.to(dev) for n, u in U.items()}


def project(V, idx, U, mu, dev, bs=64):
    """V[idx] (n, Sv, d) fp16 memmap -> features (n, Sv, d or k), stored fp16 on dev (statistics are accumulated in
    float64 downstream, so the storage precision only matters at the fp16 level of the dumped values themselves)."""
    out = []
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(np.ascontiguousarray(V[idx[i:i + bs]])).to(dev)
        out.append((x if U is None else ((x.float() - mu) @ U).half()).cpu())
    return torch.cat(out)                     # features live in host memory; chunks go to the GPU


# ------------------------------------------------------------------ ridge (token level), by chunked sufficient statistics

CH = 32768
COMPUTE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _stats(X, Y, rows):
    """n, sum x, sum y, X^T X, X^T Y (float64) over the selected rows, accumulated in chunks (X fp16/fp32 on the GPU)."""
    D, K = X.shape[1], Y.shape[1]; dev = COMPUTE
    n = 0; sx = torch.zeros(D, dtype=torch.float64, device=dev); sy = torch.zeros(K, dtype=torch.float64, device=dev)
    XX = torch.zeros(D, D, dtype=torch.float64, device=dev); XY = torch.zeros(D, K, dtype=torch.float64, device=dev)
    idx = torch.nonzero(rows).squeeze(1)
    for i in range(0, len(idx), CH):
        j = idx[i:i + CH]; x = X[j].to(dev).double(); y = Y[j].to(dev).double()
        n += len(j); sx += x.sum(0); sy += y.sum(0); XX += x.T @ x; XY += x.T @ y
    return n, sx, sy, XX, XY


def _solve(st, lam):
    n, sx, sy, XX, XY = st
    xm, ym = sx / n, sy / n
    A = XX - n * torch.outer(xm, xm); B = XY - n * torch.outer(xm, ym)
    A = A.clone(); A.diagonal().add_(lam * A.diagonal().mean())
    W = torch.linalg.solve(A, B)
    return W, xm, ym


def _predict(X, W, xm, ym, rows=None):
    idx = torch.arange(len(X)) if rows is None else torch.nonzero(rows.cpu()).squeeze(1)
    return torch.cat([((X[idx[i:i + CH]].to(COMPUTE).double() - xm) @ W + ym).float().cpu() for i in range(0, len(idx), CH)])


def ridge_select(Xtr, Ytr, groups, lossf):
    """lambda by a 20% episode hold-out of the training set; returns the refit on all training rows (chunked, float64)."""
    ug = np.unique(groups); rng = np.random.default_rng(0); hold = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False).tolist())
    hv = torch.tensor([g in hold for g in groups])
    st_h, st_r = _stats(Xtr, Ytr, hv.to(Xtr.device)), _stats(Xtr, Ytr, (~hv).to(Xtr.device))
    yh = Ytr[hv.to(Ytr.device)].float().cpu()
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        W, xm, ym = _solve(st_r, lam)
        loss = lossf(_predict(Xtr, W, xm, ym, hv), yh)
        if best is None or loss < best[0]:
            best = (loss, lam)
    st_all = tuple(a + b for a, b in zip(st_h, st_r))
    W, xm, ym = _solve(st_all, best[1])
    return (lambda X: _predict(X, W, xm, ym)), best[1]


def token_tasks(Ftr, Gtr, Fte, Gte, ep_tr):
    """Ftr/Fte: (n, Sv, D) features; returns {metric: value}."""
    hi = torch.as_tensor(head_token_index().reshape(-1), device=Ftr.device)
    res = {}
    okr, oke = np.asarray(Gtr['head_ok'], bool), np.asarray(Gte['head_ok'], bool)
    ftr, fte = Ftr[torch.as_tensor(okr, device=Ftr.device)], Fte[torch.as_tensor(oke, device=Fte.device)]
    gtr = {k: v[okr] for k, v in Gtr.items()}; gte = {k: v[oke] for k, v in Gte.items()}
    grp_tok = np.repeat(np.asarray(ep_tr)[okr], NF * GH * GW)
    dev = Ftr.device
    # depth + segmentation: one row per (window, latent frame, head token)
    Xtr, Xte = ftr[:, hi].reshape(-1, ftr.shape[-1]), fte[:, hi].reshape(-1, fte.shape[-1])
    d_tr, d_te = torch.as_tensor(gtr['depth'].reshape(-1), device=dev), torch.as_tensor(gte['depth'].reshape(-1), device=dev)
    m_tr, m_te = torch.isfinite(d_tr) & (d_tr > 0), torch.isfinite(d_te) & (d_te > 0)
    f, lam = ridge_select(Xtr[m_tr], d_tr[m_tr].log()[:, None], grp_tok[m_tr.cpu().numpy()], lambda p, y: float(((p - y) ** 2).mean()))
    pd = f(Xte[m_te])[:, 0].exp()
    res['depth_absrel'] = float(((pd - d_te[m_te]).abs() / d_te[m_te]).mean())
    t_tr, t_te = torch.as_tensor(gtr['target_frac'].reshape(-1), device=dev)[:, None], torch.as_tensor(gte['target_frac'].reshape(-1), device=dev)
    f, _ = ridge_select(Xtr, t_tr, grp_tok, lambda p, y: float(((p - y) ** 2).mean()))
    # threshold: predicted positive rate on the TRAINING tokens = their true positive rate (same rule for every input;
    # target tokens are ~3% of the head tokens, so a fixed 0.5 cut on a linear fraction regression rarely fires)
    pos = float((t_tr[:, 0] >= 0.5).float().mean())
    thr = float(torch.quantile(f(Xtr)[:, 0].float()[torch.randperm(len(Xtr))[:200000]], 1 - pos)) if pos > 0 else 0.5
    pr, gt_ = f(Xte)[:, 0] >= thr, t_te >= 0.5
    res['seg_iou'] = float((pr & gt_).sum() / (pr | gt_).sum().clamp_min(1)) if bool(gt_.any()) else float('nan')
    # flow: concat token at t and t+1 (same position) -> flow t -> t+1
    hi3 = hi.view(NF, GH * GW)
    Xf_tr = torch.cat([torch.cat([ftr[:, hi3[t]], ftr[:, hi3[t + 1]]], -1) for t in range(NF - 1)], 1).reshape(-1, 2 * ftr.shape[-1])
    Xf_te = torch.cat([torch.cat([fte[:, hi3[t]], fte[:, hi3[t + 1]]], -1) for t in range(NF - 1)], 1).reshape(-1, 2 * fte.shape[-1])
    y_tr = torch.as_tensor(gtr['flow3d'].reshape(len(ftr), -1, 3), device=dev).reshape(-1, 3); y_te = torch.as_tensor(gte['flow3d'].reshape(len(fte), -1, 3), device=dev).reshape(-1, 3)
    mt, me = torch.isfinite(y_tr).all(1), torch.isfinite(y_te).all(1)
    gflow = np.repeat(np.asarray(ep_tr)[okr], (NF - 1) * GH * GW)[mt.cpu().numpy()]
    f, _ = ridge_select(Xf_tr[mt], y_tr[mt], gflow, lambda p, y: float((p - y).norm(dim=1).mean()))
    res['flow_epe3d_mm'] = float((f(Xf_te[me]) - y_te[me]).norm(dim=1).mean())
    # tracking: concat tokens at latent 0,1,2 -> displacement (u, v) to latent 1, 2 + visibility
    Xk_tr = torch.cat([ftr[:, hi3[t]] for t in range(NF)], -1).reshape(-1, NF * ftr.shape[-1]); Xk_te = torch.cat([fte[:, hi3[t]] for t in range(NF)], -1).reshape(-1, NF * fte.shape[-1])
    cv, cu = np.arange(GH) * 30 + 15, np.arange(GW) * 32 + 16
    c0 = np.stack(np.meshgrid(cu, cv), -1).reshape(-1, 2)                                    # (80, 2) u, v
    def trk(g):
        tr = g['track'].transpose(0, 2, 3, 1, 4).reshape(len(g['track']), GH * GW, 2, 2) - c0[None, :, None, :]   # (n, 80, 2 frames, 2)
        vis = g['track_vis'].transpose(0, 2, 3, 1).reshape(len(g['track']), GH * GW, 2)
        return torch.as_tensor(np.nan_to_num(tr.reshape(-1, 4)), device=dev).float(), torch.as_tensor(vis.reshape(-1, 2), device=dev).float(), torch.as_tensor(np.isfinite(tr).reshape(-1, 4).all(1), device=dev)
    ytr, vtr, okt = trk(gtr); yte, vte, oke2 = trk(gte)
    f, _ = ridge_select(Xk_tr, torch.cat([ytr, vtr], 1), np.repeat(np.asarray(ep_tr)[okr], GH * GW), lambda p, y: float(((p - y) ** 2).mean()))
    P = f(Xk_te); pdisp, pvis = P[:, :4].view(-1, 2, 2), P[:, 4:] >= 0.5
    err = (pdisp - yte.view(-1, 2, 2)).norm(dim=-1)                                           # (n, 2)
    gv = (vte >= 0.5) & oke2[:, None]
    js = []
    for th in AJ_THR:
        within = err < th
        tp = (pvis & gv & within).sum(); fp = (pvis & ~(gv & within)).sum(); fn = (gv & ~(pvis & within)).sum()
        js.append(float(tp / (tp + fp + fn).clamp_min(1)))
    res['track_aj'] = float(np.mean(js))
    return res


# ------------------------------------------------------------------ attention-pooling probe (window level)

class AttnProbe(nn.Module):
    def __init__(self, din, dout, h=256, nq=4):
        super().__init__()
        self.k, self.v = nn.Linear(din, h), nn.Linear(din, h)
        self.q = nn.Parameter(torch.randn(nq, h) / math.sqrt(h)); self.head = nn.Linear(nq * h, dout); self.ln = nn.LayerNorm(din)

    def forward(self, x):                                          # (B, S, din)
        x = self.ln(x)
        a = (self.q @ self.k(x).transpose(1, 2) / math.sqrt(self.q.shape[1])).softmax(-1)   # (B, nq, S)
        return self.head((a @ self.v(x)).flatten(1))


def quat_to_R(q):                                                  # (..., 4) w x y z -> (..., 3, 3)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def sixd_to_R(v):
    a, b = v[..., :3], v[..., 3:]
    x = nn.functional.normalize(a, dim=-1); y = nn.functional.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def window_targets(G, dev):
    """content -> (Y (n, k) float, kind, valid (n,) bool)."""
    n = len(G['task'])
    t = lambda k: torch.as_tensor(np.asarray(G[k], np.float32), device=dev)
    out = {'task': (torch.as_tensor(G['task_id'], device=dev).long(), 'cls', torch.ones(n, dtype=torch.bool, device=dev))}
    pok = torch.as_tensor(np.asarray(G['pose_ok'], bool), device=dev)
    R6 = quat_to_R(t('link6')[..., 3:7])[..., :, :2].transpose(-1, -2).reshape(n, -1)          # first two columns, 6D
    out['ee_pose'] = (torch.cat([t('tcp').reshape(n, -1), R6], 1), 'pose', pok & torch.isfinite(t('tcp').reshape(n, -1)).all(1))
    g2t = t('grip2target').reshape(n, -1); out['grip2target'] = (g2t, 'reg_mm', pok & torch.isfinite(g2t).all(1))
    fut = t('future_tcp').reshape(n, -1); out['future_tcp'] = (fut, 'reg_mm', pok & torch.isfinite(fut).all(1))
    if 'grasp' in G and 'n_fingers_on_target' in G:
        gr = np.asarray(G['grasp'], np.float32).reshape(n, -1); nf = (np.asarray(G['n_fingers_on_target'], np.float32).reshape(n, -1) >= 1).astype(np.float32)
        ok = (np.asarray(G['n_fingers_on_target']).reshape(n, -1) >= 0).all(1)
        out['contact'] = (torch.as_tensor(np.concatenate([gr, nf], 1), device=dev), 'bin', pok & torch.as_tensor(ok, device=dev))
    return out


def train_probe(Ftr, Ytr, Fte, Yte, kind, seed, epochs=40, bs=32):
    torch.manual_seed(seed)
    dev = COMPUTE
    Ytr, Yte = Ytr.to(dev), Yte.to(dev)
    dout = int(Ytr.max()) + 1 if kind == 'cls' else Ytr.shape[1]
    if kind != 'cls':
        ym, ys = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
    net = AttnProbe(Ftr.shape[-1], dout if kind != 'cls' else 50).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(Ftr) / bs))
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(Ftr), generator=g)
        for i in range(0, len(Ftr), bs):
            b = perm[i:i + bs]; out = net(Ftr[b].to(dev).float()); b = b.to(dev)
            if kind == 'cls':
                loss = nn.functional.cross_entropy(out, Ytr[b])
            elif kind == 'bin':
                loss = nn.functional.binary_cross_entropy_with_logits(out, Ytr[b])
            else:
                loss = ((out - (Ytr[b] - ym) / ys) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        P = torch.cat([net(Fte[i:i + 64].to(dev).float()) for i in range(0, len(Fte), 64)])
    if kind == 'cls':
        return {'task_acc': float((P.argmax(1) == Yte).float().mean())}
    if kind == 'bin':
        return {'contact_acc': float(((P > 0) == (Yte > 0.5)).float().mean())}
    P = P * ys + ym
    if kind == 'pose':
        n = len(P); pos_p, pos_t = P[:, :18].view(n, 3, 2, 3), Yte[:, :18].view(n, 3, 2, 3)
        # targets were stored as the first two COLUMNS of R, column-major: [c0x c0y c0z c1x c1y c1z] (same layout sixd_to_R reads)
        Rp, R6 = sixd_to_R(P[:, 18:].view(n, 3, 2, 6)), Yte[:, 18:].view(n, 3, 2, 2, 3)
        c0, c1 = R6[..., 0, :], R6[..., 1, :]
        Rt = torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], -1)
        cosang = (((Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        return {'ee_pos_mm': float((pos_p - pos_t).norm(dim=-1).mean()), 'ee_rot_deg': float(torch.rad2deg(torch.acos(cosang)).mean())}
    k = 'grip2target_mm' if Yte.shape[1] == 18 else 'future_tcp_mm'
    return {k: float((P - Yte).view(len(P), -1, 3).norm(dim=-1).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--train-set', default='ID'); p.add_argument('--test-set', default=None)
    p.add_argument('--layers', type=int, nargs='+', default=[4, 9, 14, 19, 24, 29]); p.add_argument('--k', type=int, required=True)
    p.add_argument('--probe-seeds', type=int, default=3); p.add_argument('--inputs', nargs='+', default=['full', 'R'] + [f'rand{s}' for s in range(5)])
    p.add_argument('--skip-window', action='store_true'); p.add_argument('--device', default='cuda')
    p.add_argument('--novel', default=None, help='4b: novel_<B>vs<A>_<set>.pt; probes run on full, novel and nrand0..4 only')
    p.add_argument('--only-contents', nargs='*', default=None,
                   help='rerun only these window-level contents (e.g. contact), skip token-level tasks; written to a separate file')
    a = p.parse_args()
    novel = None
    if a.novel:
        novel = torch.load(OUT / 'R' / a.novel, map_location='cpu')['evec']
        a.inputs = ['full', 'novel'] + [f'nrand{s}' for s in range(5)]
    test_set = a.test_set or a.train_set
    dev = torch.device(a.device)
    meta = json.loads((FEAT / f'{a.model}_{a.train_set}' / 'meta.json').read_text()); dump_layers = meta['dump_layers']
    rows = []
    for layer in a.layers:
        lp = dump_layers.index(layer)
        Vtr, Gtr = load_set(a.model, a.train_set, lp); Vte, Gte = (Vtr, Gtr) if test_set == a.train_set else load_set(a.model, test_set, lp)
        itr = np.nonzero(Gtr['half'] == 'fit')[0]; ite = np.nonzero(Gte['half'] == 'eval')[0]
        gtr = {k: v[itr] for k, v in Gtr.items()}; gte = {k: v[ite] for k, v in Gte.items()}
        ep_tr = [f'{t}/{e}' for t, e in zip(gtr['task'], gtr['episode'])]
        U = bases(a.model, a.train_set, layer, a.k, dev, novel)
        # mu_V over the FIT video tokens of the training set
        mu = torch.zeros(Vtr.shape[-1], device=dev, dtype=torch.float64); cnt = 0
        for i in range(0, len(itr), 64):
            x = torch.from_numpy(np.ascontiguousarray(Vtr[itr[i:i + 64]])).to(dev).double(); mu += x.sum((0, 1)); cnt += x.shape[0] * x.shape[1]
        mu = (mu / cnt).float()
        for inp in a.inputs:
            t0 = time.time()
            Ub = None if inp == 'full' else U[inp]
            if inp == a.inputs[0]:
                # read this layer's train / test value vectors from the (slow, network) memmap ONCE, then project every
                # input from host memory (identical values; previously re-read for each of the 7 inputs)
                Vtr_c = np.ascontiguousarray(Vtr[itr]); Vte_c = Vtr_c if (test_set == a.train_set and np.array_equal(itr, ite)) else np.ascontiguousarray(Vte[ite])
            Ftr = project(Vtr_c, np.arange(len(itr)), Ub, mu, dev); Fte = project(Vte_c, np.arange(len(ite)), Ub, mu, dev)
            base = dict(model=a.model, train_set=a.train_set, test_set=test_set, layer=layer, input=inp, k=a.k if inp != 'full' else Ftr.shape[-1],
                        n_train_windows=len(itr), n_test_windows=len(ite))
            if a.only_contents is None:
                for metric, val in token_tasks(Ftr, gtr, Fte, gte, ep_tr).items():
                    rows.append(dict(base, content=metric.rsplit('_', 1)[0] if metric != 'flow_epe3d_mm' else 'flow3d', metric=metric, value=val, probe_seed=-1))
            if not a.skip_window:
                Ttr, Tte = window_targets(gtr, 'cpu'), window_targets(gte, 'cpu')
                for content, (Ytr, kind, vtr) in Ttr.items():
                    if a.only_contents is not None and content not in a.only_contents:
                        continue
                    base['n_train_windows'], base['n_test_windows'] = int(vtr.sum()), int(Tte[content][2].sum())
                    Yte, _, vte = Tte[content]
                    for s in range(a.probe_seeds):
                        for metric, val in train_probe(Ftr[vtr], Ytr[vtr], Fte[vte], Yte[vte], kind, s).items():
                            rows.append(dict(base, content=content, metric=metric, value=val, probe_seed=s))
            print(f'L{layer} {inp}: {time.time() - t0:.0f}s ' + ' '.join(f'{r["metric"]}={r["value"]:.4g}' for r in rows if r['layer'] == layer and r['input'] == inp and r['probe_seed'] in (-1, 0)), flush=True)
            del Ftr, Fte; torch.cuda.empty_cache()
    tag = (f'_novel-{Path(a.novel).stem}' if a.novel else '') + (f'_only-{"-".join(a.only_contents)}' if a.only_contents else '')
    out = OUT / 'results' / ('probes_novel' if a.novel else 'probes') / f'step3a_probe_{a.model}_{a.train_set}2{test_set}_L{"-".join(map(str, a.layers))}{tag}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
