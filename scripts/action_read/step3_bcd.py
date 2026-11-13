#!/usr/bin/env python
"""Steps 3b, 3c, 3d of wamu/action_read_subspace_plan.md from read_subspace.py --mode dump outputs + build_token_gt.py.

3b  unexplained fraction: ridge regression of the read vector's coordinates in R (top k* directions, every action token,
    every layer, the three action sigmas pooled) on ALL ground-truth quantities of the window (TCP / link6 pose,
    TCP->target, contact, task one-hot, future TCP trajectory, this token's future TCP step, head-token maps of depth,
    target fraction and 3D flow).  Lambda by a 20% episode hold-out of FIT; R^2 on EVAL (variance-weighted over the k*
    coordinates).  1 - R^2 = what the candidate contents do not linearly explain.  Windows need head and pose GT.
3c  where the action stream looks: attention mass on video tokens (mean over heads, action tokens), split by region of
    the head-camera tokens (target / gripper / rest of robot / other moving object / static scene) and by camera (head,
    left wrist, right wrist); enrichment = mass share / token share.  Per layer, latent frame, action sigma; 95% CI by
    episode bootstrap (1000).
3d  figures: for chosen windows (lowest / highest NMSE_full, ID and OOD, same windows for all models), canvas frames,
    attention heat map and the top-3 principal components of z[j] = U_k^T (V[j] - mu_V) as RGB on the token grid.
usage: python scripts/action_read/step3_bcd.py --part b|c|d --model M1 --set ID --k K [--layer 19]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path('artifacts/action_read'); FEAT = Path.home() / 'dataspace/action_read_features'
GH, GW, GRID_H, NF = 8, 10, 12, 3
REGIONS = ['target', 'gripper', 'robot_other', 'moving_other', 'static']


def load(model, set_name):
    root = FEAT / f'{model}_{set_name}'
    keys = [tuple(k) for k in json.loads((root / 'keys.json').read_text())]
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from probes import wait_for_gt
    gt = wait_for_gt(set_name, keys)
    gidx = {(str(t), int(e), int(s)): i for i, (t, e, s) in enumerate(zip(gt['task'], gt['episode'], gt['start']))}
    order = np.array([gidx[(t, int(e), int(s))] for t, e, s in keys])
    return root, keys, {k: gt[k][order] for k in gt.files}, json.loads((root / 'meta.json').read_text())


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')


def gt_matrix(G, idx):
    """Window-level GT vector (n, D) + per-action-token future step (n, 32, 6)."""
    n = len(idx); f = lambda k: np.asarray(G[k][idx], np.float64).reshape(n, -1)
    parts = [f('tcp'), f('link6')[:, [i for i in range(42) if i % 7 >= 3]], f('grip2target'), f('future_tcp'),
             np.eye(50)[np.asarray(G['task_id'][idx])], f('depth'), f('target_frac'), f('flow3d')]
    if 'grasp' in G:
        parts += [f('grasp'), (f('n_fingers_on_target') >= 1).astype(float)]
    X = np.concatenate(parts, 1)
    fut = np.asarray(G['future_tcp'][idx], np.float64)                         # (n, 2, 32, 3)
    tok = fut.transpose(0, 2, 1, 3).reshape(n, 32, 6)
    return X, tok


def part_b(a):
    import torch
    root, keys, G, meta = load(a.model, a.set)
    mc = np.load(root / 'mcoord.npy', mmap_mode='r')                            # (N, S, L, 32, K)
    ok = np.asarray(G['head_ok'], bool) & np.asarray(G['pose_ok'], bool)
    fit = np.nonzero(ok & (G['half'] == 'fit'))[0]; ev = np.nonzero(ok & (G['half'] == 'eval'))[0]
    Xw_f, tok_f = gt_matrix(G, fit); Xw_e, tok_e = gt_matrix(G, ev)
    Xw_f, Xw_e = np.nan_to_num(Xw_f), np.nan_to_num(Xw_e)
    S = mc.shape[1]
    def rows_of(Xw, tok, n):                                                     # (n * S * 32, D + 6 + S)
        X = np.concatenate([np.repeat(Xw[:, None, None], S, 1).repeat(32, 2), np.repeat(tok[:, None], S, 1),
                            np.broadcast_to(np.eye(S)[None, :, None], (n, S, 32, S))], -1)
        return X.reshape(-1, X.shape[-1])
    Xf, Xe = rows_of(Xw_f, tok_f, len(fit)), rows_of(Xw_e, tok_e, len(ev))
    sd = Xf.std(0) + 1e-6; xm = Xf.mean(0); Xf = (Xf - xm) / sd; Xe = (Xe - xm) / sd
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xf_t, Xe_t = torch.as_tensor(Xf, device=dev, dtype=torch.float64), torch.as_tensor(Xe, device=dev, dtype=torch.float64)
    eps_f = np.repeat(np.array([f'{G["task"][i]}/{G["episode"][i]}' for i in fit]), S * 32)
    rng = np.random.default_rng(0); ug = np.unique(eps_f); hold = np.isin(eps_f, rng.choice(ug, len(ug) // 5, replace=False))
    hv = torch.as_tensor(hold, device=dev)
    rows = []
    for l in range(mc.shape[2]):
        Yf = torch.as_tensor(np.asarray(mc[fit, :, l, :, :a.k], np.float64).reshape(-1, a.k), device=dev)
        Ye = torch.as_tensor(np.asarray(mc[ev, :, l, :, :a.k], np.float64).reshape(-1, a.k), device=dev)
        def fitr(X, Y, lam):
            A = X.T @ X; A.diagonal().add_(lam * A.diagonal().mean()); W = torch.linalg.solve(A, X.T @ (Y - Y.mean(0))); return W, Y.mean(0)
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            W, ym = fitr(Xf_t[~hv], Yf[~hv], lam); e = float(((Xf_t[hv] @ W + ym - Yf[hv]) ** 2).sum())
            best = (e, lam) if best is None or e < best[0] else best
        W, ym = fitr(Xf_t, Yf, best[1])
        sse = float(((Xe_t @ W + ym - Ye) ** 2).sum()); sst = float(((Ye - Ye.mean(0)) ** 2).sum())
        rows.append(dict(model=a.model, set=a.set, layer=l, k=a.k, n_fit_windows=len(fit), n_eval_windows=len(ev), n_gt_features=Xf.shape[1],
                         lambda_=best[1], r2=1 - sse / sst, unexplained=sse / sst, seed=0))
        print(f'L{l}: R2 {1 - sse / sst:.3f}', flush=True)
    write(OUT / 'results' / 'step3b' / f'step3b_r2_{a.model}_{a.set}.csv', rows)


def part_c(a):
    root, keys, G, meta = load(a.model, a.set)
    att = np.load(root / 'attn.npy', mmap_mode='r')                              # (N, S, L, Sv)
    ev = np.nonzero(np.asarray(G['head_ok'], bool) & (G['half'] == 'eval'))[0]
    reg = np.asarray(G['region'][ev])                                            # (n, 3, 8, 10)
    n, S, L = len(ev), att.shape[1], att.shape[2]
    A = np.asarray(att[ev], np.float64).reshape(n, S, L, NF, GRID_H, GW)
    lab = np.full((n, NF, GRID_H, GW), -1, np.int64); lab[:, :, :GH] = reg
    lab[:, :, GH:, :GW // 2] = 5; lab[:, :, GH:, GW // 2:] = 6                  # 5 = left wrist cam, 6 = right wrist cam
    names = REGIONS + ['wrist_left', 'wrist_right']
    eps = np.array([f'{G["task"][i]}/{G["episode"][i]}' for i in ev]); ue, inv = np.unique(eps, return_inverse=True)
    rng = np.random.default_rng(0); B = rng.integers(0, len(ue), size=(1000, len(ue)))
    rows = []
    for si, sig in enumerate(meta['action_sigmas']):
        for frame in ['all', 0, 1, 2]:
            fsel = slice(None) if frame == 'all' else slice(frame, frame + 1)
            a_ = A[:, si, :, fsel].reshape(n, L, -1); l_ = lab[:, fsel].reshape(n, -1)
            tot = a_.sum(-1)                                                     # (n, L)
            for r, name in enumerate(names):
                msk = (l_ == r)
                mass = (a_ * msk[:, None, :]).sum(-1)                            # (n, L)
                share_tok = msk.mean(1)                                          # (n,)
                # per-episode sums -> ratio of sums, bootstrap over episodes
                em = np.zeros((len(ue), L)); et = np.zeros((len(ue), L)); es = np.zeros(len(ue)); ec = np.zeros(len(ue))
                np.add.at(em, inv, mass); np.add.at(et, inv, tot); np.add.at(es, inv, share_tok); np.add.at(ec, inv, 1)
                ms_ = em.sum(0) / et.sum(0); ts = es.sum() / ec.sum()
                bms = em[B].sum(1) / et[B].sum(1); bts = es[B].sum(1) / ec[B].sum(1)
                enr = ms_ / max(ts, 1e-9); benr = bms / np.maximum(bts, 1e-9)[:, None]
                for l in range(L):
                    rows.append(dict(model=a.model, set=a.set, action_sigma=sig, layer=l, latent_frame=frame, region=name,
                                     mass_share=ms_[l], token_share=ts, enrichment=enr[l], enrichment_lo=float(np.percentile(benr[:, l], 2.5)),
                                     enrichment_hi=float(np.percentile(benr[:, l], 97.5)), n_windows=n, boot_seed=0))
    write(OUT / 'results' / 'step3c' / f'step3c_where_{a.model}_{a.set}.csv', rows)


def part_d(a):
    """Figures for a fixed list of windows (same for every model): artifacts/action_read/figs/step3d_<model>_<set>_<i>.png."""
    import h5py
    import torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import sys
    sys.path.insert(0, '.')
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    root, keys, G, meta = load(a.model, a.set)
    V = np.load(root / 'V.npy', mmap_mode='r'); att = np.load(root / 'attn.npy', mmap_mode='r')
    lp = meta['dump_layers'].index(a.layer)
    R = torch.load(OUT / 'R' / f'R_{a.model}_{a.set}.pt', map_location='cpu'); U = R['evec'][a.layer][:, :a.k].numpy()
    fit = np.nonzero(G['half'] == 'fit')[0]
    mu = np.mean([np.asarray(V[i, lp], np.float64).mean(0) for i in fit[:300]], 0)
    Z = np.concatenate([(np.asarray(V[i, lp], np.float64) - mu) @ U for i in fit[:300]])
    Zm = Z.mean(0); _, _, Vt = np.linalg.svd(Z - Zm, full_matrices=False); P3 = Vt[:3].T
    zz = (Z - Zm) @ P3; lo, hi = np.percentile(zz, 1, 0), np.percentile(zz, 99, 0)
    sel = json.loads((OUT / 'figs' / f'step3d_windows_{a.set}.json').read_text())
    kidx = {(t, int(e), int(s)): i for i, (t, e, s) in enumerate(keys)}
    var = 'clean_50' if a.set == 'ID' else 'randomized_500'
    off = Path.home() / 'dataspace/openwam/benchmark_data/robotwin2.0/dataset'
    si = meta['action_sigmas'].index(min(meta['action_sigmas'], key=lambda x: abs(x - 0.5)))
    for j, w in enumerate(sel):
        i = kidx[(w['task'], int(w['episode']), int(w['start']))]
        z = ((np.asarray(V[i, lp], np.float64) - mu) @ U - Zm) @ P3
        rgb = np.clip((z - lo) / (hi - lo + 1e-9), 0, 1).reshape(NF, GRID_H, GW, 3)
        at = np.asarray(att[i, si, a.layer], np.float64).reshape(NF, GRID_H, GW)
        with h5py.File(off / w['task'] / f'aloha-agilex_{var}' / 'data' / f'episode{w["episode"]}.hdf5', 'r') as f:
            frames = []
            for fr in (w['start'], w['start'] + 16, w['start'] + 32):
                ims = {c: Image.open(__import__('io').BytesIO(bytes(f[f'observation/{c}/rgb'][fr]))).convert('RGB') for c in ('head_camera', 'left_camera', 'right_camera')}
                frames.append(np.asarray(assemble_multiview_layout(ims, ['head_camera', 'left_camera', 'right_camera'], 384, 320)))
        fig, ax = plt.subplots(3, NF, figsize=(3 * NF, 9.5))
        for t in range(NF):
            ax[0, t].imshow(frames[t]); ax[0, t].set_title(f'latent frame {t}', fontsize=8)
            ax[1, t].imshow(frames[t]); ax[1, t].imshow(np.kron(at[t], np.ones((32, 32))), alpha=.6, cmap='magma'); ax[1, t].set_title(f'action->video attention L{a.layer}', fontsize=8)
            ax[2, t].imshow(np.kron(rgb[t], np.ones((32, 32, 1)))); ax[2, t].set_title('z top-3 PCs as RGB', fontsize=8)
            for r in range(3):
                ax[r, t].axis('off')
        fig.suptitle(f'{a.model} {a.set} {w["task"]} ep{w["episode"]} start {w["start"]}  ({w["why"]})  k={a.k}', fontsize=9)
        fig.tight_layout(); out = OUT / 'figs' / f'step3d_{a.model}_{a.set}_{j:02d}.png'; fig.savefig(out, dpi=110); plt.close(fig)
    print(f'wrote {len(sel)} figures for {a.model} {a.set}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['b', 'c', 'd'], required=True); p.add_argument('--model', required=True); p.add_argument('--set', required=True)
    p.add_argument('--k', type=int, default=0); p.add_argument('--layer', type=int, default=19)
    a = p.parse_args()
    {'b': part_b, 'c': part_c, 'd': part_d}[a.part](a)


if __name__ == '__main__':
    main()
