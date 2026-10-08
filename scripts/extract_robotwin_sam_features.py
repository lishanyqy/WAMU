#!/usr/bin/env python
"""SAM ViT-H (facebook/sam-vit-huge) on the GT head-camera frames: encoder tokens for the dense probes plus an
oracle-prompted zero-shot segmentation ceiling.

Scored frames: aligned frames 4, 8, ..., 32 (the 8 frames every dense probe is scored on). For each frame
  vit   : hidden states after the 4 global-attention blocks (8, 16, 24, 32) of the 32-block encoder,
          64 x 64 tokens average-pooled to 16 x 16 (the same order of token count as DINOv2's 16 x 21)   -> (4, 8, 16, 16, 1280) fp16
  neck  : the 256-d image embedding SAM's mask decoder consumes, pooled the same way                     -> (8, 16, 16, 256) fp16
  zs_iou: zero-shot IoU of SAM prompted with the GT target's bounding box + centroid (oracle prompts),
          best of its 3 masks by SAM's own IoU score, evaluated at the 120 x 160 GT; NaN when the target
          is not visible in that frame                                                                    -> (8,) float32
Output: artifacts/<emb>_features_gt/sam/<key>.npz (uncompressed, memory-mappable).
Env: fastwam with USE_TF=0 PYTHONNOUSERSITE=1 (a stray user-site TensorFlow breaks transformers otherwise).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_data import gt_hdf5
from robotwin_gt_frames import load_row_frames

GT_DIR = Path('artifacts/downstream/gt')
BLOCKS = (8, 16, 24, 32)          # hidden_states index = blocks applied; global attention sits in blocks 8/16/24/32
POOL = 4                          # 64 x 64 -> 16 x 16


def pool(t):                      # (B, 64, 64, C) -> (B, 16, 16, C)
    return F.avg_pool2d(t.permute(0, 3, 1, 2).float(), POOL).permute(0, 2, 3, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()
    out_dir = a.output or Path(f'artifacts/{a.embodiment}_features_gt/sam')
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import SamModel, SamProcessor
    model = SamModel.from_pretrained('facebook/sam-vit-huge', torch_dtype=torch.float16, attn_implementation='sdpa').to(a.device).eval()
    proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
    gt = np.load(GT_DIR / f'{a.embodiment}.npz')
    keys = [str(k) for k in gt['keys']][:a.limit]
    target = gt['target']                                            # (N, 9, 120, 160) bool, index 1..8 = frames 4..32
    t0 = time.time()
    for i, k in enumerate(keys):
        out = out_dir / f'{k}.npz'
        if out.exists():
            continue
        frames = load_row_frames({'gt_hdf5': str(gt_hdf5(a.embodiment, k))})[1]
        imgs = [np.asarray(frames[4 * j]) for j in range(1, 9)]
        H, W = imgs[0].shape[:2]
        vit = np.empty((4, 8, 16, 16, 1280), np.float16); neck = np.empty((8, 16, 16, 256), np.float16)
        zs = np.full(8, np.nan, np.float32)
        with torch.inference_mode():
            enc = proc(images=imgs, return_tensors='pt')
            pv = enc['pixel_values'].to(a.device).half()
            embs = []
            for b in range(0, 8, a.batch):
                vo = model.vision_encoder(pv[b:b + a.batch], output_hidden_states=True)
                for bi, blk in enumerate(BLOCKS):
                    vit[bi, b:b + a.batch] = pool(vo.hidden_states[blk]).cpu().numpy()
                neck[b:b + a.batch] = pool(vo.last_hidden_state.permute(0, 2, 3, 1)).cpu().numpy()
                embs.append(vo.last_hidden_state)
            embs = torch.cat(embs)
            for j in range(8):
                m = target[i, j + 1]
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                sy, sx = H / m.shape[0], W / m.shape[1]
                box = [[float(xs.min() * sx), float(ys.min() * sy), float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]]
                pt = [[[float(xs.mean() * sx), float(ys.mean() * sy)]]]
                pr = proc(images=[imgs[j]], input_boxes=[box], input_points=[pt], return_tensors='pt')
                o = model(image_embeddings=embs[j:j + 1], input_boxes=pr['input_boxes'].to(a.device).half(),
                          input_points=pr['input_points'].to(a.device).half(), multimask_output=True)
                masks = proc.image_processor.post_process_masks(o.pred_masks.float().cpu(), pr['original_sizes'], pr['reshaped_input_sizes'])[0]
                best = int(o.iou_scores[0, 0].argmax())
                pm = F.interpolate(masks[0, best][None, None].float(), size=m.shape, mode='area')[0, 0].numpy() > 0.5
                zs[j] = (pm & m).sum() / max((pm | m).sum(), 1)
        tmp = out.with_suffix('.tmp.npz')
        np.savez(tmp, vit=vit, neck=neck, zs_iou=zs, key=k, blocks=np.array(BLOCKS), pool=POOL)
        tmp.rename(out)
        if (i + 1) % 10 == 0 or i + 1 == len(keys):
            print(f'progress={i + 1}/{len(keys)} ({(time.time() - t0) / (i + 1):.1f} s/ep)', flush=True)
    print(f'complete={len(keys)}', flush=True)


if __name__ == '__main__':
    main()
