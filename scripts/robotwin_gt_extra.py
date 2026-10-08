"""Read episode<N>_extra.hdf5 and compute dense 3D flow / 2D tracks.

Pair each episode<N>.hdf5 with episode<N>_extra.hdf5 (same folder on the Hub).
"""
import json

import cv2
import h5py
import numpy as np

CAM = 'observation/head_camera'


def load_extra(path):
    with h5py.File(path, 'r') as x:
        seg = x['segmentation']
        return {
            'entity_id': x['entities/entity_id'][:],
            'name': x['entities/name'][:].astype(str),
            'articulation': x['entities/articulation'][:].astype(str),
            'kind': x['entities/kind'][:].astype(str),
            'is_robot': x['entities/is_robot'][:],
            'pose': x['entities/pose'][:],              # (T, N, 7) x y z qw qx qy qz, world frame
            'actor_id': x[f'{CAM}/actor_id'][:],        # (T, 240, 320) uint16 = entity_id
            'mesh_id': x[f'{CAM}/mesh_id'][:],          # (T, 240, 320) uint16
            'mesh_table': dict(zip(seg['mesh_id'][:].tolist(), seg['mesh_entity_id'][:].tolist())),
            'targets': json.loads(seg.attrs['targets']),
            'task_objects': json.loads(seg.attrs['task_objects']),
            'scene_info': json.loads(seg.attrs['scene_info']),
        }


def load_wrist_rgb(path, camera='left_camera'):
    """Wrist RGB frames as RGB uint8 arrays (same JPEG convention as head rgb: decode with cv2, no swap)."""
    with h5py.File(path, 'r') as x:
        return [cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR) for b in x[f'observation/{camera}/rgb']]


def pose_matrix(p):
    x, y, z, w, qx, qy, qz = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
                  [2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w)],
                  [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, (x, y, z)
    return M


def dense_flow(main_path, extra, t, t2, occlusion_tol_m=0.01):
    """Track every pixel of frame t to frame t2.

    Returns dict with, per pixel (H, W): world points at t and t2 (3D scene flow =
    p2 - p1), the 2D position (u2, v2) at t2, and `visible` (in view, not occluded:
    depth at t2 agrees within tolerance and the pixel still belongs to the entity).
    Pixels without a known entity (e.g. empty background) are NaN.
    """
    with h5py.File(main_path, 'r') as m:
        depth = m[f'{CAM}/depth'][[t, t2]] / 1000.0
        K = m[f'{CAM}/intrinsic_cv'][t].astype(float)
        E = np.vstack([m[f'{CAM}/extrinsic_cv'][t], [0, 0, 0, 1]]).astype(float)
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H, 0:W]
    z = depth[0]
    cam = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], -1)
    p1 = (cam.reshape(-1, 4) @ np.linalg.inv(E).T)
    ids = extra['actor_id'][t].reshape(-1)
    p2 = np.full_like(p1, np.nan)
    index = {int(e): i for i, e in enumerate(extra['entity_id'])}
    for e in np.unique(ids):
        i = index.get(int(e))
        if i is None:
            continue
        sel = ids == e
        rel = pose_matrix(extra['pose'][t2, i]) @ np.linalg.inv(pose_matrix(extra['pose'][t, i]))
        p2[sel] = p1[sel] @ rel.T
    c2 = p2 @ E.T
    u2 = K[0, 0] * c2[:, 0] / c2[:, 2] + K[0, 2] - 0.5
    v2 = K[1, 1] * c2[:, 1] / c2[:, 2] + K[1, 2] - 0.5
    ui, vi = np.round(u2), np.round(v2)
    inview = np.isfinite(ui) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (c2[:, 2] > 0)
    visible = np.zeros(H * W, bool)
    k = np.nonzero(inview)[0]
    uu, vv = ui[k].astype(int), vi[k].astype(int)
    visible[k] = (np.abs(depth[1][vv, uu] - c2[k, 2]) < occlusion_tol_m) & (extra['actor_id'][t2][vv, uu] == ids[k])
    shape = (H, W)
    return {'p1': p1[:, :3].reshape(*shape, 3), 'p2': p2[:, :3].reshape(*shape, 3),
            'u2': u2.reshape(shape), 'v2': v2.reshape(shape), 'visible': visible.reshape(shape),
            'entity_id': ids.reshape(shape)}
