"""Read feature-aligned frames and GT from an OpenWAM Robotwin GT HDF5.

Use the same helper for feature extraction and GT extraction so both sides
select identical frames.
"""
import cv2
import h5py
import numpy as np
from PIL import Image

CAM = 'observation/head_camera'


def feature_indices(total, n=33, stride=4):
    """0, 4, ..., 128 when the episode has >= 129 frames, otherwise linspace."""
    if total <= 0:
        raise ValueError('empty episode')
    ids = np.arange(n, dtype=np.int64) * stride
    if ids[-1] >= total:
        ids = np.linspace(0, total - 1, n, dtype=np.int64)
    return ids


def decode_rgb(buf):
    """Decode one stored frame to an RGB uint8 array (H, W, 3).

    RoboTwin passes RGB arrays to cv2.imencode, so the JPEG bytes hold R and B
    swapped.  cv2.imdecode therefore returns true RGB; PIL/ffmpeg/torchvision
    decoders would return BGR.
    """
    return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)


def load_rgb_frames(path, n=33, stride=4):
    """Return (indices, list of RGB PIL images) for the aligned frames."""
    with h5py.File(path, 'r') as f:
        rgb = f[f'{CAM}/rgb']
        ids = feature_indices(len(rgb), n, stride)
        frames = [Image.fromarray(decode_rgb(rgb[i])) for i in ids]
    return ids, frames


def load_gt(path, n=33, stride=4):
    """Return a dict of GT arrays at the aligned frames."""
    with h5py.File(path, 'r') as f:
        ids = feature_indices(len(f[f'{CAM}/rgb']), n, stride)
        out = {'frame_indices': ids}
        for name in ('depth', 'actor_segmentation', 'mesh_segmentation',
                     'intrinsic_cv', 'extrinsic_cv', 'cam2world_gl'):
            out[name] = f[f'{CAM}/{name}'][ids]
        for name in ('pointcloud', 'joint_action/vector', 'endpose/left_endpose', 'endpose/right_endpose',
                     'endpose/left_gripper', 'endpose/right_gripper'):
            out[name] = f[name][ids]
    return out


def load_row_frames(row, n=33, stride=4):
    """Aligned RGB frames for a manifest row built by build_robotwin_gt_manifest.py.

    Old manifests point at the official mp4, whose later frames no longer match
    the replayed GT, so refuse them instead of silently falling back.
    """
    if 'gt_hdf5' not in row:
        raise KeyError(f"row {row.get('task')}/{row.get('episode')} has no gt_hdf5; "
                       "use a manifest from scripts/build_robotwin_gt_manifest.py")
    return load_rgb_frames(row['gt_hdf5'], n, stride)
