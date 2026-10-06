"""Shared data access for the downstream probes (wamu/downstreamtask.md).

Probe frames: OpenWAM latent k is scored against the GT of aligned frame 4k (latent 0 ->
frame 0), the last frame its causal-VAE chunk covers.
"""
import json
import zipfile
from pathlib import Path

import numpy as np

GT_ROOT = Path('/home/a1905721/dataspace/openwam-robotwin-gt')
OPENWAM_ROOT = Path('/home/a1905721/dataspace/openwam_gt_features')
N_LATENTS = 9
PROBE_ALIGNED_FRAMES = np.arange(N_LATENTS) * 4  # aligned frame scored for each latent


def gt_hdf5(embodiment, key):
    task, episode = key.split('__')
    return GT_ROOT / embodiment / task / 'gt_export' / 'data' / f'{episode}.hdf5'


def npz_member_memmap(path, member='features'):
    """Memory-map one array of an uncompressed .npz (np.savez) without reading the rest."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f'{member}.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f'{path}:{member} is compressed; cannot memory-map')
        with open(path, 'rb') as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            name_len = int.from_bytes(local[26:28], 'little')
            extra_len = int.from_bytes(local[28:30], 'little')
            data_offset = info.header_offset + 30 + name_len + extra_len
            fh.seek(data_offset)
            version = np.lib.format.read_magic(fh)
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read_header(fh)
            array_offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=array_offset, shape=shape, order='F' if fortran else 'C')


def openwam_layer(embodiment, model, key, layer, root=OPENWAM_ROOT):
    """(latents, head_h, head_w, dim) head tokens of one layer, read without loading the other 29."""
    feats = npz_member_memmap(root / embodiment / model / f'{key}.npz')  # (9, 30, 8, 10, 3072)
    return np.ascontiguousarray(feats[:, layer])


def load_split(path='artifacts/downstream/split.json'):
    return json.loads(Path(path).read_text())
