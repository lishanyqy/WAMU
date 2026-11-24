"""LeRobot-v3 reader for ``WAMsU/wamu_real``.

The source is a bimanual Quest ViperX recording with joint-space 14-D
actions/states and three camera streams.  The reader keeps the native joint
representation and scatters it into OpenWAM's shared 80-D action space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec


EXPECTED_NAMES = (
    "left.waist", "left.shoulder", "left.elbow", "left.forearm_roll",
    "left.wrist_angle", "left.wrist_rotate", "left.gripper",
    "right.waist", "right.shoulder", "right.elbow", "right.forearm_roll",
    "right.wrist_angle", "right.wrist_rotate", "right.gripper",
)


class WamuRealDataset(LeRobotV3Reader):
    DATASET_NAME = "WAMUReal"
    ACTION_DIM = 14
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    HEAD_CAMERA = "observation.images.top"
    LEFT_WRIST_CAMERA = "observation.images.left_wrist"
    RIGHT_WRIST_CAMERA = "observation.images.right_wrist"
    STATS_FILENAME = "stats.json"
    STATS_DIM = 14
    STATS_STRICT_MINMAX = True
    DEPLOY_ACTION_MODE = "joint"
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + ("normalization_stats_path",)

    @classmethod
    def from_config(cls, config, split: str = "train"):
        root = Path(get_cfg(config, "dataset_dir"))
        if (root / "meta" / "info.json").is_file():
            return super().from_config(config, split=split)
        # Buckets are the immediate subdirs carrying meta/info.json: ``bowls``,
        # ``cupstack``... (WAMsU/wamu_real) or the older ``*_lerobot`` names.
        # The root ``meta/`` (shared stats only, no info.json) is never a bucket.
        subdirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "meta" / "info.json").is_file()]
        # Optional ``tasks`` filter (bucket dir names, e.g. [cupstack]); None = all.
        # Normalization still uses the shared root stats, so a single-task run is
        # directly comparable with the multi-task one.
        wanted = get_cfg(config, "tasks", None)
        if wanted is not None:
            wanted = [str(wanted)] if isinstance(wanted, str) else [str(t) for t in wanted]
            available = sorted(p.name for p in subdirs)
            missing = sorted(set(wanted) - set(available))
            if missing:
                raise ValueError(f"WAMUReal tasks {missing} not found under {root}; available: {available}")
            subdirs = [p for p in subdirs if p.name in wanted]
        buckets = []
        for subdir in subdirs:
            if not (subdir / "meta" / "info.json").is_file():
                continue
            kwargs = {"dataset_dir": str(subdir), "split": split}
            for key in cls.CONFIG_KEYS:
                value = get_cfg(config, key, None)
                if value is not None:
                    kwargs[key] = value
            # headphones is a single-left-arm 7-D bucket; the other
            # three tasks are bimanual 14-D. Keep both in the shared 80-D
            # space, with the single arm occupying the left-arm slots, i.e. the
            # first 7 entries of the configured 14-D map (gripper stays at the
            # same slot as in the bimanual buckets).
            import json
            with (subdir / "meta" / "info.json").open(encoding="utf-8") as handle:
                source_dim = int(json.load(handle)["features"]["action"]["shape"][0])
            if source_dim == 7:
                full_map = get_cfg(config, "unify_action_map", None)
                left_map = (
                    parse_unify_spec(full_map, UNIFY_DIM)[:7].tolist() if full_map is not None else list(range(7))
                )
                kwargs["unify_action_map"] = left_map
                kwargs["unify_state_map"] = left_map
            buckets.append(cls(**kwargs))
        if not buckets:
            raise FileNotFoundError(f"WAMUReal root has no <bucket>/meta/info.json buckets: {root}")
        return MultiWamuRealDataset(buckets)

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        required = (self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA)
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(f"WAMUReal missing camera features: {missing}")
        return required

    def __init__(self, dataset_dir: str, normalization_stats_path: Optional[str] = None, **kwargs: Any):
        import json
        # Optional shared stats (e.g. <root>/meta/stats.json from
        # scripts/compute_wamu_real_stats.py) so every bucket normalizes with
        # the same 14-D statistics; None -> this bucket's own meta/stats.json.
        self._shared_stats_path = Path(normalization_stats_path) if normalization_stats_path else None
        with (Path(dataset_dir) / "meta" / "info.json").open(encoding="utf-8") as handle:
            self.ACTION_DIM = int(json.load(handle)["features"]["action"]["shape"][0])
        if self.ACTION_DIM not in (7, 14):
            raise ValueError(f"WAMUReal supports only 7-D single-arm or 14-D bimanual actions, got {self.ACTION_DIM}")
        super().__init__(dataset_dir, **kwargs)

    def _post_init(self, info: dict) -> None:
        if info.get("robot_type") != "quest_viperx":
            raise ValueError(f"WAMUReal expects robot_type='quest_viperx', got {info.get('robot_type')!r}")
        features = info.get("features", {}) or {}
        for column in ("action", "observation.state"):
            feature = features.get(column)
            if feature is None or tuple(feature.get("shape", ())) != (self._raw_action_dim,):
                raise ValueError(f"WAMUReal {column} must have shape [{self._raw_action_dim}]")
            names = tuple(feature.get("names") or ())
            expected = EXPECTED_NAMES if self._raw_action_dim == 14 else EXPECTED_NAMES[:7]
            if names != expected:
                raise ValueError(f"WAMUReal {column} names do not match Quest ViperX contract: {names}")
        if self._unify:
            if self._unify_dim != UNIFY_DIM:
                raise ValueError("WAMUReal unified action width must be OpenWAM's 80-D schema")
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(f"WAMUReal requires a {self._raw_action_dim}-entry unify_action_map")

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        path = self._shared_stats_path or self._dataset_dir / "meta" / self.STATS_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"WAMUReal stats file missing: {path}")
        import json
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        block = raw.get("action") if isinstance(raw, dict) else None
        if not isinstance(block, dict):
            raise ValueError("WAMUReal stats.json must contain an action block")
        if self._shared_stats_path is not None:
            # Deploy denormalizer next to the shared stats; the trainer copies it
            # into the checkpoint dir via ``normalization_stats_path``.
            deploy = path.parent / "normalization_stats.npy"
            if not deploy.is_file():
                raise FileNotFoundError(
                    f"WAMUReal deploy stats missing: {deploy}; run scripts/compute_wamu_real_stats.py"
                )
            self.normalization_stats_path = str(deploy)
        # Shared stats are 14-D; a 7-D single-arm bucket takes the left-arm dims 0-6.
        dim = self._raw_action_dim
        block = {k: v[:dim] if isinstance(v, list) else v for k, v in block.items()}
        return materialize_eef_stats(
            block, self._normalize_mode, dim=dim,
            strict_minmax=self.STATS_STRICT_MINMAX, source_hint=str(path),
        )

    def _values(self, win, column: str) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != self._raw_action_dim or not np.isfinite(values).all():
            raise ValueError(f"WAMUReal {column} must be finite (T,{self._raw_action_dim}), got {values.shape}")
        return values

    def _action_20d(self, win) -> np.ndarray:
        return apply_normalization(self._values(win, "action"), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        values = self._values(win, "observation.state")[:1]
        return apply_normalization(values, self._normalization_stats, self._normalize_mode)


class MultiWamuRealDataset(MultiLeRobotV3Reader):
    """Aggregate all task buckets in one ``wamu_real`` checkout."""

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return WamuRealDataset.from_config(config, split=split)

    @property
    def normalization_stats_path(self) -> Optional[str]:
        """Shared deploy stats path; None unless every bucket uses the same file."""
        paths = {getattr(bucket, "normalization_stats_path", None) for bucket in self.buckets}
        return paths.pop() if len(paths) == 1 else None


__all__ = ["WamuRealDataset", "MultiWamuRealDataset", "EXPECTED_NAMES"]
