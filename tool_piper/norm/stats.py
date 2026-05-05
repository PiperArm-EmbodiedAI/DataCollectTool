from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import json

import numpy as np

from tool_piper.constants import DEFAULT_ASSETS_ROOT, LEROBOT_ACTION_KEY, LEROBOT_STATE_KEY

ProgressCallback = Callable[[int, int, str, float | None], None]
CancelCheck = Callable[[], None]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def _load_local_arrays(
    root: Path,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to read local LeRobot parquet files.") from exc

    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")

    states = []
    actions = []
    total = len(files)
    for file_index, path in enumerate(files, start=1):
        if cancel_check is not None:
            cancel_check()
        if progress is not None:
            progress(file_index, total, f"Reading parquet {file_index}/{total}: {path.name}", None)
        table = pq.read_table(path, columns=[LEROBOT_STATE_KEY, LEROBOT_ACTION_KEY])
        for row_index in range(table.num_rows):
            if cancel_check is not None and row_index % 500 == 0:
                cancel_check()
            if max_frames is not None and len(states) >= max_frames:
                if progress is not None:
                    progress(file_index, total, f"Reached max_frames={max_frames}", 0)
                return states, actions
            states.append(_to_numpy(table[LEROBOT_STATE_KEY][row_index].as_py()))
            actions.append(_to_numpy(table[LEROBOT_ACTION_KEY][row_index].as_py()))
    return states, actions


def compute_norm_stats(
    repo_id: str,
    root: Path | None = None,
    out_dir: Path | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    if root is None:
        raise ValueError("root is required for local norm stats computation")

    if cancel_check is not None:
        cancel_check()
    states, actions = _load_local_arrays(Path(root), max_frames=max_frames, progress=progress, cancel_check=cancel_check)
    if not states:
        raise ValueError("Cannot compute norm stats for an empty dataset")

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(1, 2, f"Computing statistics for {len(states)} frames", None)
    stats = {
        "state": _stats(np.stack(states, axis=0)),
        "actions": _stats(np.stack(actions, axis=0)),
    }

    if cancel_check is not None:
        cancel_check()
    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_ASSETS_ROOT / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "norm_stats.json"
    if progress is not None:
        progress(2, 2, f"Writing norm_stats.json: {out_path}", None)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return out_path
