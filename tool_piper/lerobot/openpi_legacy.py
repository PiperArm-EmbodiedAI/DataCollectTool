from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np

from tool_piper.constants import (
    ACTION_DIM,
    FPS,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
    LEROBOT_TOP_KEY,
    LEROBOT_WRIST_KEY,
    ROBOT_TYPE,
    STATE_DIM,
)

ProgressCallback = Callable[[int, int, str, float | None], None]
CancelCheck = Callable[[], None]

LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000


def _clean_json(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_clean_json(row), ensure_ascii=False) + "\n")


def _parquet_files(root: Path, relative: str) -> list[Path]:
    files = sorted((root / relative).glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / relative}")
    return files


def _load_episode_rows(source_root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to export OpenPI legacy datasets.") from exc

    rows: list[dict[str, Any]] = []
    for path in _parquet_files(source_root, "meta/episodes"):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _load_tasks(source_root: Path, fallback_task: str) -> dict[int, str]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to export OpenPI legacy datasets.") from exc

    tasks_path = source_root / "meta" / "tasks.parquet"
    tasks: dict[int, str] = {}
    if tasks_path.exists():
        table = pq.read_table(tasks_path)
        for row in table.to_pylist():
            task_index = int(row.get("task_index", len(tasks)))
            task_text = row.get("task") or row.get("__index_level_0__") or fallback_task
            tasks[task_index] = str(task_text)
    if not tasks:
        tasks[0] = fallback_task
    return tasks


def _task_for_episode(row: dict[str, Any], tasks: dict[int, str], fallback_task: str) -> str:
    row_tasks = row.get("tasks")
    if row_tasks:
        return str(row_tasks[0])
    task_index = int(row.get("task_index", 0) or 0)
    return tasks.get(task_index, fallback_task)


def _legacy_features(source_root: Path) -> dict[str, Any]:
    source_info = _read_json(source_root / "meta" / "info.json")
    source_features = source_info.get("features", {})
    features = {}
    for key in (LEROBOT_TOP_KEY, LEROBOT_WRIST_KEY):
        features[key] = source_features.get(
            key,
            {
                "dtype": "image",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
        )
    features[LEROBOT_STATE_KEY] = source_features.get(
        LEROBOT_STATE_KEY,
        {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]},
    )
    features[LEROBOT_ACTION_KEY] = source_features.get(
        LEROBOT_ACTION_KEY,
        {"dtype": "float32", "shape": [ACTION_DIM], "names": ["action"]},
    )
    features["timestamp"] = source_features.get("timestamp", {"dtype": "float32", "shape": [1], "names": None})
    for key in ("frame_index", "episode_index", "index", "task_index"):
        features[key] = source_features.get(key, {"dtype": "int64", "shape": [1], "names": None})
    return _clean_json(features)


def _episode_stats(row: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        stat_path = key[len("stats/") :]
        feature_name, stat_name = stat_path.rsplit("/", 1)
        stats.setdefault(feature_name, {})[stat_name] = _clean_json(value)
    return stats


def _legacy_hf_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        LEROBOT_TOP_KEY: {"_type": "Image"},
        LEROBOT_WRIST_KEY: {"_type": "Image"},
        LEROBOT_STATE_KEY: {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": STATE_DIM,
            "_type": "Sequence",
        },
        LEROBOT_ACTION_KEY: {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": ACTION_DIM,
            "_type": "Sequence",
        },
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }


def _patch_table_huggingface_metadata(table: Any, features: dict[str, Any]) -> Any:
    metadata = dict(table.schema.metadata or {})
    payload = metadata.get(b"huggingface")
    if payload is not None:
        try:
            info = json.loads(payload.decode("utf-8"))
        except Exception:
            info = {}
    else:
        info = {}
    info.setdefault("info", {})["features"] = _legacy_hf_features(features)
    metadata[b"huggingface"] = json.dumps(info, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _ensure_columns(table: Any, episode_index: int, global_start: int) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    count = table.num_rows
    names = set(table.column_names)
    additions = []
    if "timestamp" not in names:
        additions.append(("timestamp", pa.array([i / FPS for i in range(count)], type=pa.float32())))
    if "frame_index" not in names:
        additions.append(("frame_index", pa.array(range(count), type=pa.int64())))
    if "episode_index" not in names:
        additions.append(("episode_index", pa.array([episode_index] * count, type=pa.int64())))
    if "index" not in names:
        additions.append(("index", pa.array(range(global_start, global_start + count), type=pa.int64())))
    if "task_index" not in names:
        additions.append(("task_index", pa.array([0] * count, type=pa.int64())))
    for name, array in additions:
        table = table.append_column(name, array)

    if "episode_index" in table.column_names:
        table = table.set_column(
            table.column_names.index("episode_index"),
            "episode_index",
            pa.array([episode_index] * count, type=pa.int64()),
        )
    if "frame_index" in table.column_names:
        table = table.set_column(table.column_names.index("frame_index"), "frame_index", pa.array(range(count), type=pa.int64()))

    target_columns = [
        LEROBOT_TOP_KEY,
        LEROBOT_WRIST_KEY,
        LEROBOT_STATE_KEY,
        LEROBOT_ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    missing = [column for column in target_columns if column not in table.column_names]
    if missing:
        raise KeyError(f"Missing required columns in source dataset: {missing}")
    return table.select(target_columns)


def _write_episode_parquets(
    source_root: Path,
    output_root: Path,
    episode_rows: list[dict[str, Any]],
    progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to export OpenPI legacy datasets.") from exc

    data_files = _parquet_files(source_root, "data")
    data_tables = [pq.read_table(path) for path in data_files]
    source_table = pa.concat_tables(data_tables, promote_options="default")
    legacy_features = _legacy_features(source_root)
    if "episode_index" not in source_table.column_names:
        raise KeyError("Source data parquet does not contain episode_index; cannot split legacy episodes.")

    total = len(episode_rows)
    for current, episode in enumerate(episode_rows, start=1):
        if cancel_check is not None:
            cancel_check()
        episode_index = int(episode["episode_index"])
        mask = pc.equal(source_table["episode_index"], episode_index)
        table = source_table.filter(mask)
        expected_length = int(episode["length"])
        if table.num_rows != expected_length:
            raise ValueError(
                f"Episode {episode_index} length mismatch: metadata={expected_length}, data={table.num_rows}"
            )
        global_start = int(episode.get("dataset_from_index", 0) or 0)
        table = _ensure_columns(table, episode_index, global_start)
        table = _patch_table_huggingface_metadata(table, legacy_features)
        chunk_index = episode_index // CHUNKS_SIZE
        out_path = output_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        if progress is not None:
            progress(current, total, f"Writing legacy episode {episode_index}/{total - 1}: {table.num_rows} frames", None)


def _write_metadata(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    task: str,
    tasks: dict[int, str],
    episode_rows: list[dict[str, Any]],
) -> None:
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    total_episodes = len(episode_rows)
    total_frames = sum(int(row["length"]) for row in episode_rows)
    total_chunks = max(1, (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE)
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH,
        "features": _legacy_features(source_root),
    }
    (meta_root / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4))

    _write_jsonl(
        meta_root / "tasks.jsonl",
        [{"task_index": task_index, "task": task_text} for task_index, task_text in sorted(tasks.items())],
    )
    _write_jsonl(
        meta_root / "episodes.jsonl",
        [
            {
                "episode_index": int(row["episode_index"]),
                "tasks": [_task_for_episode(row, tasks, task)],
                "length": int(row["length"]),
            }
            for row in episode_rows
        ],
    )
    _write_jsonl(
        meta_root / "episodes_stats.jsonl",
        [{"episode_index": int(row["episode_index"]), "stats": _episode_stats(row)} for row in episode_rows],
    )


def export_openpi_legacy_dataset(
    source_root: Path,
    repo_id: str,
    task: str,
    output_root: Path | None = None,
    overwrite: bool = True,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    source_root = Path(source_root)
    target_repo_id = f"{repo_id}_openpi_legacy"
    output_root = Path(output_root) if output_root is not None else source_root.parent / target_repo_id
    if cancel_check is not None:
        cancel_check()
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if output_root.exists():
        raise FileExistsError(f"OpenPI legacy output already exists: {output_root}")

    if progress is not None:
        progress(0, 4, "Reading LeRobot 0.4 metadata", None)
    episode_rows = _load_episode_rows(source_root)
    if not episode_rows:
        raise ValueError(f"No episode metadata found under {source_root}")
    tasks = _load_tasks(source_root, task)

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(1, 4, f"Exporting {len(episode_rows)} legacy episode parquet files", None)
    _write_episode_parquets(source_root, output_root, episode_rows, progress=None, cancel_check=cancel_check)

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(3, 4, "Writing OpenPI legacy metadata", None)
    _write_metadata(source_root, output_root, target_repo_id, task, tasks, episode_rows)

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(4, 4, f"OpenPI legacy export complete: {output_root}", 0)
    return output_root
