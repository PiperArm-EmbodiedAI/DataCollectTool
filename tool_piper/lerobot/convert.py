from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import time

from tool_piper.constants import (
    DEFAULT_LEROBOT_ROOT,
    FPS,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
    LEROBOT_TOP_KEY,
    LEROBOT_WRIST_KEY,
    ROBOT_TYPE,
)
from tool_piper.raw.sync import synced_frames


@dataclass(frozen=True)
class ConversionProgress:
    current: int
    total: int
    episode: str
    frames: int
    message: str
    elapsed_seconds: float
    eta_seconds: float | None


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def episode_dirs(raw_root: Path, latest_n: int | None = None) -> list[Path]:
    folders = [path for path in Path(raw_root).iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.stat().st_mtime)
    if latest_n is not None:
        folders = folders[-latest_n:]
    return folders


def fix_parquet_sequence_metadata(dataset_root: Path, cancel_check: Callable[[], None] | None = None) -> None:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to patch LeRobot parquet metadata.") from exc

    for path in sorted((Path(dataset_root) / "data").glob("chunk-*/episode_*.parquet")):
        if cancel_check is not None:
            cancel_check()
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        payload = metadata.get(b"huggingface")
        if payload is None:
            continue
        info = json.loads(payload.decode("utf-8"))
        changed = False
        for key in (LEROBOT_STATE_KEY, LEROBOT_ACTION_KEY):
            feature = info.get("info", {}).get("features", {}).get(key)
            if feature and feature.get("_type") == "List":
                feature["_type"] = "Sequence"
                changed = True
        if changed:
            metadata[b"huggingface"] = json.dumps(info, separators=(",", ":")).encode("utf-8")
            table = table.replace_schema_metadata(metadata)
            pq.write_table(table, path)


def convert_raw_to_lerobot(
    raw_root: Path,
    repo_id: str,
    task: str,
    output_root: Path | None = None,
    latest_n: int | None = None,
    overwrite: bool = True,
    progress: Callable[[ConversionProgress], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError("LeRobot is required for conversion. Install Tool/requirements.txt in your environment.") from exc

    raw_root = Path(raw_root)
    dataset_root = Path(output_root) if output_root is not None else DEFAULT_LEROBOT_ROOT / repo_id
    if cancel_check is not None:
        cancel_check()
    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)
    if cancel_check is not None:
        cancel_check()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            LEROBOT_TOP_KEY: {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            LEROBOT_WRIST_KEY: {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            LEROBOT_STATE_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            LEROBOT_ACTION_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
        use_videos=True,
    )

    folders = episode_dirs(raw_root, latest_n)
    if not folders:
        raise FileNotFoundError(f"No episode folders found under {raw_root}")

    started_at = time.monotonic()
    converted = 0
    total_folders = len(folders)

    def notify(index: int, folder: Path, frames_count: int, message: str) -> None:
        elapsed = time.monotonic() - started_at
        eta = None
        if index > 0:
            eta = (elapsed / index) * max(total_folders - index, 0)
        progress_event = ConversionProgress(
            current=index,
            total=total_folders,
            episode=folder.name,
            frames=frames_count,
            message=message,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )
        if progress is not None:
            progress(progress_event)
        else:
            print(
                f"[{index}/{total_folders}] {message} | elapsed {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}",
                flush=True,
            )

    for index, folder in enumerate(folders, start=1):
        if cancel_check is not None:
            cancel_check()
        frames = synced_frames(folder)
        if len(frames) < 2:
            notify(index, folder, len(frames), f"Skipping {folder.name}: only {len(frames)} synced frames")
            continue

        episode = frames[3:-3] if len(frames) > 8 else frames
        notify(index, folder, len(episode), f"Converting {folder.name}: {len(episode)} frames")
        for frame_index, (state, action, top_image, wrist_image) in enumerate(episode):
            if cancel_check is not None and frame_index % 20 == 0:
                cancel_check()
            dataset.add_frame(
                {
                    LEROBOT_TOP_KEY: top_image,
                    LEROBOT_WRIST_KEY: wrist_image,
                    LEROBOT_STATE_KEY: state,
                    LEROBOT_ACTION_KEY: action,
                    "task": task,
                }
            )
        if cancel_check is not None:
            cancel_check()
        dataset.save_episode()
        converted += 1

    if converted == 0:
        raise RuntimeError(f"No episodes converted from {raw_root}")

    if cancel_check is not None:
        cancel_check()
    fix_parquet_sequence_metadata(dataset_root, cancel_check=cancel_check)
    return dataset_root
