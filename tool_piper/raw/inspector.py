from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

from tool_piper.constants import (
    RAW_OPTIONAL_STREAMS,
    RAW_REQUIRED_STREAMS,
    TOP_IMAGE_STREAM,
    WRIST_IMAGE_STREAM,
)
from tool_piper.raw.binary_io import ENDPOSE_STRUCT, GRIPPER_STRUCT, JOINT_STRUCT, count_records
from tool_piper.raw.sync import synced_frame_records
from tool_piper.reports import DatasetReport, EpisodeReport

ProgressCallback = Callable[[int, int, str, float | None], None]
CancelCheck = Callable[[], None]

_RECORD_SIZES = {
    "follower_joint": JOINT_STRUCT.size,
    "leader_joint": JOINT_STRUCT.size,
    "follower_gripper": GRIPPER_STRUCT.size,
    "leader_gripper": GRIPPER_STRUCT.size,
    "follower_endpose": ENDPOSE_STRUCT.size,
}


def _image_count(episode_dir: Path, stream_name: str) -> int:
    image_dir = episode_dir / "img" / stream_name
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _episode_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    folders = [path for path in raw_root.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.stat().st_mtime)
    return folders


def check_raw_episode(episode_dir: Path, cancel_check: CancelCheck | None = None) -> EpisodeReport:
    episode_dir = Path(episode_dir)
    errors: list[str] = []
    warnings: list[str] = []
    stream_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}

    if cancel_check is not None:
        cancel_check()

    if not episode_dir.exists():
        return EpisodeReport(
            episode=episode_dir.name,
            path=str(episode_dir),
            ok=False,
            errors=["episode directory does not exist"],
        )

    for stream in RAW_REQUIRED_STREAMS:
        if cancel_check is not None:
            cancel_check()
        path = episode_dir / stream
        if not path.exists():
            errors.append(f"missing required stream: {stream}")
            stream_counts[stream] = 0
            continue
        if path.stat().st_size == 0:
            errors.append(f"empty required stream: {stream}")
        stream_counts[stream] = count_records(path, _RECORD_SIZES[stream])

    for stream in RAW_OPTIONAL_STREAMS:
        if cancel_check is not None:
            cancel_check()
        path = episode_dir / stream
        stream_counts[stream] = count_records(path, _RECORD_SIZES[stream]) if path.exists() else 0
        if not path.exists():
            warnings.append(f"missing optional stream: {stream}")

    for stream in (TOP_IMAGE_STREAM, WRIST_IMAGE_STREAM):
        if cancel_check is not None:
            cancel_check()
        image_counts[stream] = _image_count(episode_dir, stream)
        if image_counts[stream] == 0:
            errors.append(f"missing or empty image stream: img/{stream}")

    synced_count = 0
    if not errors:
        try:
            if cancel_check is not None:
                cancel_check()
            synced_count = len(synced_frame_records(episode_dir))
            if synced_count < 2:
                errors.append(f"only {synced_count} synced frames")
            elif synced_count < 20:
                warnings.append(f"short episode: only {synced_count} synced frames")
        except Exception as exc:
            errors.append(f"sync failed: {exc}")

    return EpisodeReport(
        episode=episode_dir.name,
        path=str(episode_dir),
        ok=not errors,
        warnings=warnings,
        errors=errors,
        stream_counts=stream_counts,
        image_counts=image_counts,
        synced_frames=synced_count,
    )


def check_raw_dataset(
    raw_root: Path,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> DatasetReport:
    raw_root = Path(raw_root)
    errors: list[str] = []
    warnings: list[str] = []
    if cancel_check is not None:
        cancel_check()
    if not raw_root.exists():
        errors.append(f"raw root does not exist: {raw_root}")
        return DatasetReport(raw_root=str(raw_root), ok=False, episodes=[], errors=errors)

    episode_dirs = _episode_dirs(raw_root)
    total = len(episode_dirs)
    started_at = time.monotonic()
    episodes = []
    if progress is not None:
        progress(0, max(total, 1), f"Scanning raw episodes under {raw_root}", None)
    for index, path in enumerate(episode_dirs, start=1):
        if cancel_check is not None:
            cancel_check()
        episode = check_raw_episode(path, cancel_check=cancel_check)
        episodes.append(episode)
        if progress is not None:
            elapsed = time.monotonic() - started_at
            eta = (elapsed / index) * max(total - index, 0) if index else None
            progress(index, max(total, 1), f"Checked raw episode {index}/{total}: {path.name}", eta)
    if not episodes:
        errors.append(f"no episode folders under: {raw_root}")

    if episodes and not any(ep.ok for ep in episodes):
        errors.append("no valid episodes found")

    return DatasetReport(
        raw_root=str(raw_root),
        ok=not errors and all(ep.ok for ep in episodes),
        episodes=episodes,
        warnings=warnings,
        errors=errors,
    )
