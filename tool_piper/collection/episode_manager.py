from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    path: Path
    modified_at: datetime
    size_bytes: int
    file_count: int
    dir_count: int


def _dir_size(path: Path) -> tuple[int, int, int]:
    size = 0
    files = 0
    dirs = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            size += child.stat().st_size
        elif child.is_dir():
            dirs += 1
    return size, files, dirs


def list_episodes(raw_root: Path) -> list[EpisodeInfo]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return []
    episodes = []
    for path in sorted((item for item in raw_root.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime):
        size, files, dirs = _dir_size(path)
        episodes.append(
            EpisodeInfo(
                name=path.name,
                path=path,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime),
                size_bytes=size,
                file_count=files,
                dir_count=dirs,
            )
        )
    return episodes


def delete_episode(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Episode path is not a directory: {path}")
    shutil.rmtree(path)


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
