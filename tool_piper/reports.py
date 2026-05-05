from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class EpisodeReport:
    episode: str
    path: str
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stream_counts: dict[str, int] = field(default_factory=dict)
    image_counts: dict[str, int] = field(default_factory=dict)
    synced_frames: int = 0


@dataclass
class DatasetReport:
    raw_root: str
    ok: bool
    episodes: list[EpisodeReport]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def num_ok(self) -> int:
        return sum(1 for ep in self.episodes if ep.ok)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


@dataclass
class GenericReport:
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)


def write_json_report(report: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(report), ensure_ascii=False, indent=2))
    return path
