from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tool_piper.constants import DEFAULT_ASSETS_ROOT, DEFAULT_LEROBOT_ROOT, DEFAULT_OUTPUTS_ROOT


@dataclass
class GuiConfig:
    raw_root: Path | None = None
    repo_id: str = "piper_tool_pickup_100"
    task: str = "pick up the egg and put into the yellow plate"
    lerobot_root: Path | None = None
    assets_root: Path = DEFAULT_ASSETS_ROOT
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT
    latest_n: int | None = 100
    episode_index: int = 0
    frame_index: int = 0
    max_frames: int | None = None
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None

    def dataset_root(self) -> Path:
        if self.lerobot_root is not None:
            return self.lerobot_root
        return DEFAULT_LEROBOT_ROOT / self.repo_id

    def norm_stats_dir(self) -> Path:
        return self.assets_root / self.repo_id
