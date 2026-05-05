from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class CameraSource(Protocol):
    def read(self) -> dict[str, np.ndarray]: ...


class RobotStateSource(Protocol):
    def read_state(self) -> np.ndarray: ...

    def read_action(self) -> np.ndarray: ...


class EpisodeWriter(Protocol):
    def start(self, episode_dir: Path, task: str) -> None: ...

    def write_frame(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]) -> None: ...

    def stop(self) -> None: ...


class CollectionSession(Protocol):
    def start(self, dataset_root: Path, task: str) -> Path: ...

    def stop(self) -> None: ...
