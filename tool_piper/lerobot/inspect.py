from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tool_piper.constants import (
    ACTION_DIM,
    FPS,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
    LEROBOT_TOP_KEY,
    LEROBOT_WRIST_KEY,
    STATE_DIM,
)
from tool_piper.reports import GenericReport

ProgressCallback = Callable[[int, int, str, float | None], None]
CancelCheck = Callable[[], None]


def check_lerobot_dataset(
    repo_id: str,
    root: Path | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> GenericReport:
    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(0, 4, "Importing LeRobot dataset support", None)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except Exception as exc:
            return GenericReport(ok=False, errors=[f"LeRobot import failed: {exc}"])

    if root is None:
        from tool_piper.constants import DEFAULT_LEROBOT_ROOT

        root = DEFAULT_LEROBOT_ROOT / repo_id
    root = Path(root)

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(1, 4, f"Opening LeRobot dataset: {repo_id}", None)
    try:
        dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    except Exception as exc:
        return GenericReport(ok=False, errors=[f"Failed to open dataset: {exc}"])

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(2, 4, "Reading LeRobot metadata", None)
    errors: list[str] = []
    warnings: list[str] = []
    summary = {
        "repo_id": repo_id,
        "root": str(root),
        "num_episodes": getattr(dataset, "num_episodes", None),
        "num_frames": getattr(dataset, "num_frames", None),
        "fps": getattr(dataset, "fps", None),
        "features": list(getattr(dataset, "features", {}).keys()),
    }

    if summary["num_episodes"] in (None, 0):
        errors.append("dataset has no episodes")
    if summary["fps"] != FPS:
        warnings.append(f"unexpected fps: {summary['fps']} (expected {FPS})")

    if summary["num_frames"] and summary["num_frames"] > 0:
        if cancel_check is not None:
            cancel_check()
        if progress is not None:
            progress(3, 4, "Validating first LeRobot sample", None)
        try:
            sample = dataset[0]
        except Exception as exc:
            return GenericReport(ok=False, summary=summary, errors=[f"Failed to read sample: {exc}"])

        state = sample.get(LEROBOT_STATE_KEY)
        action = sample.get(LEROBOT_ACTION_KEY)
        top = sample.get(LEROBOT_TOP_KEY)
        wrist = sample.get(LEROBOT_WRIST_KEY)

        if state is None or getattr(state, "shape", None) != (STATE_DIM,):
            errors.append(f"state shape mismatch: {getattr(state, 'shape', None)}")
        if action is None or getattr(action, "shape", None) != (ACTION_DIM,):
            errors.append(f"action shape mismatch: {getattr(action, 'shape', None)}")
        if top is None:
            errors.append("missing top image")
        if wrist is None:
            errors.append("missing wrist image")

    if cancel_check is not None:
        cancel_check()
    if progress is not None:
        progress(4, 4, "LeRobot check complete", 0)
    return GenericReport(ok=not errors, summary=summary, warnings=warnings, errors=errors)
