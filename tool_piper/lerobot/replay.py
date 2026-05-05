from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tool_piper.constants import (
    DEFAULT_OUTPUTS_ROOT,
    FPS,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
    LEROBOT_TOP_KEY,
    LEROBOT_WRIST_KEY,
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc_rgb(value: Any) -> np.ndarray:
    if isinstance(value, dict) and value.get("bytes") is not None:
        data = np.frombuffer(value["bytes"], dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = _to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _format_vector(name: str, value: np.ndarray) -> str:
    text = np.array2string(value.astype(float), precision=2, suppress_small=True, max_line_width=200)
    return f"{name}: {text}"


def _put_lines(image: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 24
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return image


ProgressCallback = Callable[[int, int, str, float | None], None]
CancelCheck = Callable[[], None]


def _load_episode_samples(root: Path, episode_index: int, cancel_check: CancelCheck | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to read local LeRobot parquet files.") from exc

    path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    if cancel_check is not None:
        cancel_check()
    table = pq.read_table(path)
    rows = []
    for row_index in range(table.num_rows):
        if cancel_check is not None and row_index % 100 == 0:
            cancel_check()
        rows.append(
            {
                LEROBOT_TOP_KEY: table[LEROBOT_TOP_KEY][row_index].as_py(),
                LEROBOT_WRIST_KEY: table[LEROBOT_WRIST_KEY][row_index].as_py(),
                LEROBOT_STATE_KEY: table[LEROBOT_STATE_KEY][row_index].as_py(),
                LEROBOT_ACTION_KEY: table[LEROBOT_ACTION_KEY][row_index].as_py(),
            }
        )
    return rows


def make_replay_video(
    repo_id: str,
    root: Path | None = None,
    episode_index: int = 0,
    out_path: Path | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    if root is None:
        raise ValueError("root is required for local replay generation")
    if cancel_check is not None:
        cancel_check()
    samples = _load_episode_samples(Path(root), int(episode_index), cancel_check=cancel_check)
    if not samples:
        raise ValueError(f"Episode {episode_index} has no frames")

    if out_path is None:
        out_path = DEFAULT_OUTPUTS_ROOT / "replay" / repo_id / f"episode_{int(episode_index):03d}.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = len(samples)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))

    writer = None
    try:
        if progress is not None:
            progress(0, max(frame_count, 1), f"Preparing replay for episode {episode_index}", None)
        for frame_index in range(frame_count):
            if cancel_check is not None:
                cancel_check()
            if progress is not None:
                progress(frame_index + 1, max(frame_count, 1), f"Rendering replay frame {frame_index + 1}/{frame_count}", None)
            sample = samples[frame_index]
            top = _image_hwc_rgb(sample[LEROBOT_TOP_KEY])
            wrist = _image_hwc_rgb(sample[LEROBOT_WRIST_KEY])
            if top.shape[:2] != wrist.shape[:2]:
                wrist = cv2.resize(wrist, (top.shape[1], top.shape[0]))

            state = _to_numpy(sample[LEROBOT_STATE_KEY]).astype(np.float32)
            action = _to_numpy(sample[LEROBOT_ACTION_KEY]).astype(np.float32)
            diff = action - state
            canvas = np.concatenate([top, wrist], axis=1)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            lines = [
                f"repo={repo_id} episode={episode_index} frame={frame_index}/{frame_count - 1} fps={FPS}",
                _format_vector("state", state),
                _format_vector("action", action),
                _format_vector("diff", diff),
                f"diff_norm: {float(np.linalg.norm(diff)):.4f}",
            ]
            canvas_bgr = _put_lines(canvas_bgr, lines)

            if writer is None:
                height, width = canvas_bgr.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create video writer: {out_path}")
            writer.write(canvas_bgr)
    finally:
        if writer is not None:
            writer.release()

    return out_path
