from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tool_piper.constants import (
    LEROBOT_STATE_KEY,
    LEROBOT_TOP_KEY,
    LEROBOT_WRIST_KEY,
    MODEL_TOP_KEY,
    MODEL_WRIST_KEY,
    STATE_DIM,
)


def ensure_chw_uint8(image: Any) -> np.ndarray:
    if isinstance(image, dict) and image.get("bytes") is not None:
        data = np.frombuffer(image["bytes"], dtype=np.uint8)
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image bytes")
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        out = image
    elif image.shape[-1] == 3:
        out = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}")
    if np.issubdtype(out.dtype, np.floating):
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    elif out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def build_piper_observation(state: Any, top_image: Any, wrist_image: Any, prompt: str) -> dict:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (STATE_DIM,):
        raise ValueError(f"Expected 7D Piper state, got shape {state_array.shape}")
    return {
        "state": np.ascontiguousarray(state_array),
        "images": {
            MODEL_TOP_KEY: ensure_chw_uint8(top_image),
            MODEL_WRIST_KEY: ensure_chw_uint8(wrist_image),
        },
        "prompt": prompt,
    }


def load_lerobot_dataset(repo_id: str, root: Path | None = None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except Exception as exc:
            raise RuntimeError("LeRobot is required to load model observations.") from exc
    return LeRobotDataset(repo_id, root=root, download_videos=False)


def _load_local_sample(root: Path, frame_index: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to read local LeRobot parquet files.") from exc

    remaining = int(frame_index)
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    for path in files:
        table = pq.read_table(path)
        if remaining >= table.num_rows:
            remaining -= table.num_rows
            continue
        return {
            LEROBOT_STATE_KEY: table[LEROBOT_STATE_KEY][remaining].as_py(),
            LEROBOT_TOP_KEY: table[LEROBOT_TOP_KEY][remaining].as_py(),
            LEROBOT_WRIST_KEY: table[LEROBOT_WRIST_KEY][remaining].as_py(),
        }
    raise IndexError(f"frame_index out of range: {frame_index}")


def load_sample_observation(
    repo_id: str,
    root: Path | None = None,
    frame_index: int = 0,
    prompt: str | None = None,
) -> dict:
    if root is None:
        dataset = load_lerobot_dataset(repo_id, root)
        sample = dataset[int(frame_index)]
        sample_prompt = prompt or sample.get("task") or ""
    else:
        sample = _load_local_sample(Path(root), int(frame_index))
        sample_prompt = prompt or ""
    return build_piper_observation(
        sample[LEROBOT_STATE_KEY],
        sample[LEROBOT_TOP_KEY],
        sample[LEROBOT_WRIST_KEY],
        str(sample_prompt),
    )


def observation_summary(observation: dict) -> dict:
    return {
        "state_shape": tuple(np.asarray(observation["state"]).shape),
        "state_dtype": str(np.asarray(observation["state"]).dtype),
        "cam_high_shape": tuple(np.asarray(observation["images"][MODEL_TOP_KEY]).shape),
        "cam_high_dtype": str(np.asarray(observation["images"][MODEL_TOP_KEY]).dtype),
        "cam_left_wrist_shape": tuple(np.asarray(observation["images"][MODEL_WRIST_KEY]).shape),
        "cam_left_wrist_dtype": str(np.asarray(observation["images"][MODEL_WRIST_KEY]).dtype),
        "prompt": observation.get("prompt", ""),
    }
