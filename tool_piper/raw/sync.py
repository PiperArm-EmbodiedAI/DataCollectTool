from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np

from tool_piper.constants import IMAGE_MAX_OFFSET_SECONDS, TOP_IMAGE_STREAM, WRIST_IMAGE_STREAM
from tool_piper.raw.binary_io import (
    parser_gripper_ctrl_from_stream,
    parser_gripper_from_stream,
    parser_joint_ctrl_from_stream,
    parser_joint_from_stream,
)


def timestamp(value: float) -> float:
    return value / 1e9 if value > 1e12 else value


def stream_times(items) -> list[float]:
    return [timestamp(item.time_stamp) for item in items]


def image_timestamp(name: str) -> float:
    return float(name[:-4].replace("_", "."))


def image_names_and_times(folder: Path, stream_name: str) -> tuple[list[str], list[float]]:
    image_dir = folder / "img" / stream_name
    if not image_dir.exists():
        return [], []
    names = [path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    names.sort(key=image_timestamp)
    return names, [image_timestamp(name) for name in names]


def nearest_index(times: list[float], anchor: float) -> int | None:
    if not times:
        return None
    position = bisect.bisect_left(times, anchor)
    candidates = []
    if position < len(times):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    return min(candidates, key=lambda index: abs(times[index] - anchor))


def nearest_item(items: list, times: list[float], anchor: float):
    index = nearest_index(times, anchor)
    return None if index is None else items[index]


def nearest_image_name(names: list[str], times: list[float], anchor: float) -> str | None:
    index = nearest_index(times, anchor)
    if index is None or abs(times[index] - anchor) > IMAGE_MAX_OFFSET_SECONDS:
        return None
    return names[index]


def joint_to_array(joint) -> np.ndarray:
    return np.asarray(
        [
            joint.joint_state.joint_1,
            joint.joint_state.joint_2,
            joint.joint_state.joint_3,
            joint.joint_state.joint_4,
            joint.joint_state.joint_5,
            joint.joint_state.joint_6,
        ],
        dtype=np.float32,
    )


def gripper_to_array(gripper) -> np.ndarray:
    return np.asarray([gripper.gripper_state.grippers_angle], dtype=np.float32)


def leader_joint_to_array(joint) -> np.ndarray:
    return np.asarray(
        [
            joint.joint_ctrl.joint_1,
            joint.joint_ctrl.joint_2,
            joint.joint_ctrl.joint_3,
            joint.joint_ctrl.joint_4,
            joint.joint_ctrl.joint_5,
            joint.joint_ctrl.joint_6,
        ],
        dtype=np.float32,
    )


def leader_gripper_to_array(gripper) -> np.ndarray:
    return np.asarray([gripper.gripper_ctrl.grippers_angle], dtype=np.float32)


def piper_state(joint, gripper) -> np.ndarray:
    return np.concatenate([joint_to_array(joint), gripper_to_array(gripper)]).astype(np.float32)


def piper_action(joint, gripper) -> np.ndarray:
    return np.concatenate([leader_joint_to_array(joint), leader_gripper_to_array(gripper)]).astype(np.float32)


def read_image_chw(folder: Path, stream_name: str, image_name: str) -> np.ndarray:
    image_path = folder / "img" / stream_name / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def load_required_stream(folder: Path, filename: str, parser) -> list:
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required stream: {path}")
    items = list(parser(path))
    if not items:
        raise ValueError(f"Empty required stream: {path}")
    return items


def synced_frame_records(folder: Path) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    follower_joints = load_required_stream(folder, "follower_joint", parser_joint_from_stream)
    follower_grippers = load_required_stream(folder, "follower_gripper", parser_gripper_from_stream)
    leader_joints = load_required_stream(folder, "leader_joint", parser_joint_ctrl_from_stream)
    leader_grippers = load_required_stream(folder, "leader_gripper", parser_gripper_ctrl_from_stream)

    follower_gripper_times = stream_times(follower_grippers)
    leader_joint_times = stream_times(leader_joints)
    leader_gripper_times = stream_times(leader_grippers)
    top_image_names, top_image_times = image_names_and_times(folder, TOP_IMAGE_STREAM)
    wrist_image_names, wrist_image_times = image_names_and_times(folder, WRIST_IMAGE_STREAM)

    frames = []
    for follower_joint in follower_joints:
        anchor = timestamp(follower_joint.time_stamp)
        follower_gripper = nearest_item(follower_grippers, follower_gripper_times, anchor)
        leader_joint = nearest_item(leader_joints, leader_joint_times, anchor)
        leader_gripper = nearest_item(leader_grippers, leader_gripper_times, anchor)
        top_image_name = nearest_image_name(top_image_names, top_image_times, anchor)
        wrist_image_name = nearest_image_name(wrist_image_names, wrist_image_times, anchor)

        if None in (follower_gripper, leader_joint, leader_gripper, top_image_name, wrist_image_name):
            continue

        frames.append(
            (
                piper_state(follower_joint, follower_gripper),
                piper_action(leader_joint, leader_gripper),
                top_image_name,
                wrist_image_name,
            )
        )
    return frames


def synced_frames(folder: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    frames = []
    for state, action, top_name, wrist_name in synced_frame_records(folder):
        frames.append(
            (
                state,
                action,
                read_image_chw(folder, TOP_IMAGE_STREAM, top_name),
                read_image_chw(folder, WRIST_IMAGE_STREAM, wrist_name),
            )
        )
    return frames
