from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import struct
from typing import BinaryIO

from .piper_datastructures import (
    ArmEndPose,
    ArmEndPoseCtrl,
    ArmGripper,
    ArmGripperCtrl,
    ArmJoint,
    ArmJointCtrl,
)

JOINT_STRUCT = struct.Struct(">ddiiiiii")
GRIPPER_STRUCT = struct.Struct(">ddiii")
ENDPOSE_STRUCT = struct.Struct(">ddiiiiii")


def _open_if_path(source: str | Path | BinaryIO):
    if hasattr(source, "read"):
        return source, False
    return Path(source).open("rb"), True


def _parse_blocks(source: str | Path | BinaryIO, block: struct.Struct, parser) -> Iterator:
    stream, should_close = _open_if_path(source)
    try:
        while True:
            data = stream.read(block.size)
            if not data:
                break
            if len(data) != block.size:
                break
            yield parser(data)
    finally:
        if should_close:
            stream.close()


def _bytes_to_joint(data: bytes) -> ArmJoint:
    ts, hz, j1, j2, j3, j4, j5, j6 = JOINT_STRUCT.unpack(data)
    item = ArmJoint(time_stamp=ts, Hz=hz)
    item.joint_state.joint_1 = j1
    item.joint_state.joint_2 = j2
    item.joint_state.joint_3 = j3
    item.joint_state.joint_4 = j4
    item.joint_state.joint_5 = j5
    item.joint_state.joint_6 = j6
    return item


def _bytes_to_joint_ctrl(data: bytes) -> ArmJointCtrl:
    ts, hz, j1, j2, j3, j4, j5, j6 = JOINT_STRUCT.unpack(data)
    item = ArmJointCtrl(time_stamp=ts, Hz=hz)
    item.joint_ctrl.joint_1 = j1
    item.joint_ctrl.joint_2 = j2
    item.joint_ctrl.joint_3 = j3
    item.joint_ctrl.joint_4 = j4
    item.joint_ctrl.joint_5 = j5
    item.joint_ctrl.joint_6 = j6
    return item


def _bytes_to_gripper(data: bytes) -> ArmGripper:
    ts, hz, angle, effort, status = GRIPPER_STRUCT.unpack(data)
    item = ArmGripper(time_stamp=ts, Hz=hz)
    item.gripper_state.grippers_angle = angle
    item.gripper_state.grippers_effort = effort
    item.gripper_state.status_code = status
    return item


def _bytes_to_gripper_ctrl(data: bytes) -> ArmGripperCtrl:
    ts, hz, angle, effort, status = GRIPPER_STRUCT.unpack(data)
    item = ArmGripperCtrl(time_stamp=ts, Hz=hz)
    item.gripper_ctrl.grippers_angle = angle
    item.gripper_ctrl.grippers_effort = effort
    item.gripper_ctrl.status_code = status
    return item


def _bytes_to_endpose(data: bytes) -> ArmEndPose:
    ts, hz, x, y, z, rx, ry, rz = ENDPOSE_STRUCT.unpack(data)
    item = ArmEndPose(time_stamp=ts, Hz=hz)
    item.end_pose.X_axis = x
    item.end_pose.Y_axis = y
    item.end_pose.Z_axis = z
    item.end_pose.RX_axis = rx
    item.end_pose.RY_axis = ry
    item.end_pose.RZ_axis = rz
    return item


def _bytes_to_endpose_ctrl(data: bytes) -> ArmEndPoseCtrl:
    ts, hz, x, y, z, rx, ry, rz = ENDPOSE_STRUCT.unpack(data)
    item = ArmEndPoseCtrl(time_stamp=ts, Hz=hz)
    item.endpose_ctrl.X_axis = x
    item.endpose_ctrl.Y_axis = y
    item.endpose_ctrl.Z_axis = z
    item.endpose_ctrl.RX_axis = rx
    item.endpose_ctrl.RY_axis = ry
    item.endpose_ctrl.RZ_axis = rz
    return item


def parser_joint_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmJoint]:
    return _parse_blocks(source, JOINT_STRUCT, _bytes_to_joint)


def parser_joint_ctrl_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmJointCtrl]:
    return _parse_blocks(source, JOINT_STRUCT, _bytes_to_joint_ctrl)


def parser_gripper_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmGripper]:
    return _parse_blocks(source, GRIPPER_STRUCT, _bytes_to_gripper)


def parser_gripper_ctrl_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmGripperCtrl]:
    return _parse_blocks(source, GRIPPER_STRUCT, _bytes_to_gripper_ctrl)


def parser_endpose_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmEndPose]:
    return _parse_blocks(source, ENDPOSE_STRUCT, _bytes_to_endpose)


def parser_endpose_ctrl_from_stream(source: str | Path | BinaryIO) -> Iterator[ArmEndPoseCtrl]:
    return _parse_blocks(source, ENDPOSE_STRUCT, _bytes_to_endpose_ctrl)


def count_records(path: Path, record_size: int) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size // record_size
