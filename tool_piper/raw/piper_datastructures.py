from dataclasses import dataclass, field


@dataclass
class ArmMsgJointFeedBack:
    joint_1: int = 0
    joint_2: int = 0
    joint_3: int = 0
    joint_4: int = 0
    joint_5: int = 0
    joint_6: int = 0


@dataclass
class ArmMsgJointCtrl:
    joint_1: int = 0
    joint_2: int = 0
    joint_3: int = 0
    joint_4: int = 0
    joint_5: int = 0
    joint_6: int = 0


@dataclass
class ArmMsgGripperFeedBack:
    grippers_angle: int = 0
    grippers_effort: int = 0
    status_code: int = 0


@dataclass
class ArmMsgGripperCtrl:
    grippers_angle: int = 0
    grippers_effort: int = 0
    status_code: int = 0


@dataclass
class ArmMsgEndPoseFeedBack:
    X_axis: int = 0
    Y_axis: int = 0
    Z_axis: int = 0
    RX_axis: int = 0
    RY_axis: int = 0
    RZ_axis: int = 0


@dataclass
class ArmMsgEndPoseCtrl:
    X_axis: int = 0
    Y_axis: int = 0
    Z_axis: int = 0
    RX_axis: int = 0
    RY_axis: int = 0
    RZ_axis: int = 0


@dataclass
class ArmJoint:
    time_stamp: float = 0.0
    Hz: float = 0.0
    joint_state: ArmMsgJointFeedBack = field(default_factory=ArmMsgJointFeedBack)


@dataclass
class ArmJointCtrl:
    time_stamp: float = 0.0
    Hz: float = 0.0
    joint_ctrl: ArmMsgJointCtrl = field(default_factory=ArmMsgJointCtrl)


@dataclass
class ArmGripper:
    time_stamp: float = 0.0
    Hz: float = 0.0
    gripper_state: ArmMsgGripperFeedBack = field(default_factory=ArmMsgGripperFeedBack)


@dataclass
class ArmGripperCtrl:
    time_stamp: float = 0.0
    Hz: float = 0.0
    gripper_ctrl: ArmMsgGripperCtrl = field(default_factory=ArmMsgGripperCtrl)


@dataclass
class ArmEndPose:
    time_stamp: float = 0.0
    Hz: float = 0.0
    end_pose: ArmMsgEndPoseFeedBack = field(default_factory=ArmMsgEndPoseFeedBack)


@dataclass
class ArmEndPoseCtrl:
    time_stamp: float = 0.0
    Hz: float = 0.0
    endpose_ctrl: ArmMsgEndPoseCtrl = field(default_factory=ArmMsgEndPoseCtrl)
