from pathlib import Path

FPS = 20
STATE_DIM = 7
ACTION_DIM = 7
IMAGE_MAX_OFFSET_SECONDS = 0.05
ROBOT_TYPE = "piper_single_arm"

RAW_REQUIRED_STREAMS = (
    "follower_joint",
    "follower_gripper",
    "leader_joint",
    "leader_gripper",
)
RAW_OPTIONAL_STREAMS = ("follower_endpose",)
IMAGE_FOLDER = "img"
TOP_IMAGE_STREAM = "front_color"
WRIST_IMAGE_STREAM = "wrist_color"

LEROBOT_TOP_KEY = "observation.images.top"
LEROBOT_WRIST_KEY = "observation.images.cam_left_wrist"
LEROBOT_STATE_KEY = "observation.state"
LEROBOT_ACTION_KEY = "action"

MODEL_TOP_KEY = "cam_high"
MODEL_WRIST_KEY = "cam_left_wrist"

TOOL_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TOOL_ROOT / "data"
DEFAULT_LEROBOT_ROOT = DATA_ROOT / "lerobot"
DEFAULT_ASSETS_ROOT = DATA_ROOT / "assets"
DEFAULT_OUTPUTS_ROOT = DATA_ROOT / "outputs"
