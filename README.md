# Piper Tool

Standalone Piper data collection and processing toolchain.

This project extracts the useful Piper data path from the original OpenPI workspace into a smaller independent Tool. The Tool can inspect raw Piper episodes, convert them to LeRobot format, generate replay videos, compute normalization statistics, build model-ready observations, and prepare data for later OpenPI training on a separate PC.

## What this Tool does

- Manage raw Piper episode folders.
- Check raw dataset integrity.
- Convert raw Piper data to LeRobot format.
- Check LeRobot dataset schema.
- Generate replay videos with state/action overlays.
- Compute `norm_stats.json` for OpenPI-compatible normalization.
- Build and summarize model-ready observations.
- Show a simple GUI for staged data processing.

## What this Tool does not require

The data-processing GUI does not require OpenPI, ROS, CAN, or robot hardware. Hardware collection is represented in the GUI and has clean interfaces reserved for future camera/CAN integration.

## Main Piper semantics

The current supported mainline is:

- Piper single arm.
- Joint absolute action.
- No eef action.
- No delta action.
- `state = follower_joint(6) + follower_gripper(1)`.
- `action = leader_joint(6) + leader_gripper(1)`.
- State/action are both 7D.
- FPS is 20.
- Images use top/front camera and wrist camera.
- Model-ready keys are `state`, `images.cam_high`, `images.cam_left_wrist`, and `prompt`.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gui]"
```

Or use the helper script:

```bash
bash scripts/create_venv.sh
```

## Run GUI

```bash
bash scripts/run_gui.sh
```

If installed in an active environment:

```bash
tool-piper-gui
```

## Run CLI

```bash
tool-piper --help
```

The old module form also works:

```bash
PYTHONPATH="$PWD" python -m tool_piper.cli --help
```

## Typical workflow

```bash
tool-piper raw-check --raw-root data/raw/pickup

tool-piper convert \
  --raw-root data/raw/pickup \
  --repo-id piper_tool_pickup_100 \
  --task "pick up the egg and put into the yellow plate" \
  --latest-n 100

tool-piper lerobot-check \
  --repo-id piper_tool_pickup_100 \
  --root data/lerobot/piper_tool_pickup_100

tool-piper replay \
  --repo-id piper_tool_pickup_100 \
  --root data/lerobot/piper_tool_pickup_100 \
  --episode-index 0

tool-piper norm-stats \
  --repo-id piper_tool_pickup_100 \
  --root data/lerobot/piper_tool_pickup_100

tool-piper build-observation \
  --repo-id piper_tool_pickup_100 \
  --root data/lerobot/piper_tool_pickup_100 \
  --frame-index 0 \
  --prompt "pick up the egg and put into the yellow plate"
```

## Output locations

Default outputs are under `Tool/data`:

- Raw data: `data/raw/<dataset>`
- LeRobot dataset: `data/lerobot/<repo_id>`
- Normalization stats: `data/assets/<repo_id>/norm_stats.json`
- Replay videos: `data/outputs/replay/<repo_id>/episode_XXX.mp4`
- Reports: `data/outputs/reports/`

## PC training layout

For PC-side OpenPI training, copy data to a structure like:

```text
~/Desktop/project/OpenPi/datasets/<repo_id>/
~/Desktop/project/OpenPi/assets/<repo_id>/norm_stats.json
~/Desktop/project/OpenPi/checkpoints/
```

OpenPI config should use:

```text
repo_id = "<repo_id>"
asset_id = "<repo_id>"
default_prompt = "<task prompt>"
```

Recommended workflow:

1. Use this Tool locally or on the PC to generate LeRobot data and `norm_stats.json`.
2. Use a clean official OpenPI checkout on the PC.
3. Add a Piper training config that points to the dataset and asset id.
4. Run a short smoke test before full training.
5. Export checkpoint/assets back to the robot-side machine for local inference.

## GitHub note

Generated data, raw episodes, videos, packages, caches, and checkpoints are ignored by `.gitignore`. Upload code only; move datasets separately with a disk or explicit transfer.
