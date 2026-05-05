from __future__ import annotations

from pathlib import Path


def stage_hint(stage: str, repo_id: str, task: str, dataset_root: Path, assets_root: Path, outputs_root: Path) -> str:
    norm_path = assets_root / repo_id / "norm_stats.json"
    replay_dir = outputs_root / "replay" / repo_id
    legacy_root = dataset_root.parent / f"{repo_id}_openpi_legacy"
    hints = {
        "raw-check": f"Raw check finished. If ok, convert raw data to LeRobot for repo_id '{repo_id}'.",
        "convert": f"Conversion finished. LeRobot dataset is at:\n{dataset_root}\nNext: run LeRobot Check, Replay, and Norm Stats.",
        "lerobot-check": "LeRobot check finished. If ok, export OpenPI Legacy before using the current pinned OpenPI trainer.",
        "openpi-legacy": f"OpenPI legacy export finished. Copy this dataset to PC OpenPI datasets and use repo_id '{repo_id}_openpi_legacy':\n{legacy_root}",
        "replay": f"Replay generation finished. Check videos under:\n{replay_dir}",
        "norm-stats": f"Norm stats finished. Use this file for OpenPI assets:\n{norm_path}",
        "observation": "Observation check finished. Model-ready keys should be state, images.cam_high, images.cam_left_wrist, and prompt.",
        "policy": "Policy dry-run finished. If actions returned, the local/remote policy server accepted this observation format.",
    }
    return hints.get(stage, f"Stage '{stage}' finished for repo_id '{repo_id}' and task '{task}'.")


def pc_training_guide(repo_id: str, task: str) -> str:
    return f"""PC/OpenPI placement guide

Copy LeRobot dataset to:
~/Desktop/project/OpenPi/datasets/{repo_id}/

Copy norm stats to:
~/Desktop/project/OpenPi/assets/{repo_id}/norm_stats.json

OpenPI config values:
repo_id = "{repo_id}"
asset_id = "{repo_id}"
default_prompt = "{task}"

Recommended PC flow:
1. Use clean official OpenPI source.
2. Activate myOpenPi.
3. Verify LeRobotDataset can read ~/Desktop/project/OpenPi/datasets/{repo_id}.
4. Add a Piper training config using the values above.
5. Run a 1-5 step smoke test before full training.
6. Export checkpoint/assets back to the robot-side machine for local inference.
"""


def copy_paths_guide(repo_id: str, local_dataset: Path, local_assets: Path) -> str:
    return f"""Move/copy these outputs for PC training:

Dataset:
{local_dataset}
  -> ~/Desktop/project/OpenPi/datasets/{repo_id}/

Assets:
{local_assets / repo_id / 'norm_stats.json'}
  -> ~/Desktop/project/OpenPi/assets/{repo_id}/norm_stats.json

If using a mobile disk, keep the target folder names exactly matching repo_id/asset_id.
"""
