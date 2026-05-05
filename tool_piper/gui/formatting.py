from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import shlex
import traceback
from typing import Any

from tool_piper.reports import to_dict


def pretty_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    else:
        try:
            value = to_dict(value)
        except Exception:
            pass
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def command_raw_check(raw_root: str, episode: bool = False) -> str:
    parts = ["tool-piper", "raw-check", "--raw-root", raw_root]
    if episode:
        parts.append("--episode")
    return shell_join(parts)


def command_convert(raw_root: str, repo_id: str, task: str, latest_n: int | None) -> str:
    parts = ["tool-piper", "convert", "--raw-root", raw_root, "--repo-id", repo_id, "--task", task]
    if latest_n is not None:
        parts.extend(["--latest-n", str(latest_n)])
    return shell_join(parts)


def command_lerobot_check(repo_id: str, root: str) -> str:
    return shell_join(["tool-piper", "lerobot-check", "--repo-id", repo_id, "--root", root])


def command_export_openpi_legacy(repo_id: str, root: str, task: str, output_root: str) -> str:
    return shell_join(
        [
            "tool-piper",
            "export-openpi-legacy",
            "--repo-id",
            repo_id,
            "--root",
            root,
            "--task",
            task,
            "--output-root",
            output_root,
        ]
    )


def command_replay(repo_id: str, root: str, episode_index: int, max_frames: int | None) -> str:
    parts = ["tool-piper", "replay", "--repo-id", repo_id, "--root", root, "--episode-index", str(episode_index)]
    if max_frames is not None:
        parts.extend(["--max-frames", str(max_frames)])
    return shell_join(parts)


def command_norm_stats(repo_id: str, root: str, out_dir: str) -> str:
    return shell_join(["tool-piper", "norm-stats", "--repo-id", repo_id, "--root", root, "--out-dir", out_dir])


def command_build_observation(repo_id: str, root: str, frame_index: int, prompt: str) -> str:
    return shell_join(
        [
            "tool-piper",
            "build-observation",
            "--repo-id",
            repo_id,
            "--root",
            root,
            "--frame-index",
            str(frame_index),
            "--prompt",
            prompt,
        ]
    )


def command_policy_dry_run(host: str, port: int, repo_id: str, root: str, frame_index: int, prompt: str) -> str:
    return shell_join(
        [
            "tool-piper",
            "policy-dry-run",
            "--host",
            host,
            "--port",
            str(port),
            "--repo-id",
            repo_id,
            "--root",
            root,
            "--frame-index",
            str(frame_index),
            "--prompt",
            prompt,
        ]
    )
