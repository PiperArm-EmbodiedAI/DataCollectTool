from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

from tool_piper.constants import DEFAULT_OUTPUTS_ROOT
from tool_piper.lerobot.convert import convert_raw_to_lerobot
from tool_piper.lerobot.inspect import check_lerobot_dataset
from tool_piper.lerobot.replay import make_replay_video
from tool_piper.model.observation import load_sample_observation, observation_summary
from tool_piper.norm.stats import compute_norm_stats
from tool_piper.raw.inspector import check_raw_dataset, check_raw_episode
from tool_piper.reports import to_dict, write_json_report


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def raw_check(args: argparse.Namespace) -> None:
    target = Path(args.raw_root)
    report = check_raw_episode(target) if args.episode else check_raw_dataset(target)
    print(_json(to_dict(report)))
    if args.out is not None:
        out_path = write_json_report(report, Path(args.out))
    else:
        out_dir = DEFAULT_OUTPUTS_ROOT / "reports"
        name = f"raw_episode_{target.name}.json" if args.episode else f"raw_dataset_{target.name}.json"
        out_path = write_json_report(report, out_dir / name)
    print(f"report: {out_path}")


def convert(args: argparse.Namespace) -> None:
    path = convert_raw_to_lerobot(
        raw_root=Path(args.raw_root),
        repo_id=args.repo_id,
        task=args.task,
        output_root=Path(args.output_root) if args.output_root else None,
        latest_n=args.latest_n,
        overwrite=not args.no_overwrite,
    )
    print(f"dataset: {path}")


def lerobot_check(args: argparse.Namespace) -> None:
    report = check_lerobot_dataset(args.repo_id, Path(args.root) if args.root else None)
    print(_json(to_dict(report)))
    if args.out:
        print(f"report: {write_json_report(report, Path(args.out))}")


def replay(args: argparse.Namespace) -> None:
    path = make_replay_video(
        repo_id=args.repo_id,
        root=Path(args.root) if args.root else None,
        episode_index=args.episode_index,
        out_path=Path(args.out) if args.out else None,
        max_frames=args.max_frames,
    )
    print(f"video: {path}")


def norm_stats(args: argparse.Namespace) -> None:
    path = compute_norm_stats(
        repo_id=args.repo_id,
        root=Path(args.root) if args.root else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        max_frames=args.max_frames,
    )
    print(f"norm_stats: {path}")


def build_observation(args: argparse.Namespace) -> None:
    observation = load_sample_observation(
        repo_id=args.repo_id,
        root=Path(args.root) if args.root else None,
        frame_index=args.frame_index,
        prompt=args.prompt,
    )
    print(_json(observation_summary(observation)))


def policy_dry_run_cmd(args: argparse.Namespace) -> None:
    from tool_piper.model.policy_client import policy_dry_run

    observation = load_sample_observation(
        repo_id=args.repo_id,
        root=Path(args.root) if args.root else None,
        frame_index=args.frame_index,
        prompt=args.prompt,
    )
    actions = policy_dry_run(args.host, args.port, observation, api_key=args.api_key)
    print(_json({"actions_shape": list(actions.shape), "first_action": actions[0].astype(float).tolist()}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Piper data toolchain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw-check", help="Check raw Piper episode folders")
    raw_parser.add_argument("--raw-root", required=True, help="Raw dataset root or one episode directory with --episode")
    raw_parser.add_argument("--episode", action="store_true", help="Treat --raw-root as a single episode directory")
    raw_parser.add_argument("--out", help="Optional JSON report path")
    raw_parser.set_defaults(func=raw_check)

    convert_parser = subparsers.add_parser("convert", help="Convert raw Piper data to LeRobot")
    convert_parser.add_argument("--raw-root", required=True)
    convert_parser.add_argument("--repo-id", required=True)
    convert_parser.add_argument("--task", required=True)
    convert_parser.add_argument("--output-root")
    convert_parser.add_argument("--latest-n", type=int)
    convert_parser.add_argument("--no-overwrite", action="store_true")
    convert_parser.set_defaults(func=convert)

    check_parser = subparsers.add_parser("lerobot-check", help="Check LeRobot Piper schema")
    check_parser.add_argument("--repo-id", required=True)
    check_parser.add_argument("--root")
    check_parser.add_argument("--out")
    check_parser.set_defaults(func=lerobot_check)

    replay_parser = subparsers.add_parser("replay", help="Create replay mp4 with overlays")
    replay_parser.add_argument("--repo-id", required=True)
    replay_parser.add_argument("--root")
    replay_parser.add_argument("--episode-index", type=int, default=0)
    replay_parser.add_argument("--out")
    replay_parser.add_argument("--max-frames", type=int)
    replay_parser.set_defaults(func=replay)

    norm_parser = subparsers.add_parser("norm-stats", help="Compute OpenPI-compatible Piper norm stats")
    norm_parser.add_argument("--repo-id", required=True)
    norm_parser.add_argument("--root")
    norm_parser.add_argument("--out-dir")
    norm_parser.add_argument("--max-frames", type=int)
    norm_parser.set_defaults(func=norm_stats)

    obs_parser = subparsers.add_parser("build-observation", help="Build and summarize model-ready observation")
    obs_parser.add_argument("--repo-id", required=True)
    obs_parser.add_argument("--root")
    obs_parser.add_argument("--frame-index", type=int, default=0)
    obs_parser.add_argument("--prompt")
    obs_parser.set_defaults(func=build_observation)

    dry_parser = subparsers.add_parser("policy-dry-run", help="Send one sample observation to policy server")
    dry_parser.add_argument("--host", default="0.0.0.0")
    dry_parser.add_argument("--port", type=int, default=8000)
    dry_parser.add_argument("--api-key")
    dry_parser.add_argument("--repo-id", required=True)
    dry_parser.add_argument("--root")
    dry_parser.add_argument("--frame-index", type=int, default=0)
    dry_parser.add_argument("--prompt")
    dry_parser.set_defaults(func=policy_dry_run_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
