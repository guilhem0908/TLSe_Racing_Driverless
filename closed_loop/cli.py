"""
closed_loop/cli.py

Command line of the closed loop: ``python -m closed_loop --help``.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from closed_loop.loop import LoopConfig, RunResult, run
from closed_loop.track import available_tracks, load_start_pose, track_path
from simulation.vision import DEFAULT_VISION, VisionConfig
from track_utils import load_track


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m closed_loop``."""
    parser = argparse.ArgumentParser(
        prog="python -m closed_loop",
        description="Drive the simulated car around a cone track with the "
        "field-of-view sensor, a cone memory and pure pursuit.",
    )
    parser.add_argument(
        "--track",
        default="belgium",
        help="bundled track name (%s) or path to a CSV file" % ", ".join(available_tracks()),
    )
    parser.add_argument(
        "--range", type=float, default=DEFAULT_VISION.range_m, dest="range_m",
        help="sensor range in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--fov", type=float, default=DEFAULT_VISION.fov_deg, dest="fov_deg",
        help="sensor field of view in degrees (default: %(default)s)",
    )
    parser.add_argument("--laps", type=int, default=3, help="laps to complete (default: 3)")
    parser.add_argument(
        "--memory", type=float, default=LoopConfig.memory_ttl_s, dest="memory_ttl_s",
        help="seconds a cone is remembered, 0 disables the memory (default: %(default)s)",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="do not extend the centre line from cones seen on one side only",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, dest="noise_std_m",
        help="standard deviation of the detection noise in metres (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the detection noise")
    parser.add_argument(
        "--headless", action="store_true",
        help="run without a window and print the lap table",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> LoopConfig:
    """Translate parsed arguments into a run configuration."""
    return LoopConfig(
        vision=VisionConfig(range_m=args.range_m, fov_deg=args.fov_deg),
        memory_ttl_s=args.memory_ttl_s,
        one_sided_fallback=not args.no_fallback,
        noise_std_m=args.noise_std_m,
        seed=args.seed,
        target_laps=args.laps,
    )


def format_result(result: RunResult) -> List[str]:
    """Human-readable summary of a run."""
    lines = [
        f"status: {result.status}",
        f"simulated time: {result.sim_time_s:.1f} s, distance: {result.distance_m:.0f} m, "
        f"top speed: {result.max_speed_mps:.1f} m/s",
    ]
    for lap in result.laps:
        note = "" if lap.valid else "  (not valid)"
        lines.append(f"lap {lap.number}: {lap.time_s:.2f} s, {lap.cones_hit} cones hit{note}")
    lines.append(f"cones hit in total: {result.cones_hit_total}")
    if result.off_course_at is not None:
        t, x, y = result.off_course_at
        lines.append(f"left the track at t = {t:.1f} s, x = {x:.1f} m, y = {y:.1f} m")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    path = track_path(args.track)
    cones = load_track(str(path))
    start = load_start_pose(path)
    config = config_from_args(args)

    if args.headless:
        result = run(cones, start, config)
        print("\n".join(format_result(result)))
        return 0 if result.status == "finished" else 1

    from closed_loop.viewer import run_viewer  # needs a display

    run_viewer(path.stem, cones, start, config)
    return 0
