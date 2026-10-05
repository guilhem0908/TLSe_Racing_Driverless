"""
scripts/benchmark.py

Run the closed loop on every bundled track and write the results.

Two families of runs:
- sensor sweep: the full controller with several range / field-of-view
  settings of the sensor model;
- ablations: the default sensor with one part of the controller removed, or
  with noise added to the detections.

Everything is deterministic: the simulation has no randomness except the
detection noise, which uses a fixed seed. Outputs, in ``results/``:
- ``benchmark.json``: every run with all its laps, plus the environment;
- ``benchmark.csv``: one line per run;
- ``benchmark.md``: the tables shown in the README.

Usage (from the repository root):
    python scripts/benchmark.py            # full benchmark
    python scripts/benchmark.py --quick    # one lap, three short tracks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from closed_loop.centerline import pair_cones  # noqa: E402
from closed_loop.loop import LoopConfig, run  # noqa: E402
from closed_loop.pure_pursuit import menger_curvature, resample  # noqa: E402
from closed_loop.track import (  # noqa: E402
    available_tracks,
    load_start_pose,
    reference_gates,
    split_cones,
    track_path,
)
from closed_loop.vehicle import VehicleParams  # noqa: E402
from simulation.vision import VisionConfig  # noqa: E402
from track_utils import load_track  # noqa: E402

SEED = 0
TARGET_LAPS = 3

# (key, range in metres, field of view in degrees, note)
SENSORS: List[Tuple[str, float, float, str]] = [
    ("default", 4.0, 100.0, "repository default"),
    ("first", 7.0, 60.0, "first setting, November 2025"),
    ("wide", 8.0, 120.0, ""),
    ("long", 12.0, 120.0, ""),
]

# (key, label, LoopConfig overrides), all with the default sensor
VARIANTS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("full", "full controller", {}),
    ("no_memory", "no cone memory", {"memory_ttl_s": 0.0}),
    ("no_fallback", "no one-sided fallback", {"one_sided_fallback": False}),
    ("noise_0.1", "detection noise, sigma 0.1 m", {"noise_std_m": 0.1}),
    ("noise_0.2", "detection noise, sigma 0.2 m", {"noise_std_m": 0.2}),
]

QUICK_TRACKS = ["belgium", "peanut", "small_track"]


def describe_track(name: str) -> Dict[str, Any]:
    """Facts about a track computed from its cone map."""
    path = track_path(name)
    cones = load_track(str(path))
    start = load_start_pose(path)
    blue, yellow = split_cones(cones)
    gates = reference_gates(cones, start)

    mids = [g.mid for g in gates]
    closed = mids + mids[:1]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(closed, closed[1:]))

    # Tightest bend of the centre line: radius of the circle through three
    # points 3 m apart along the closed loop. The long arms keep the small
    # zigzag of the gate middles from showing up as a bend.
    step, arm = 0.5, 6
    samples = resample(closed, step)
    wrapped = samples[-arm:] + samples + samples[:arm]
    kappa = max(
        menger_curvature(wrapped[i - arm], wrapped[i], wrapped[i + arm])
        for i in range(arm, len(wrapped) - arm)
    )

    widths = [g.width for g in gates]
    return {
        "track": name,
        "blue_cones": len(blue),
        "yellow_cones": len(yellow),
        "big_orange_cones": sum(1 for c in cones if c["tag"] == "big_orange"),
        "paired_gates": len(pair_cones(blue, yellow)),
        "reference_gates": len(gates),
        "centerline_length_m": round(length, 1),
        "gate_width_min_m": round(min(widths), 2),
        "gate_width_median_m": round(statistics.median(widths), 2),
        "gate_width_max_m": round(max(widths), 2),
        "tightest_radius_m": round(1.0 / kappa, 2) if kappa > 1e-9 else None,
    }


def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one configuration on one track and return a flat record."""
    path = track_path(job["track"])
    cones = load_track(str(path))
    start = load_start_pose(path)
    config = replace(
        LoopConfig(
            vision=VisionConfig(range_m=job["range_m"], fov_deg=job["fov_deg"]),
            target_laps=job["laps"],
            seed=SEED,
        ),
        **job["overrides"],
    )

    started = time.perf_counter()
    result = run(cones, start, config)
    wall = time.perf_counter() - started

    valid = result.valid_laps
    times = [lap.time_s for lap in valid]
    record: Dict[str, Any] = {
        "track": job["track"],
        "family": job["family"],
        "sensor": job["sensor"],
        "range_m": job["range_m"],
        "fov_deg": job["fov_deg"],
        "variant": job["variant"],
        "status": result.status,
        "laps_target": job["laps"],
        "laps_valid": len(valid),
        "lap_times_s": [round(lap.time_s, 2) for lap in result.laps],
        "lap_cones_hit": [lap.cones_hit for lap in result.laps],
        "lap_valid": [lap.valid for lap in result.laps],
        "best_lap_s": round(min(times), 2) if times else None,
        "mean_lap_s": round(statistics.fmean(times), 2) if times else None,
        "cones_per_lap": (
            round(statistics.fmean(lap.cones_hit for lap in valid), 2) if valid else None
        ),
        "cones_hit_total": result.cones_hit_total,
        "mean_speed_mps": (
            round(result.track_length_m / statistics.fmean(times), 2) if times else None
        ),
        "max_speed_mps": round(result.max_speed_mps, 2),
        "max_lat_accel_mps2": round(result.max_lat_accel_mps2, 2),
        "sim_time_s": round(result.sim_time_s, 1),
        "distance_m": round(result.distance_m, 1),
        "off_course_at": (
            [round(v, 1) for v in result.off_course_at] if result.off_course_at else None
        ),
        "wall_time_s": round(wall, 1),
    }
    return record


def build_jobs(tracks: Sequence[str], laps: int, quick: bool) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    sensors = SENSORS[:1] if quick else SENSORS
    variants = VARIANTS[:1] if quick else VARIANTS
    for key, range_m, fov_deg, _ in sensors:
        for track in tracks:
            jobs.append(
                {
                    "family": "sensor",
                    "track": track,
                    "sensor": key,
                    "range_m": range_m,
                    "fov_deg": fov_deg,
                    "variant": "full",
                    "overrides": {},
                    "laps": laps,
                }
            )
    default_key, default_range, default_fov, _ = SENSORS[0]
    for key, _, overrides in variants[1:]:
        for track in tracks:
            jobs.append(
                {
                    "family": "ablation",
                    "track": track,
                    "sensor": default_key,
                    "range_m": default_range,
                    "fov_deg": default_fov,
                    "variant": key,
                    "overrides": overrides,
                    "laps": laps,
                }
            )
    return jobs


# -----------------------------
# Output
# -----------------------------
def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _outcome(record: Dict[str, Any]) -> str:
    if record["status"] == "finished":
        return "finished"
    if record["status"] == "off_course":
        t, x, y = record["off_course_at"]
        return f"left the track at {t:.0f} s"
    return record["status"]


def _cones(record: Dict[str, Any]) -> str:
    if record["cones_per_lap"] is not None:
        return _fmt(record["cones_per_lap"], 1)
    return f"- ({record['cones_hit_total']} before stopping)"


def render_markdown(data: Dict[str, Any]) -> str:
    """Tables for the README."""
    lines: List[str] = []
    sensor_notes = {key: (rng, fov, note) for key, rng, fov, note in SENSORS}

    lines.append("#### Tracks")
    lines.append("")
    lines.append(
        "| Track | Blue / yellow cones | Centre line (m) | Width min / median / max (m) "
        "| Tightest bend radius (m) |"
    )
    lines.append("|---|---|---|---|---|")
    for t in data["tracks"]:
        lines.append(
            f"| `{t['track']}` | {t['blue_cones']} / {t['yellow_cones']} "
            f"| {t['centerline_length_m']:.1f} "
            f"| {t['gate_width_min_m']:.2f} / {t['gate_width_median_m']:.2f} / "
            f"{t['gate_width_max_m']:.2f} | {_fmt(t['tightest_radius_m'])} |"
        )

    def rows(family: str) -> List[Dict[str, Any]]:
        return [r for r in data["runs"] if r["family"] == family]

    lines.append("")
    lines.append("#### Sensor sweep (full controller)")
    lines.append("")
    lines.append(
        "| Sensor (range / field of view) | Track | Outcome | Valid laps | Best lap (s) "
        "| Mean lap (s) | Mean speed (m/s) | Cones hit per lap | Peak lateral acc. (m/s^2) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows("sensor"):
        rng, fov, note = sensor_notes[r["sensor"]]
        label = f"{rng:g} m / {fov:g} deg" + (f" ({note})" if note else "")
        lines.append(
            f"| {label} | `{r['track']}` | {_outcome(r)} "
            f"| {r['laps_valid']} / {r['laps_target']} | {_fmt(r['best_lap_s'])} "
            f"| {_fmt(r['mean_lap_s'])} | {_fmt(r['mean_speed_mps'], 1)} | {_cones(r)} "
            f"| {r['max_lat_accel_mps2']:.1f} |"
        )

    ablations = rows("ablation")
    if ablations:
        labels = {key: label for key, label, _ in VARIANTS}
        default = [r for r in rows("sensor") if r["sensor"] == SENSORS[0][0]]
        rng, fov, _ = sensor_notes[SENSORS[0][0]]
        lines.append("")
        lines.append(f"#### Ablations (sensor {rng:g} m / {fov:g} deg)")
        lines.append("")
        lines.append(
            "| Variant | Track | Outcome | Valid laps | Best lap (s) | Cones hit per lap |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in default + ablations:
            lines.append(
                f"| {labels[r['variant']]} | `{r['track']}` | {_outcome(r)} "
                f"| {r['laps_valid']} / {r['laps_target']} | {_fmt(r['best_lap_s'])} "
                f"| {_cones(r)} |"
            )

    env = data["environment"]
    lines.append("")
    lines.append(
        f"{len(data['runs'])} runs, {data['simulated_time_s']:.0f} s of simulated driving. "
        f"Computing time: {data['runs_time_s']:.0f} s summed over the single-threaded runs, "
        f"{data['wall_time_s']:.0f} s wall clock with {data['jobs']} processes "
        f"(Python {env['python']}, NumPy {env['numpy']}, {env['system']} {env['machine']}, "
        f"{env['cpu_count']} logical CPUs, no GPU)."
    )
    return "\n".join(lines) + "\n"


def write_outputs(data: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    columns = [
        "family", "sensor", "range_m", "fov_deg", "variant", "track", "status",
        "laps_valid", "laps_target", "best_lap_s", "mean_lap_s", "mean_speed_mps",
        "cones_per_lap", "cones_hit_total", "max_speed_mps", "max_lat_accel_mps2",
        "sim_time_s", "distance_m", "wall_time_s",
    ]
    with open(out_dir / "benchmark.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data["runs"])

    with open(out_dir / "benchmark.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracks", nargs="*", default=None, help="tracks to run (default: all)")
    parser.add_argument("--laps", type=int, default=TARGET_LAPS)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--quick", action="store_true",
        help="one lap on the three short tracks with the default sensor only",
    )
    args = parser.parse_args()

    tracks = args.tracks or (QUICK_TRACKS if args.quick else available_tracks())
    laps = 1 if args.quick else args.laps
    jobs = build_jobs(tracks, laps, args.quick)

    started = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(run_one, jobs))
    else:
        runs = [run_one(job) for job in jobs]
    wall = time.perf_counter() - started

    defaults = asdict(LoopConfig(target_laps=laps, seed=SEED))
    del defaults["vision"]  # set per run
    data = {
        "seed": SEED,
        "default_config": defaults,
        "min_turn_radius_m": round(VehicleParams().min_turn_radius_m, 2),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "jobs": args.jobs,
        "wall_time_s": round(wall, 1),
        "runs_time_s": round(sum(r["wall_time_s"] for r in runs), 1),
        "simulated_time_s": round(sum(r["sim_time_s"] for r in runs), 1),
        "tracks": [describe_track(name) for name in tracks],
        "runs": runs,
    }
    write_outputs(data, args.out_dir)

    for r in runs:
        print(
            f"{r['family']:8s} {r['sensor']:8s} {r['variant']:12s} {r['track']:32s} "
            f"{r['status']:10s} laps {r['laps_valid']}/{r['laps_target']} "
            f"best {_fmt(r['best_lap_s'])} s  cones/lap {_fmt(r['cones_per_lap'], 1)}"
        )
    print(f"wrote {args.out_dir / 'benchmark.md'} ({wall:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
