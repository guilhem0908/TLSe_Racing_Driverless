"""
scripts/show_track.py

Open the track viewer of ``simulation/main_simulation.py`` on a bundled track.

The viewer animates the car along a given path and draws its field-of-view
sector. The path used here is the reference centre line of the whole track,
computed from the full map: the car in this viewer does not drive, it is moved
along the line. Use ``python -m closed_loop`` to watch it drive from what the
sensor sees.

Usage (from the repository root):
    python scripts/show_track.py --track peanut --range 7 --fov 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from closed_loop.pure_pursuit import resample  # noqa: E402
from closed_loop.track import (  # noqa: E402
    available_tracks,
    load_start_pose,
    reference_gates,
    track_path,
)
from simulation.main_simulation import process_pygame  # noqa: E402
from simulation.vision import DEFAULT_VISION, VisionConfig  # noqa: E402
from track_utils import compute_world_bounds, load_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--track", default="belgium", help=", ".join(available_tracks()))
    parser.add_argument("--range", type=float, default=DEFAULT_VISION.range_m, dest="range_m")
    parser.add_argument("--fov", type=float, default=DEFAULT_VISION.fov_deg, dest="fov_deg")
    parser.add_argument(
        "--step", type=float, default=0.08,
        help="distance in metres the car advances per displayed frame (default: %(default)s)",
    )
    args = parser.parse_args()

    path = track_path(args.track)
    cones = load_track(str(path))
    gates = reference_gates(cones, load_start_pose(path))
    mids = [gate.mid for gate in gates]
    line = resample(mids + mids[:1], args.step)

    process_pygame(
        cones,
        compute_world_bounds(cones),
        line,
        vision=VisionConfig(range_m=args.range_m, fov_deg=args.fov_deg),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
