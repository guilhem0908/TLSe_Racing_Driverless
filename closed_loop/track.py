"""
closed_loop/track.py

Track facts that the closed loop needs beyond ``track_utils.load_track``:
the starting pose (with heading), the timing line and the ordered list of
reference gates used by the referee.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from closed_loop.centerline import CenterlineParams, Gate, build_centerline
from track_utils import Cone

Point2D = Tuple[float, float]
Segment = Tuple[Point2D, Point2D]

TRACKS_DIR = Path(__file__).resolve().parents[1] / "tracks"


@dataclass(frozen=True)
class StartPose:
    """Starting position and heading (radians) of the car."""

    x: float
    y: float
    heading_rad: float = 0.0

    @property
    def position(self) -> Point2D:
        """Starting position."""
        return self.x, self.y

    @property
    def direction(self) -> Point2D:
        """Unit vector of the starting heading."""
        return math.cos(self.heading_rad), math.sin(self.heading_rad)


def available_tracks() -> List[str]:
    """Names (without extension) of the bundled track files."""
    return sorted(p.stem for p in TRACKS_DIR.glob("*.csv"))


def track_path(name: str) -> Path:
    """
    Resolve a track name or a path to a CSV file.

    Args:
        name: Bundled track name such as ``"peanut"``, or a path to a CSV.

    Raises:
        FileNotFoundError: If neither exists.
    """
    direct = Path(name)
    if direct.suffix == ".csv" and direct.is_file():
        return direct
    bundled = TRACKS_DIR / f"{name}.csv"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"Unknown track {name!r}. Bundled tracks: {', '.join(available_tracks())}"
    )


def load_start_pose(csv_path: Union[str, Path]) -> StartPose:
    """
    Read the starting pose from the ``car_start`` row of a track file.

    The ``direction`` column, which ``track_utils.load_track`` ignores, is
    read as a heading in radians. Without a ``car_start`` row the pose is the
    origin facing +x, like ``track_utils.get_start_pos``.

    Args:
        csv_path: Path to the track CSV.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tag") == "car_start":
                heading = float(row.get("direction") or 0.0)
                return StartPose(float(row["x"]), float(row["y"]), heading)
    return StartPose(0.0, 0.0, 0.0)


def split_cones(cones: Sequence[Cone]) -> Tuple[List[Point2D], List[Point2D]]:
    """Blue (left) and yellow (right) cone positions of a track."""
    blue = [(c["x"], c["y"]) for c in cones if c["tag"] == "blue"]
    yellow = [(c["x"], c["y"]) for c in cones if c["tag"] == "yellow"]
    return blue, yellow


def timing_line(
    cones: Sequence[Cone], start: StartPose, fallback_half_width_m: float = 3.0
) -> Segment:
    """
    Start/finish line as a (left end, right end) segment.

    If big orange cones stand on both sides of the starting direction, the
    line joins the centroid of the left ones to the centroid of the right
    ones. Otherwise it is a segment through the starting position,
    perpendicular to the starting heading.
    """
    hx, hy = start.direction
    left: List[Point2D] = []
    right: List[Point2D] = []
    for c in cones:
        if c["tag"] != "big_orange":
            continue
        side = hx * (c["y"] - start.y) - hy * (c["x"] - start.x)
        (left if side > 0.0 else right).append((c["x"], c["y"]))

    if left and right:
        return _centroid(left), _centroid(right)

    w = fallback_half_width_m
    return (start.x - hy * w, start.y + hx * w), (start.x + hy * w, start.y - hx * w)


def reference_gates(
    cones: Sequence[Cone],
    start: StartPose,
    params: CenterlineParams = CenterlineParams(),
) -> List[Gate]:
    """
    Gates of the whole track, ordered in the driving direction.

    This uses the full map and is meant for the referee, never for the car.
    """
    blue, yellow = split_cones(cones)
    whole_track = replace(params, max_length_m=math.inf)
    line = build_centerline(
        blue, yellow, start.position, start.direction, whole_track, allow_virtual=False
    )
    return line.gates


def _centroid(points: Sequence[Point2D]) -> Point2D:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )
