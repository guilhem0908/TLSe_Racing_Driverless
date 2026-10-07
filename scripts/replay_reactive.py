"""
scripts/replay_reactive.py

Replay the November 2025 reactive controller (realtime.py) off-screen.

realtime.py is run unmodified with a fixed 60 Hz clock and no window. The
positions it computes are logged by wrapping its sensor test, and compared
with the middle of the track: the script prints when the car is first more
than STRAY_DISTANCE_M from the middle of every gate of the track, and how far
from it the car is after the replay. Nothing is written to disk.

Usage (from the repository root):
    python scripts/replay_reactive.py
    python scripts/replay_reactive.py --tracks belgium --seconds 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

import realtime  # noqa: E402
from closed_loop.track import load_start_pose, reference_gates, track_path  # noqa: E402
from track_utils import compute_world_bounds, load_track  # noqa: E402

FPS = 60
STRAY_DISTANCE_M = 4.0
TRACKS = ["belgium", "hairpins_increasing_difficulty", "peanut", "small_track"]

Point2D = Tuple[float, float]


class FixedClock:
    """Stand-in for pygame.time.Clock: fixed step, posts QUIT after a number of ticks."""

    def __init__(self, ticks: int) -> None:
        self.remaining = ticks

    def tick(self, fps: int) -> float:
        self.remaining -= 1
        if self.remaining < 0:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return 1000.0 / fps


def replay(track: str, seconds: float) -> List[Point2D]:
    """Positions of the car, one per simulation step, for ``seconds`` of simulated time."""
    path = track_path(track)
    cones = load_track(str(path))
    positions: List[Point2D] = []
    sensor_test = realtime.point_in_vision_cone

    def spy(point, car_pos, heading, vision):
        if not positions or positions[-1] != car_pos:
            positions.append(car_pos)
        return sensor_test(point, car_pos, heading, vision)

    clock_class = pygame.time.Clock
    realtime.point_in_vision_cone = spy
    pygame.time.Clock = lambda: FixedClock(int(seconds * FPS))
    try:
        realtime.run_realtime(cones, compute_world_bounds(cones))
    finally:
        realtime.point_in_vision_cone = sensor_test
        pygame.time.Clock = clock_class
    return positions


def distance_to_track_middle(position: Point2D, middles: List[Point2D]) -> float:
    return min(math.hypot(position[0] - m[0], position[1] - m[1]) for m in middles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracks", nargs="+", default=TRACKS)
    parser.add_argument("--seconds", type=float, default=40.0, help="simulated time per track")
    args = parser.parse_args()

    print(f"{'track':34s} {'first > %.0f m from the middle' % STRAY_DISTANCE_M:>32s}"
          f" {'distance at the end':>20s}")
    for track in args.tracks:
        path = track_path(track)
        cones = load_track(str(path))
        middles = [gate.mid for gate in reference_gates(cones, load_start_pose(path))]
        positions = replay(track, args.seconds)
        stray: Optional[float] = None
        for step, position in enumerate(positions):
            if distance_to_track_middle(position, middles) > STRAY_DISTANCE_M:
                stray = step / FPS
                break
        left = f"{stray:.1f} s" if stray is not None else "never"
        end = distance_to_track_middle(positions[-1], middles)
        print(f"{track:34s} {left:>32s} {end:>18.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
