"""
closed_loop/referee.py

Lap timer, cone-hit counter and off-course check.

The referee knows the whole map; the car never reads anything from it.

Definitions used here:
- A *lap* runs from one forward crossing of the timing line to the next. The
  first crossing, a few metres after the standing start, starts lap 1.
- A lap is *valid* if the centre of the car went through at least
  ``min_gate_ratio`` of the reference gates of the track during that lap.
- A cone is *hit* when its base circle touches the rectangular footprint of
  the car. A cone counts once per lap and stays where it is.
- The car is *off course* when its centre is outside the corridor between the
  two rows of cones by more than half a car width, i.e. roughly when the whole
  car has left the track. The run stops there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from closed_loop.centerline import Gate
from closed_loop.track import StartPose, reference_gates, timing_line
from closed_loop.vehicle import (
    VehicleParams,
    VehicleState,
    body_center,
    distance_to_footprint,
)
from track_utils import Cone

Point2D = Tuple[float, float]

# Half of a 228 mm wide cone base.
CONE_RADIUS_M: float = 0.114


@dataclass(frozen=True)
class LapRecord:
    """
    Outcome of one lap.

    Attributes:
        number: 1-based lap number.
        time_s: Lap time.
        cones_hit: Distinct cones touched during the lap.
        gate_ratio: Share of the reference gates the car went through.
        valid: Whether ``gate_ratio`` reached the required minimum.
    """

    number: int
    time_s: float
    cones_hit: int
    gate_ratio: float
    valid: bool


def segment_crossing(p0: Point2D, p1: Point2D, q0: Point2D, q1: Point2D) -> Optional[float]:
    """
    Where the segment p0-p1 crosses the segment q0-q1.

    Returns:
        The fraction along p0-p1 at which the two segments intersect, or None
        if they do not (parallel segments never count).
    """
    rx, ry = p1[0] - p0[0], p1[1] - p0[1]
    sx, sy = q1[0] - q0[0], q1[1] - q0[1]
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None
    qpx, qpy = q0[0] - p0[0], q0[1] - p0[1]
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Even-odd test of a point against a simple polygon."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def distance_to_segment(point: Point2D, a: Point2D, b: Point2D) -> float:
    """Distance from a point to the segment a-b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    length2 = abx * abx + aby * aby
    if length2 < 1e-12:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(point[0] - (a[0] + t * abx), point[1] - (a[1] + t * aby))


class Referee:
    """
    Scores a run from the true vehicle states.

    Args:
        cones: Full track map.
        start: Starting pose of the car.
        vehicle: Vehicle geometry (footprint).
        cone_radius_m: Radius of a cone base.
        min_gate_ratio: Share of the reference gates a valid lap must cross.
        off_course_margin_m: Distance outside the corridor at which the car is
            declared off course. Defaults to half the body width.
    """

    _LOOK_BACK = 4
    _LOOK_AHEAD = 10

    def __init__(
        self,
        cones: Sequence[Cone],
        start: StartPose,
        vehicle: VehicleParams,
        cone_radius_m: float = CONE_RADIUS_M,
        min_gate_ratio: float = 0.95,
        off_course_margin_m: Optional[float] = None,
    ) -> None:
        self.vehicle = vehicle
        self.cone_radius_m = cone_radius_m
        self.min_gate_ratio = min_gate_ratio
        self.off_course_margin_m = (
            vehicle.body_width_m / 2.0 if off_course_margin_m is None else off_course_margin_m
        )

        self.gates: List[Gate] = reference_gates(cones, start)
        if len(self.gates) < 3:
            raise ValueError("Track has too few blue/yellow pairs to be refereed")
        self.line = timing_line(cones, start)
        line_gate = Gate(self.line[0], self.line[1])
        self._line_forward = line_gate.forward

        self._cones: List[Point2D] = [
            (c["x"], c["y"]) for c in cones if c["tag"] in ("blue", "yellow", "big_orange")
        ]
        self._cell = 4.0
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        for index, (x, y) in enumerate(self._cones):
            self._grid.setdefault(self._key(x, y), []).append(index)

        self._next_gate = 0
        self._crossed: Set[int] = set()
        self._lap_start: Optional[float] = None
        self._lap_cones: Set[int] = set()

        self.laps: List[LapRecord] = []
        self.cones_hit_total = 0
        self.hit_positions: List[Point2D] = []
        self.off_course = False
        self.off_course_at: Optional[Tuple[float, float, float]] = None

    # -----------------------------
    # Public API
    # -----------------------------
    @property
    def valid_laps(self) -> int:
        """Number of valid laps completed so far."""
        return sum(1 for lap in self.laps if lap.valid)

    @property
    def timing(self) -> bool:
        """True once the timing line has been crossed for the first time."""
        return self._lap_start is not None

    @property
    def lap_start_time(self) -> Optional[float]:
        """Time at which the current lap started, if any."""
        return self._lap_start

    @property
    def current_lap_cones(self) -> int:
        """Cones touched since the current lap (or the run) started."""
        return len(self._lap_cones)

    def update(
        self, previous: VehicleState, current: VehicleState, t_previous: float, t_current: float
    ) -> None:
        """
        Account for one simulation step.

        Args:
            previous: State at the start of the step.
            current: State at the end of the step.
            t_previous: Time at the start of the step.
            t_current: Time at the end of the step.
        """
        p0 = body_center(previous, self.vehicle)
        p1 = body_center(current, self.vehicle)

        self._count_cone_hits(current)
        self._track_gates(p0, p1)

        crossing = segment_crossing(p0, p1, self.line[0], self.line[1])
        if crossing is not None:
            moving_forward = (
                (p1[0] - p0[0]) * self._line_forward[0]
                + (p1[1] - p0[1]) * self._line_forward[1]
            ) > 0.0
            if moving_forward:
                self._on_line(t_previous + crossing * (t_current - t_previous))

        if not self.off_course and self._is_off_course(p1):
            self.off_course = True
            self.off_course_at = (t_current, p1[0], p1[1])

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return math.floor(x / self._cell), math.floor(y / self._cell)

    def _count_cone_hits(self, state: VehicleState) -> None:
        cx, cy = body_center(state, self.vehicle)
        kx, ky = self._key(cx, cy)
        for gx in (kx - 1, kx, kx + 1):
            for gy in (ky - 1, ky, ky + 1):
                for index in self._grid.get((gx, gy), ()):
                    if index in self._lap_cones:
                        continue
                    gap = distance_to_footprint(self._cones[index], state, self.vehicle)
                    if gap <= self.cone_radius_m:
                        self._lap_cones.add(index)
                        self.cones_hit_total += 1
                        self.hit_positions.append(self._cones[index])

    def _track_gates(self, p0: Point2D, p1: Point2D) -> None:
        """
        Follow the progress of the car through the reference gates.

        A gate counts as crossed when the car passes between its two cones.
        Progress also advances when the car passes just outside one of them
        (within the off-course margin), so that a lap with a missed gate is
        scored as such instead of freezing the progress.
        """
        n = len(self.gates)
        first = self._next_gate
        margin = self.off_course_margin_m
        for offset in range(self._LOOK_AHEAD):
            index = (first + offset) % n
            gate = self.gates[index]
            if segment_crossing(p0, p1, gate.left, gate.right) is not None:
                self._crossed.add(index)
                self._next_gate = (index + 1) % n
                continue
            width = gate.width
            if width < 1e-9:
                continue
            ux = (gate.left[0] - gate.right[0]) / width * margin
            uy = (gate.left[1] - gate.right[1]) / width * margin
            wide_left = (gate.left[0] + ux, gate.left[1] + uy)
            wide_right = (gate.right[0] - ux, gate.right[1] - uy)
            if segment_crossing(p0, p1, wide_left, wide_right) is not None:
                self._next_gate = (index + 1) % n

    def _on_line(self, t_cross: float) -> None:
        if self._lap_start is not None:
            ratio = len(self._crossed) / len(self.gates)
            self.laps.append(
                LapRecord(
                    number=len(self.laps) + 1,
                    time_s=t_cross - self._lap_start,
                    cones_hit=len(self._lap_cones),
                    gate_ratio=ratio,
                    valid=ratio >= self.min_gate_ratio,
                )
            )
        self._lap_start = t_cross
        self._crossed = set()
        self._lap_cones = set()

    def _is_off_course(self, point: Point2D) -> bool:
        n = len(self.gates)
        nearest_edge = math.inf
        for offset in range(-self._LOOK_BACK, self._LOOK_AHEAD):
            a = self.gates[(self._next_gate + offset - 1) % n]
            b = self.gates[(self._next_gate + offset) % n]
            cell = (a.left, b.left, b.right, a.right)
            if point_in_polygon(point, cell):
                return False
            for i in range(4):
                nearest_edge = min(
                    nearest_edge, distance_to_segment(point, cell[i], cell[(i + 1) % 4])
                )
        return nearest_edge > self.off_course_margin_m
