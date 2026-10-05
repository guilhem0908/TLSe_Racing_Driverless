"""Tests of the track helpers and of the referee (laps, cone hits, off course)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

from closed_loop.centerline import pair_cones
from closed_loop.referee import (
    CONE_RADIUS_M,
    Referee,
    distance_to_segment,
    point_in_polygon,
    segment_crossing,
)
from closed_loop.track import (
    StartPose,
    available_tracks,
    load_start_pose,
    reference_gates,
    split_cones,
    timing_line,
    track_path,
)
from closed_loop.vehicle import VehicleParams, VehicleState
from tests.synthetic import RING_CENTER_M, RING_INNER_M, RING_OUTER_M
from track_utils import load_track

VEHICLE = VehicleParams()
LINE_ANGLE = 0.25  # where make_ring puts the timing line


def circle_states(radius: float, start_angle: float, end_angle: float, step: float = 0.01):
    """States of a car whose body centre runs counter-clockwise on a circle."""
    states: List[VehicleState] = []
    n = round((end_angle - start_angle) / step)
    for i in range(n + 1):
        a = start_angle + (end_angle - start_angle) * i / n
        yaw = a + math.pi / 2.0
        cx, cy = radius * math.cos(a), radius * math.sin(a)
        states.append(
            VehicleState(
                cx - VEHICLE.center_offset_m * math.cos(yaw),
                cy - VEHICLE.center_offset_m * math.sin(yaw),
                yaw,
            )
        )
    return states


def drive(referee: Referee, states, dt: float = 0.05, t0: float = 0.0) -> float:
    t = t0
    for previous, current in zip(states, states[1:]):
        referee.update(previous, current, t, t + dt)
        t += dt
    return t


# -----------------------------
# Geometry helpers
# -----------------------------
def test_segment_crossing():
    assert segment_crossing((0, 0), (2, 0), (0.5, -1), (0.5, 1)) == pytest.approx(0.25)
    assert segment_crossing((0, 0), (2, 0), (3, -1), (3, 1)) is None  # beyond the end
    assert segment_crossing((0, 0), (2, 0), (1, 1), (1, 2)) is None  # misses the gate
    assert segment_crossing((0, 0), (2, 0), (0, 1), (2, 1)) is None  # parallel


def test_point_in_polygon_and_distance_to_segment():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert point_in_polygon((1, 1), square)
    assert not point_in_polygon((3, 1), square)
    assert not point_in_polygon((1, -0.1), square)
    assert distance_to_segment((1, 1), (0, 0), (2, 0)) == pytest.approx(1.0)
    assert distance_to_segment((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)
    assert distance_to_segment((5, 0), (0, 0), (2, 0)) == pytest.approx(3.0)


# -----------------------------
# Track helpers
# -----------------------------
def test_bundled_tracks_are_listed():
    assert available_tracks() == [
        "belgium",
        "hairpins_increasing_difficulty",
        "peanut",
        "small_track",
    ]
    assert track_path("peanut").name == "peanut.csv"
    with pytest.raises(FileNotFoundError):
        track_path("monaco")


def test_start_pose_reads_the_direction_column(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "tag,x,y,direction,x_variance,y_variance,xy_covariance\n"
        "blue,0,1,0,0,0,0\n"
        "car_start,2.5,-1.0,1.25,0,0,0\n",
        encoding="utf-8",
    )
    pose = load_start_pose(path)
    assert (pose.x, pose.y, pose.heading_rad) == (2.5, -1.0, 1.25)
    assert pose.direction == pytest.approx((math.cos(1.25), math.sin(1.25)))

    path.write_text("tag,x,y\nblue,0,1\n", encoding="utf-8")
    assert load_start_pose(path) == StartPose(0.0, 0.0, 0.0)


def test_timing_line_joins_the_big_orange_cones(ring, ring_start):
    left, right = timing_line(ring, ring_start)
    assert math.hypot(*left) == pytest.approx(RING_INNER_M - 0.4)
    assert math.hypot(*right) == pytest.approx(RING_OUTER_M + 0.4)
    assert math.atan2(left[1], left[0]) == pytest.approx(LINE_ANGLE)


def test_timing_line_without_orange_cones(ring, ring_start):
    plain = [c for c in ring if c["tag"] != "big_orange"]
    left, right = timing_line(plain, ring_start, fallback_half_width_m=2.0)
    # Heading +y at (10, 0): the left end is towards -x.
    assert left == pytest.approx((RING_CENTER_M - 2.0, 0.0))
    assert right == pytest.approx((RING_CENTER_M + 2.0, 0.0))


@pytest.mark.parametrize("name", available_tracks())
def test_reference_gates_cover_each_bundled_track(name):
    path = track_path(name)
    cones = load_track(str(path))
    gates = reference_gates(cones, load_start_pose(path))
    blue, yellow = split_cones(cones)

    # The ordering walks through every pair found on the full map ...
    assert len(gates) == len(pair_cones(blue, yellow))
    # ... without ever jumping far, including from the last gate to the first.
    mids = [g.mid for g in gates]
    for a, b in zip(mids, mids[1:] + mids[:1]):
        assert math.dist(a, b) < 6.0
    # Nearly every cone belongs to a gate (one yellow cone of peanut does not).
    unused = (set(blue) - {g.left for g in gates}) | (set(yellow) - {g.right for g in gates})
    assert len(unused) <= 1


# -----------------------------
# Referee
# -----------------------------
def test_lap_time_on_the_ring(ring, ring_start):
    referee = Referee(ring, ring_start, VEHICLE)
    assert not referee.timing

    step, dt = 0.01, 0.05  # 0.2 rad/s
    states = circle_states(RING_CENTER_M, 0.0, 2.0 * (2.0 * math.pi) + 0.5, step)
    drive(referee, states, dt)

    assert referee.timing
    assert not referee.off_course
    assert [lap.number for lap in referee.laps] == [1, 2]
    for lap in referee.laps:
        assert lap.valid
        assert lap.gate_ratio == 1.0
        assert lap.cones_hit == 0
        assert lap.time_s == pytest.approx(2.0 * math.pi / (step / dt), abs=0.02)
    assert referee.valid_laps == 2
    assert referee.cones_hit_total == 0


def test_line_crossed_backwards_does_not_count(ring, ring_start):
    referee = Referee(ring, ring_start, VEHICLE)
    states = circle_states(RING_CENTER_M, 0.0, 0.5)
    drive(referee, states)
    assert referee.timing
    start_time = referee.lap_start_time

    # Same arc driven in reverse order: the line is crossed the wrong way.
    backwards = [VehicleState(s.x, s.y, s.yaw + math.pi) for s in reversed(states)]
    drive(referee, backwards, t0=100.0)
    assert referee.laps == []
    assert referee.lap_start_time == start_time


def test_cone_hits_count_once_per_lap(ring, ring_start):
    # Running 0.9 m inside the centre line puts the left side of the body over
    # the blue cones.
    radius = RING_CENTER_M - 0.9
    assert radius - VEHICLE.body_width_m / 2.0 < RING_INNER_M + CONE_RADIUS_M

    referee = Referee(ring, ring_start, VEHICLE)
    drive(referee, circle_states(radius, 0.0, 2.0 * (2.0 * math.pi) + 0.5))

    n_blue = sum(1 for c in ring if c["tag"] == "blue")
    assert [lap.cones_hit for lap in referee.laps] == [n_blue, n_blue]
    assert all(lap.valid for lap in referee.laps)
    # Total = the cones touched before the first crossing and after the last one too.
    assert referee.cones_hit_total > 2 * n_blue


def test_no_hit_when_the_body_just_clears_the_cones(ring, ring_start):
    # Lowest radius at which the left side of the body stays clear of the blue cones.
    clear = RING_INNER_M + CONE_RADIUS_M + VEHICLE.body_width_m / 2.0 + 0.05
    referee = Referee(ring, ring_start, VEHICLE)
    drive(referee, circle_states(clear, 0.0, 2.0 * math.pi + 0.5))
    assert referee.cones_hit_total == 0


def test_lap_with_missed_gates_is_not_valid(ring, ring_start):
    referee = Referee(ring, ring_start, VEHICLE)
    outside = RING_OUTER_M + 0.4  # beyond the yellow cones, within the off-course margin

    states = circle_states(RING_CENTER_M, 0.0, 1.0)
    states += circle_states(outside, 1.0, 3.0)[1:]
    states += circle_states(RING_CENTER_M, 3.0, 2.0 * math.pi + 0.5)[1:]
    drive(referee, states)

    assert not referee.off_course
    assert len(referee.laps) == 1
    lap = referee.laps[0]
    assert not lap.valid
    assert 0.6 < lap.gate_ratio < 0.8
    assert referee.valid_laps == 0


def test_leaving_the_corridor_is_off_course(ring, ring_start):
    referee = Referee(ring, ring_start, VEHICLE)
    states = circle_states(RING_CENTER_M, 0.0, 1.0)
    drive(referee, states)
    assert not referee.off_course

    # Drift outwards until the centre is 2 m beyond the yellow cones.
    last = states[-1]
    out = []
    for k in range(1, 41):
        r = RING_CENTER_M + k * 0.1
        out.append(circle_states(r, 1.0 + 0.005 * k, 1.0 + 0.005 * k + 0.01)[0])
    drive(referee, [last] + out, t0=10.0)

    assert referee.off_course
    t, x, y = referee.off_course_at
    assert t > 10.0
    # Declared when the centre is about half a car width outside the cones.
    assert math.hypot(x, y) == pytest.approx(
        RING_OUTER_M + VEHICLE.body_width_m / 2.0, abs=0.25
    )


def test_referee_needs_a_real_track(ring_start):
    with pytest.raises(ValueError):
        Referee([{"tag": "blue", "x": 0.0, "y": 0.0}], ring_start, VEHICLE)


def test_track_files_are_where_the_package_expects_them():
    assert track_path("belgium") == Path(__file__).resolve().parents[1] / "tracks" / "belgium.csv"
