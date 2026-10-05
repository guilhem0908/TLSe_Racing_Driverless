"""Tests of pure-pursuit steering and of the speed target."""

from __future__ import annotations

import math

import pytest

from closed_loop.pure_pursuit import (
    PursuitParams,
    lookahead_distance,
    lookahead_point,
    menger_curvature,
    resample,
    speed_target,
    steering_angle,
)

PARAMS = PursuitParams()
WHEELBASE = 1.53


def arc(radius: float, sweep: float, n: int = 60):
    """Left-turning arc starting at the origin, heading +x."""
    return [
        (radius * math.sin(sweep * i / n), radius * (1.0 - math.cos(sweep * i / n)))
        for i in range(n + 1)
    ]


def test_lookahead_distance_grows_with_speed_within_limits():
    assert lookahead_distance(0.0, PARAMS) == PARAMS.lookahead_min_m
    mid = lookahead_distance(5.0, PARAMS)
    assert mid == pytest.approx(PARAMS.lookahead_base_m + PARAMS.lookahead_gain_s * 5.0)
    assert lookahead_distance(100.0, PARAMS) == PARAMS.lookahead_max_m


def test_lookahead_point_on_a_straight_path():
    path = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)]
    assert lookahead_point(path, (1.0, 0.0), (0.0, 0.0), 3.0) == pytest.approx((3.0, 0.0))


def test_lookahead_point_is_at_the_lookahead_distance_on_a_bend():
    path = [(0.0, 0.0), (2.0, 0.0), (4.0, 1.0), (5.0, 3.0), (5.0, 6.0)]
    origin = (0.3, -0.2)
    for lookahead in (1.0, 2.5, 4.0, 5.5):
        target = lookahead_point(path, (0.0, 1.0), origin, lookahead)
        assert math.dist(target, origin) == pytest.approx(lookahead)


def test_lookahead_point_extends_a_short_path():
    path = [(0.0, 0.0), (1.0, 0.0)]
    target = lookahead_point(path, (0.0, 1.0), (0.0, 0.0), 2.0)
    # The path is continued upwards from (1, 0) until 2 m from the origin.
    assert target == pytest.approx((1.0, math.sqrt(3.0)))


def test_lookahead_point_with_only_the_car():
    target = lookahead_point([(2.0, 1.0)], (0.0, -1.0), (2.0, 1.0), 3.0)
    assert target == pytest.approx((2.0, -2.0))


def test_steering_is_zero_straight_ahead_and_signed():
    assert steering_angle((0.0, 0.0), 0.0, (5.0, 0.0), WHEELBASE) == pytest.approx(0.0)
    assert steering_angle((0.0, 0.0), 0.0, (5.0, 1.0), WHEELBASE) > 0.0  # target on the left
    assert steering_angle((0.0, 0.0), 0.0, (5.0, -1.0), WHEELBASE) < 0.0
    assert steering_angle((1.0, 1.0), 0.3, (1.0, 1.0), WHEELBASE) == 0.0  # degenerate


@pytest.mark.parametrize("radius", [3.0, 8.0, 25.0])
def test_steering_matches_the_circle_through_the_target(radius):
    # A target on a circle tangent to the heading needs atan(L / R).
    for sweep in (0.2, 0.6, 1.0):
        target = (radius * math.sin(sweep), radius * (1.0 - math.cos(sweep)))
        steer = steering_angle((0.0, 0.0), 0.0, target, WHEELBASE)
        assert steer == pytest.approx(math.atan(WHEELBASE / radius))


def test_resample_spacing():
    samples = resample([(0.0, 0.0), (1.2, 0.0), (1.2, 2.0)], 0.5)
    for a, b in zip(samples[:2], samples[1:3]):
        assert math.dist(a, b) == pytest.approx(0.5)
    assert samples[0] == (0.0, 0.0)
    assert samples[-1] == pytest.approx((1.2, 1.8))
    assert len(samples) == 7
    assert resample([], 0.5) == []
    assert resample([(1.0, 1.0)], 0.5) == [(1.0, 1.0)]


def test_menger_curvature():
    radius = 4.0
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in (0.1, 0.5, 1.3)]
    assert menger_curvature(*points) == pytest.approx(1.0 / radius)
    assert menger_curvature((0.0, 0.0), (1.0, 1.0), (3.0, 3.0)) == 0.0
    assert menger_curvature((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == 0.0


def test_speed_without_a_line_is_the_search_speed():
    assert speed_target([], 0.0, 0.0, 10.0, PARAMS) == PARAMS.search_speed_mps


def test_speed_on_a_long_straight_is_the_cap():
    line = [(float(x), 0.0) for x in range(0, 60)]
    assert speed_target(line, 0.5, 0.0, 10.0, PARAMS) == 10.0


def test_speed_is_limited_by_the_end_of_the_known_line():
    line = [(1.0, 0.0), (3.0, 0.0)]
    to_end = 1.0 + 2.0
    expected = math.sqrt(PARAMS.end_speed_mps**2 + 2.0 * PARAMS.plan_brake_mps2 * to_end)
    assert speed_target(line, 1.0, 0.0, 10.0, PARAMS) == pytest.approx(expected)
    # A longer known line allows a higher speed.
    longer = [(1.0, 0.0), (6.0, 0.0)]
    assert speed_target(longer, 1.0, 0.0, 10.0, PARAMS) > expected


def test_speed_in_a_bend_respects_the_lateral_acceleration():
    radius = 6.0
    line = arc(radius, sweep=math.pi * 1.5, n=200)
    expected = math.sqrt(PARAMS.max_lat_accel_mps2 * radius)
    assert speed_target(line, 0.0, 0.0, 10.0, PARAMS) == pytest.approx(expected, rel=0.02)


def test_speed_anticipates_a_bend_further_ahead():
    radius = 3.0
    straight = [(float(x), 0.0) for x in range(0, 13)]
    bend = [(12.0 + x, y) for x, y in arc(radius, sweep=math.pi)[1:]]
    bend_speed = math.sqrt(PARAMS.max_lat_accel_mps2 * radius)

    cap = 30.0
    far = speed_target(straight + bend, 0.0, 0.0, cap, PARAMS)
    near = speed_target(straight[8:] + bend, 0.0, 0.0, cap, PARAMS)
    assert bend_speed < near < far < cap
    # Braking at the planned rate from `far` reaches the bend speed in time.
    reachable = math.sqrt(max(far**2 - 2.0 * PARAMS.plan_brake_mps2 * 12.0, 0.0))
    assert reachable <= bend_speed * 1.05


def test_speed_is_limited_by_the_current_steering_arc():
    line = [(float(x), 0.0) for x in range(0, 60)]
    curvature = 1.0 / 2.0
    expected = math.sqrt(PARAMS.max_lat_accel_mps2 / curvature)
    assert speed_target(line, 0.0, curvature, 10.0, PARAMS) == pytest.approx(expected)
    assert speed_target(line, 0.0, -curvature, 10.0, PARAMS) == pytest.approx(expected)
