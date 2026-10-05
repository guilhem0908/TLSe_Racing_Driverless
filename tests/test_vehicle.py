"""Tests of the kinematic bicycle model."""

from __future__ import annotations

import math

import pytest

from closed_loop.vehicle import (
    VehicleParams,
    VehicleState,
    body_center,
    distance_to_footprint,
    footprint_corners,
    path_curvature,
    step,
    wrap_angle,
)

PARAMS = VehicleParams()
DT = 0.01


def test_wrap_angle():
    assert wrap_angle(0.0) == 0.0
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi / 2.0 - 2.0 * math.pi) == pytest.approx(-math.pi / 2.0)
    assert -math.pi < wrap_angle(123.456) <= math.pi


def test_straight_line_at_constant_speed():
    state = VehicleState(1.0, 2.0, math.radians(30.0), speed=4.0)
    for _ in range(100):
        state = step(state, 0.0, 0.0, DT, PARAMS)
    assert state.x == pytest.approx(1.0 + 4.0 * math.cos(math.radians(30.0)))
    assert state.y == pytest.approx(2.0 + 4.0 * math.sin(math.radians(30.0)))
    assert state.yaw == pytest.approx(math.radians(30.0))
    assert state.speed == pytest.approx(4.0)


def test_constant_steer_follows_the_bicycle_circle():
    steer = math.radians(15.0)
    radius = PARAMS.wheelbase_m / math.tan(steer)
    speed = 2.0  # lateral acceleration v^2 / R stays far below the cap
    state = VehicleState(0.0, 0.0, 0.0, speed=speed, steer=steer)

    steps = round(2.0 * math.pi * radius / speed / DT)
    centre = (0.0, radius)  # turning left from the origin, heading +x
    for _ in range(steps):
        state = step(state, steer, 0.0, DT, PARAMS)
        assert math.hypot(state.x - centre[0], state.y - centre[1]) == pytest.approx(
            radius, abs=1e-3
        )
    # Back at the start after one revolution (the step count is rounded).
    assert math.hypot(state.x, state.y) < speed * DT


def test_minimum_turn_radius():
    assert PARAMS.min_turn_radius_m == pytest.approx(1.53 / math.tan(math.radians(30.0)))


def test_steering_is_rate_and_amplitude_limited():
    state = VehicleState(0.0, 0.0, 0.0, speed=1.0)
    state = step(state, math.radians(90.0), 0.0, DT, PARAMS)
    assert state.steer == pytest.approx(math.radians(PARAMS.max_steer_rate_deg_s) * DT)
    for _ in range(200):
        state = step(state, math.radians(90.0), 0.0, DT, PARAMS)
    assert state.steer == pytest.approx(PARAMS.max_steer_rad)


def test_acceleration_and_speed_limits():
    state = VehicleState(0.0, 0.0, 0.0)
    state = step(state, 0.0, 100.0, 0.1, PARAMS)
    assert state.speed == pytest.approx(PARAMS.max_accel_mps2 * 0.1)
    for _ in range(200):
        state = step(state, 0.0, 100.0, 0.1, PARAMS)
    assert state.speed == pytest.approx(PARAMS.max_speed_mps)

    state = step(state, 0.0, -100.0, 0.1, PARAMS)
    assert state.speed == pytest.approx(PARAMS.max_speed_mps - PARAMS.max_brake_mps2 * 0.1)
    for _ in range(200):
        state = step(state, 0.0, -100.0, 0.1, PARAMS)
    assert state.speed == 0.0  # never reverses


def test_lateral_acceleration_is_capped():
    steer = PARAMS.max_steer_rad
    slow = path_curvature(steer, 1.0, PARAMS)
    assert slow == pytest.approx(math.tan(steer) / PARAMS.wheelbase_m)

    fast = path_curvature(steer, 10.0, PARAMS)
    assert fast * 10.0**2 == pytest.approx(PARAMS.max_lat_accel_mps2)
    assert path_curvature(-steer, 10.0, PARAMS) == pytest.approx(-fast)

    # At speed the car turns on a wider circle than the steering angle asks.
    state = VehicleState(0.0, 0.0, 0.0, speed=10.0, steer=steer)
    after = step(state, steer, 0.0, DT, PARAMS)
    yaw_rate = (after.yaw - state.yaw) / DT
    assert 10.0 * yaw_rate == pytest.approx(PARAMS.max_lat_accel_mps2)


def test_footprint_geometry():
    state = VehicleState(5.0, -1.0, 0.0)
    corners = footprint_corners(state, PARAMS)
    rear = 5.0 - PARAMS.rear_overhang_m
    front = rear + PARAMS.body_length_m
    half = PARAMS.body_width_m / 2.0
    assert corners == pytest.approx(
        [(rear, -1.0 + half), (front, -1.0 + half), (front, -1.0 - half), (rear, -1.0 - half)]
    )
    assert body_center(state, PARAMS) == pytest.approx(((rear + front) / 2.0, -1.0))


def test_footprint_rotates_with_the_heading():
    state = VehicleState(0.0, 0.0, math.pi / 2.0)
    front = PARAMS.body_length_m - PARAMS.rear_overhang_m
    assert body_center(state, PARAMS) == pytest.approx((0.0, PARAMS.center_offset_m))
    assert distance_to_footprint((0.0, front + 0.5), state, PARAMS) == pytest.approx(0.5)
    assert distance_to_footprint((PARAMS.body_width_m / 2.0 + 0.2, 1.0), state, PARAMS) == (
        pytest.approx(0.2)
    )


def test_distance_to_footprint():
    state = VehicleState(0.0, 0.0, 0.0)
    front = PARAMS.body_length_m - PARAMS.rear_overhang_m
    half = PARAMS.body_width_m / 2.0
    assert distance_to_footprint((1.0, 0.0), state, PARAMS) == 0.0  # inside
    assert distance_to_footprint((front + 1.0, 0.0), state, PARAMS) == pytest.approx(1.0)
    assert distance_to_footprint((-PARAMS.rear_overhang_m - 0.3, 0.0), state, PARAMS) == (
        pytest.approx(0.3)
    )
    assert distance_to_footprint((front + 3.0, half + 4.0), state, PARAMS) == pytest.approx(5.0)
