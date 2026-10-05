"""
closed_loop/vehicle.py

Kinematic bicycle model of the simulated car.

The reference point of the state is the centre of the rear axle. The model has
no tyre model, so it is only a reasonable stand-in for a real car at low
lateral acceleration. Steering angle, steering rate, acceleration and speed
are saturated. The lateral acceleration is capped as well: above the cap the
car turns less than the steering angle asks for and runs wide, which is the
only way this model can "lose grip".

All default dimensions are assumptions chosen to be of the order of a Formula
Student car. They are not measurements of an actual vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class VehicleParams:
    """
    Geometry and actuator limits of the simulated car.

    Attributes:
        wheelbase_m: Distance between the axles.
        body_length_m: Overall length of the rectangular footprint.
        body_width_m: Overall width of the rectangular footprint.
        rear_overhang_m: Distance from the rear axle to the rear of the body.
        max_steer_deg: Largest front-wheel steering angle (each side).
        max_steer_rate_deg_s: Largest steering speed.
        max_accel_mps2: Largest forward acceleration.
        max_brake_mps2: Largest deceleration (positive number).
        max_speed_mps: Speed cap.
        max_lat_accel_mps2: Largest lateral acceleration the car can hold.
    """

    wheelbase_m: float = 1.53
    body_length_m: float = 2.8
    body_width_m: float = 1.4
    rear_overhang_m: float = 0.5
    max_steer_deg: float = 30.0
    max_steer_rate_deg_s: float = 180.0
    max_accel_mps2: float = 3.0
    max_brake_mps2: float = 6.0
    max_speed_mps: float = 10.0
    max_lat_accel_mps2: float = 8.0

    @property
    def max_steer_rad(self) -> float:
        """Steering limit in radians."""
        return math.radians(self.max_steer_deg)

    @property
    def min_turn_radius_m(self) -> float:
        """Smallest radius the rear axle can follow at full steering lock."""
        return self.wheelbase_m / math.tan(self.max_steer_rad)

    @property
    def center_offset_m(self) -> float:
        """Distance from the rear axle forward to the centre of the body."""
        return self.body_length_m / 2.0 - self.rear_overhang_m


@dataclass(frozen=True)
class VehicleState:
    """
    State of the bicycle model.

    Attributes:
        x, y: Position of the rear-axle centre (world units, metres).
        yaw: Heading in radians, counter-clockwise from the +x axis.
        speed: Forward speed in m/s.
        steer: Current front-wheel steering angle in radians (left positive).
    """

    x: float
    y: float
    yaw: float
    speed: float = 0.0
    steer: float = 0.0

    @property
    def rear_axle(self) -> Point2D:
        """Rear-axle centre."""
        return self.x, self.y

    @property
    def heading(self) -> Point2D:
        """Unit vector along the heading."""
        return math.cos(self.yaw), math.sin(self.yaw)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def step(
    state: VehicleState,
    steer_cmd: float,
    accel_cmd: float,
    dt: float,
    params: VehicleParams,
) -> VehicleState:
    """
    Advance the bicycle model by one time step.

    The steering command is first limited in rate and in amplitude, the
    acceleration command in amplitude, and the resulting curvature by the
    lateral acceleration cap. Position is integrated with the heading at the
    middle of the step, which keeps a constant-steer run on its circle.

    Args:
        state: Current state.
        steer_cmd: Desired steering angle (rad).
        accel_cmd: Desired longitudinal acceleration (m/s^2, negative brakes).
        dt: Time step (s).
        params: Vehicle parameters.

    Returns:
        The state after ``dt`` seconds.
    """
    max_delta = math.radians(params.max_steer_rate_deg_s) * dt
    steer = state.steer + _clamp(steer_cmd - state.steer, -max_delta, max_delta)
    steer = _clamp(steer, -params.max_steer_rad, params.max_steer_rad)

    accel = _clamp(accel_cmd, -params.max_brake_mps2, params.max_accel_mps2)
    speed = _clamp(state.speed + accel * dt, 0.0, params.max_speed_mps)
    mean_speed = 0.5 * (state.speed + speed)

    curvature = path_curvature(steer, mean_speed, params)
    yaw_rate = mean_speed * curvature
    mid_yaw = state.yaw + 0.5 * yaw_rate * dt

    return VehicleState(
        x=state.x + mean_speed * math.cos(mid_yaw) * dt,
        y=state.y + mean_speed * math.sin(mid_yaw) * dt,
        yaw=wrap_angle(state.yaw + yaw_rate * dt),
        speed=speed,
        steer=steer,
    )


def path_curvature(steer: float, speed: float, params: VehicleParams) -> float:
    """
    Curvature the rear axle follows for a steering angle at a given speed.

    This is ``tan(steer) / wheelbase``, reduced if needed so that
    ``speed**2 * curvature`` stays within the lateral acceleration cap.
    """
    curvature = math.tan(steer) / params.wheelbase_m
    if speed > 1e-6:
        limit = params.max_lat_accel_mps2 / (speed * speed)
        curvature = _clamp(curvature, -limit, limit)
    return curvature


def body_center(state: VehicleState, params: VehicleParams) -> Point2D:
    """Centre of the rectangular footprint (also where the sensor sits)."""
    hx, hy = state.heading
    return (
        state.x + params.center_offset_m * hx,
        state.y + params.center_offset_m * hy,
    )


def footprint_corners(state: VehicleState, params: VehicleParams) -> List[Point2D]:
    """
    Corners of the rectangular footprint in world coordinates.

    Returns:
        [rear-left, front-left, front-right, rear-right].
    """
    hx, hy = state.heading
    lx, ly = -hy, hx  # unit vector to the left of the heading
    rear = -params.rear_overhang_m
    front = params.body_length_m - params.rear_overhang_m
    half_w = params.body_width_m / 2.0

    corners: List[Point2D] = []
    for along, side in ((rear, half_w), (front, half_w), (front, -half_w), (rear, -half_w)):
        corners.append(
            (state.x + along * hx + side * lx, state.y + along * hy + side * ly)
        )
    return corners


def distance_to_footprint(
    point: Point2D, state: VehicleState, params: VehicleParams
) -> float:
    """
    Distance from a world point to the rectangular footprint.

    Returns:
        0.0 if the point is inside the footprint, otherwise the distance to
        its nearest edge.
    """
    hx, hy = state.heading
    dx = point[0] - state.x
    dy = point[1] - state.y
    along = dx * hx + dy * hy
    side = -dx * hy + dy * hx

    rear = -params.rear_overhang_m
    front = params.body_length_m - params.rear_overhang_m
    half_w = params.body_width_m / 2.0

    out_along = max(rear - along, 0.0, along - front)
    out_side = max(abs(side) - half_w, 0.0)
    return math.hypot(out_along, out_side)
