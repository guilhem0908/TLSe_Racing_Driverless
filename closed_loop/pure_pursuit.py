"""
closed_loop/pure_pursuit.py

Pure-pursuit steering and a speed target for the local centre line.

Steering: the car aims at the point of the centre line that lies one lookahead
distance away from its rear axle and takes the circular arc through that point.

Speed: the car may not exceed the speed at which a bend of the known line
would need more than the allowed lateral acceleration, and it must be able to
slow down to a low "end speed" where its knowledge of the track stops. The
second rule ties the speed to how far the sensor and the memory let it see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class PursuitParams:
    """
    Tuning of the path follower.

    Attributes:
        lookahead_base_m: Lookahead distance at standstill.
        lookahead_gain_s: Lookahead added per m/s of speed.
        lookahead_min_m: Shortest lookahead distance.
        lookahead_max_m: Longest lookahead distance.
        max_lat_accel_mps2: Lateral acceleration allowed when choosing speed.
        plan_brake_mps2: Deceleration assumed when planning to slow down.
        end_speed_mps: Speed allowed where the known centre line ends.
        search_speed_mps: Speed used when there is no centre line at all.
        speed_gain: Proportional gain of the speed loop (1/s).
        resample_step_m: Spacing used to estimate the curvature of the line.
        curvature_arm_pts: Number of resampled points on each side of the
            point where the curvature is estimated.
    """

    lookahead_base_m: float = 1.0
    lookahead_gain_s: float = 0.4
    lookahead_min_m: float = 2.0
    lookahead_max_m: float = 5.0
    max_lat_accel_mps2: float = 4.0
    plan_brake_mps2: float = 4.0
    end_speed_mps: float = 1.5
    search_speed_mps: float = 1.5
    speed_gain: float = 8.0
    resample_step_m: float = 0.5
    curvature_arm_pts: int = 3


def lookahead_distance(speed: float, params: PursuitParams) -> float:
    """Speed-dependent lookahead distance, clamped to its limits."""
    distance = params.lookahead_base_m + params.lookahead_gain_s * speed
    return max(params.lookahead_min_m, min(params.lookahead_max_m, distance))


def _circle_exit(a: Point2D, b: Point2D, center: Point2D, radius: float) -> float:
    """
    Parameter t >= 0 at which the ray a + t (b - a) leaves the circle.

    Returns the largest root of |a + t (b - a) - center| = radius, or -1 if
    the line through a and b never reaches the circle.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - center[0], a[1] - center[1]
    qa = dx * dx + dy * dy
    if qa < 1e-12:
        return -1.0
    qb = 2.0 * (fx * dx + fy * dy)
    qc = fx * fx + fy * fy - radius * radius
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return -1.0
    return (-qb + math.sqrt(disc)) / (2.0 * qa)


def lookahead_point(
    points: Sequence[Point2D],
    end_direction: Point2D,
    origin: Point2D,
    lookahead: float,
) -> Point2D:
    """
    Point of the path one lookahead distance away from ``origin``.

    The path is walked from its first point; the result is where it first
    leaves the circle of radius ``lookahead`` around ``origin``. If the path
    ends inside the circle it is extended in ``end_direction``.

    Args:
        points: Path polyline (at least one point).
        end_direction: Unit vector used to extend the path past its end.
        origin: Centre of the lookahead circle (the rear axle).
        lookahead: Lookahead distance.

    Returns:
        The target point.
    """
    r2 = lookahead * lookahead
    for a, b in zip(points, points[1:]):
        if (b[0] - origin[0]) ** 2 + (b[1] - origin[1]) ** 2 >= r2:
            t = _circle_exit(a, b, origin, lookahead)
            if 0.0 <= t <= 1.0:
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            return b

    last = points[-1]
    ahead = (last[0] + end_direction[0], last[1] + end_direction[1])
    t = _circle_exit(last, ahead, origin, lookahead)
    if t < 0.0:
        return last
    return last[0] + t * end_direction[0], last[1] + t * end_direction[1]


def steering_angle(
    rear_axle: Point2D, yaw: float, target: Point2D, wheelbase_m: float
) -> float:
    """
    Pure-pursuit steering angle.

    Args:
        rear_axle: Rear-axle position.
        yaw: Heading in radians.
        target: Lookahead point.
        wheelbase_m: Wheelbase of the car.

    Returns:
        Front-wheel angle in radians (left positive) of the arc that goes from
        the rear axle, tangent to the heading, through the target.
    """
    dx = target[0] - rear_axle[0]
    dy = target[1] - rear_axle[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0
    alpha = math.atan2(dy, dx) - yaw
    return math.atan2(2.0 * wheelbase_m * math.sin(alpha), distance)


def resample(points: Sequence[Point2D], step: float) -> List[Point2D]:
    """
    Resample a polyline at a constant spacing.

    Args:
        points: Polyline.
        step: Spacing between returned points.

    Returns:
        Points every ``step`` along the polyline, starting at its first point.
        The last point of the polyline is not included unless it falls on the
        spacing.
    """
    if not points:
        return []
    out: List[Point2D] = [points[0]]
    carried = 0.0
    for a, b in zip(points, points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-9:
            continue
        position = step - carried
        while position <= seg + 1e-9:
            t = position / seg
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            position += step
        carried = seg - (position - step)
    return out


def menger_curvature(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Unsigned curvature of the circle through three points (0 if aligned).
    """
    ab = math.hypot(b[0] - a[0], b[1] - a[1])
    bc = math.hypot(c[0] - b[0], c[1] - b[1])
    ca = math.hypot(a[0] - c[0], a[1] - c[1])
    if ab < 1e-9 or bc < 1e-9 or ca < 1e-9:
        return 0.0
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 2.0 * abs(cross) / (ab * bc * ca)


def speed_target(
    line: Sequence[Point2D],
    distance_to_line: float,
    steer_curvature: float,
    max_speed: float,
    params: PursuitParams,
) -> float:
    """
    Speed the car should aim for right now.

    Three limits apply, and the lowest wins:
    - every known bend must be reachable, braking at ``plan_brake_mps2``, at
      the speed that keeps it within the lateral-acceleration limit;
    - the end of the known line must be reachable at ``end_speed_mps``;
    - the arc the steering is asking for right now must itself stay within
      the lateral-acceleration limit.

    Args:
        line: Known centre line ahead (gate middles, without the car itself).
        distance_to_line: Distance from the car to the first point of ``line``.
        steer_curvature: Curvature of the arc commanded by pure pursuit.
        max_speed: Speed cap.
        params: Tuning.

    Returns:
        The target speed in m/s.
    """
    target = max_speed
    if abs(steer_curvature) > 1e-6:
        target = min(target, math.sqrt(params.max_lat_accel_mps2 / abs(steer_curvature)))
    if not line:
        return min(target, params.search_speed_mps)

    samples = resample(line, params.resample_step_m)
    arm = params.curvature_arm_pts
    brake = 2.0 * params.plan_brake_mps2

    for i in range(arm, len(samples) - arm):
        kappa = menger_curvature(samples[i - arm], samples[i], samples[i + arm])
        if kappa < 1e-6:
            continue
        bend_speed = math.sqrt(params.max_lat_accel_mps2 / kappa)
        # The bend is taken to start where the three-point window starts.
        distance = distance_to_line + (i - arm) * params.resample_step_m
        target = min(target, math.sqrt(bend_speed**2 + brake * distance))

    line_length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line, line[1:])
    )
    to_end = distance_to_line + line_length
    target = min(target, math.sqrt(params.end_speed_mps**2 + brake * to_end))
    return target
