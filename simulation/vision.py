"""
simulation/vision.py

Vision cone utilities:
- Configuration dataclass
- World-space polygon generation
- Point-in-cone test

All computations are done in WORLD coordinates (meters if your world is meters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class VisionConfig:
    """
    Vision cone configuration in world units.

    Attributes:
        range_m: Cone range in meters (world units).
        fov_deg: Total field-of-view angle in degrees.
        fill_rgba: RGBA color used for translucent filling.
        edge_samples: Polygon arc resolution.
    """
    range_m: float = 7.0
    fov_deg: float = 60.0
    fill_rgba: Tuple[int, int, int, int] = (160, 160, 160, 60)
    edge_samples: int = 28


DEFAULT_VISION = VisionConfig()


def vision_cone_polygon_world(
    car_pos: Point2D,
    heading_deg: float,
    vision: VisionConfig = DEFAULT_VISION,
) -> List[Point2D]:
    """
    Build a polygon approximating the vision cone in world coordinates.

    Args:
        car_pos: Car position in world.
        heading_deg: Car heading angle (deg).
        vision: Vision configuration.

    Returns:
        List of polygon world points (apex + arc).
    """
    cx, cy = car_pos
    half_fov = vision.fov_deg * 0.5
    start_ang = math.radians(heading_deg - half_fov)
    end_ang = math.radians(heading_deg + half_fov)

    points: List[Point2D] = [(cx, cy)]
    for t in range(vision.edge_samples + 1):
        a = start_ang + (end_ang - start_ang) * (t / vision.edge_samples)
        x = cx + vision.range_m * math.cos(a)
        y = cy + vision.range_m * math.sin(a)
        points.append((x, y))

    return points


def point_in_vision_cone(
    point: Point2D,
    car_pos: Point2D,
    heading_deg: float,
    vision: VisionConfig = DEFAULT_VISION,
) -> bool:
    """
    Check if a world point lies inside the vision cone.

    Args:
        point: Target point (world).
        car_pos: Car position (world).
        heading_deg: Car heading (deg).
        vision: Vision configuration.

    Returns:
        True if point is inside the cone, otherwise False.
    """
    px, py = point
    cx, cy = car_pos

    vx = px - cx
    vy = py - cy

    # Range check
    dist_sq = vx * vx + vy * vy
    if dist_sq > vision.range_m * vision.range_m:
        return False

    # Angle check
    heading_rad = math.radians(heading_deg)
    hx = math.cos(heading_rad)
    hy = math.sin(heading_rad)

    v_norm = math.hypot(vx, vy)
    if v_norm < 1e-6:
        return True

    vx_u = vx / v_norm
    vy_u = vy / v_norm

    dot = hx * vx_u + hy * vy_u
    dot = max(-1.0, min(1.0, dot))
    ang = math.degrees(math.acos(dot))

    return ang <= (vision.fov_deg * 0.5)
