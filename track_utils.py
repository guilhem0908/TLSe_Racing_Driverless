"""
track_utils.py

Utility functions to load a cone-based track from CSV and derive common
navigation helpers (world bounds, start/goal positions, obstacles).
"""

from __future__ import annotations

import csv
from typing import List, Sequence, Tuple, TypedDict


# -----------------------------
# Public types
# -----------------------------
WorldBounds = Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
Point2D = Tuple[float, float]
Obstacle = Tuple[float, float, float]  # (x, y, radius)


class Cone(TypedDict):
    """A cone item loaded from CSV."""
    tag: str
    x: float
    y: float


# -----------------------------
# Public API
# -----------------------------
def load_track(csv_path: str) -> List[Cone]:
    """
    Load cones from a CSV file.

    The CSV is expected to contain at least the columns: "tag", "x", "y".

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A list of cones, each with fields {tag, x, y}.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If x/y cannot be converted to float.
        KeyError: If required columns are missing.
    """
    cones: List[Cone] = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Basic schema check early for clearer errors
            required = {"tag", "x", "y"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise KeyError(
                    f"CSV must contain columns {sorted(required)}, "
                    f"got {reader.fieldnames!r}"
                )

            for row in reader:
                cones.append(
                    {
                        "tag": row["tag"],
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                    }
                )

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found {csv_path!r}") from e

    return cones


def compute_world_bounds(cones: Sequence[Cone], margin: float = 2.0) -> WorldBounds:
    """
    Compute world bounds that include all cones plus a margin.

    Args:
        cones: Track cones.
        margin: Extra space added on each side (world units).

    Returns:
        (min_x, max_x, min_y, max_y).
        If cones is empty, returns a default box (-10, 10, -10, 10).
    """
    if not cones:
        return -10.0, 10.0, -10.0, 10.0

    xs = [c["x"] for c in cones]
    ys = [c["y"] for c in cones]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def get_start_pos(cones: Sequence[Cone]) -> Point2D:
    """
    Get the car start position if present.

    Args:
        cones: Track cones.

    Returns:
        (x, y) of the cone tagged "car_start", or (0.0, 0.0) if not found.
    """
    for c in cones:
        if c["tag"] == "car_start":
            return c["x"], c["y"]
    return 0.0, 0.0


def get_goal_pos(cones: Sequence[Cone]) -> Point2D:
    """
    Compute a goal position.

    Strategy:
        - If cones is empty, return (10.0, 10.0).
        - Otherwise pick the cone farthest from origin (0,0),
          and return 90% of its coordinates (slightly inside).

    Args:
        cones: Track cones.

    Returns:
        Goal position (x, y).
    """
    if not cones:
        return 10.0, 10.0

    furthest = max(cones, key=_dist_sq_from_origin)
    return furthest["x"] * 0.9, furthest["y"] * 0.9


def get_obstacles(cones: Sequence[Cone], size: float = 1.0) -> List[Obstacle]:
    """
    Build obstacle circles from cones.

    Args:
        cones: Track cones.
        size: Obstacle radius to assign to each cone (world units).

    Returns:
        List of (x, y, radius) obstacles, excluding the "car_start" cone.
    """
    return [(c["x"], c["y"], size) for c in cones if c["tag"] != "car_start"]


# -----------------------------
# Internal helpers
# -----------------------------
def _dist_sq_from_origin(cone: Cone) -> float:
    """
    Squared distance of a cone from origin, used for comparisons.

    Args:
        cone: A cone.

    Returns:
        x^2 + y^2.
    """
    return cone["x"] ** 2 + cone["y"] ** 2
