"""Synthetic tracks with a known geometry, shared by the tests."""

from __future__ import annotations

import math
from typing import List

from track_utils import Cone

RING_INNER_M = 8.5
RING_OUTER_M = 11.5
RING_CENTER_M = (RING_INNER_M + RING_OUTER_M) / 2.0


def make_ring(n_inner: int = 28, n_outer: int = 36) -> List[Cone]:
    """
    Circular track driven counter-clockwise.

    Blue cones (left edge) are on the inner circle, yellow cones (right edge)
    on the outer one. Two big orange cones mark the timing line at an angle of
    0.25 rad after the start.
    """
    cones: List[Cone] = []
    for i in range(n_inner):
        a = 2.0 * math.pi * i / n_inner
        cones.append(
            {"tag": "blue", "x": RING_INNER_M * math.cos(a), "y": RING_INNER_M * math.sin(a)}
        )
    for i in range(n_outer):
        a = 2.0 * math.pi * (i + 0.5) / n_outer
        cones.append(
            {"tag": "yellow", "x": RING_OUTER_M * math.cos(a), "y": RING_OUTER_M * math.sin(a)}
        )
    line_angle = 0.25
    for radius in (RING_INNER_M - 0.4, RING_OUTER_M + 0.4):
        cones.append(
            {
                "tag": "big_orange",
                "x": radius * math.cos(line_angle),
                "y": radius * math.sin(line_angle),
            }
        )
    cones.append({"tag": "car_start", "x": RING_CENTER_M, "y": 0.0})
    return cones
