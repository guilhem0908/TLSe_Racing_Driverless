"""
closed_loop/centerline.py

Local centre line from paired cones.

Convention (the one all bundled tracks follow): blue cones mark the left edge
of the track and yellow cones the right edge, seen in the driving direction.

Steps:
1. Pairing. Every blue cone is paired with its nearest yellow cone and every
   yellow cone with its nearest blue cone. A pair is kept as a *gate* if its
   width is plausible and if no other cone lies inside the circle whose
   diameter is the pair (the Gabriel-graph test). That test is what rejects a
   pair that would reach across a neighbouring piece of track.
2. Ordering. Starting from the car, gates are chained greedily: the next gate
   is the nearest one that lies ahead and faces the same way as the current
   direction of travel.
3. One-sided fallback. A cone that found no partner (the other edge has not
   been seen yet) can still extend the line: it is offset by half a track
   width towards the inside, perpendicular to the local direction of its own
   edge.

The same functions run on the full map to give the referee its ordered gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class CenterlineParams:
    """
    Tuning of the centre-line builder.

    Attributes:
        min_gate_width_m: Narrowest blue-yellow pair accepted as a gate.
        max_gate_width_m: Widest blue-yellow pair accepted as a gate.
        gabriel_shrink: Factor (< 1) applied to the radius of the emptiness
            test so that a cone sitting exactly on the circle does not count.
        max_step_m: Largest jump between two consecutive centre-line points.
        max_length_m: Length at which the line is cut.
        default_half_width_m: Half track width assumed for the one-sided
            fallback while no real gate has been seen.
        edge_neighbor_max_m: Largest distance to the same-colour neighbour
            used to estimate the direction of an edge.
        min_edge_alignment: Lower bound on the cosine between an edge direction
            and the direction of travel for the one-sided fallback.
        virtual_clearance_ratio: A fallback point must stay at least this
            fraction of the half width away from every known cone.
    """

    min_gate_width_m: float = 2.0
    max_gate_width_m: float = 6.5
    gabriel_shrink: float = 0.95
    max_step_m: float = 6.0
    max_length_m: float = 25.0
    default_half_width_m: float = 1.5
    edge_neighbor_max_m: float = 5.5
    min_edge_alignment: float = 0.5
    virtual_clearance_ratio: float = 0.7


@dataclass(frozen=True)
class Gate:
    """
    A left/right pair that the car should drive through.

    Attributes:
        left: Left cone (blue).
        right: Right cone (yellow).
        virtual: True when one of the two ends was not observed but inferred
            by the one-sided fallback.
    """

    left: Point2D
    right: Point2D
    virtual: bool = False

    @property
    def mid(self) -> Point2D:
        """Middle of the gate."""
        return (self.left[0] + self.right[0]) / 2.0, (self.left[1] + self.right[1]) / 2.0

    @property
    def width(self) -> float:
        """Distance between the two ends."""
        return math.hypot(self.left[0] - self.right[0], self.left[1] - self.right[1])

    @property
    def forward(self) -> Point2D:
        """Unit vector normal to the gate, pointing in the driving direction."""
        lx = self.left[0] - self.right[0]
        ly = self.left[1] - self.right[1]
        norm = math.hypot(lx, ly)
        if norm < 1e-9:
            return 1.0, 0.0
        # (lx, ly) points to the left; rotating it by -90 degrees points ahead.
        return ly / norm, -lx / norm


@dataclass(frozen=True)
class CenterLine:
    """
    Ordered centre line ahead of a starting point.

    Attributes:
        points: Starting point followed by the middle of each ordered gate.
        gates: Ordered gates (``len(points) - 1`` of them).
        end_direction: Unit vector in which the line may be extended.
    """

    points: List[Point2D]
    gates: List[Gate]
    end_direction: Point2D

    @property
    def length(self) -> float:
        """Polyline length."""
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(self.points, self.points[1:])
        )


def _as_array(points: Sequence[Point2D]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float).reshape(-1, 2)


def pair_cones(
    blue: Sequence[Point2D],
    yellow: Sequence[Point2D],
    params: CenterlineParams = CenterlineParams(),
) -> List[Tuple[int, int]]:
    """
    Pair blue and yellow cones into gates.

    Args:
        blue: Left-edge cones.
        yellow: Right-edge cones.
        params: Width limits and tolerance of the emptiness test.

    Returns:
        Sorted list of ``(blue_index, yellow_index)`` pairs.
    """
    b = _as_array(blue)
    y = _as_array(yellow)
    if len(b) == 0 or len(y) == 0:
        return []

    dist = np.hypot(b[:, None, 0] - y[None, :, 0], b[:, None, 1] - y[None, :, 1])
    candidates = {(i, int(j)) for i, j in enumerate(dist.argmin(axis=1))}
    candidates |= {(int(i), j) for j, i in enumerate(dist.argmin(axis=0))}

    pairs = np.array(sorted(candidates), dtype=int)
    bi, yj = pairs[:, 0], pairs[:, 1]
    width = dist[bi, yj]
    plausible = (width >= params.min_gate_width_m) & (width <= params.max_gate_width_m)

    # Gabriel test: no third cone inside the circle whose diameter is the pair.
    # The two ends lie on the circle itself and the radius is shrunk slightly,
    # so they never count.
    everything = np.vstack((b, y))
    centers = (b[bi] + y[yj]) / 2.0
    radius = params.gabriel_shrink * width / 2.0
    to_center = np.hypot(
        centers[:, None, 0] - everything[None, :, 0],
        centers[:, None, 1] - everything[None, :, 1],
    )
    empty = ~(to_center < radius[:, None]).any(axis=1)

    return [(int(i), int(j)) for i, j in pairs[plausible & empty]]


def _edge_directions(points: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Unoriented direction of the edge at each cone.

    For every cone, the unit vector towards its nearest same-colour neighbour,
    or (0, 0) when no neighbour lies within ``max_dist``.
    """
    n = len(points)
    directions = np.zeros((n, 2), dtype=float)
    if n < 2:
        return directions
    diff = points[None, :, :] - points[:, None, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(axis=1)
    for i, j in enumerate(nearest):
        d = dist[i, j]
        if 1e-6 < d <= max_dist:
            directions[i] = diff[i, j] / d
    return directions


def build_centerline(
    blue: Sequence[Point2D],
    yellow: Sequence[Point2D],
    start: Point2D,
    heading: Point2D,
    params: CenterlineParams = CenterlineParams(),
    allow_virtual: bool = True,
) -> CenterLine:
    """
    Build the ordered centre line ahead of ``start``.

    Args:
        blue: Known left-edge cones.
        yellow: Known right-edge cones.
        start: Point the line starts from (the car).
        heading: Unit vector of the initial direction of travel.
        params: Tuning.
        allow_virtual: Enable the one-sided fallback.

    Returns:
        The centre line. With no usable cone it contains only ``start``.
    """
    b = _as_array(blue)
    y = _as_array(yellow)
    pairs = pair_cones(blue, yellow, params)

    if pairs:
        bi = np.array([p[0] for p in pairs])
        yi = np.array([p[1] for p in pairs])
        left = b[bi]
        right = y[yi]
        mids = (left + right) / 2.0
        across = left - right
        widths = np.hypot(across[:, 0], across[:, 1])
        forwards = np.column_stack((across[:, 1], -across[:, 0])) / widths[:, None]
        half_width = float(np.median(widths)) / 2.0
    else:
        left = right = mids = forwards = np.zeros((0, 2), dtype=float)
        half_width = params.default_half_width_m

    # Cones without a partner, candidates for the one-sided fallback.
    if allow_virtual:
        paired_b = {p[0] for p in pairs}
        paired_y = {p[1] for p in pairs}
        dir_b = _edge_directions(b, params.edge_neighbor_max_m)
        dir_y = _edge_directions(y, params.edge_neighbor_max_m)
        free_b = [i for i in range(len(b)) if i not in paired_b]
        free_y = [i for i in range(len(y)) if i not in paired_y]
        lone = np.vstack((b[free_b], y[free_y])) if free_b or free_y else np.zeros((0, 2))
        lone_dir = (
            np.vstack((dir_b[free_b], dir_y[free_y])) if free_b or free_y else np.zeros((0, 2))
        )
        # +1 for a left-edge cone (the track is on its right), -1 otherwise.
        lone_side = np.array([1.0] * len(free_b) + [-1.0] * len(free_y))
        everything = np.vstack((b, y)) if len(b) + len(y) else np.zeros((0, 2))
    else:
        lone = lone_dir = everything = np.zeros((0, 2), dtype=float)
        lone_side = np.zeros(0)

    used = np.zeros(len(mids), dtype=bool)
    lone_used = np.zeros(len(lone), dtype=bool)

    cur = np.array(start, dtype=float)
    direction = np.array(heading, dtype=float)
    points: List[Point2D] = [(float(cur[0]), float(cur[1]))]
    gates: List[Gate] = []
    length = 0.0

    while length < params.max_length_m:
        best_real: Optional[int] = None
        best_real_dist = math.inf
        if len(mids):
            delta = mids - cur
            dist = np.hypot(delta[:, 0], delta[:, 1])
            ok = (
                ~used
                & (dist <= params.max_step_m)
                & (delta @ direction > 0.0)
                & (forwards @ direction > 0.0)
            )
            if ok.any():
                best_real = int(np.where(ok, dist, np.inf).argmin())
                best_real_dist = float(dist[best_real])

        best_virtual: Optional[Tuple[int, np.ndarray, np.ndarray]] = None
        best_virtual_dist = math.inf
        if len(lone):
            # Orient each edge direction along the current direction of travel.
            dots = lone_dir @ direction
            tangents = np.where(dots[:, None] < 0.0, -lone_dir, lone_dir)
            no_dir = (np.abs(lone_dir).sum(axis=1) < 1e-9)
            tangents[no_dir] = direction
            alignment = tangents @ direction
            # Rotate by -90 degrees (to the right) for a left-edge cone.
            inward = np.column_stack((tangents[:, 1], -tangents[:, 0])) * lone_side[:, None]
            virtual_mids = lone + inward * half_width
            delta = virtual_mids - cur
            dist = np.hypot(delta[:, 0], delta[:, 1])
            ok = (
                ~lone_used
                & (dist <= params.max_step_m)
                & (delta @ direction > 0.0)
                & (alignment >= params.min_edge_alignment)
            )
            min_clear = params.virtual_clearance_ratio * half_width
            for k in np.argsort(np.where(ok, dist, np.inf)):
                if not ok[k] or dist[k] >= best_real_dist:
                    break
                vm = virtual_mids[k]
                clear = np.hypot(everything[:, 0] - vm[0], everything[:, 1] - vm[1]).min()
                if clear >= min_clear:
                    best_virtual = (int(k), vm, tangents[k])
                    best_virtual_dist = float(dist[k])
                    break

        if best_virtual is not None and best_virtual_dist < best_real_dist:
            k, vm, tangent = best_virtual
            lone_used[k] = True
            cone = lone[k]
            other = cone + 2.0 * (vm - cone)
            if lone_side[k] > 0:
                gate = Gate(_pt(cone), _pt(other), virtual=True)
            else:
                gate = Gate(_pt(other), _pt(cone), virtual=True)
            step_to, new_direction = vm, tangent
        elif best_real is not None:
            used[best_real] = True
            gate = Gate(_pt(left[best_real]), _pt(right[best_real]))
            step_to, new_direction = mids[best_real], forwards[best_real]
        else:
            break

        length += float(np.hypot(step_to[0] - cur[0], step_to[1] - cur[1]))
        cur = np.array(step_to, dtype=float)
        direction = np.array(new_direction, dtype=float)
        points.append(_pt(cur))
        gates.append(gate)

    return CenterLine(points, gates, (float(direction[0]), float(direction[1])))


def _pt(values: np.ndarray) -> Point2D:
    return float(values[0]), float(values[1])
