"""Tests of the centre-line builder (pairing, ordering, one-sided fallback)."""

from __future__ import annotations

import math

import pytest

from closed_loop.centerline import CenterlineParams, Gate, build_centerline, pair_cones
from closed_loop.track import split_cones
from tests.synthetic import RING_CENTER_M

PARAMS = CenterlineParams()


def corridor(length: int = 6, width: float = 3.0, spacing: float = 2.0, stagger: float = 0.0):
    """Straight corridor along +x: blue on the left (y > 0), yellow on the right."""
    blue = [(i * spacing, width / 2.0) for i in range(length)]
    yellow = [(i * spacing + stagger, -width / 2.0) for i in range(length)]
    return blue, yellow


def test_gate_properties():
    gate = Gate(left=(0.0, 2.0), right=(0.0, -1.0))
    assert gate.mid == (0.0, 0.5)
    assert gate.width == 3.0
    assert gate.forward == pytest.approx((1.0, 0.0))
    # Swapping the sides reverses the driving direction.
    assert Gate(left=(0.0, -1.0), right=(0.0, 2.0)).forward == pytest.approx((-1.0, 0.0))


def test_pairing_of_a_straight_corridor():
    blue, yellow = corridor()
    assert pair_cones(blue, yellow) == [(i, i) for i in range(6)]


def test_pairing_needs_both_colours():
    blue, yellow = corridor()
    assert pair_cones(blue, []) == []
    assert pair_cones([], yellow) == []


def test_pairing_respects_the_width_limits():
    blue = [(0.0, 0.0)]
    assert pair_cones(blue, [(0.0, -1.0)]) == []  # narrower than min_gate_width_m
    assert pair_cones(blue, [(0.0, -3.0)]) == [(0, 0)]
    assert pair_cones(blue, [(0.0, -8.0)]) == []  # wider than max_gate_width_m


def test_pairing_does_not_reach_across_a_neighbouring_row():
    # Two parallel pieces of track whose blue rows are 2 m apart, as at the
    # waist of the peanut track. Only the lower yellow row is known.
    lower_blue = [(float(x), 1.5) for x in range(0, 9, 2)]
    upper_blue = [(float(x) + 1.0, 3.5) for x in range(0, 9, 2)]
    yellow = [(float(x), -1.5) for x in range(0, 9, 2)]
    blue = lower_blue + upper_blue

    pairs = pair_cones(blue, yellow)
    assert pairs
    assert all(b < len(lower_blue) for b, _ in pairs)


def test_centerline_runs_down_the_middle_in_order():
    blue, yellow = corridor(stagger=0.7)
    line = build_centerline(blue, yellow, start=(-1.0, 0.2), heading=(1.0, 0.0))

    assert line.points[0] == (-1.0, 0.2)
    xs = [p[0] for p in line.points]
    assert xs == sorted(xs)
    assert all(abs(p[1]) < 1e-9 for p in line.points[1:])
    assert all(not g.virtual for g in line.gates)
    assert line.end_direction[0] > 0.9
    assert line.length == pytest.approx(
        sum(math.dist(a, b) for a, b in zip(line.points, line.points[1:]))
    )


def test_gates_behind_the_car_are_ignored():
    blue, yellow = corridor()
    line = build_centerline(blue, yellow, start=(5.0, 0.0), heading=(1.0, 0.0))
    assert [p[0] for p in line.points[1:]] == [6.0, 8.0, 10.0]


def test_gates_facing_the_other_way_are_ignored():
    blue, yellow = corridor()
    # Driving towards -x, the colours are on the wrong sides: nothing to follow.
    line = build_centerline(
        blue, yellow, start=(11.0, 0.0), heading=(-1.0, 0.0), allow_virtual=False
    )
    assert line.points == [(11.0, 0.0)]
    assert line.gates == []
    assert line.end_direction == (-1.0, 0.0)


def test_line_is_cut_at_max_length():
    blue, yellow = corridor(length=30)
    short = CenterlineParams(max_length_m=10.0)
    line = build_centerline(blue, yellow, (-1.0, 0.0), (1.0, 0.0), short)
    assert 10.0 <= line.length <= 12.0


def test_no_cones_gives_a_line_with_only_the_start():
    line = build_centerline([], [], (1.0, 2.0), (0.0, 1.0))
    assert line.points == [(1.0, 2.0)]
    assert line.end_direction == (0.0, 1.0)


def test_ring_is_fully_ordered_counter_clockwise(ring, ring_start):
    blue, yellow = split_cones(ring)
    whole = CenterlineParams(max_length_m=math.inf)
    line = build_centerline(
        blue, yellow, ring_start.position, ring_start.direction, whole, allow_virtual=False
    )

    assert len(line.gates) == len(pair_cones(blue, yellow))
    angles = [math.atan2(g.mid[1], g.mid[0]) % (2.0 * math.pi) for g in line.gates]
    assert angles == sorted(angles)
    for gate in line.gates:
        assert math.hypot(*gate.mid) == pytest.approx(RING_CENTER_M, abs=0.2)
    assert line.length == pytest.approx(2.0 * math.pi * RING_CENTER_M, rel=0.03)


def test_one_sided_fallback_offsets_blue_cones_to_the_right():
    blue, _ = corridor()
    line = build_centerline(blue, [], start=(-1.0, 0.0), heading=(1.0, 0.0))

    assert len(line.gates) == len(blue)
    assert all(g.virtual for g in line.gates)
    for (x, y), cone in zip(line.points[1:], blue):
        assert x == pytest.approx(cone[0])
        assert y == pytest.approx(cone[1] - PARAMS.default_half_width_m)


def test_one_sided_fallback_offsets_yellow_cones_to_the_left():
    _, yellow = corridor()
    line = build_centerline([], yellow, start=(-1.0, 0.0), heading=(1.0, 0.0))

    assert all(g.virtual for g in line.gates)
    for (x, y), cone in zip(line.points[1:], yellow):
        assert x == pytest.approx(cone[0])
        assert y == pytest.approx(cone[1] + PARAMS.default_half_width_m)
    # The inferred end of a virtual gate mirrors the real cone.
    assert line.gates[0].right == yellow[0]
    assert line.gates[0].left == pytest.approx((0.0, 1.5))


def test_fallback_uses_the_width_of_the_gates_already_seen():
    # Both edges known for x <= 4, only the blue edge beyond.
    blue = [(float(x), 2.0) for x in range(0, 13, 2)]
    yellow = [(float(x), -2.0) for x in range(0, 5, 2)]
    line = build_centerline(blue, yellow, start=(-1.0, 0.0), heading=(1.0, 0.0))

    real = [g for g in line.gates if not g.virtual]
    virtual = [g for g in line.gates if g.virtual]
    # The blue cone at x = 6 still pairs, diagonally, with the last yellow one.
    assert [g.left[0] for g in real] == [0.0, 2.0, 4.0, 6.0]
    assert [g.left[0] for g in virtual] == [8.0, 10.0, 12.0]
    assert all(g.mid[1] == pytest.approx(0.0) for g in virtual)  # half of 4 m, not 1.5 m


def test_fallback_can_be_disabled():
    blue, _ = corridor()
    line = build_centerline(blue, [], (-1.0, 0.0), (1.0, 0.0), allow_virtual=False)
    assert line.gates == []


def test_fallback_rejects_points_next_to_a_known_cone():
    # Blue and yellow rows only 1.3 m apart: too narrow to be gates, and the
    # point inferred from either row would land 0.2 m from the other one.
    blue = [(float(x), 1.5) for x in range(0, 9, 2)]
    yellow = [(float(x), 0.2) for x in range(0, 9, 2)]
    assert pair_cones(blue, yellow) == []
    line = build_centerline(blue, yellow, start=(-1.0, 0.8), heading=(1.0, 0.0))
    assert line.gates == []
