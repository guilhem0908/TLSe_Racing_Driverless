"""Tests of the simulated cone sensor and of the cone memory."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from closed_loop.perception import ConeMemory, ConeSensor, Detection
from simulation.vision import VisionConfig, point_in_vision_cone
from track_utils import load_track

TRACKS = Path(__file__).resolve().parents[1] / "tracks"


def test_sensor_agrees_with_the_sector_test_on_every_cone():
    cones = load_track(str(TRACKS / "hairpins_increasing_difficulty.csv"))
    real = [c for c in cones if c["tag"] != "car_start"]
    rng = random.Random(1)

    for vision in (VisionConfig(), VisionConfig(range_m=9.0, fov_deg=140.0)):
        sensor = ConeSensor(cones, vision)
        for _ in range(60):
            anchor = rng.choice(real)
            pos = (anchor["x"] + rng.uniform(-3, 3), anchor["y"] + rng.uniform(-3, 3))
            heading = rng.uniform(-360.0, 360.0)
            expected = sorted(
                (c["tag"], c["x"], c["y"])
                for c in real
                if point_in_vision_cone((c["x"], c["y"]), pos, heading, vision)
            )
            got = sorted((d.tag, d.x, d.y) for d in sensor.detect(pos, heading))
            assert got == expected


def test_sensor_never_reports_the_start_marker():
    cones = [{"tag": "car_start", "x": 1.0, "y": 0.0}, {"tag": "blue", "x": 2.0, "y": 0.5}]
    detections = ConeSensor(cones, VisionConfig()).detect((0.0, 0.0), 0.0)
    assert [d.tag for d in detections] == ["blue"]


def test_noise_is_seeded_and_has_the_requested_spread():
    cones = [{"tag": "blue", "x": 2.0, "y": 0.0}]
    vision = VisionConfig()

    def sample(seed: int):
        sensor = ConeSensor(cones, vision, noise_std_m=0.1, seed=seed)
        return [sensor.detect((0.0, 0.0), 0.0)[0] for _ in range(2000)]

    first, again, other = sample(3), sample(3), sample(4)
    assert first == again
    assert first != other

    errors = [d.x - 2.0 for d in first] + [d.y for d in first]
    mean = sum(errors) / len(errors)
    std = math.sqrt(sum((e - mean) ** 2 for e in errors) / len(errors))
    assert abs(mean) < 0.01
    assert std == pytest.approx(0.1, rel=0.1)


def test_memory_merges_repeated_detections():
    memory = ConeMemory(ttl_s=5.0, merge_radius_m=0.5, min_observations=1)
    memory.update([Detection("blue", 1.0, 1.0)], now=0.0)
    memory.update([Detection("blue", 1.2, 1.0)], now=0.1)
    assert len(memory) == 1
    assert memory.points("blue") == [pytest.approx((1.1, 1.0))]


def test_memory_keeps_colours_and_distant_cones_apart():
    memory = ConeMemory(ttl_s=5.0, merge_radius_m=0.5, min_observations=1)
    memory.update(
        [Detection("blue", 0.0, 0.0), Detection("yellow", 0.1, 0.0), Detection("blue", 2.0, 0.0)],
        now=0.0,
    )
    assert len(memory) == 3
    assert sorted(memory.points("blue")) == [(0.0, 0.0), (2.0, 0.0)]
    assert memory.points("yellow") == [(0.1, 0.0)]


def test_memory_forgets_after_ttl():
    memory = ConeMemory(ttl_s=2.0, min_observations=1)
    memory.update([Detection("blue", 0.0, 0.0)], now=0.0)
    memory.update([Detection("yellow", 5.0, 0.0)], now=1.5)
    assert len(memory) == 2
    memory.update([], now=2.5)  # the blue cone was last seen 2.5 s ago
    assert memory.points("blue") == []
    assert memory.points("yellow") == [(5.0, 0.0)]


def test_zero_ttl_means_no_memory():
    memory = ConeMemory(ttl_s=0.0, min_observations=1)
    memory.update([Detection("blue", 0.0, 0.0)], now=0.0)
    assert len(memory) == 1
    memory.update([Detection("blue", 3.0, 0.0)], now=0.02)
    assert memory.points("blue") == [(3.0, 0.0)]


def test_cone_needs_confirmation_before_use():
    memory = ConeMemory(ttl_s=5.0, min_observations=3)
    for k in range(2):
        memory.update([Detection("blue", 1.0, 0.0)], now=0.02 * k)
        assert memory.points("blue") == []
    memory.update([Detection("blue", 1.0, 0.0)], now=0.04)
    assert memory.points("blue") == [(1.0, 0.0)]


def test_memory_averages_noise_away():
    rng = random.Random(0)
    memory = ConeMemory(ttl_s=10.0, merge_radius_m=0.5, min_observations=3)
    for k in range(200):
        memory.update(
            [Detection("blue", 4.0 + rng.gauss(0, 0.1), -2.0 + rng.gauss(0, 0.1))], now=0.02 * k
        )
    confirmed = memory.points("blue")
    assert len(confirmed) == 1
    assert math.hypot(confirmed[0][0] - 4.0, confirmed[0][1] + 2.0) < 0.05


def test_merge_radius_must_be_positive():
    with pytest.raises(ValueError):
        ConeMemory(merge_radius_m=0.0)
