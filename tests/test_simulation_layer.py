"""Tests of the November 2025 simulation layer: sensor model, camera, loader."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from simulation.camera import Camera, compute_fit_zoom
from simulation.vision import (
    DEFAULT_VISION,
    VisionConfig,
    point_in_vision_cone,
    vision_cone_polygon_world,
)
from track_utils import (
    compute_world_bounds,
    get_goal_pos,
    get_obstacles,
    get_start_pos,
    load_track,
)

TRACKS = Path(__file__).resolve().parents[1] / "tracks"
HEADER = "tag,x,y,direction,x_variance,y_variance,xy_covariance\n"


# -----------------------------
# Field-of-view sensor model
# -----------------------------
def test_default_sensor_is_four_metres_and_hundred_degrees():
    assert DEFAULT_VISION.range_m == 4.0
    assert DEFAULT_VISION.fov_deg == 100.0


@pytest.mark.parametrize(
    "point, expected",
    [
        ((3.9, 0.0), True),  # straight ahead, inside the range
        ((4.1, 0.0), False),  # straight ahead, beyond the range
        ((0.0, 0.0), True),  # the apex itself
        ((-1.0, 0.0), False),  # behind
        ((2.0, 2.0 * math.tan(math.radians(49.0))), True),  # just inside the edge
        ((2.0, 2.0 * math.tan(math.radians(51.0))), False),  # just outside the edge
        ((2.0, -2.0 * math.tan(math.radians(49.0))), True),  # symmetric on the right
    ],
)
def test_point_in_sector_facing_x(point, expected):
    assert point_in_vision_cone(point, (0.0, 0.0), 0.0, DEFAULT_VISION) is expected


def test_sector_follows_position_and_heading():
    vision = VisionConfig(range_m=5.0, fov_deg=60.0)
    car = (10.0, -3.0)
    # Heading +y: a point 4 m "above" is seen, a point 4 m to the right is not.
    assert point_in_vision_cone((10.0, 1.0), car, 90.0, vision)
    assert not point_in_vision_cone((14.0, -3.0), car, 90.0, vision)


def test_sector_handles_heading_wrap_around():
    vision = VisionConfig(range_m=5.0, fov_deg=40.0)
    # Heading 179 deg, target at -175 deg from the car: 6 degrees apart.
    target = (3.0 * math.cos(math.radians(-175.0)), 3.0 * math.sin(math.radians(-175.0)))
    assert point_in_vision_cone(target, (0.0, 0.0), 179.0, vision)
    assert point_in_vision_cone(target, (0.0, 0.0), 179.0 + 360.0, vision)


def test_sector_polygon_shape():
    vision = VisionConfig(range_m=6.0, fov_deg=90.0, edge_samples=12)
    polygon = vision_cone_polygon_world((1.0, 2.0), 30.0, vision)

    assert len(polygon) == vision.edge_samples + 2
    assert polygon[0] == (1.0, 2.0)
    for x, y in polygon[1:]:
        assert math.hypot(x - 1.0, y - 2.0) == pytest.approx(6.0)

    first = math.degrees(math.atan2(polygon[1][1] - 2.0, polygon[1][0] - 1.0))
    last = math.degrees(math.atan2(polygon[-1][1] - 2.0, polygon[-1][0] - 1.0))
    assert first == pytest.approx(30.0 - 45.0)
    assert last == pytest.approx(30.0 + 45.0)


def test_sector_polygon_points_are_inside_the_sector():
    vision = VisionConfig(range_m=4.0, fov_deg=100.0)
    arc = vision_cone_polygon_world((0.0, 0.0), 75.0, vision)[1:]
    # The two end points sit exactly on the angular edges; the others are
    # pulled slightly inwards to stay clear of rounding on the range.
    for x, y in arc[1:-1]:
        assert point_in_vision_cone((0.999 * x, 0.999 * y), (0.0, 0.0), 75.0, vision)


# -----------------------------
# Camera
# -----------------------------
BOUNDS = (-10.0, 30.0, -5.0, 15.0)
SCREEN = (1200, 800)


def test_fit_zoom_uses_the_tighter_axis():
    assert compute_fit_zoom(BOUNDS, SCREEN) == pytest.approx(min(1200 / 40.0, 800 / 20.0))
    assert compute_fit_zoom((0.0, 0.0, 0.0, 1.0), SCREEN) == 50.0  # degenerate box


def test_camera_centres_the_world():
    camera = Camera(BOUNDS, SCREEN)
    assert camera.world_to_screen(10.0, 5.0, SCREEN) == (600, 400)


def test_camera_y_axis_points_up():
    camera = Camera(BOUNDS, SCREEN)
    _, sy_low = camera.world_to_screen(10.0, 0.0, SCREEN)
    _, sy_high = camera.world_to_screen(10.0, 10.0, SCREEN)
    assert sy_high < sy_low


def test_camera_round_trip():
    camera = Camera(BOUNDS, SCREEN)
    camera.change_zoom(1.7, (300, 500), SCREEN)
    camera.pan_pixels(40.0, -25.0)
    for sx, sy in [(0, 0), (600, 400), (1199, 799), (37, 512)]:
        wx, wy = camera.screen_to_world(sx, sy, SCREEN)
        assert camera.world_to_screen(wx, wy, SCREEN) == pytest.approx((sx, sy), abs=1)


def test_zoom_keeps_the_point_under_the_cursor():
    camera = Camera(BOUNDS, SCREEN)
    cursor = (900, 200)
    before = camera.screen_to_world(cursor[0], cursor[1], SCREEN)
    camera.change_zoom(1.5, cursor, SCREEN)
    after = camera.screen_to_world(cursor[0], cursor[1], SCREEN)
    assert after == pytest.approx(before)
    assert camera.zoom == pytest.approx(1.5 * camera.base_zoom)


def test_zoom_is_clamped():
    camera = Camera(BOUNDS, SCREEN)
    for _ in range(50):
        camera.change_zoom(2.0, (600, 400), SCREEN)
    assert camera.zoom == pytest.approx(10.0 * camera.base_zoom)
    for _ in range(100):
        camera.change_zoom(0.5, (600, 400), SCREEN)
    assert camera.zoom == pytest.approx(0.1 * camera.base_zoom)
    camera.change_zoom(0.0, (600, 400), SCREEN)  # ignored
    assert camera.zoom == pytest.approx(0.1 * camera.base_zoom)


def test_pan_moves_the_world_with_the_drag():
    camera = Camera(BOUNDS, SCREEN)
    before = camera.world_to_screen(10.0, 5.0, SCREEN)
    camera.pan_pixels(50.0, 20.0)  # drag right and down
    after = camera.world_to_screen(10.0, 5.0, SCREEN)
    assert after == pytest.approx((before[0] + 50, before[1] + 20), abs=1)


# -----------------------------
# Track loader
# -----------------------------
@pytest.mark.parametrize(
    "name, blue, yellow, orange",
    [
        ("belgium", 67, 63, 2),
        ("small_track", 37, 30, 4),
        ("peanut", 54, 64, 4),
        ("hairpins_increasing_difficulty", 490, 490, 4),
    ],
)
def test_bundled_tracks_load(name, blue, yellow, orange):
    cones = load_track(str(TRACKS / f"{name}.csv"))
    tags = [c["tag"] for c in cones]
    assert tags.count("blue") == blue
    assert tags.count("yellow") == yellow
    assert tags.count("big_orange") == orange
    assert tags.count("car_start") == 1
    assert all(isinstance(c["x"], float) and isinstance(c["y"], float) for c in cones)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_track(str(tmp_path / "nowhere.csv"))


def test_missing_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("tag,x\nblue,1.0\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_track(str(path))


def test_non_numeric_coordinate_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(HEADER + "blue,abc,1.0,0,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_track(str(path))


def test_world_bounds_and_helpers():
    cones = [
        {"tag": "blue", "x": -2.0, "y": 1.0},
        {"tag": "yellow", "x": 6.0, "y": -3.0},
        {"tag": "car_start", "x": 1.0, "y": 0.5},
    ]
    assert compute_world_bounds(cones, margin=1.0) == (-3.0, 7.0, -4.0, 2.0)
    assert compute_world_bounds([]) == (-10.0, 10.0, -10.0, 10.0)
    assert get_start_pos(cones) == (1.0, 0.5)
    assert get_start_pos(cones[:2]) == (0.0, 0.0)
    assert get_obstacles(cones, size=0.5) == [(-2.0, 1.0, 0.5), (6.0, -3.0, 0.5)]
    assert get_goal_pos(cones) == pytest.approx((5.4, -2.7))
    assert get_goal_pos([]) == (10.0, 10.0)
