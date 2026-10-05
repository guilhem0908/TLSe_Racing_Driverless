"""End-to-end tests of the closed loop, its command line and its drawing code."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest

from closed_loop.cli import build_parser, config_from_args, format_result, main
from closed_loop.loop import ClosedLoop, LoopConfig, run
from closed_loop.track import load_start_pose, track_path
from closed_loop.vehicle import body_center
from simulation.vision import DEFAULT_VISION, VisionConfig
from tests.synthetic import RING_CENTER_M, RING_INNER_M, RING_OUTER_M
from track_utils import load_track

ONE_LAP = LoopConfig(target_laps=1)


def load(name: str):
    path = track_path(name)
    return load_track(str(path)), load_start_pose(path)


def test_car_starts_with_its_body_centre_on_the_start_marker(ring, ring_start):
    loop = ClosedLoop(ring, ring_start, ONE_LAP)
    assert body_center(loop.state, ONE_LAP.vehicle) == pytest.approx(ring_start.position)
    assert loop.state.yaw == ring_start.heading_rad
    assert loop.status is None


def test_one_clean_lap_of_the_ring(ring, ring_start):
    loop = ClosedLoop(ring, ring_start, ONE_LAP)
    radii = []
    while loop.status is None:
        loop.step()
        radii.append(math.hypot(*body_center(loop.state, ONE_LAP.vehicle)))
    result = loop.result()

    assert result.status == "finished"
    assert len(result.laps) == 1 and result.laps[0].valid
    assert result.cones_hit_total == 0
    assert result.track_length_m == pytest.approx(2.0 * math.pi * RING_CENTER_M, rel=0.03)
    # The body centre stays well inside the corridor all the way round.
    assert RING_INNER_M + 0.8 < min(radii) and max(radii) < RING_OUTER_M - 0.8
    # Constant radius: the speed settles near sqrt(a_lat * R), below the cap.
    assert result.max_speed_mps < ONE_LAP.vehicle.max_speed_mps
    assert result.max_lat_accel_mps2 <= ONE_LAP.vehicle.max_lat_accel_mps2 + 1e-9
    lap_speed = result.track_length_m / result.laps[0].time_s
    assert 3.0 < lap_speed < math.sqrt(ONE_LAP.pursuit.max_lat_accel_mps2 * RING_CENTER_M) * 1.1


@pytest.mark.parametrize("name", ["belgium", "peanut", "small_track"])
def test_one_clean_lap_of_a_bundled_track_with_the_default_sensor(name):
    cones, start = load(name)
    result = run(cones, start, ONE_LAP)
    assert result.status == "finished"
    assert result.laps[0].valid
    assert result.laps[0].gate_ratio == 1.0
    assert result.cones_hit_total == 0
    assert result.off_course_at is None


def test_runs_are_deterministic():
    cones, start = load("small_track")
    first = run(cones, start, ONE_LAP)
    second = run(cones, start, ONE_LAP)
    assert first == second


def test_noisy_runs_depend_only_on_the_seed():
    cones, start = load("peanut")
    noisy = replace(ONE_LAP, noise_std_m=0.1, seed=5)
    first = run(cones, start, noisy)
    assert first == run(cones, start, noisy)
    assert first != run(cones, start, replace(noisy, seed=6))
    assert first.status == "finished"


def test_without_the_one_sided_fallback_the_car_leaves_the_track():
    cones, start = load("peanut")
    result = run(cones, start, replace(ONE_LAP, one_sided_fallback=False))
    assert result.status == "off_course"
    assert result.valid_laps == []
    assert result.off_course_at is not None


def test_run_stops_at_the_time_limit(ring, ring_start):
    result = run(ring, ring_start, replace(ONE_LAP, target_laps=50, max_time_s=3.0))
    assert result.status == "timeout"
    assert result.sim_time_s == pytest.approx(3.0, abs=ONE_LAP.dt)


def test_frame_reports_what_the_car_saw_and_planned(ring, ring_start):
    wide = replace(ONE_LAP, vision=VisionConfig(range_m=8.0, fov_deg=120.0))
    loop = ClosedLoop(ring, ring_start, wide)
    frame = None
    for _ in range(100):
        frame = loop.step()

    assert frame.t == pytest.approx(100 * wide.dt)
    assert frame.detections
    assert {tag for tag, _, _ in frame.remembered} <= {"blue", "yellow", "big_orange"}
    assert len(frame.remembered) >= len(frame.detections)
    assert frame.line.points[0] != frame.line.points[-1]
    assert len(frame.line.points) == len(frame.line.gates) + 1
    lookahead = math.dist(frame.target, frame.line.points[0])
    assert wide.pursuit.lookahead_min_m <= lookahead <= wide.pursuit.lookahead_max_m + 1e-6
    assert 0.0 < frame.speed_target <= wide.vehicle.max_speed_mps


# -----------------------------
# Command line
# -----------------------------
def test_cli_defaults_match_the_repository_sensor():
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert args.track == "belgium"
    assert config.vision.range_m == DEFAULT_VISION.range_m
    assert config.vision.fov_deg == DEFAULT_VISION.fov_deg
    assert config.memory_ttl_s == LoopConfig().memory_ttl_s
    assert config.one_sided_fallback and config.noise_std_m == 0.0


def test_cli_options_reach_the_configuration():
    args = build_parser().parse_args(
        ["--track", "peanut", "--range", "9", "--fov", "70", "--laps", "2", "--memory", "0",
         "--no-fallback", "--noise", "0.05", "--seed", "4"]
    )
    config = config_from_args(args)
    assert (config.vision.range_m, config.vision.fov_deg) == (9.0, 70.0)
    assert (config.target_laps, config.memory_ttl_s) == (2, 0.0)
    assert not config.one_sided_fallback
    assert (config.noise_std_m, config.seed) == (0.05, 4)


def test_cli_headless_run(capsys):
    assert main(["--track", "small_track", "--laps", "1", "--headless"]) == 0
    out = capsys.readouterr().out
    assert "status: finished" in out
    assert "lap 1:" in out and "0 cones hit" in out


def test_cli_headless_run_reports_a_failure(capsys):
    code = main(["--track", "peanut", "--laps", "1", "--no-fallback", "--headless"])
    assert code == 1
    out = capsys.readouterr().out
    assert "status: off_course" in out
    assert "left the track at" in out


def test_format_result_lists_every_lap(ring, ring_start):
    result = run(ring, ring_start, replace(ONE_LAP, target_laps=2))
    lines = format_result(result)
    assert lines[0] == "status: finished"
    assert sum(line.startswith("lap ") for line in lines) == 2


# -----------------------------
# Drawing (off-screen)
# -----------------------------
def test_scene_can_be_drawn_without_a_window(ring, ring_start):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from closed_loop.viewer import BACKGROUND, draw_scene, follow_camera, status_lines
    from simulation.camera import Camera
    from track_utils import compute_world_bounds

    pygame.init()
    try:
        loop = ClosedLoop(ring, ring_start, ONE_LAP)
        surface = pygame.Surface((320, 240))
        camera = Camera(compute_world_bounds(ring), surface.get_size())

        draw_scene(surface, camera, ring, loop, None, [])  # before the first step
        frame, trail = None, []
        for _ in range(60):
            frame = loop.step()
            trail.append(body_center(loop.state, ONE_LAP.vehicle))
        draw_scene(surface, camera, ring, loop, frame, trail)

        colors = {
            tuple(surface.get_at((x, y)))[:3] for x in range(0, 320, 4) for y in range(0, 240, 4)
        }
        assert BACKGROUND in colors and len(colors) > 3

        centre = body_center(loop.state, ONE_LAP.vehicle)
        follow = follow_camera(ring, surface.get_size(), centre, span_m=20.0)
        assert follow.world_to_screen(centre[0], centre[1], surface.get_size()) == (160, 120)
        assert follow.zoom == pytest.approx(320 / 20.0)

        lines = status_lines("ring", loop)
        assert lines[0].startswith("ring") and "sensor 4 m / 100 deg" in lines[0]
    finally:
        pygame.quit()
