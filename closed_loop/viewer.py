"""
closed_loop/viewer.py

Pygame view of the closed loop.

It draws, with the camera of ``simulation/camera.py``:
- the field-of-view sector of ``simulation/vision.py``;
- every cone of the map, dimmed, since the car does not know them;
- the cones the car remembers, in full colour, and the ones it sees right
  now with a white outline;
- the gates and centre line it planned, and the pure-pursuit target;
- the footprint of the car, its trail, the timing line and a text panel with
  lap times and cone hits.

Controls: mouse wheel zooms, left-drag pans, F toggles the follow camera,
SPACE pauses, R restarts, ESC quits.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import pygame

from closed_loop.loop import ClosedLoop, Frame, LoopConfig
from closed_loop.track import StartPose
from closed_loop.vehicle import body_center, footprint_corners
from simulation.camera import Camera
from simulation.vision import vision_cone_polygon_world
from track_utils import Cone, compute_world_bounds

Color = Tuple[int, int, int]
Point2D = Tuple[float, float]

BACKGROUND: Color = (30, 30, 30)
CONE_COLORS: Dict[str, Color] = {
    "blue": (50, 100, 255),
    "yellow": (255, 255, 0),
    "big_orange": (255, 150, 0),
}
OUTLINE: Color = (255, 255, 255)
GATE: Color = (95, 95, 95)
VIRTUAL_GATE: Color = (150, 90, 150)
CENTER_LINE: Color = (255, 60, 60)
TARGET: Color = (255, 255, 255)
CAR_BODY: Color = (255, 0, 255)
CAR_FRONT: Color = (160, 0, 160)
TRAIL: Color = (130, 70, 130)
TIMING_LINE: Color = (235, 235, 235)
TEXT: Color = (220, 220, 220)

UNKNOWN_DIM = 0.45
FPS = 50
FOLLOW_SPAN_M = 26.0
TRAIL_POINTS = 4000


def _dim(color: Color, factor: float) -> Color:
    return int(color[0] * factor), int(color[1] * factor), int(color[2] * factor)


def draw_scene(
    surface: pygame.Surface,
    camera: Camera,
    cones: Sequence[Cone],
    loop: ClosedLoop,
    frame: Optional[Frame],
    trail: Sequence[Point2D],
) -> None:
    """
    Draw the track and what the car currently knows onto ``surface``.

    Args:
        surface: Target surface (cleared first).
        camera: World-to-screen camera.
        cones: Full track map.
        loop: The running closed loop.
        frame: Last frame returned by ``loop.step`` (None before the first).
        trail: Past positions of the body centre, oldest first.
    """
    size = surface.get_size()
    surface.fill(BACKGROUND)
    cone_px = max(2, int(0.16 * camera.zoom))

    def to_screen(point: Point2D) -> Tuple[int, int]:
        return camera.world_to_screen(point[0], point[1], size)

    if frame is not None:
        sector = vision_cone_polygon_world(
            frame.sensor_pos, frame.sensor_heading_deg, loop.config.vision
        )
        overlay = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, loop.config.vision.fill_rgba, [to_screen(p) for p in sector])
        surface.blit(overlay, (0, 0))

    pygame.draw.line(
        surface, TIMING_LINE, to_screen(loop.referee.line[0]), to_screen(loop.referee.line[1]), 1
    )

    for cone in cones:
        color = CONE_COLORS.get(cone["tag"])
        if color is not None:
            pygame.draw.circle(
                surface, _dim(color, UNKNOWN_DIM), to_screen((cone["x"], cone["y"])), cone_px
            )

    if len(trail) >= 2:
        pygame.draw.lines(surface, TRAIL, False, [to_screen(p) for p in trail], 1)

    if frame is not None:
        for gate in frame.line.gates:
            pygame.draw.line(
                surface,
                VIRTUAL_GATE if gate.virtual else GATE,
                to_screen(gate.left),
                to_screen(gate.right),
                1,
            )
        for tag, x, y in frame.remembered:
            pygame.draw.circle(surface, CONE_COLORS[tag], to_screen((x, y)), cone_px)
        for detection in frame.detections:
            pygame.draw.circle(
                surface, OUTLINE, to_screen((detection.x, detection.y)), cone_px + 2, width=2
            )

    corners = footprint_corners(loop.state, loop.config.vehicle)
    pygame.draw.polygon(surface, CAR_BODY, [to_screen(p) for p in corners])
    rear_left, front_left, front_right, rear_right = corners
    nose = [
        front_left,
        front_right,
        _between(front_right, rear_right, 0.3),
        _between(front_left, rear_left, 0.3),
    ]
    pygame.draw.polygon(surface, CAR_FRONT, [to_screen(p) for p in nose])

    if frame is not None:
        if len(frame.line.points) >= 2:
            pygame.draw.lines(
                surface, CENTER_LINE, False, [to_screen(p) for p in frame.line.points], 2
            )
        pygame.draw.circle(surface, TARGET, to_screen(frame.target), max(3, cone_px), width=1)


def _between(a: Point2D, b: Point2D, t: float) -> Point2D:
    return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])


def status_lines(track_name: str, loop: ClosedLoop) -> List[str]:
    """Text of the information panel."""
    vision = loop.config.vision
    referee = loop.referee
    lines = [
        f"{track_name}   sensor {vision.range_m:g} m / {vision.fov_deg:g} deg",
        f"t {loop.t:6.1f} s   speed {loop.state.speed:4.1f} m/s",
    ]
    if referee.lap_start_time is None:
        lines.append("lap -   waiting for the timing line")
    else:
        lines.append(
            f"lap {len(referee.laps) + 1}   {loop.t - referee.lap_start_time:5.1f} s"
            f"   cones this lap {referee.current_lap_cones}"
        )
    for lap in referee.laps[-3:]:
        note = "" if lap.valid else "  (not valid)"
        lines.append(f"lap {lap.number}: {lap.time_s:6.2f} s, {lap.cones_hit} cones{note}")
    if loop.status is not None:
        lines.append(f"run over: {loop.status}")
    return lines


def draw_text(
    surface: pygame.Surface, font: pygame.font.Font, lines: Sequence[str], pos: Tuple[int, int]
) -> None:
    """Draw text lines one below the other."""
    x, y = pos
    for line in lines:
        surface.blit(font.render(line, True, TEXT), (x, y))
        y += font.get_linesize()


def follow_camera(
    cones: Sequence[Cone], size: Tuple[int, int], center: Point2D, span_m: float = FOLLOW_SPAN_M
) -> Camera:
    """Camera centred on ``center`` that shows about ``span_m`` metres across."""
    camera = Camera(compute_world_bounds(cones), size)
    camera.cx, camera.cy = center
    camera.zoom = size[0] / span_m
    return camera


def run_viewer(
    track_name: str,
    cones: Sequence[Cone],
    start: StartPose,
    config: LoopConfig = LoopConfig(),
) -> None:
    """
    Open a window and drive the closed loop in real time.

    Args:
        track_name: Name shown in the information panel.
        cones: Track map.
        start: Starting pose.
        config: Run configuration.
    """
    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    pygame.display.set_caption("Closed loop - " + track_name)
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    bounds = compute_world_bounds(cones)

    loop = ClosedLoop(cones, start, config)
    frame: Optional[Frame] = None
    trail: Deque[Point2D] = deque(maxlen=TRAIL_POINTS)
    camera = Camera(bounds, screen.get_size())
    follow = False
    paused = False
    dragging = False
    last_mouse: Optional[Tuple[int, int]] = None
    steps_per_frame = max(1, round(1.0 / (FPS * config.dt)))

    running = True
    while running:
        clock.tick(FPS)
        size = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                camera = Camera(bounds, screen.get_size())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    follow = not follow
                    camera = Camera(bounds, size)
                elif event.key == pygame.K_r:
                    loop = ClosedLoop(cones, start, config)
                    frame = None
                    trail.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging, last_mouse = True, event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging, last_mouse = False, None
            elif event.type == pygame.MOUSEMOTION and dragging and last_mouse is not None:
                camera.pan_pixels(event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1])
                last_mouse = event.pos
            elif event.type == pygame.MOUSEWHEEL and event.y != 0:
                factor = 1.1 if event.y > 0 else 1 / 1.1
                camera.change_zoom(factor, pygame.mouse.get_pos(), size)

        if not paused and loop.status is None:
            for _ in range(steps_per_frame):
                frame = loop.step()
                trail.append(body_center(loop.state, config.vehicle))
                if loop.status is not None:
                    break

        if follow:
            camera = follow_camera(cones, size, body_center(loop.state, config.vehicle))

        draw_scene(screen, camera, cones, loop, frame, trail)
        lines = status_lines(track_name, loop)
        lines.append("wheel zoom | drag pan | F follow | SPACE pause | R restart | ESC quit")
        draw_text(screen, font, lines, (10, 10))
        pygame.display.flip()

    pygame.quit()
