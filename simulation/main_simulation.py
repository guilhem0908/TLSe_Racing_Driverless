"""
process_pygame.py

Interactive visualization of cones (track) and an optional path using Pygame.

Controls:
- Zoom: mouse wheel
- Pan camera: left click + drag
- Visual scale: arrow keys
- Reset view: R
- Toggle fullscreen: F11
- Quit: ESC / window close
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TypedDict

import pygame

from simulation.camera import Camera


# -----------------------------
# Display / UX constants
# -----------------------------
DEFAULT_WIDTH: int = 1200
DEFAULT_HEIGHT: int = 800
BACKGROUND_COLOR: Tuple[int, int, int] = (30, 30, 30)
FPS: int = 60

CAR_SIZE_M: float = 1.5

COLORS: Dict[str, Tuple[int, int, int]] = {
    "yellow": (255, 255, 0),
    "blue": (50, 100, 255),
    "big_orange": (255, 150, 0),
    "car_start": (0, 255, 0),
    "path_line": (255, 50, 50),
    "car_body": (255, 0, 255),
    "car_front": (200, 0, 200),
}


# -----------------------------
# Utility types
# -----------------------------
WorldBounds = Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
Point2D = Tuple[float, float]


class Cone(TypedDict):
    """A cone item loaded from track data."""
    tag: str
    x: float
    y: float


def process_pygame(
    cones: Sequence[Cone],
    world_bounds: WorldBounds,
    path: Optional[Sequence[Point2D]] = None,
) -> None:
    """
    Open a Pygame window to visualize the track and (optionally) a path.

    Args:
        cones: List of cones with fields "tag", "x", "y".
        world_bounds: World bounds (min_x, max_x, min_y, max_y).
        path: List of (x, y) points representing a path to draw/animate.
    """

    pygame.init()
    screen = _create_screen(DEFAULT_WIDTH, DEFAULT_HEIGHT)
    pygame.display.set_caption("Track and Path Visualization")

    clock = pygame.time.Clock()
    running = True

    camera = Camera(world_bounds, screen.get_size())
    display_scale = 1.0  # visual scaling with arrow keys (purely visual)

    fullscreen = False
    f11_pressed_during_resize = False
    windowed_size = screen.get_size()

    dragging = False
    last_mouse_pos: Optional[Tuple[int, int]] = None

    car_path_index = 0

    # Base pixel sizes adjusted to the current window size
    base_sizes = _compute_base_sizes(screen.get_size())

    while running:
        dt = clock.tick(FPS) / 1000.0
        screen_size = screen.get_size()

        (
            running,
            screen,
            camera,
            base_sizes,
            display_scale,
            fullscreen,
            f11_pressed_during_resize,
            windowed_size,
            dragging,
            last_mouse_pos,
            car_path_index,
        ) = _handle_events(
            running=running,
            screen=screen,
            camera=camera,
            world_bounds=world_bounds,
            base_sizes=base_sizes,
            display_scale=display_scale,
            fullscreen=fullscreen,
            f11_pressed_during_resize=f11_pressed_during_resize,
            windowed_size=windowed_size,
            dragging=dragging,
            last_mouse_pos=last_mouse_pos,
            car_path_index=car_path_index,
        )

        display_scale = _update_display_scale(display_scale, dt)

        # Render scene
        screen.fill(BACKGROUND_COLOR)

        if path and len(path) > 1:
            car_path_index = _draw_path_and_car(
                screen=screen,
                camera=camera,
                path=path,
                screen_size=screen_size,
                display_scale=display_scale,
                car_path_index=car_path_index,
            )

        _draw_cones(
            screen=screen,
            camera=camera,
            cones=cones,
            screen_size=screen_size,
            base_cone_radius_px=base_sizes["cone_radius_px"],
            base_start_radius_px=base_sizes["start_radius_px"],
            display_scale=display_scale,
        )

        _draw_hud(screen)
        pygame.display.flip()

    pygame.quit()


# -----------------------------
# Internal helpers
# -----------------------------
def _create_screen(width: int, height: int) -> pygame.Surface:
    """
    Create a resizable Pygame window.
    """
    return pygame.display.set_mode((width, height), pygame.RESIZABLE)


def _compute_base_sizes(screen_size: Tuple[int, int]) -> Dict[str, float]:
    """
    Compute base sizes (cone radius) based on the smallest window dimension.

    Args:
        screen_size: (width_px, height_px)

    Returns:
        Dict with base cone sizes in pixels.
    """
    w, h = screen_size
    min_dim = min(w, h)
    return {
        "cone_radius_px": min_dim * 0.0075,
        "start_radius_px": min_dim * 0.0125,
    }


def _handle_events(
    *,
    running: bool,
    screen: pygame.Surface,
    camera: Camera,
    world_bounds: WorldBounds,
    base_sizes: Dict[str, float],
    display_scale: float,
    fullscreen: bool,
    f11_pressed_during_resize: bool,
    windowed_size: Tuple[int, int],
    dragging: bool,
    last_mouse_pos: Optional[Tuple[int, int]],
    car_path_index: int,
) -> Tuple[
    bool,
    pygame.Surface,
    Camera,
    Dict[str, float],
    float,
    bool,
    bool,
    Tuple[int, int],
    bool,
    Optional[Tuple[int, int]],
    int,
]:
    """
    Process Pygame events and return updated state.
    """
    screen_size = screen.get_size()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEORESIZE:
            # Ignore resize triggered by fullscreen toggle
            if f11_pressed_during_resize:
                f11_pressed_during_resize = False
            else:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            base_sizes = _compute_base_sizes((event.w, event.h))
            camera = Camera(world_bounds, screen.get_size())

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
            last_mouse_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
            last_mouse_pos = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging and last_mouse_pos is not None:
                mx, my = event.pos
                lx, ly = last_mouse_pos
                camera.pan_pixels(mx - lx, my - ly)
                last_mouse_pos = event.pos

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    windowed_size = screen.get_size()
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)

                f11_pressed_during_resize = True
                base_sizes = _compute_base_sizes(screen.get_size())
                camera = Camera(world_bounds, screen.get_size())

            elif event.key == pygame.K_r:
                camera = Camera(world_bounds, screen.get_size())
                display_scale = 1.0
                car_path_index = 0

        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                camera.change_zoom(1.1, pygame.mouse.get_pos(), screen_size)
            elif event.y < 0:
                camera.change_zoom(1 / 1.1, pygame.mouse.get_pos(), screen_size)

    return (
        running,
        screen,
        camera,
        base_sizes,
        display_scale,
        fullscreen,
        f11_pressed_during_resize,
        windowed_size,
        dragging,
        last_mouse_pos,
        car_path_index,
    )


def _update_display_scale(display_scale: float, dt: float) -> float:
    """
    Update visual scale using keyboard arrows.

    Note:
        This is a visual-only multiplier. If you want strictly physical sizes,
        remove display_scale from car/cone size computations.
    """
    keys = pygame.key.get_pressed()
    scale_speed = 2.0 * dt

    if keys[pygame.K_UP] or keys[pygame.K_RIGHT]:
        display_scale += scale_speed
    if keys[pygame.K_DOWN] or keys[pygame.K_LEFT]:
        display_scale -= scale_speed

    return max(0.1, min(display_scale, 10.0))


def _draw_path_and_car(
    *,
    screen: pygame.Surface,
    camera: Camera,
    path: Sequence[Point2D],
    screen_size: Tuple[int, int],
    display_scale: float,
    car_path_index: int,
) -> int:
    """
    Draw the path and animate a car along it, WITHOUT any orientation.
    The car is always drawn with a fixed rotation.

    The car size is defined in world units (CAR_SIZE_M × CAR_SIZE_M).

    Returns:
        Updated car index along the path.
    """
    screen_points = [camera.world_to_screen(x, y, screen_size) for (x, y) in path]

    if len(screen_points) >= 2:
        pygame.draw.lines(screen, COLORS["path_line"], False, screen_points, 3)
        pygame.draw.aalines(screen, COLORS["path_line"], False, screen_points)

    if 0 <= car_path_index < len(path):
        cx, cy = path[car_path_index]
        scx, scy = camera.world_to_screen(cx, cy, screen_size)

        # Convert world size (meters) to pixels: Zoom is px per world unit
        car_px = int(CAR_SIZE_M * camera.zoom * display_scale)
        car_px = max(4, car_px)  # safety clamp

        car_w = car_px
        car_l = car_px

        car_surf = _make_car_surface(car_l, car_w)
        rect = car_surf.get_rect(center=(scx, scy))
        screen.blit(car_surf, rect)

        car_path_index += 1
        if car_path_index >= len(path):
            car_path_index = 0

    return car_path_index


def _make_car_surface(length_px: int, width_px: int) -> pygame.Surface:
    """
    Build a simple car surface (rectangle + front highlight).
    The front is always on the right side of the screen.
    """
    surf = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
    pygame.draw.rect(surf, COLORS["car_body"], (0, 0, length_px, width_px))
    pygame.draw.rect(
        surf,
        COLORS["car_front"],
        (length_px * 0.7, 0, length_px * 0.3, width_px),
    )
    return surf


def _draw_cones(
    *,
    screen: pygame.Surface,
    camera: Camera,
    cones: Sequence[Cone],
    screen_size: Tuple[int, int],
    base_cone_radius_px: float,
    base_start_radius_px: float,
    display_scale: float,
) -> None:
    """
    Draw all cones on screen.
    """
    for cone in cones:
        tag = cone["tag"]
        wx = cone["x"]
        wy = cone["y"]

        sx, sy = camera.world_to_screen(wx, wy, screen_size)
        color = COLORS.get(tag, (200, 200, 200))

        base_r = base_start_radius_px if tag == "car_start" else base_cone_radius_px
        final_radius = int(base_r * (camera.zoom / camera.base_zoom) * display_scale)
        radius = max(2, final_radius)

        pygame.draw.circle(screen, color, (sx, sy), radius)


def _draw_hud(screen: pygame.Surface) -> None:
    """
    Draw a small HUD with usage help.
    """
    font = pygame.font.SysFont("Arial", 20)
    txt = font.render(
        "Zoom: wheel | Pan: left-drag | Scale: arrows | Reset: R",
        True,
        (200, 200, 200),
    )
    screen.blit(txt, (10, 10))
