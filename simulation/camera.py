"""
camera.py

Simple 2D camera handling world ↔ screen conversion, mouse-centered zoom,
and panning in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


WorldBounds = Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)


def compute_fit_zoom(world_bounds: WorldBounds, screen_size: Tuple[int, int]) -> float:
    """
    Compute a zoom factor so that the world bounds fit inside the screen.

    Args:
        world_bounds: (min_x, max_x, min_y, max_y)
        screen_size: (width_px, height_px)

    Returns:
        Zoom factor in pixels per world unit.
    """
    min_x, max_x, min_y, max_y = world_bounds
    screen_w, screen_h = screen_size

    w_world = max_x - min_x
    h_world = max_y - min_y

    if w_world <= 0 or h_world <= 0:
        return 50.0  # reasonable fallback

    zoom_x = screen_w / w_world
    zoom_y = screen_h / h_world
    return min(zoom_x, zoom_y)


@dataclass
class Camera:
    """
    2D camera utility.

    Attributes:
        cx, cy: camera center in world coordinates.
        base_zoom: initial zoom chosen to fit the world into the screen.
        zoom: current zoom factor.
    """

    world_bounds: WorldBounds
    screen_size: Tuple[int, int]

    cx: float = 0.0
    cy: float = 0.0
    base_zoom: float = 1.0
    zoom: float = 1.0

    def __post_init__(self) -> None:
        """
        Initialize camera center and zoom after dataclass construction.
        """
        min_x, max_x, min_y, max_y = self.world_bounds

        self.cx = (min_x + max_x) / 2.0
        self.cy = (min_y + max_y) / 2.0

        self.base_zoom = compute_fit_zoom(self.world_bounds, self.screen_size)
        self.zoom = self.base_zoom

    def world_to_screen(
        self, wx: float, wy: float, screen_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Convert a world point (wx, wy) to screen coordinates.

        Args:
            wx: world x coordinate.
            wy: world y coordinate.
            screen_size: actual screen size (width, height).

        Returns:
            (sx, sy) in pixels.
        """
        sw, sh = screen_size
        sx = (wx - self.cx) * self.zoom + sw / 2.0
        sy = -(wy - self.cy) * self.zoom + sh / 2.0
        return int(sx), int(sy)

    def screen_to_world(
        self, sx: float, sy: float, screen_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Convert a screen point (sx, sy) to world coordinates.

        Args:
            sx: screen x coordinate in pixels.
            sy: screen y coordinate in pixels.
            screen_size: actual screen size (width, height).

        Returns:
            (wx, wy) in world units.
        """
        sw, sh = screen_size
        wx = (sx - sw / 2.0) / self.zoom + self.cx
        wy = -(sy - sh / 2.0) / self.zoom + self.cy
        return wx, wy

    def change_zoom(
        self,
        factor: float,
        mouse_pos: Tuple[int, int],
        screen_size: Tuple[int, int],
    ) -> None:
        """
        Change zoom while keeping the world point under the mouse stable.

        Args:
            factor: multiplicative zoom factor.
            mouse_pos: mouse position on screen (x, y).
            screen_size: actual screen size (width, height).
        """
        if factor == 0:
            return

        before = self.screen_to_world(mouse_pos[0], mouse_pos[1], screen_size)

        new_zoom = self.zoom * factor
        min_zoom = self.base_zoom * 0.1
        max_zoom = self.base_zoom * 10.0
        self.zoom = max(min_zoom, min(max_zoom, new_zoom))

        after = self.screen_to_world(mouse_pos[0], mouse_pos[1], screen_size)

        # Shift center so the mouse keeps pointing at the same world location
        self.cx += before[0] - after[0]
        self.cy += before[1] - after[1]

    def pan_pixels(self, dx_px: float, dy_px: float) -> None:
        """
        Pan the camera by a delta expressed in pixels.

        Args:
            dx_px: horizontal pixel delta (positive means drag right).
            dy_px: vertical pixel delta (positive means drag down).
        """
        self.cx -= dx_px / self.zoom
        self.cy += dy_px / self.zoom
