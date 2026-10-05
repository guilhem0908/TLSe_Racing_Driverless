"""
scripts/render_laps.py

Draw the path driven on every bundled track, with the cones that were hit.

One run per track with the default configuration (repository sensor, three
laps), rendered off-screen with pygame into ``docs/laps.png``. Every cone the
car touched is circled in white.

Usage (from the repository root):
    python scripts/render_laps.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

from closed_loop.loop import ClosedLoop, LoopConfig  # noqa: E402
from closed_loop.track import load_start_pose, track_path  # noqa: E402
from closed_loop.vehicle import body_center  # noqa: E402
from closed_loop.viewer import BACKGROUND, CONE_COLORS, OUTLINE, TEXT, TIMING_LINE  # noqa: E402
from simulation.camera import Camera  # noqa: E402
from simulation.vision import DEFAULT_VISION, VisionConfig  # noqa: E402
from track_utils import compute_world_bounds, load_track  # noqa: E402

PATH_COLOR = (255, 0, 255)
FRAME_COLOR = (70, 70, 70)
CAPTION_H = 46

# (track, panel rectangle as x, y, width, height) on a 1400 x 960 canvas
LAYOUT: List[Tuple[str, Tuple[int, int, int, int]]] = [
    ("belgium", (0, 0, 860, 300)),
    ("peanut", (0, 300, 860, 330)),
    ("small_track", (0, 630, 860, 330)),
    ("hairpins_increasing_difficulty", (860, 0, 540, 960)),
]


def render_panel(name: str, size: Tuple[int, int], config: LoopConfig, font) -> pygame.Surface:
    path = track_path(name)
    cones = load_track(str(path))
    loop = ClosedLoop(cones, load_start_pose(path), config)
    trail = [body_center(loop.state, config.vehicle)]
    while loop.status is None:
        loop.step()
        trail.append(body_center(loop.state, config.vehicle))
    result = loop.result()

    panel = pygame.Surface(size)
    panel.fill(BACKGROUND)
    view = (size[0], size[1] - CAPTION_H)
    camera = Camera(compute_world_bounds(cones, margin=2.0), view)

    def to_screen(point) -> Tuple[int, int]:
        sx, sy = camera.world_to_screen(point[0], point[1], view)
        return sx, sy + CAPTION_H

    cone_px = max(2, int(0.2 * camera.zoom))
    pygame.draw.line(
        panel, TIMING_LINE, to_screen(loop.referee.line[0]), to_screen(loop.referee.line[1]), 1
    )
    pygame.draw.aalines(panel, PATH_COLOR, False, [to_screen(p) for p in trail])
    for cone in cones:
        color = CONE_COLORS.get(cone["tag"])
        if color is not None:
            pygame.draw.circle(panel, color, to_screen((cone["x"], cone["y"])), cone_px)
    for hit in set(result.hit_positions):
        pygame.draw.circle(panel, OUTLINE, to_screen(hit), cone_px + 4, width=2)

    valid = result.valid_laps
    caption = f"{result.status}, {len(valid)} valid laps"
    if valid:
        best = min(lap.time_s for lap in valid)
        per_lap = sum(lap.cones_hit for lap in valid) / len(valid)
        caption += f", best {best:.2f} s, {per_lap:.1f} cones hit per lap"
    panel.blit(font.render(name, True, TEXT), (8, 5))
    panel.blit(font.render(caption, True, TEXT), (8, 5 + font.get_linesize()))
    pygame.draw.rect(panel, FRAME_COLOR, panel.get_rect(), 1)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--range", type=float, default=DEFAULT_VISION.range_m, dest="range_m")
    parser.add_argument("--fov", type=float, default=DEFAULT_VISION.fov_deg, dest="fov_deg")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "laps.png")
    args = parser.parse_args()

    config = LoopConfig(
        vision=VisionConfig(range_m=args.range_m, fov_deg=args.fov_deg), target_laps=args.laps
    )

    pygame.init()
    pygame.display.set_mode((1, 1))
    font = pygame.font.Font(None, 22)
    canvas = pygame.Surface((1400, 960))
    for name, (x, y, w, h) in LAYOUT:
        canvas.blit(render_panel(name, (w, h), config, font), (x, y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, str(args.out))
    pygame.quit()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
