"""
scripts/record_gif.py

Record the closed loop to a GIF without opening a window.

Pygame renders to an off-screen surface (SDL "dummy" video driver), each frame
is written as a PNG and ffmpeg assembles the GIF. The left panel shows the
whole track, the right panel follows the car.

Usage (from the repository root, ffmpeg on the PATH):
    python scripts/record_gif.py --track belgium --out docs/closed_loop.gif
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

from closed_loop.loop import ClosedLoop, Frame, LoopConfig  # noqa: E402
from closed_loop.track import load_start_pose, track_path  # noqa: E402
from closed_loop.vehicle import body_center  # noqa: E402
from closed_loop.viewer import (  # noqa: E402
    TRAIL_POINTS,
    draw_scene,
    draw_text,
    follow_camera,
    status_lines,
)
from simulation.camera import Camera  # noqa: E402
from simulation.vision import DEFAULT_VISION, VisionConfig  # noqa: E402
from track_utils import compute_world_bounds, load_track  # noqa: E402

Point2D = Tuple[float, float]
SEPARATOR = (70, 70, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--track", default="belgium")
    parser.add_argument("--range", type=float, default=DEFAULT_VISION.range_m, dest="range_m")
    parser.add_argument("--fov", type=float, default=DEFAULT_VISION.fov_deg, dest="fov_deg")
    parser.add_argument("--laps", type=int, default=1, help="valid laps to record")
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed factor")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument("--follow-span", type=float, default=16.0,
                        help="metres shown across the follow panel")
    parser.add_argument("--colors", type=int, default=48, help="GIF palette size")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "closed_loop.gif")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on the PATH", file=sys.stderr)
        return 1

    path = track_path(args.track)
    cones = load_track(str(path))
    start = load_start_pose(path)
    config = LoopConfig(
        vision=VisionConfig(range_m=args.range_m, fov_deg=args.fov_deg),
        target_laps=args.laps,
    )
    loop = ClosedLoop(cones, start, config)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    font = pygame.font.Font(None, 20)

    follow_w = int(args.width * 0.36)
    track_size = (args.width - follow_w - 1, args.height)
    track_view = pygame.Surface(track_size)
    follow_view = pygame.Surface((follow_w, args.height))
    track_camera = Camera(compute_world_bounds(cones, margin=2.5), track_size)

    frames_dir = ROOT / "build" / "gif_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    steps_per_frame = max(1, round(args.speed / (args.fps * config.dt)))
    trail: Deque[Point2D] = deque(maxlen=TRAIL_POINTS)
    frame: Optional[Frame] = None
    index = 0
    hold = args.fps  # one second on the final picture

    while hold > 0:
        if loop.status is None:
            for _ in range(steps_per_frame):
                frame = loop.step()
                trail.append(body_center(loop.state, config.vehicle))
                if loop.status is not None:
                    break
        else:
            hold -= 1

        center = body_center(loop.state, config.vehicle)
        draw_scene(track_view, track_camera, cones, loop, frame, trail)
        draw_scene(
            follow_view,
            follow_camera(cones, follow_view.get_size(), center, args.follow_span),
            cones,
            loop,
            frame,
            trail,
        )
        screen.blit(track_view, (0, 0))
        screen.blit(follow_view, (track_size[0] + 1, 0))
        pygame.draw.line(
            screen, SEPARATOR, (track_size[0], 0), (track_size[0], args.height), 1
        )
        draw_text(screen, font, status_lines(path.stem, loop), (8, 6))
        pygame.image.save(screen, str(frames_dir / f"{index:05d}.png"))
        index += 1

    pygame.quit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    palette = (
        f"split[a][b];[a]palettegen=max_colors={args.colors}:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "%05d.png"),
        "-vf", palette,
        str(args.out),
    ]
    subprocess.run(command, check=True)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    result = loop.result()
    size_kb = args.out.stat().st_size / 1024
    print(f"{args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}: "
          f"{index} frames, {size_kb:.0f} kB, run status {result.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
