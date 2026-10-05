"""
closed_loop/perception.py

Simulated cone sensor and short-term cone memory.

The sensor is the field-of-view model of ``simulation/vision.py``: a cone is
detected when ``point_in_vision_cone`` says it lies inside the circular sector
in front of the car. Detections carry the cone colour and a world position.
Placing them in the world frame uses the simulator's exact pose, so the memory
below never has to cope with odometry drift.

The memory keeps every cone for a fixed time after it was last seen. That is
what lets the car keep using the inside cones of a hairpin once they have left
the sector.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from simulation.vision import VisionConfig, point_in_vision_cone
from track_utils import Cone

Point2D = Tuple[float, float]

CONE_TAGS: Tuple[str, ...] = ("blue", "yellow", "big_orange")


@dataclass(frozen=True)
class Detection:
    """One cone reported by the sensor, in world coordinates."""

    tag: str
    x: float
    y: float


class ConeSensor:
    """
    Field-of-view cone sensor over a known map.

    Args:
        cones: Track cones as returned by ``track_utils.load_track``.
        vision: Range and opening angle of the sector.
        noise_std_m: Standard deviation of the Gaussian noise added to each
            detected position (0 disables it).
        seed: Seed of the noise generator.
    """

    def __init__(
        self,
        cones: Sequence[Cone],
        vision: VisionConfig,
        noise_std_m: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.vision = vision
        self.noise_std_m = noise_std_m
        self._rng = random.Random(seed)
        self._cones: List[Tuple[str, float, float]] = [
            (c["tag"], c["x"], c["y"]) for c in cones if c["tag"] in CONE_TAGS
        ]
        # Uniform grid so that a detection only tests the cones near the car.
        self._cell = max(vision.range_m, 1.0)
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        for index, (_, x, y) in enumerate(self._cones):
            self._grid.setdefault(self._key(x, y), []).append(index)

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return math.floor(x / self._cell), math.floor(y / self._cell)

    def visible_indices(self, sensor_pos: Point2D, heading_deg: float) -> List[int]:
        """Indices (into the cone list) of the cones inside the sector."""
        kx, ky = self._key(sensor_pos[0], sensor_pos[1])
        visible: List[int] = []
        for gx in (kx - 1, kx, kx + 1):
            for gy in (ky - 1, ky, ky + 1):
                for index in self._grid.get((gx, gy), ()):
                    _, x, y = self._cones[index]
                    if point_in_vision_cone((x, y), sensor_pos, heading_deg, self.vision):
                        visible.append(index)
        visible.sort()
        return visible

    def detect(self, sensor_pos: Point2D, heading_deg: float) -> List[Detection]:
        """
        Cones currently inside the sector.

        Args:
            sensor_pos: Sensor position in the world.
            heading_deg: Sensor heading in degrees.

        Returns:
            One detection per visible cone, with noise if configured.
        """
        detections: List[Detection] = []
        for index in self.visible_indices(sensor_pos, heading_deg):
            tag, x, y = self._cones[index]
            if self.noise_std_m > 0.0:
                x += self._rng.gauss(0.0, self.noise_std_m)
                y += self._rng.gauss(0.0, self.noise_std_m)
            detections.append(Detection(tag, x, y))
        return detections


@dataclass
class RememberedCone:
    """A cone held in memory."""

    tag: str
    x: float
    y: float
    last_seen: float
    observations: int = 1


class ConeMemory:
    """
    Short-term memory of detected cones.

    A detection is merged into a remembered cone of the same colour if one lies
    within ``merge_radius_m``; the stored position is then the running mean of
    the observations. Cones not seen for ``ttl_s`` seconds are forgotten.

    Args:
        ttl_s: Time a cone is kept after it was last seen. 0 keeps only the
            cones seen at the current instant (no memory).
        merge_radius_m: Association distance between a detection and a
            remembered cone.
        min_observations: Number of detections a cone needs before
            ``points`` reports it. Values above 1 filter out one-off outliers
            when the sensor is noisy.
    """

    _MAX_WEIGHT = 20

    def __init__(
        self, ttl_s: float = 6.0, merge_radius_m: float = 0.5, min_observations: int = 3
    ) -> None:
        if merge_radius_m <= 0.0:
            raise ValueError("merge_radius_m must be positive")
        self.ttl_s = ttl_s
        self.merge_radius_m = merge_radius_m
        self.min_observations = min_observations
        self._cones: List[RememberedCone] = []

    def __len__(self) -> int:
        return len(self._cones)

    @property
    def cones(self) -> List[RememberedCone]:
        """Remembered cones (the list is owned by the memory)."""
        return self._cones

    def points(self, tag: str) -> List[Point2D]:
        """Positions of the confirmed remembered cones of one colour."""
        return [
            (c.x, c.y)
            for c in self._cones
            if c.tag == tag and c.observations >= self.min_observations
        ]

    def update(self, detections: Iterable[Detection], now: float) -> None:
        """
        Merge new detections and forget stale cones.

        Args:
            detections: Detections of the current instant.
            now: Current time in seconds.
        """
        for detection in detections:
            match = self._nearest(detection)
            if match is None:
                self._cones.append(
                    RememberedCone(detection.tag, detection.x, detection.y, now)
                )
                continue
            weight = min(match.observations, self._MAX_WEIGHT)
            match.x = (match.x * weight + detection.x) / (weight + 1)
            match.y = (match.y * weight + detection.y) / (weight + 1)
            match.observations += 1
            match.last_seen = now

        self._cones = [c for c in self._cones if now - c.last_seen <= self.ttl_s]

    def _nearest(self, detection: Detection) -> Optional[RememberedCone]:
        best: Optional[RememberedCone] = None
        best_d2 = self.merge_radius_m * self.merge_radius_m
        for cone in self._cones:
            if cone.tag != detection.tag:
                continue
            d2 = (cone.x - detection.x) ** 2 + (cone.y - detection.y) ** 2
            if d2 <= best_d2:
                best = cone
                best_d2 = d2
        return best
