"""
closed_loop/loop.py

The closed loop itself: sense, remember, plan, steer, move, score.

``ClosedLoop.step`` runs one cycle and returns a ``Frame`` describing what the
car saw and decided, which the viewer draws. ``run`` iterates until the target
number of laps is reached, the car leaves the track or the time limit expires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from closed_loop.centerline import CenterLine, CenterlineParams, build_centerline
from closed_loop.perception import ConeMemory, ConeSensor, Detection
from closed_loop.pure_pursuit import (
    PursuitParams,
    lookahead_distance,
    lookahead_point,
    speed_target,
    steering_angle,
)
from closed_loop.referee import LapRecord, Referee
from closed_loop.track import StartPose
from closed_loop.vehicle import (
    VehicleParams,
    VehicleState,
    body_center,
    path_curvature,
    step,
)
from simulation.vision import DEFAULT_VISION, VisionConfig
from track_utils import Cone

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class LoopConfig:
    """
    Everything that defines a run.

    Attributes:
        vision: Field-of-view sensor model.
        vehicle: Vehicle geometry and limits.
        centerline: Centre-line builder tuning.
        pursuit: Path follower tuning.
        memory_ttl_s: How long a cone is remembered after it was last seen.
        merge_radius_m: Association distance of the cone memory.
        min_observations: Detections needed before a remembered cone is used.
        one_sided_fallback: Extend the line from cones seen on one side only.
        noise_std_m: Standard deviation of the detection noise.
        seed: Seed of the detection noise.
        dt: Simulation and control period.
        target_laps: Number of valid laps after which the run stops.
        max_time_s: Simulated time after which the run is abandoned.
    """

    vision: VisionConfig = DEFAULT_VISION
    vehicle: VehicleParams = VehicleParams()
    centerline: CenterlineParams = CenterlineParams()
    pursuit: PursuitParams = PursuitParams()
    memory_ttl_s: float = 6.0
    merge_radius_m: float = 0.5
    min_observations: int = 3
    one_sided_fallback: bool = True
    noise_std_m: float = 0.0
    seed: int = 0
    dt: float = 0.02
    target_laps: int = 3
    max_time_s: float = 900.0


@dataclass(frozen=True)
class Frame:
    """
    Snapshot of one cycle, for display and inspection.

    Attributes:
        t: Time at the end of the cycle.
        state: Vehicle state at the end of the cycle.
        sensor_pos: Where the sensor was when it looked.
        sensor_heading_deg: Sensor heading when it looked.
        detections: Cones seen during the cycle.
        remembered: Cones in memory after the cycle, as (tag, x, y).
        line: Centre line the car planned.
        target: Pure-pursuit lookahead point.
        speed_target: Speed the car was aiming for.
    """

    t: float
    state: VehicleState
    sensor_pos: Point2D
    sensor_heading_deg: float
    detections: List[Detection]
    remembered: List[Tuple[str, float, float]]
    line: CenterLine
    target: Point2D
    speed_target: float


@dataclass
class RunResult:
    """
    Summary of a finished run.

    Attributes:
        status: ``"finished"``, ``"off_course"`` or ``"timeout"``.
        laps: Every lap that was timed, valid or not.
        cones_hit_total: Cones touched over the whole run, a cone counting
            once per lap.
        sim_time_s: Simulated duration.
        distance_m: Distance travelled by the rear axle.
        max_speed_mps: Highest speed reached.
        max_lat_accel_mps2: Highest lateral acceleration of the model.
        track_gates: Number of reference gates of the track.
        track_length_m: Length of the reference centre line.
        off_course_at: (time, x, y) where the car left the track, if it did.
        hit_positions: Position of the cone of every counted contact.
    """

    status: str
    laps: List[LapRecord] = field(default_factory=list)
    cones_hit_total: int = 0
    sim_time_s: float = 0.0
    distance_m: float = 0.0
    max_speed_mps: float = 0.0
    max_lat_accel_mps2: float = 0.0
    track_gates: int = 0
    track_length_m: float = 0.0
    off_course_at: Optional[Tuple[float, float, float]] = None
    hit_positions: List[Point2D] = field(default_factory=list)

    @property
    def valid_laps(self) -> List[LapRecord]:
        """Laps that count."""
        return [lap for lap in self.laps if lap.valid]


class ClosedLoop:
    """
    One car on one track.

    Args:
        cones: Track map from ``track_utils.load_track``.
        start: Starting pose.
        config: Run configuration.
    """

    def __init__(
        self, cones: Sequence[Cone], start: StartPose, config: LoopConfig = LoopConfig()
    ) -> None:
        self.config = config
        self.sensor = ConeSensor(cones, config.vision, config.noise_std_m, config.seed)
        self.memory = ConeMemory(
            config.memory_ttl_s, config.merge_radius_m, config.min_observations
        )
        self.referee = Referee(cones, start, config.vehicle)

        # The starting position of the track file is where the body centre sits.
        offset = config.vehicle.center_offset_m
        hx, hy = start.direction
        self.state = VehicleState(
            x=start.x - offset * hx, y=start.y - offset * hy, yaw=start.heading_rad
        )
        self.t = 0.0
        self.distance_m = 0.0
        self.max_speed_mps = 0.0
        self.max_lat_accel_mps2 = 0.0

    @property
    def status(self) -> Optional[str]:
        """Final status once the run is over, otherwise None."""
        if self.referee.off_course:
            return "off_course"
        if self.referee.valid_laps >= self.config.target_laps:
            return "finished"
        if self.t >= self.config.max_time_s:
            return "timeout"
        return None

    def step(self) -> Frame:
        """Run one sense-plan-act cycle and advance the simulation."""
        cfg = self.config
        state = self.state

        # Sense.
        sensor_pos = body_center(state, cfg.vehicle)
        sensor_heading_deg = math.degrees(state.yaw)
        detections = self.sensor.detect(sensor_pos, sensor_heading_deg)

        # Remember.
        self.memory.update(detections, self.t)

        # Plan.
        line = build_centerline(
            self.memory.points("blue"),
            self.memory.points("yellow"),
            state.rear_axle,
            state.heading,
            cfg.centerline,
            allow_virtual=cfg.one_sided_fallback,
        )

        # Steer and choose a speed.
        lookahead = lookahead_distance(state.speed, cfg.pursuit)
        target = lookahead_point(line.points, line.end_direction, state.rear_axle, lookahead)
        steer_cmd = steering_angle(state.rear_axle, state.yaw, target, cfg.vehicle.wheelbase_m)

        ahead = line.points[1:]
        to_line = (
            math.hypot(ahead[0][0] - state.x, ahead[0][1] - state.y) if ahead else 0.0
        )
        steer_curvature = math.tan(steer_cmd) / cfg.vehicle.wheelbase_m
        v_target = speed_target(
            ahead, to_line, steer_curvature, cfg.vehicle.max_speed_mps, cfg.pursuit
        )
        accel_cmd = cfg.pursuit.speed_gain * (v_target - state.speed)

        # Move.
        new_state = step(state, steer_cmd, accel_cmd, cfg.dt, cfg.vehicle)
        t_new = self.t + cfg.dt

        # Score.
        self.referee.update(state, new_state, self.t, t_new)
        self.distance_m += math.hypot(new_state.x - state.x, new_state.y - state.y)
        self.max_speed_mps = max(self.max_speed_mps, new_state.speed)
        lat_accel = new_state.speed**2 * abs(
            path_curvature(new_state.steer, new_state.speed, cfg.vehicle)
        )
        self.max_lat_accel_mps2 = max(self.max_lat_accel_mps2, lat_accel)

        self.state = new_state
        self.t = t_new

        return Frame(
            t=t_new,
            state=new_state,
            sensor_pos=sensor_pos,
            sensor_heading_deg=sensor_heading_deg,
            detections=detections,
            remembered=[(c.tag, c.x, c.y) for c in self.memory.cones],
            line=line,
            target=target,
            speed_target=v_target,
        )

    def result(self) -> RunResult:
        """Summary of the run so far."""
        gates = self.referee.gates
        mids = [g.mid for g in gates]
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(mids, mids[1:] + mids[:1])
        )
        return RunResult(
            status=self.status or "running",
            laps=list(self.referee.laps),
            cones_hit_total=self.referee.cones_hit_total,
            sim_time_s=self.t,
            distance_m=self.distance_m,
            max_speed_mps=self.max_speed_mps,
            max_lat_accel_mps2=self.max_lat_accel_mps2,
            track_gates=len(gates),
            track_length_m=length,
            off_course_at=self.referee.off_course_at,
            hit_positions=list(self.referee.hit_positions),
        )


def run(cones: Sequence[Cone], start: StartPose, config: LoopConfig = LoopConfig()) -> RunResult:
    """
    Run the closed loop until it ends.

    Args:
        cones: Track map.
        start: Starting pose.
        config: Run configuration.

    Returns:
        The summary of the run.
    """
    loop = ClosedLoop(cones, start, config)
    while loop.status is None:
        loop.step()
    return loop.result()
