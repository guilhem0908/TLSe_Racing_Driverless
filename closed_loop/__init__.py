"""
closed_loop

Closed-loop driving in the 2D simulator, added in October 2026 on top of the
November 2025 simulation layer (``simulation/`` and ``track_utils.py``).

The car only knows the cones that the field-of-view sensor model of
``simulation/vision.py`` lets it see. From those it keeps a short-term memory,
pairs blue and yellow cones into a local centre line, and follows that line
with pure-pursuit steering on a kinematic bicycle model. A referee that knows
the full map times the laps and counts the cones the car touches.

Everything in this package is a 2D simulation: there is no tyre model, no
sensor noise unless requested, and the pose used to place detections in the
world frame is the simulator's own.
"""
