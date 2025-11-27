"""
simulation/algorithms/planner.py
Orchestrateur de la génération de trajectoire.
"""

import numpy as np
from .math_lib import opt_min_curv, calc_splines
from .track_processor import process_track_to_centerline
from ..params import VEHICLE_PARAMS, OPTIM_OPTS

class PathPlanner:
    def __init__(self):
        self.reftrack = None # [x, y, w_r, w_l]
        self.trajectory = None
        self.norm_vecs = None
        self.alpha_opt = None

    def plan(self, cones, start_pos=(0,0)):
        """
        Exécute la pipeline complète : Cônes -> Centerline -> QP Optimization -> Trajectoire.
        """
        print("--- [Planner] Start Planning ---")

        # 1. Processing
        print(f"[Planner] Processing {len(cones)} cones...")
        self.reftrack = process_track_to_centerline(cones, start_pos)

        if self.reftrack.shape[0] < 10:
            print("[Planner] Failed: Track too short or generation failed.")
            return []

        print(f"[Planner] Centerline generated: {len(self.reftrack)} points.")

        # 2. Optimization (Minimum Curvature)
        print("[Planner] Running QP Optimization (Min Curvature)...")
        try:
            self.alpha_opt, self.norm_vecs = opt_min_curv(
                self.reftrack,
                w_veh=OPTIM_OPTS["width_opt"],
                kappa_bound=VEHICLE_PARAMS["curvlim"]
            )
        except Exception as e:
            print(f"[Planner] Optimization CRASHED: {e}")
            return self.reftrack[:, :2].tolist() # Fallback centerline

        # 3. Reconstruction de la trajectoire optimale
        # Trajectoire = Centerline + alpha * Normale
        raceline = self.reftrack[:, :2] + self.alpha_opt[:, None] * self.norm_vecs

        # 4. Lissage final (Splines pour rendu Pygame fluide)
        # On augmente la résolution pour la visualisation
        try:
            t_orig = np.arange(len(raceline))
            cs_x, cs_y = calc_splines(raceline)

            t_new = np.linspace(0, len(raceline)-1, len(raceline)*4) # 4x plus de points
            x_new = cs_x(t_new)
            y_new = cs_y(t_new)
            self.trajectory = np.column_stack((x_new, y_new))
        except Exception:
             self.trajectory = raceline

        print("[Planner] Trajectory Ready.")
        return self.trajectory