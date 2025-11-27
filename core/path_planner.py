import numpy as np
import math
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class RaceLineOptimizer:
    def __init__(self, cones, car_width=1.2, safety_margin=0.2):
        self.cones = cones
        self.car_width = car_width
        self.safety_dist = safety_margin

        # MARGE TOTALE = Demi-voiture + Sécurité
        # Si la voiture fait 1.2m, la demi-largeur est 0.6m.
        # Avec 0.2m de sécu, le CENTRE doit être à 0.8m du cône.
        self.effective_margin = (self.car_width / 2.0) + self.safety_dist

        self.blue_cones = np.array([[c['x'], c['y']] for c in cones if c['tag'] == 'blue'])
        self.yellow_cones = np.array([[c['x'], c['y']] for c in cones if c['tag'] == 'yellow'])

        # Création d'arbres de recherche rapide pour la distance exacte
        self.tree_blue = cKDTree(self.blue_cones) if len(self.blue_cones) > 0 else None
        self.tree_yellow = cKDTree(self.yellow_cones) if len(self.yellow_cones) > 0 else None

        self.centerline = []
        self.gates = []
        self.norm_vecs = []

    def dist_sq(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def extract_track_gates(self, max_edge_len=15.0, start_pos_hint=None, start_yaw_hint=None):
        if self.tree_blue is None or self.tree_yellow is None: return False

        # --- 1. Triangulation (inchangé) ---
        all_points = np.vstack((self.blue_cones, self.yellow_cones))
        n_blue = len(self.blue_cones)
        try:
            tri = Delaunay(all_points)
        except:
            return False
        gates_raw = []
        seen_edges = set()
        for simplex in tri.simplices:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                idx1, idx2 = simplex[i], simplex[j]
                is_blue1 = idx1 < n_blue
                is_blue2 = idx2 < n_blue
                if is_blue1 != is_blue2:
                    edge_id = tuple(sorted((idx1, idx2)))
                    if edge_id in seen_edges: continue
                    seen_edges.add(edge_id)
                    p1, p2 = all_points[idx1], all_points[idx2]
                    if self.dist_sq(p1, p2) < max_edge_len ** 2:
                        if is_blue1:
                            pb, py = p1, p2
                        else:
                            pb, py = p2, p1
                        gates_raw.append({'blue': pb, 'yellow': py, 'mid': (pb + py) / 2, 'vec': py - pb})
        if not gates_raw: return False

        # --- 2. Ordonnancement (inchangé) ---
        if start_pos_hint is None:
            ref_pos = np.mean([g['mid'] for g in gates_raw], axis=0)
        else:
            ref_pos = np.array(start_pos_hint)
        forward_vec = None
        if start_yaw_hint is not None: forward_vec = np.array([np.cos(start_yaw_hint), np.sin(start_yaw_hint)])

        curr_idx = min(range(len(gates_raw)), key=lambda i: self.dist_sq(gates_raw[i]['mid'], ref_pos))
        self.gates = [gates_raw.pop(curr_idx)]

        while gates_raw:
            last_pos = self.gates[-1]['mid']
            best_idx = -1
            best_score = float('inf')
            for i, g in enumerate(gates_raw):
                d2 = self.dist_sq(g['mid'], last_pos)
                if d2 > max_edge_len ** 2 * 2.5: continue
                score = d2
                if forward_vec is not None:
                    step_vec = g['mid'] - last_pos
                    norm = np.linalg.norm(step_vec)
                    if norm > 0.1 and np.dot(step_vec / norm, forward_vec) < -0.3: score = float('inf')
                if score < best_score:
                    best_score = score
                    best_idx = i
            if best_idx != -1 and best_score < float('inf'):
                self.gates.append(gates_raw.pop(best_idx))
            else:
                break
        return True

    def generate_spline_centerline(self, smoothing=0.0):
        if len(self.gates) < 2: return
        mids = np.array([g['mid'] for g in self.gates])

        # Nettoyage doublons
        unique_mids = [mids[0]]
        for i in range(1, len(mids)):
            if self.dist_sq(mids[i], unique_mids[-1]) > 0.05:
                unique_mids.append(mids[i])
        unique_mids = np.array(unique_mids)

        if len(unique_mids) < 2:
            self.centerline = unique_mids
            return

        try:
            k = 3 if len(unique_mids) > 3 else (2 if len(unique_mids) > 2 else 1)
            # s=0 force la ligne à passer EXACTEMENT au milieu des portes (sécurité max initiale)
            tck, u = splprep(unique_mids.T, s=0.0, k=k)
            total_len = np.sum(np.sqrt(np.sum(np.diff(unique_mids, axis=0) ** 2, axis=1)))
            num_points = max(5, int(total_len * 1.5))
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            self.centerline = np.column_stack((x_new, y_new))
        except:
            self.centerline = unique_mids
            return

        if len(self.centerline) < 2: return

        # Calcul Normales
        dx = np.gradient(self.centerline[:, 0])
        dy = np.gradient(self.centerline[:, 1])
        self.norm_vecs = []
        for i in range(len(self.centerline)):
            n = np.array([-dy[i], dx[i]])
            nm = np.linalg.norm(n)
            self.norm_vecs.append(n / nm if nm > 0 else n)
        self.norm_vecs = np.array(self.norm_vecs)

    def optimize_trajectory(self, weight_curv=1.0, weight_dist=2.0):
        if len(self.centerline) < 3: return self.centerline
        N = len(self.centerline)

        # --- NOUVEAU CALCUL DES BORNES (KD-Tree) ---
        # Pour chaque point de la centerline, on cherche la VRAIE distance au cône le plus proche
        # et on contraint le mouvement latéral pour ne jamais violer la marge.

        bounds = []
        for i in range(N):
            pt = self.centerline[i]
            norm = self.norm_vecs[i]

            # Distance au bleu le plus proche
            dist_b, _ = self.tree_blue.query(pt)
            # Distance au jaune le plus proche
            dist_y, _ = self.tree_yellow.query(pt)

            # Projection pour savoir si le cône est à gauche ou à droite de la normale
            # Astuce : On assume que la centerline initiale est "sûre" et au milieu.
            # Donc l'espace dispo vers le bleu est (dist_b - marge)
            # et vers le jaune est (dist_y - marge).

            # On identifie quel côté est "positif" selon la normale
            # On teste un point déplacé dans la direction de la normale
            pt_test = pt + 0.1 * norm
            dist_b_test, _ = self.tree_blue.query(pt_test)

            # Si on s'est rapproché du bleu, le bleu est du côté positif
            if dist_b_test < dist_b:
                # Normale pointe vers Bleu
                max_move_pos = dist_b - self.effective_margin
                max_move_neg = dist_y - self.effective_margin  # Vers jaune (sens opposé)

                # Bornes : [-vers_jaune, +vers_bleu]
                upper = max(0.0, max_move_pos)
                lower = -max(0.0, max_move_neg)
            else:
                # Normale pointe vers Jaune
                max_move_pos = dist_y - self.effective_margin
                max_move_neg = dist_b - self.effective_margin

                upper = max(0.0, max_move_pos)
                lower = -max(0.0, max_move_neg)

            # Sécurité piste trop étroite
            if upper < lower:
                upper = lower = 0.0  # On bouge pas, on reste au milieu

            bounds.append((lower, upper))

        def objective(alphas):
            pts = self.centerline + alphas[:, np.newaxis] * self.norm_vecs
            # Courbure (lissage)
            diff2 = pts[:-2] - 2 * pts[1:-1] + pts[2:]
            k_cost = np.sum(diff2 ** 2)
            # Distance (couper les virages)
            diff1 = pts[:-1] - pts[1:]
            d_cost = np.sum(diff1 ** 2)
            return weight_curv * k_cost + weight_dist * d_cost

        x0 = np.zeros(N)
        # On garde SLSQP car il respecte strictement les bornes
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'ftol': 1e-2, 'maxiter': 15, 'disp': False})

        final_alphas = res.x if res.success else x0
        return self.centerline + final_alphas[:, np.newaxis] * self.norm_vecs

    def get_smooth_path(self, start_pos_hint=None, start_yaw_hint=None):
        if not self.extract_track_gates(start_pos_hint=start_pos_hint, start_yaw_hint=start_yaw_hint):
            return []
        self.generate_spline_centerline()
        path = self.optimize_trajectory()
        return path.tolist() if len(path) > 0 else []


class RecedingHorizonPlanner:
    # ... (Le reste de la classe RecedingHorizonPlanner reste EXACTEMENT identique au code précédent) ...
    # ... (Copiez simplement la classe RecedingHorizonPlanner du message d'avant ici) ...
    # Pour être sûr, je la remets ici pour un copier-coller facile :

    def __init__(self, all_cones, car_width=1.2, safety_margin=0.2):
        self.all_cones = all_cones
        self.car_width = car_width
        self.safety_margin = safety_margin
        self.step_size = 0.5
        self.max_view_dist = 30.0
        self.fov_angle = np.deg2rad(160)
        self.memory = {}
        self.memory_duration = 80

    def get_visible_and_remembered_cones(self, car_pos, car_yaw, current_step):
        car_dir = np.array([np.cos(car_yaw), np.sin(car_yaw)])
        for c in self.all_cones:
            p = np.array([c['x'], c['y']])
            vec = p - car_pos
            dist = np.linalg.norm(vec)
            if dist > self.max_view_dist: continue
            if dist > 0.1:
                dot = np.dot(vec / dist, car_dir)
                angle = np.arccos(np.clip(dot, -1.0, 1.0))
                if angle < self.fov_angle / 2.0:
                    cone_key = (c['x'], c['y'], c['tag'])
                    self.memory[cone_key] = current_step
        active_cones = []
        keys_to_remove = []
        for key, last_seen in self.memory.items():
            if current_step - last_seen > self.memory_duration:
                keys_to_remove.append(key)
                continue
            cx, cy, tag = key
            dist_to_car = np.sqrt((cx - car_pos[0]) ** 2 + (cy - car_pos[1]) ** 2)
            if dist_to_car < self.max_view_dist * 1.5:
                active_cones.append({'x': cx, 'y': cy, 'tag': tag})
        for k in keys_to_remove: del self.memory[k]
        return active_cones

    def simulate_lap(self):
        print(f"Sim: Portée {self.max_view_dist}m | FOV {np.rad2deg(self.fov_angle):.0f}°")
        temp_opt = RaceLineOptimizer(self.all_cones)
        if not temp_opt.extract_track_gates(): return []
        start_gate = temp_opt.gates[0]
        car_pos = start_gate['mid']
        vec = start_gate['vec']
        car_yaw = np.arctan2(-vec[0], vec[1])
        if len(temp_opt.gates) > 1:
            diff = temp_opt.gates[1]['mid'] - car_pos
            if np.dot(diff, [np.cos(car_yaw), np.sin(car_yaw)]) < 0: car_yaw += np.pi
        full_path = [tuple(car_pos)]
        max_steps = 4000
        t0 = time.time()
        print("Calcul en cours...")
        for step in range(max_steps):
            if step % 100 == 0: print(f"Step {step} | Dist: {step * self.step_size:.0f}m")
            local_cones = self.get_visible_and_remembered_cones(car_pos, car_yaw, step)
            if len(local_cones) < 4:
                print(f"Arrêt (Pas assez de cônes).")
                break
            planner = RaceLineOptimizer(local_cones, self.car_width, self.safety_margin)
            path_segment = planner.get_smooth_path(start_pos_hint=car_pos, start_yaw_hint=car_yaw)
            if path_segment and len(path_segment) > 1:
                path_arr = np.array(path_segment)
                dists = np.sum((path_arr - car_pos) ** 2, axis=1)
                idx_closest = np.argmin(dists)
                if dists[idx_closest] > 4.0: break
                lookahead_idx = min(idx_closest + 8, len(path_segment) - 1)
                target = path_arr[lookahead_idx]
                move_vec = target - car_pos
                dist_move = np.linalg.norm(move_vec)
                if dist_move > 0.01:
                    direction = move_vec / dist_move
                    next_pos = car_pos + direction * self.step_size
                    target_yaw = np.arctan2(direction[1], direction[0])
                    ang_diff = (target_yaw - car_yaw + np.pi) % (2 * np.pi) - np.pi
                    car_yaw += ang_diff * 0.5
                else:
                    next_pos = car_pos
            else:
                next_pos = car_pos + np.array([np.cos(car_yaw), np.sin(car_yaw)]) * self.step_size
            car_pos = next_pos
            full_path.append(tuple(car_pos))
            if step > 300 and np.linalg.norm(car_pos - full_path[0]) < 5.0:
                print(f"Tour bouclé en {step} étapes !")
                break
        return full_path


def compute_limited_vision_path(cones, car_width=1.2, safety_margin=0.2):
    sim = RecedingHorizonPlanner(cones, car_width, safety_margin)
    return sim.simulate_lap()