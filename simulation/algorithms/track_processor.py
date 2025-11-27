"""
simulation/algorithms/track_processor.py
Transforme des cônes non triés en une Centerline ordonnée avec largeurs.
Utilise un algorithme de plus proche voisin itératif (Zipper Method).
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Dict

def process_track_to_centerline(cones: List[Dict], start_pos: Tuple[float, float]) -> np.ndarray:
    """
    Prend les cônes bruts et retourne un tableau [x, y, w_right, w_left].
    """
    # 1. Séparation des cônes
    blues = np.array([[c['x'], c['y']] for c in cones if c['tag'] == 'blue'])
    yellows = np.array([[c['x'], c['y']] for c in cones if c['tag'] == 'yellow'])

    if len(blues) < 3 or len(yellows) < 3:
        print("Erreur: Pas assez de cônes pour générer une piste.")
        return np.array([])

    # 2. Trouver le départ (paire bleue/jaune la plus proche de la voiture)
    tree_b = KDTree(blues)
    tree_y = KDTree(yellows)

    _, idx_b_start = tree_b.query(start_pos)
    _, idx_y_start = tree_y.query(start_pos)

    # 3. Algorithme "Zipper" pour ordonner la piste
    # On part du départ et on avance point par point
    ordered_centerline = []
    ordered_widths = [] # [right, left]

    curr_b_idx = idx_b_start
    curr_y_idx = idx_y_start

    used_b = {curr_b_idx}
    used_y = {curr_y_idx}

    # Calcul du premier point central
    p_b = blues[curr_b_idx]
    p_y = yellows[curr_y_idx]
    mid = (p_b + p_y) / 2
    ordered_centerline.append(mid)

    # Largeurs initiales
    w_r = np.linalg.norm(p_y - mid) # Jaune à droite
    w_l = np.linalg.norm(p_b - mid) # Bleu à gauche
    ordered_widths.append([w_r, w_l])

    # Boucle principale (Tant qu'on trouve des cônes proches non visités)
    max_iter = len(blues) + len(yellows)
    for _ in range(max_iter):
        # Trouver le prochain bleu le plus proche du bleu actuel
        dists_b, idxs_b = tree_b.query(blues[curr_b_idx], k=5)
        next_b = None
        for idx in idxs_b:
            if idx not in used_b:
                next_b = idx
                break

        # Trouver le prochain jaune
        dists_y, idxs_y = tree_y.query(yellows[curr_y_idx], k=5)
        next_y = None
        for idx in idxs_y:
            if idx not in used_y:
                next_y = idx
                break

        # Critère d'arrêt : on a fait le tour ou plus de suite
        if next_b is None or next_y is None:
            break

        # Mise à jour
        curr_b_idx = next_b
        curr_y_idx = next_y
        used_b.add(curr_b_idx)
        used_y.add(curr_y_idx)

        # Calcul nouveau point central
        p_b = blues[curr_b_idx]
        p_y = yellows[curr_y_idx]

        # Robustesse : vérifier que le midpoint n'est pas aberrant (croisement)
        mid = (p_b + p_y) / 2
        last_mid = ordered_centerline[-1]

        # Si le saut est trop grand (> 20m), c'est probablement une erreur ou la fin de piste
        if np.linalg.norm(mid - last_mid) > 20.0:
            break

        ordered_centerline.append(mid)
        w_r = np.linalg.norm(p_y - mid)
        w_l = np.linalg.norm(p_b - mid)
        ordered_widths.append([w_r, w_l])

    # 4. Conversion en format reftrack pour math_lib [x, y, w_right, w_left]
    center = np.array(ordered_centerline)
    widths = np.array(ordered_widths)

    # Smoothing basique (Moyenne mobile) pour éviter les zigzags du zipper
    window = 3
    if len(center) > window:
        kernel = np.ones(window) / window
        center[:, 0] = np.convolve(center[:, 0], kernel, mode='same')
        center[:, 1] = np.convolve(center[:, 1], kernel, mode='same')
        widths[:, 0] = np.convolve(widths[:, 0], kernel, mode='same') # w_right
        widths[:, 1] = np.convolve(widths[:, 1], kernel, mode='same') # w_left

    reftrack = np.column_stack((center, widths))
    return reftrack