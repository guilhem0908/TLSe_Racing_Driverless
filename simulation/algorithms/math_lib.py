"""
simulation/algorithms/math_lib.py
Adaptation de math_utils.py pour l'intégration projet.
Contient: Splines, Calcul de Courbure, Optimisation QP.
"""

import numpy as np
import math
import quadprog
from scipy import interpolate, optimize, spatial

def calc_splines(path):
    """Calcule les coefficients des splines cubiques pour un chemin donné."""
    # Version simplifiée et robuste de ta fonction calc_splines
    # On suppose que le chemin est [x, y]
    t = np.arange(path.shape[0])
    cs_x = interpolate.CubicSpline(t, path[:, 0], bc_type='natural')
    cs_y = interpolate.CubicSpline(t, path[:, 1], bc_type='natural')
    return cs_x, cs_y

def calc_head_curv_num(path, stepsize=1.0):
    """Calcul numérique du heading (psi) et de la courbure (kappa)."""
    # Calcul des tangentes par différences finies
    diffs = np.diff(path, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.unwrap(headings) # Gestion des sauts +/- pi

    # Calcul de la courbure : d(heading)/ds
    dists = np.linalg.norm(diffs, axis=1)
    d_headings = np.diff(headings)
    # Ajustement de taille pour matcher la longueur
    kappa = np.zeros(len(path))
    # Approximation simple centrée
    if len(dists) > 1:
        kappa[1:-1] = d_headings / dists[:-1]

    return headings, kappa

def opt_min_curv(reftrack, w_veh, kappa_bound):
    """
    Résout le problème d'optimisation quadratique (QP) pour minimiser la courbure.
    C'est le coeur de 'l'algo de fou'.

    Args:
        reftrack: np.array [[x, y, w_right, w_left]]
        w_veh: Largeur véhicule
        kappa_bound: Limite physique courbure

    Returns:
        alpha_opt: Déviation latérale optimale par rapport à la centerline.
    """
    no_points = reftrack.shape[0]

    # 1. Construction des vecteurs normaux
    diffs = np.diff(reftrack[:, :2], axis=0)
    tangents = np.vstack([diffs, diffs[-1]]) # Répéter le dernier
    norms = np.linalg.norm(tangents, axis=1)
    tangents /= norms[:, None]
    # Normale = rotation 90 deg de la tangente
    normvectors = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # 2. Formulation QP : 1/2 x^T H x + f^T x
    # On veut minimiser la dérivée seconde de la déviation (lissage ~ min courbure)
    # H (Hessienne) : Matrice de différences finies du second ordre
    H = np.eye(no_points) * 2
    H -= np.eye(no_points, k=1)
    H -= np.eye(no_points, k=-1)
    # On renforce la diagonale pour la stabilité numérique
    H = np.dot(H.T, H) + np.eye(no_points) * 0.01

    f = np.zeros(no_points) # Pas de terme linéaire dominant

    # 3. Contraintes (Limites de piste)
    # reftrack[:, 2] est w_right, reftrack[:, 3] est w_left
    # La solution alpha doit être : -w_right + w_veh/2 < alpha < w_left - w_veh/2

    margin = w_veh / 2.0
    lb = -reftrack[:, 2] + margin
    ub = reftrack[:, 3] - margin

    # Vérification de faisabilité basique
    if np.any(lb > ub):
        print("WARN: La piste est trop étroite pour le véhicule à certains endroits !")
        # Fix temporaire pour éviter le crash : élargir virtuellement
        mask = lb > ub
        ub[mask] = lb[mask] + 0.1

    # Quadprog attend : min 1/2 x^T G x - a^T x
    # s.t. C^T x >= b

    # Construction de C et b pour Quadprog
    # Contraintes bornées: I*x >= lb  ET  -I*x >= -ub
    C = np.vstack([np.eye(no_points), -np.eye(no_points)]).T
    b = np.hstack([lb, -ub])

    # Résolution
    try:
        # quadprog.solve_qp(G, a, C, b)
        alpha_opt = quadprog.solve_qp(H, -f, C, b)[0]
    except ValueError as e:
        print(f"QP Solver Error: {e}. Fallback to centerline.")
        alpha_opt = np.zeros(no_points)

    return alpha_opt, normvectors