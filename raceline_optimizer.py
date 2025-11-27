"""
raceline_optimizer.py

Intégration de l'algo de raceline / min-time (math_utils + core_logic)
dans ton projet TLSe_Racing_Driverless.

Idée :
    - on part des cônes (blue / yellow / big_orange / car_start) chargés
      par track_utils.load_track
    - on reconstruit deux bordures gauche/droite du circuit
    - on prend tous les points milieux = centre de piste + largeur locale
    - on écrit ce "reftrack" dans un CSV temporaire
    - on appelle core_logic.run_optimization(...) qui fait tout le gros boulot
    - on récupère la trajectoire optimisée et on la transforme en liste de (x, y)
      pour l'affichage Pygame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import os
from pathlib import Path

import numpy as np

import core_logic
import track_utils
from track_utils import Cone, Point2D


# ---------------------------------------------------------------------------
# Petites structures utilitaires
# ---------------------------------------------------------------------------
# --- paramètres géométriques de ta voiture ---
VOITURE_LARGEUR = 1.5   # m
MARGE_CONE = 0.25  # 25 cm  # m

demie_largeur_piste = 0.5 * largeur_piste   # centre -> bord
marge_centre = 0.5 * VOITURE_LARGEUR + MARGE_CONE
w_dispo = demie_largeur_piste - marge_centre

@dataclass
class BorduresCircuit:
    """Bordures géométriques du circuit dans le plan monde."""
    gauche: np.ndarray  # (N, 2)
    droite: np.ndarray  # (N, 2)


@dataclass
class RefTrack:
    """
    Représentation de la piste pour core_logic.import_track :

        [x_centre, y_centre, largeur_droite, largeur_gauche]
    """
    donnees: np.ndarray  # (N, 4)


# ---------------------------------------------------------------------------
# 1) Extraction / tri des cônes
# ---------------------------------------------------------------------------

def _extraire_cones_par_couleur(cones: Sequence[Cone], tag: str) -> List[Cone]:
    """Filtre une liste de cônes par tag."""
    return [c for c in cones if c["tag"] == tag]


def _chaine_voisins_proches(points: np.ndarray) -> np.ndarray:
    """
    Ordonne des points (N, 2) en une chaîne continue en utilisant
    un algo glouton "nearest neighbor".

    Ce n'est pas optimal mathématiquement, mais largement suffisant pour
    reconstruire le contour d'un circuit à partir de cônes.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n <= 2:
        return pts.copy()

    # point de départ : le plus proche de l'origine (0,0)
    d2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    idx_depart = int(np.argmin(d2))

    visite = np.zeros(n, dtype=bool)
    ordre: List[int] = []

    idx_courant = idx_depart
    for _ in range(n):
        visite[idx_courant] = True
        ordre.append(idx_courant)

        # distances vers les points non visités
        non_visites = np.where(~visite)[0]
        if non_visites.size == 0:
            break
        diff = pts[non_visites] - pts[idx_courant]
        d2_nv = np.sum(diff * diff, axis=1)
        idx_suivant = non_visites[int(np.argmin(d2_nv))]
        idx_courant = idx_suivant

    return pts[ordre]


def _ordonner_par_voisin_proche(points: np.ndarray) -> np.ndarray:
    """
    Ordonne des points (N, 2) en une chaîne continue avec un algo glouton.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n <= 2:
        return pts.copy()

    d2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    idx_depart = int(np.argmin(d2))

    visite = np.zeros(n, dtype=bool)
    ordre = []
    idx_courant = idx_depart
    for _ in range(n):
        visite[idx_courant] = True
        ordre.append(idx_courant)

        non_visites = np.where(~visite)[0]
        if non_visites.size == 0:
            break
        diff = pts[non_visites] - pts[idx_courant]
        d2_nv = np.sum(diff * diff, axis=1)
        idx_courant = non_visites[int(np.argmin(d2_nv))]

    return pts[ordre]


def _resampler_bordure(points: np.ndarray, nombre_points: int) -> np.ndarray:
    """
    Ordonne la bordure puis la ré-échantillonne sur un nombre fixe de points,
    espacés régulièrement en abscisse curviligne.
    """
    pts_ord = _ordonner_par_voisin_proche(points)
    # longueurs de segments
    seg = pts_ord[1:] - pts_ord[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    longueur_totale = s[-1]

    # grille régulière en s
    s_new = np.linspace(0.0, longueur_totale, nombre_points)

    x_new = np.interp(s_new, s, pts_ord[:, 0])
    y_new = np.interp(s_new, s, pts_ord[:, 1])
    return np.column_stack((x_new, y_new))


def _construire_bordures(cones: Sequence[Cone]) -> BorduresCircuit:
    """
    Construit des bordures gauche/droite propres en :
      - filtrant blue / yellow
      - ordonnant chaque bordure
      - ré-échantillonnant les deux sur la même grille en s.
    """
    cones_bleus = _extraire_cones_par_couleur(cones, "blue")
    cones_jaunes = _extraire_cones_par_couleur(cones, "yellow")

    if not cones_bleus or not cones_jaunes:
        raise ValueError(
            "Impossible de construire les bordures : il faut au moins un cône "
            "blue et un cône yellow dans le CSV."
        )

    pts_bleus = np.array([[c["x"], c["y"]] for c in cones_bleus], dtype=float)
    pts_jaunes = np.array([[c["x"], c["y"]] for c in cones_jaunes], dtype=float)

    # nombre de points commun aux 2 bordures (tu peux augmenter si tu veux plus de résolution)
    nombre_points = max(len(pts_bleus), len(pts_jaunes)) * 4

    bordure_gauche = _resampler_bordure(pts_bleus, nombre_points)
    bordure_droite = _resampler_bordure(pts_jaunes, nombre_points)

    return BorduresCircuit(gauche=bordure_gauche, droite=bordure_droite)


def _construire_reftrack(bordures: BorduresCircuit, start_pos: Point2D) -> RefTrack:
    """
    À partir des bordures ré-échantillonnées sur la même grille, on construit :
        centre = (gauche + droite) / 2

    Puis on définit w_tr_right / w_tr_left de manière à garantir que la
    voiture (largeur VOITURE_LARGEUR) reste à au moins MARGE_CONE des cônes.
    """
    gauche = np.asarray(bordures.gauche, dtype=float)
    droite = np.asarray(bordures.droite, dtype=float)

    if gauche.shape != droite.shape:
        raise ValueError("Les bordures gauche et droite doivent avoir la même forme (N, 2).")

    # centre de piste pur géométrique
    centre = 0.5 * (gauche + droite)

    # largeur physique de piste (distance entre les bordures)
    diff = droite - gauche
    largeur_piste = np.sqrt(np.sum(diff * diff, axis=1))      # [m]
    demie_largeur_piste = 0.5 * largeur_piste                 # centre -> bord

    # marge totale centre -> bord = (demi-largeur voiture + marge cônes)
    marge_centre = 0.5 * VOITURE_LARGEUR + MARGE_CONE         # ex : 0.75 + 0.20 = 0.95 m

    # demi-largeur disponible pour le centre de la voiture
    w_dispo = demie_largeur_piste - marge_centre

    # si jamais la piste est trop étroite, on clippe à un minimum
    w_dispo = np.clip(w_dispo, 0.05, None)

    w_droite = w_dispo.copy()
    w_gauche = w_dispo.copy()

    # on recale la piste autour de la position de départ
    sx, sy = start_pos
    distances2 = (centre[:, 0] - sx) ** 2 + (centre[:, 1] - sy) ** 2
    idx_start = int(np.argmin(distances2))

    centre = np.roll(centre, -idx_start, axis=0)
    w_droite = np.roll(w_droite, -idx_start)
    w_gauche = np.roll(w_gauche, -idx_start)

    # boucle fermée
    if not np.allclose(centre[0], centre[-1]):
        centre = np.vstack([centre, centre[0]])
        w_droite = np.concatenate([w_droite, [w_droite[0]]])
        w_gauche = np.concatenate([w_gauche, [w_gauche[0]]])

    reftrack = np.column_stack((centre, w_droite, w_gauche))
    return RefTrack(donnees=reftrack)


# ---------------------------------------------------------------------------
# 3) Passage reftrack -> core_logic.run_optimization
# ---------------------------------------------------------------------------

def _ecrire_reftrack_temp(reftrack: RefTrack, dossier_tmp: Path) -> str:
    """
    Sauvegarde le reftrack dans un CSV temporaire qui sera relu par
    core_logic.import_track à l'intérieur de core_logic.run_optimization.
    """
    dossier_tmp.mkdir(parents=True, exist_ok=True)
    chemin_csv = dossier_tmp / "reftrack_tmp.csv"
    np.savetxt(chemin_csv, reftrack.donnees, delimiter=",")
    return str(chemin_csv)


def calculer_trajectoire_optimisee_depuis_cones(
    cones: Sequence[Cone],
    pars: Dict,
    opt_type: str = "mintime",
    dossier_tmp: str | Path = "data/tmp_raceline",
) -> List[Point2D]:
    """
    Pipeline complet :
        cônes -> bordures -> reftrack -> core_logic.run_optimization -> path (x,y)

    Args:
        cones: liste de cônes (track_utils.load_track).
        pars: dictionnaire de paramètres pour l'algo (même structure que dans
              ton projet d'origine où tu utilisais déjà math_utils/core_logic).
        opt_type: type d'optimisation ('mintime', 'mincurv', etc.).
        dossier_tmp: dossier où écrire le CSV temporaire de reftrack.

    Returns:
        Liste de points (x, y) pour l'animation Pygame.
    """
    if not cones:
        raise ValueError("La liste de cônes est vide, impossible de calculer une raceline.")

    # 1) Bordures gauche / droite
    bordures = _construire_bordures(cones)

    # 2) Position de départ de la voiture (cone "car_start" si présent)
    pos_depart = track_utils.get_start_pos(cones)

    # 3) Reftrack pour core_logic
    reftrack = _construire_reftrack(bordures, pos_depart)

    # 4) Sauvegarde dans un CSV temporaire
    chemin_csv = _ecrire_reftrack_temp(reftrack, Path(dossier_tmp))

    # 5) Appel de ton gros algo CasADi
    #    imp_opts et mintime_opts = None -> core_logic utilisera ses valeurs par défaut.
    trajectoire_opt = core_logic.run_optimization(
        track_file_path=chemin_csv,
        pars=pars,
        opt_type=opt_type,
        imp_opts=None,
        mintime_opts=None,
    )

    trajectoire_opt = np.asarray(trajectoire_opt)

    if trajectoire_opt.ndim != 2 or trajectoire_opt.shape[1] < 3:
        raise RuntimeError(
            "La trajectoire retournée par core_logic.run_optimization "
            "n'a pas le format attendu (au moins 3 colonnes : s, x, y)."
        )

    # Dans la plupart des implémentations de ce genre d'algo :
    #   colonne 0 = abscisse curviligne s
    #   colonne 1 = x
    #   colonne 2 = y
    x = trajectoire_opt[:, 1]
    y = trajectoire_opt[:, 2]

    path: List[Point2D] = [(float(xi), float(yi)) for xi, yi in zip(x, y)]
    return path


def calculer_trajectoire_optimisee_depuis_csv(
    csv_path: str,
    pars: Dict,
    opt_type: str = "mintime",
    dossier_tmp: str | Path = "data/tmp_raceline",
) -> List[Point2D]:
    """
    Variante pratique : on part directement du chemin du CSV de cônes.

    Args:
        csv_path: chemin vers un fichier de la forme tracks/belgium.csv, etc.
        pars: dictionnaire de paramètres pour core_logic.
        opt_type: type d'optimisation.
        dossier_tmp: dossier où écrire le CSV temporaire.

    Returns:
        Liste de (x, y) pour Pygame.
    """
    cones = track_utils.load_track(csv_path)
    return calculer_trajectoire_optimisee_depuis_cones(
        cones=cones,
        pars=pars,
        opt_type=opt_type,
        dossier_tmp=dossier_tmp,
    )
