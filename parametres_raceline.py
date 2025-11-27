"""
parametres_raceline.py

Définition du dictionnaire `pars` attendu par core_logic.run_optimization.

On met ici :
    - veh_params : paramètres simples du véhicule (masse, drag, etc.)
    - vehicle_params_mintime : paramètres pour le modèle dynamique min-time
    - tire_params_mintime : paramètres pneus simplifiés
    - optim_opts : options d’optimisation (mu, largeur, etc.)
    - pwr_params_mintime : modèle puissance/thermique (désactivé pour l’instant)
    - reg_smooth_opts / stepsize_opts / curv_calc_opts / vel_calc_opts :
      options de préparation de piste, courbure, profil vitesse…

C’est volontairement un peu générique mais complet pour que core_logic.py
ne plante plus en min-time. Tu pourras ensuite affiner les valeurs
en fonction de ton vrai modèle véhicule.
"""

from __future__ import annotations
from typing import Any, Dict
import math


def construire_parametres() -> Dict[str, Any]:
    pars: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # 1) Paramètres véhicule "généraux"
    # ---------------------------------------------------------
    pars["veh_params"] = {
        "g": 9.81,
        "mass": 300.0,
        "dragcoeff": 0.8,
        "curvlim": 40.0,
        "width": 1.5,  # <<< cohérent
        "length": 3.0,  # <<< cohérent
        "v_max": 40.0,
    }

    # ---------------------------------------------------------
    # 2) Paramètres modèle dynamique min-time
    # ---------------------------------------------------------
    pars["vehicle_params_mintime"] = {
        # géométrie
        "wheelbase": 1.6,          # empattement [m]
        "wheelbase_front": 0.8,    # CG -> essieu avant [m]
        "wheelbase_rear": 0.8,     # CG -> essieu arrière [m]
        "track_width_front": 1.2,  # voie avant [m]
        "track_width_rear": 1.2,   # voie arrière [m]
        "cog_z": 0.25,             # hauteur du CG [m]

        # aérodynamique (downforce simplifié)
        "liftcoeff_front": 0.0,
        "liftcoeff_rear": 0.0,

        # inertie / roulis
        "I_z": 150.0,              # inertie en lacet [kg.m²]
        "k_roll": 0.5,             # répartition roulis (50/50)

        # répartition traction / freinage
        "k_drive_front": 0.0,      # 0 = RWD, 1 = FWD
        "k_brake_front": 0.6,      # 60% frein à l’avant

        # limites actuateurs
        "delta_max": math.radians(35.0),  # braquage max [rad]
        "f_drive_max": 7500.0,            # traction max [N]
        "f_brake_max": 15000.0,           # freinage max [N]
        "power_max": 80000.0,             # puissance max [W]

        # temps caractéristiques actuateurs
        "t_delta": 0.3,   # s pour aller de -delta_max à +delta_max
        "t_drive": 0.3,   # s pour aller de 0 à f_drive_max
        "t_brake": 0.3,   # s pour aller de 0 à f_brake_max
    }

    # ---------------------------------------------------------
    # 3) Pneus pour le min-time (modèle simplifié mais complet)
    # ---------------------------------------------------------
    pars["tire_params_mintime"] = {
        # résistance au roulement
        "c_roll": 0.015,

        # charge nominale par roue (N) pour le modèle Pacejka
        # (exemple : ~1500 N, à adapter à ton véhicule si tu as la vraie valeur)
        "f_z0": 1500.0,

        # paramètres "Pacejka-like" avant
        "B_front": 10.0,
        "C_front": 1.3,
        "E_front": 0.97,
        "eps_front": 0.0,   # correction de mu en fonction de la charge (0 pour simplifier)

        # paramètres "Pacejka-like" arrière
        "B_rear": 12.0,
        "C_rear": 1.3,
        "E_rear": 0.97,
        "eps_rear": 0.0,    # idem
    }


    # ---------------------------------------------------------
    # 4) Options d’optimisation
    # ---------------------------------------------------------
    pars["optim_opts"] = {
        # fraction de largeur de piste utilisable [0..1]
        "width_opt": 0.8,

        # sampling non régulier (désactivé pour l’instant)
        "step_non_reg": 0,
        "eps_kappa": 0.01,

        # friction globale
        "mue": 1.3,

        # options de sécurité (désactivées au début)
        "safe_traj": False,
        "ax_pos_safe": 4.0,     # m/s²
        "ax_neg_safe": -6.0,    # m/s²
        "ay_safe": 8.0,         # m/s²

        # contrainte énergie (désactivée)
        "limit_energy": False,
        "energy_limit": 1e6,    # Wh

        # pénalités (si tu veux lisser les commandes)
        "penalty_F": 0.0,
        "penalty_delta": 0.0,

        # warm start / reopt
        "warm_start": False,    # on désactive pour commencer
        "iqp_curverror_allowed": 0.01,
        "w_tr_reopt": 0.8,
        "w_veh_reopt": 0.8,
        "iqp_iters_min": 3,     # valeur mini d’itérations pour l’IQP
    }

    # ---------------------------------------------------------
    # 5) Modèle puissance / thermique (désactivé => on met juste les clés)
    # ---------------------------------------------------------
    pars["pwr_params_mintime"] = {
        "pwr_behavior": False,   # on ne tient pas compte de la thermique / batterie

        # Ces clés ne sont lues que si pwr_behavior == True,
        # mais on les laisse complètes au cas où tu les actives plus tard.
        "T_mot_ini": 40.0,
        "T_batt_ini": 30.0,
        "T_inv_ini": 30.0,
        "T_cool_mi_ini": 30.0,
        "T_cool_b_ini": 30.0,
        "SOC_ini": 0.8,
    }

    # ---------------------------------------------------------
    # 6) Lissage / discrétisation de la piste
    # ---------------------------------------------------------
    pars["reg_smooth_opts"] = {
        "k_reg": 3,       # spline cubique
        "s_reg": 10.0,    # lissage (plus grand = plus lisse)
    }

    pars["stepsize_opts"] = {
        "stepsize_prep": 1.0,                # pas pour l’interpolation de la piste brute
        "stepsize_reg": 3.0,                 # pas pour la piste régularisée
        "stepsize_interp_after_opt": 1.0,    # pas pour la trajectoire finale
    }

    # ---------------------------------------------------------
    # 7) Options de calcul de courbure / angle de cap
    # ---------------------------------------------------------
    pars["curv_calc_opts"] = {
        # distance (en mètres) utilisée pour le calcul de l’angle de cap
        "d_preview_head": 5.0,
        "d_review_head": 5.0,

        # distance (en mètres) utilisée pour le calcul de la courbure
        "d_preview_curv": 2.0,
        "d_review_curv": 2.0,
    }

    # ---------------------------------------------------------
    # 8) Options pour le profil de vitesse (utile si on n’est pas en mintime)
    # ---------------------------------------------------------
    pars["vel_calc_opts"] = {
        "vel_profile_conv_filt_window": 15,  # fenêtre borne pour conv_filt (doit être impaire)
        "dyn_model_exp": 1.5,
    }

    return pars
