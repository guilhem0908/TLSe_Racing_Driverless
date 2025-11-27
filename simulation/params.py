"""
simulation/params.py
Paramètres physiques complets pour le véhicule et l'optimisation.
"""

import numpy as np

# Paramètres du véhicule (Formula Student Type)
VEHICLE_PARAMS = {
    "mass": 230.0,          # kg
    "length": 2.5,          # m
    "width": 1.2,           # m
    "wheelbase": 1.53,      # m
    "track_width_front": 1.2,
    "track_width_rear": 1.2,
    "cog_z": 0.3,           # Centre de gravité hauteur
    "g": 9.81,
    "dragcoeff": 0.9,       # Aero drag
    "curvlim": 5.0,         # Limite de courbure physique
    "v_max": 35.0,          # m/s
    "acc_max": 10.0,        # m/s^2
    "dec_max": -12.0,       # m/s^2
}

# Options pour le lisseur de trajectoire (Splines)
REG_SMOOTH_OPTS = {
    "k_reg": 3,             # Ordre des splines (cubique)
    "s_reg": 10.0,          # Facteur de lissage
}

# Options pour l'optimisation (Quadprog / MinCurvature)
OPTIM_OPTS = {
    "width_opt": 1.2,       # Largeur du véhicule pour l'opti
    "safe_margin": 0.5,     # Marge de sécurité par rapport aux cônes
    "stepsize_prep": 1.0,   # Discrétisation de la piste (m)
    "stepsize_reg": 3.0,    # Pas pour la régularisation
    "eps_kappa": 1e-3,      # Tolérance courbure
}