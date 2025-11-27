from track_utils import load_track, compute_world_bounds
from simulation.main_simulation import process_pygame

from raceline_optimizer import calculer_trajectoire_optimisee_depuis_cones
from parametres_raceline import construire_parametres


def main():
    csv_path = "tracks/belgium.csv"
    cones = load_track(csv_path)
    world_bounds = compute_world_bounds(cones)

    pars = construire_parametres()

    # ICI : on passe à la vraie version min-time
    path = calculer_trajectoire_optimisee_depuis_cones(
        cones=cones,
        pars=pars,
        opt_type="mintime",    # <<< changée ici
    )

    process_pygame(cones, world_bounds, path=path)


if __name__ == "__main__":
    main()
