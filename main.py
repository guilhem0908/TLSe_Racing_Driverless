from track_utils import load_track, compute_world_bounds
# from simulation.main_simulation import process_pygame  <-- ANCIEN
from realtime import run_realtime                        # <-- NOUVEAU

def main():
    csv_path = "tracks/belgium.csv"  # Essayez aussi 'tracks/small_track.csv'
    cones = load_track(csv_path)
    world_bounds = compute_world_bounds(cones)

    # Lancer la simulation temps réel
    run_realtime(cones, world_bounds)

if __name__ == "__main__":
    main()