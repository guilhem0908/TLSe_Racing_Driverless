from track_utils import load_track, compute_world_bounds
from simulation.main_simulation import process_pygame

def main():
    csv_path = "tracks/belgium.csv"
    cones = load_track(csv_path)
    world_bounds = compute_world_bounds(cones)

    process_pygame(cones, world_bounds, path=None)

if __name__ == "__main__":
    main()
