#### Tracks

| Track | Blue / yellow cones | Centre line (m) | Width min / median / max (m) | Tightest bend radius (m) |
|---|---|---|---|---|
| `belgium` | 67 / 63 | 130.6 | 2.99 / 3.56 / 4.52 | 4.86 |
| `hairpins_increasing_difficulty` | 490 / 490 | 831.1 | 2.96 / 3.00 / 3.00 | 2.13 |
| `peanut` | 54 / 64 | 104.3 | 3.82 / 4.23 / 5.12 | 3.62 |
| `small_track` | 37 / 30 | 104.7 | 4.06 / 4.78 / 5.26 | 3.48 |

#### Sensor sweep (full controller)

| Sensor (range / field of view) | Track | Outcome | Valid laps | Best lap (s) | Mean lap (s) | Mean speed (m/s) | Cones hit per lap | Peak lateral acc. (m/s^2) |
|---|---|---|---|---|---|---|---|---|
| 4 m / 100 deg (repository default) | `belgium` | finished | 3 / 3 | 25.28 | 25.29 | 5.2 | 0.0 | 4.6 |
| 4 m / 100 deg (repository default) | `hairpins_increasing_difficulty` | finished | 3 / 3 | 161.30 | 161.40 | 5.2 | 11.0 | 6.3 |
| 4 m / 100 deg (repository default) | `peanut` | finished | 3 / 3 | 21.06 | 21.06 | 5.0 | 0.0 | 5.4 |
| 4 m / 100 deg (repository default) | `small_track` | finished | 3 / 3 | 25.68 | 25.78 | 4.1 | 0.0 | 4.5 |
| 7 m / 60 deg (first setting, November 2025) | `belgium` | finished | 3 / 3 | 21.04 | 21.10 | 6.2 | 0.0 | 4.5 |
| 7 m / 60 deg (first setting, November 2025) | `hairpins_increasing_difficulty` | left the track at 52 s | 0 / 3 | - | - | - | - (1 before stopping) | 4.6 |
| 7 m / 60 deg (first setting, November 2025) | `peanut` | finished | 3 / 3 | 18.96 | 19.04 | 5.5 | 0.0 | 5.0 |
| 7 m / 60 deg (first setting, November 2025) | `small_track` | finished | 3 / 3 | 17.72 | 17.77 | 5.9 | 0.0 | 5.0 |
| 8 m / 120 deg | `belgium` | finished | 3 / 3 | 20.32 | 20.39 | 6.4 | 0.0 | 4.3 |
| 8 m / 120 deg | `hairpins_increasing_difficulty` | finished | 3 / 3 | 130.08 | 130.29 | 6.4 | 11.0 | 5.1 |
| 8 m / 120 deg | `peanut` | finished | 3 / 3 | 18.86 | 18.87 | 5.5 | 0.0 | 4.3 |
| 8 m / 120 deg | `small_track` | finished | 3 / 3 | 16.80 | 16.86 | 6.2 | 0.0 | 4.9 |
| 12 m / 120 deg | `belgium` | finished | 3 / 3 | 19.54 | 19.61 | 6.7 | 0.0 | 4.3 |
| 12 m / 120 deg | `hairpins_increasing_difficulty` | finished | 3 / 3 | 121.12 | 121.43 | 6.8 | 11.0 | 5.1 |
| 12 m / 120 deg | `peanut` | finished | 3 / 3 | 18.88 | 18.88 | 5.5 | 0.0 | 4.6 |
| 12 m / 120 deg | `small_track` | finished | 3 / 3 | 16.20 | 16.29 | 6.4 | 0.0 | 4.7 |

#### Ablations (sensor 4 m / 100 deg)

| Variant | Track | Outcome | Valid laps | Best lap (s) | Cones hit per lap |
|---|---|---|---|---|---|
| full controller | `belgium` | finished | 3 / 3 | 25.28 | 0.0 |
| full controller | `hairpins_increasing_difficulty` | finished | 3 / 3 | 161.30 | 11.0 |
| full controller | `peanut` | finished | 3 / 3 | 21.06 | 0.0 |
| full controller | `small_track` | finished | 3 / 3 | 25.68 | 0.0 |
| no cone memory | `belgium` | finished | 3 / 3 | 25.36 | 0.0 |
| no cone memory | `hairpins_increasing_difficulty` | left the track at 77 s | 0 / 3 | - | - (1 before stopping) |
| no cone memory | `peanut` | finished | 3 / 3 | 21.24 | 3.0 |
| no cone memory | `small_track` | finished | 3 / 3 | 28.84 | 2.0 |
| no one-sided fallback | `belgium` | left the track at 29 s | 0 / 3 | - | - (1 before stopping) |
| no one-sided fallback | `hairpins_increasing_difficulty` | left the track at 28 s | 0 / 3 | - | - (1 before stopping) |
| no one-sided fallback | `peanut` | left the track at 8 s | 0 / 3 | - | - (1 before stopping) |
| no one-sided fallback | `small_track` | left the track at 32 s | 0 / 3 | - | - (0 before stopping) |
| detection noise, sigma 0.1 m | `belgium` | finished | 3 / 3 | 25.25 | 0.0 |
| detection noise, sigma 0.1 m | `hairpins_increasing_difficulty` | finished | 3 / 3 | 161.09 | 11.0 |
| detection noise, sigma 0.1 m | `peanut` | finished | 3 / 3 | 21.03 | 0.0 |
| detection noise, sigma 0.1 m | `small_track` | finished | 3 / 3 | 26.01 | 0.0 |
| detection noise, sigma 0.2 m | `belgium` | left the track at 28 s | 0 / 3 | - | - (1 before stopping) |
| detection noise, sigma 0.2 m | `hairpins_increasing_difficulty` | left the track at 54 s | 0 / 3 | - | - (4 before stopping) |
| detection noise, sigma 0.2 m | `peanut` | finished | 3 / 3 | 21.73 | 0.0 |
| detection noise, sigma 0.2 m | `small_track` | finished | 3 / 3 | 26.06 | 1.3 |

32 runs, 3387 s of simulated driving. Computing time: 385 s summed over the single-threaded runs, 98 s wall clock with 8 processes (Python 3.14.2, NumPy 2.3.5, Windows AMD64, 32 logical CPUs, no GPU).
