# Methodology

Working draft of the methodology chapter. Cross-references `docs/related_work.md` (what's adopted
vs. adapted vs. diverged from prior work) and `docs/decisions.md` (why each choice was made).

## 1. System Architecture (SA-TCP)

Two decoupled ResNet-based networks process the front-camera RGB frame:

- **Branch A — control network** (`EndtoEnd` in `run_iter.py`/`train.py`): ResNet18 backbone,
  concatenated with a small MLP over `(atan2(v_perp, v), sqrt(v**2 + v_perp**2))` (heading and
  magnitude of the velocity vector), outputting a raw steering command `steer_ref` with an
  MC-dropout uncertainty `steer_var`.
- **Branch B — state network** (`model_safety_1/2/3`): three independent ResNet18 heads
  predicting curvature, cross-track error (`x`), and heading error (`theta`) relative to the
  track centerline, each with MC-dropout uncertainty.
- **Safety gate — CBF** (`get_optimal_control()`): takes Branch A's suggestion and Branch B's
  state estimate, and searches for the steering angle closest to `steer_ref` that keeps the
  predicted trajectory inside the lane corridor. See Section 2.

Training (`train.py`) uses expert-autopilot-collected data (`RUN_NO == 0`), with an 80/10/10
train/val/test split by contiguous frame-block (not per-frame, since adjacent frames are
near-duplicates of the same driving moment) and early stopping on validation loss.

## 2. Control Barrier Function Safety Gate

Implemented in `get_optimal_control()`, `run_iter.py`. Full line-by-line comparison against
Kalaria et al. (2023) is in `docs/related_work.md` — summary here:

- Two barrier functions bound cross-track error on each side of the lane:
  `h_left = LANE_WIDTH/2 - x`, `h_right = LANE_WIDTH/2 + x`.
- A higher-order CBF (HOCBF) condition is enforced on `h` and its time-derivatives under the
  bicycle-model dynamics, using a shared K-class decay constant `lambda_`.
- Model uncertainty (`x_var`, `theta_var`, `curvature_var` from Branch B) is propagated into the
  barrier constraint and used to tighten it by a Gaussian-quantile safety margin
  (`norm.ppf(1-BETA)`), turning it into a chance constraint.
- The exact QP `argmin ||u - u_ref||^2 s.t. barrier >= 0` is approximated by a brute-force grid
  search over candidate steering angles (`np.arange(-MAX_STEER, MAX_STEER, 0.01)`) with a
  quadratic penalty on constraint violation.
- **Known open issue** (see `docs/decisions.md`, 2026-08-16): the current implementation's
  relative-degree-3 branch and several coefficients do not match a correct derivation (from either
  our own dynamics or the reference paper's). Not yet resolved — flagged for future work.

## 3. Track Environments

### Town04
CARLA's built-in highway map. Route defined by `town04_waypoints.txt` (1m-spaced waypoints walked
from a fixed spawn point via `next(1.0)` on the map's road graph — see `generate_town04.py`).

### Austin (COTA)
Custom track built from real-world centerline+width data (TUMFTM/racetrack-database, see
`docs/related_work.md`), since no CARLA map corresponds to a real F1 circuit and no existing
converter does this.

**Pipeline** (`tools/track_to_opendrive.py`):
1. Source: `racetrack_source/Austin.csv` — `x_m, y_m, w_tr_right_m, w_tr_left_m`, 1102 points,
   ~5m uniform spacing, closed loop, real-world scale (~5.5km lap, 11-27.6m width range).
2. `planView`: one `<geometry>` line segment per consecutive point pair, each with its own
   explicit `x, y, hdg` (heading = `atan2(dy, dx)`) — avoids needing spline-continuity math, since
   every point already carries its own pose.
3. `lanes`: single driving lane, width following the track's actual per-point corridor width
   (`w_tr_left + w_tr_right`) via a `<width sOffset a b c d>` breakpoint at every point.
4. The single `<road>` self-links (predecessor/successor both pointing at its own road id) to
   form a closed loop, matching the source data's topology.
5. Loaded live via `client.generate_opendrive_world()` — no UE4 rebuild, no map content package.

**Track scenery** (`spawn_track_scenery()`, `run_iter.py`): a custom OpenDRIVE world is otherwise
bare road mesh. Since the vision-based state predictor (Branch B) needs a visual signal for
"where the lane edge is," hay-bale/barrel markers are placed every ~40m along both track edges
(from the same per-point width data), traffic cones mark the sharper corners
(heading-change > 12deg between consecutive ~5m segments), and sparse background props (garden
lamps, street signs) are scattered further out for visual variety. Re-spawned once per episode,
since `generate_opendrive_world()` rebuilds the world from scratch each time.

**Route-following caveat**: Austin is a closed loop, but `run_iter.py`'s waypoint-following and
episode-termination logic (`closest_index`, "reached the end" distance check) was written for
Town04's open-ended route and doesn't have closed-loop wraparound handling. Not an issue at
current episode lengths (~200s, well under one full 5.5km lap), but would need addressing before
running full-lap episodes.

## 4. Data Collection Protocol

- **Weather**: cycled through a curated "standard" daytime/dusk set (ClearNoon, CloudyNoon,
  WetNoon, ClearSunset, CloudySunset, WetSunset) — night and heavy-rain long-tail conditions
  deliberately held back until the CBF controller is stabilized (see `docs/decisions.md`).
- **Route position + perturbation**: `--randomize-spawn` picks a random point along the route,
  perturbed by up to 1.5m laterally and 15deg in yaw, so the built-in autopilot's recovery back to
  lane center becomes DAgger-style training signal at no extra collection cost.
- **Multi-episode driver**: `collect_dataset.py` loops episodes, cycling weather and seeds,
  tagging each episode's output folder (`run<N>_ep<i>_images/video`) so they don't collide, and
  stopping once a target frame count is reached.
- **Storage**: collection output is written directly over SMB (`--output-dir '\\<host>\<share>'`)
  to the Windows CARLA machine's local disk, since the collection client's own disk (a shared dev
  sandbox) has far less free space than a full dataset needs. Training data should be staged
  locally before running `train.py` (see `docs/decisions.md`).

## 5. Reproducibility

All commands, flags, and their defaults are documented in `README.md`'s Execution Commands
section, kept current as the single source of truth for "how to actually run this." This document
is the *why*; the README is the *how*.
