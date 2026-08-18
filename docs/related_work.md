# Related Work

## Kalaria, Lin, Dolan — "Towards Safety Assured End-to-End Vision-Based Control for Autonomous Racing" (IFAC 2023)

This is the primary reference for the SA-TCP architecture's safety-controller design: a decoupled
end-to-end control network (`M_com`) and end-to-end state network (`M_st`), with a probabilistic
control barrier function (CBF) built on `M_st`'s predicted state safeguarding `M_com`'s raw
steering output. Full citation:

> D. Kalaria, Q. Lin, and J. M. Dolan. *Towards Safety Assured End-to-End Vision-Based Control for
> Autonomous Racing.* IFAC-PapersOnLine, 56(2):2767-2773, 2023.

### What we adopted directly
- The overall decoupled architecture: one network for control, one for state, a CBF safety gate
  between them (our `EndtoEnd` + `model_safety_1/2/3` + `get_optimal_control()` in `run_iter.py`).
- The MC-dropout epistemic-uncertainty estimation scheme (Gal and Ghahramani, 2016) — mean/variance
  over `n` stochastic forward passes (paper Eq. 15) — our `N_ITERS`-pass loop in `run_iter.py`.
- The general chance-constraint approach: tighten a CBF inequality by a confidence-scaled margin
  derived from predicted-state uncertainty (paper Eq. 13-14; our `norm.ppf(1-BETA)` term).
- The Frenet-frame state representation `[x, theta, omega, v, v_perp, curvature]` (paper Sec. 3.1).

### Where our implementation diverges from the paper (see `docs/decisions.md` for the full analysis)
- **Relative degree**: the paper uses a single degree-2 CBF formulation throughout (both its
  dynamic-model Carla experiments and its kinematic-model RC-car experiments — Sec. 3.1: *"We
  consider using 2nd-order CBF, as we have a system with a relative degree of 2"*). Our code has a
  degree-2 branch (`VEHICLE_MODEL == 'Kinematic'`) *and* a degree-3 branch (else), which the paper
  does not have. Root cause traced to our `v_perp_dot` omitting the paper's direct steering
  (`delta`) coupling term (paper Eq. 10) — see `docs/decisions.md`, entry 2026-08-16.
- **Coefficient bug in our kinematic-model branch**: paper Eq. 11 is `h_dd + 2*lambda*h_d +
  lambda**2*h >= 0`; our code has `h_dd + lambda*h_d + lambda**2*h` (missing the factor of 2).
- **Heading-barrier order mismatch**: paper Eq. 12/13 treat `h_theta` with the *same* degree-2 form
  as position. Our code uses degree-3-style coefficients (`3*lambda`, `3*lambda**2`) for a
  quantity that only has two derivatives available — matches neither a correct degree-2 nor a
  correct degree-3 derivation.
- **Uncertainty propagation method**: the paper (Eq. 14) draws `n` Monte Carlo rollouts of the
  *barrier expression itself* and takes its empirical mean/variance. Our code instead analytically
  combines the individual state-variable variances (`x_var`, `theta_var`, `curvature_var`) through
  what looks like a linearized/delta-method weighting — a different (not necessarily worse, but
  different) way of getting to the same kind of margin.
- **Hyperparameters**: paper reports `lambda=2.5`, `theta_max=pi/4` (45deg), `eta=0.95`. Our code
  has `lambda_=5`, `THETA_LIM=37deg`, `BETA=0.1` (i.e. confidence 0.9). Not necessarily wrong, but
  not a reproduction of the paper's tuning either, and should be justified independently (via our
  own ablations) rather than left as an unexplained mismatch.
- **Solver**: the paper solves an actual QP with slack variables per constraint (Eq. 16). Our code
  approximates this with a brute-force grid search over candidate steering angles plus a quadratic
  penalty on constraint violation (`alpha=20`) — a legitimate practical substitution for a
  1-D control problem, but a soft constraint rather than an exact QP solve.

**Stance going forward** (per project decision, 2026-08-16): this paper is a *reference for the
idea*, not a formula to copy verbatim. Any future CBF changes should be independently re-derived
from the vehicle model we actually use, with the paper cited as motivation/comparison rather than
as ground truth to match line-for-line.

## TUMFTM/racetrack-database

> TUM Institute of Automotive Technology. *racetrack-database.*
> https://github.com/TUMFTM/racetrack-database (LGPLv3).

Source of the Austin (COTA) centerline+width data used to generate `Carla/tracks/austin.xodr` (see
`tools/track_to_opendrive.py`). Centerlines originally sourced from OpenStreetMap GPS points, track
widths from satellite-image processing, per the repository's own README. Vendored copy (with its
license) kept at `Carla/racetrack_source/`.

## CARLA Simulator

> A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun. *CARLA: An Open Urban Driving
> Simulator.* Proceedings of the 1st Annual Conference on Robot Learning, 2017.

Used both for its built-in Town04 map and, for the Austin track, its standalone OpenDRIVE world
generation (`carla.Client.generate_opendrive_world`), which builds a drivable road mesh live from
an arbitrary `.xodr` description without needing a UE4 rebuild or the full map content package.
