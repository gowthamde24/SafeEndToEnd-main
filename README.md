# SafeEndToEnd: SA-TCP Architecture & Control Barrier Functions in CARLA

##  Overview
This repository contains the implementation and Phase 1 baseline replication of the Safe Autonomous Trajectory Control Prediction (SA-TCP) architecture. The system executes closed-loop autonomous driving in the CARLA simulator (Town 04, and a custom real-world-derived Austin/COTA track) by combining an End-to-End (E2E) neural network with a mathematical Control Barrier Function (CBF) safety gate.

This README is the reproducibility reference (how to run things). For the research record --
related work comparison, methodology writeup, experiment log, a dated decision trail with
rationale, the results-chapter scaffold, and citations (useful for a thesis defense) -- see
[`docs/`](docs/related_work.md):
[`docs/related_work.md`](docs/related_work.md) &middot;
[`docs/methodology.md`](docs/methodology.md) &middot;
[`docs/experiments.md`](docs/experiments.md) &middot;
[`docs/decisions.md`](docs/decisions.md) &middot;
[`docs/results.md`](docs/results.md) &middot;
[`docs/references.bib`](docs/references.bib)

##  Architecture Breakdown
The pipeline processes live RGB camera feeds via a PyTorch ResNet backbone, split into two primary branches:

* **Branch A (Primary Control):** Extracts visual features and concatenates them with current velocity vectors to predict raw steering commands.
* **Branch B (Safety Predictors):** Three separate ResNet-18 models predict spatial telemetry:
  * Cross-Track Error (X)
  * Heading Error (Theta)
  * Road Curvature
* **Control Barrier Function (CBF):** Acts as the system's "Safety Gate." It evaluates the raw steering command from Branch A against the spatial state predicted by Branch B. By projecting the vehicle's trajectory using a Kinematic or Dynamic bicycle model, the CBF actively overrides the neural network with a mathematically guaranteed safe steering angle if lane boundaries are threatened.

---

##  CBF Safety Gate: How It Works

A walkthrough of `get_optimal_control()` (`run_iter.py`, `get_optimal_control` -- currently lines
277-346), mapping each term in the code to standard control-barrier-function theory. No code
behavior is described here beyond what's actually implemented.

**The problem it solves.** Branch A outputs a raw steering command `steer_ref` (plus an
MC-dropout uncertainty `steer_var`); Branch B outputs curvature/`x`/`theta` state estimates
(each with their own predicted variance). `get_optimal_control()` never trusts Branch A
blindly -- it searches for the steering angle closest to `steer_ref` that keeps the predicted
trajectory inside the lane corridor, using Branch B's state estimate to check.

**The barrier functions.** Two barrier functions bound the cross-track error `x` on each side
of the lane:
```
h_left  = LANE_WIDTH/2 - x
h_right = LANE_WIDTH/2 + x
```
Both are >= 0 exactly when `|x| <= LANE_WIDTH/2`. Being inside the corridor right now isn't
enough for a CBF -- it also has to guarantee the car *stays* inside (forward invariance), so
the code also computes `hd`/`hdd` (the barrier's rate of change under the bicycle-model
dynamics), not just `h` itself.

**Relative degree (why two different formulas).** Steering doesn't affect `x` directly -- it
works through heading, then lateral velocity, then position. How many integrators sit between
steering and `h` (the relative degree) decides how many derivatives of `h` the barrier needs:

- **Kinematic model** (`VEHICLE_MODEL == 'Kinematic'`), relative degree 2:
  ```
  hdd_left + lambda*hd_left + lambda**2*h_left  >=  0
  ```
  `lambda = lambda_ = 5`. The coefficients `1, lambda, lambda**2` are `(s+lambda)**2`
  expanded -- both roots of the constraint's characteristic polynomial at `-lambda`, so a
  violation decays exponentially rather than being clamped once.

- **Dynamic model** (used at higher speed, where tire slip matters -- adds cornering
  stiffness `Cf`/`Cr`), relative degree 3:
  ```
  hddd_left + 3*lambda*hdd_left + 3*lambda**2*hd_left + lambda**3*h_left  >=  0
  ```
  `1, 3*lambda, 3*lambda**2, lambda**3` is `(s+lambda)**3` expanded -- same idea, one degree
  higher.

  *Worth checking against the paper:* the dynamic branch also bounds the heading angle
  (`h_theta_left/right`, `|theta| <= THETA_LIM`), which only has two derivatives available
  (`hd_theta`, `hdd_theta` -- no third), yet its constraint reuses the same `1, 3*lambda,
  3*lambda**2` coefficients as the degree-3 position barrier rather than the degree-2 form
  (`1, 2*lambda, lambda**2`) that strict HOCBF theory would call for. Not necessarily wrong
  (it's a more conservative, differently-tuned constraint either way), but it's the one place
  the implementation visibly diverges from a by-the-book derivation.

**Turning it into a chance constraint.** `x`, `theta`, and `curvature` are MC-dropout
estimates with predicted variances, not exact values. The code propagates that uncertainty
through the barrier and shaves a safety margin off the top:
```
var_left = sqrt( (lambda**2*x_var)**2 + (v**2*sin(theta)*steer/L)**2
                + (v**2*curvature_var)**2 + (lambda*v*cos(theta)*theta_var)**2 )

hdd_left + lambda*hd_left + lambda**2*h_left - var_left*Phi_inv(1-BETA)  >=  0
```
`var_left` is a linearized (delta-method) propagation of `x_var`/`curvature_var`/`theta_var`
through the barrier expression; `Phi_inv(1-BETA)` is `norm.ppf(1-BETA)`, the one-sided
Gaussian quantile for confidence `1-BETA`. Mean-minus-`z*sigma` is exactly how
`P(h >= 0) >= 1-BETA` becomes a deterministic inequality under a Gaussian approximation --
when the network is less confident, the margin grows and the CBF intervenes earlier.

**Solving it: a grid search standing in for a QP.** The textbook approach is a small convex
QP minimizing `||u - u_ref||**2` subject to the barrier inequality. Steering here is
one-dimensional and bounded, so instead of a QP solver the code sweeps the full range at 0.01
resolution:
```
for steer in arange(-MAX_STEER, MAX_STEER, 0.01):
    cost = (steer - steer_ref)**2 / steer_var**2
    if barrier_left(steer)  < 0: cost += alpha * barrier_left(steer)**2
    if barrier_right(steer) < 0: cost += alpha * barrier_right(steer)**2
    keep steer with the lowest cost
```
`alpha = 20`. The tracking term is Branch A's deviation cost, inversely weighted by its own
confidence (`steer_var`) -- a confident prediction is expensive to deviate from, an uncertain
one is cheap. This is a **penalty-method approximation** of the CBF-QP, not an exact solve:
violating candidates aren't excluded, just made expensive. With `alpha` heavily outweighing
the variance-normalized tracking cost, a barrier-violating steer will almost never win in
practice -- but it's a soft constraint, not a hard guarantee.

**The off switch.** `if SAFEGUARD == False: return steer_ref` bypasses all of the above --
this is what `--cbf`/`--no-cbf` and the `RUN_NO >= K_ITERS` ramp-up default control, and
exactly the switch `compare_cbf.py` uses to produce its with/without-CBF comparison.

**Symbol map:**

| Code | Meaning | CBF-theory role |
|---|---|---|
| `h_left` / `h_right` | Distance to each lane edge | Barrier function `h` |
| `lambda_` (= 5) | Shared decay-rate constant | K-class coefficient, roots at `-lambda` |
| `alpha` (= 20) | Violation penalty weight | Penalty-method weight, not part of CBF theory itself |
| `BETA` | Target confidence level | Chance-constraint risk bound, `P(h>=0) >= 1-BETA` |
| `norm.ppf(1-BETA)` | Gaussian one-sided quantile | The "z" in `mean - z*sigma` margin tightening |
| `x_var`, `theta_var`, `curvature_var` | Branch B's predictive variances | Propagated into `var_left`/`var_right` via linearization |
| `steer_var` | Branch A's predictive variance | Inverse-confidence weight on the tracking cost |
| `L`, `Cf`, `Cr`, `mass`, `lf`, `lr`, `Iz` | Bicycle-model physical constants | Plant model used to compute `hd`/`hdd`/`hddd` |

---

##  Phase 1 
The baseline environment, training loop, and inference pipeline have been successfully stabilized. The following critical work has been completed:

1. **Phantom Error Resolution (X):** Investigated and eliminated a scaling artifact that caused the network to hallucinate a 300-meter lateral offset. The model now predicts cross-track error with centimeter-level precision.
2. **Angular Discontinuity Handling (Theta):** Fixed the heading-error label at its source (`run_iter.py` wraps into [-180, 180] before it's ever written to a frame filename), not just in evaluation -- see Known Limitations below for why the earlier evaluation-only wrap wasn't sufficient.
3. **Legacy CARLA 0.9.x Bridge:** Deployed adapter classes (`Carla09Bridge` and `MockImageConverter`) to cleanly interface modern 0.9.x sensor queues and transforms with the legacy 0.8.x physics and controller logic.
4. **Closed-Loop Evaluation:** Successfully executed full inference loops (`RUN_NO = 1`) both with and without the CBF override, against a live CARLA server, and generated a quantitative with/without-CBF comparison -- see Results below.
5. **Automated Telemetry Processing:** Upgraded `plots_curves.py` to automatically generate loss/accuracy convergence curves and "Predicted vs. Observed" spatial tracking graphs, outputting directly to the `results/` directory.
6. **Video Compilation Pipeline:** Optimized `create_video.py` to stitch real-time third-person inference frames into a final `.mp4` format, with built-in safety checks to handle dropped simulation frames.
7. **Train/Val/Test Split + Early Stopping:** Replaced train-on-everything with an 80/10/10 block-based split, per-epoch validation loss/accuracy tracking, and early stopping on val loss (see Execution Commands below).

---

##  Results (`saved_models_iter0`, run 1)

**Training (`train.py -r 0`):** early stopping selected **epoch 17** (of 26 run, ceiling 50) as the
best checkpoint, `model-best*.ckpt`. Macro-averaged across all 4 heads (steering, curvature,
X, theta) at that epoch:

| | Train Loss | Train Acc % | Val Loss | Val Acc % |
|---|---|---|---|---|
| **Overall** | 0.098 | 90.8% | 0.305 | 79.4% |

Theta generalizes essentially perfectly (0 train/val accuracy gap); curvature and X show the
largest train/val gaps (16-20 points), consistent with `run0_images` being a small (468-frame),
single-session dataset. Full per-head breakdown: `results/train_val_summary_table_run0.png`.
Per-epoch curves: `results/accuracy_{steering,curvature,x,theta,overall}_run0.png`,
`results/safety_{cross_track_error_x,curvature,heading_error_theta}_convergence.png`.

**Closed-loop with vs. without CBF** (`run_iter.py -r 1 --cbf` / `--no-cbf` against a live
CARLA server, then `compare_cbf.py --run 1`), one lap of the Town04 route, 469 evaluation frames:

| Metric | With CBF | Without CBF |
|---|---|---|
| Lane violations (`\|x\|` > 5.5m) | 0 | 0 |
| Mean `\|x\|` | 0.258 m | 0.246 m |
| Max `\|x\|` | 0.932 m | 0.552 m |
| Real CBF interventions (>0.05 correction) | 17 / 2342 frames | -- |
| Max intervention magnitude | 1.08 (near full steering lock) | -- |

The two trajectories are visually indistinguishable at track scale
(`results/comparison/trajectory_overlay_run1.png`) -- the CBF stayed silent almost the entire
lap. It did engage in one sustained ~130-frame episode with large, repeated corrections
(`results/comparison/cbf_intervention_run1.png`), but that episode coincides with the
**single worst cross-track moment of either run** (with-CBF hit -0.93m vs. without-CBF's
worst of -0.52m; `results/comparison/cross_track_error_run1.png`) -- neither ever crossed the
actual lane-violation threshold, but on this route the override didn't reduce peak deviation,
and momentarily coincided with a larger swing than doing nothing. **Honest takeaway:** the CBF
is verified to work correctly (silent when safe, forceful when its own state estimate says a
boundary is threatened), but this route didn't stress the base policy enough to demonstrate a
clear safety *benefit* -- a route with sharper corners or an adversarial starting offset would
better showcase it.

---

##  Dataset Naming Convention
To prevent desynchronization between image frames and telemetry labels, ground-truth physics are embedded directly into the dataset file strings during expert Autopilot collection (`-r 0`). 

The extraction format is:
`frame_<ID>_<Steering>_<Curvature>_<X>_<Theta>_<Speed>_<Perp_Speed>.png`

| Variable | Physical Meaning | Integer Scaling Math |
| :--- | :--- | :--- |
| **`ID`** | Sequential frame index | `frame//5` |
| **`Steering`** | Expert steering command | Degrees * 100 |
| **`Curvature`** | True road map curvature | Curvature * 10000 |
| **`X`** | Lateral cross-track error | Meters * 100 |
| **`Theta`** | Vehicle heading error | Degrees * 100 |
| **`Speed`** | Longitudinal velocity | m/s * 100 |
| **`Perp_Speed`** | Lateral slip velocity | m/s * 100 |

---

##  Setup

The full pipeline (training + CARLA inference) needs a Python 3.9 environment with the
CARLA 0.9.15 client, PyTorch, and a running CARLA server (a separate ~10GB game-engine
binary, not part of this repo) for anything under `Carla/`.

```bash
cd Carla
python3 -m venv e2e_env
source e2e_env/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu  # drop the index-url if you have a CUDA GPU
```

`vis_gradcam.py` additionally needs `pip install grad-cam` (optional, only for saliency-map visualization).

---

##  Execution Commands

All commands below are run from `Carla/` with `e2e_env` activated.

**1. Train the E2E steering model + 3 safety (curvature/X/theta) models:**
Trains on `run<N>_images/*.png` (labels are encoded in the filenames, see table above).
```bash
python train.py -r 0
```
Frames are split 80/10/10 train/val/test by contiguous 20-frame block (`BLOCK_SIZE` in
train.py) rather than by individual frame, since adjacent frames are near-duplicates of the
same driving moment -- a per-frame random split would leak near-identical scenes across
splits and make validation loss meaningless. Training runs up to `num_epochs` (ceiling 50)
with early stopping (`EARLY_STOP_PATIENCE = 8` epochs with no val-loss improvement), saving
both `model-last*.ckpt` (final epoch) and `model-best*.ckpt` (lowest combined val loss) --
`run_iter.py` prefers `model-best*.ckpt` and only falls back to `model-last*.ckpt` if no
val split existed for that training run (e.g. too little data to form 3+ blocks). Per-epoch
train/val loss (`val_losses_run<N>.csv`) and tolerance-based accuracy (`train_val_accuracy_run<N>.csv`,
`% of predictions within TOLERANCES` in train.py -- steering +/-2.0, curvature +/-0.002,
X +/-0.15m, theta +/-5 degrees) are logged, and a one-time held-out test evaluation
(`test_metrics_run<N>.csv`) runs after training stops, using the best checkpoint.

**2. Run Closed-Loop Inference (with/without CBF):**
Executes the real-time driving loop against a live CARLA server, using the trained PyTorch
models. `--cbf`/`--no-cbf` force the safety gate on/off; with neither flag it defaults to
the `RUN_NO >= K_ITERS` ramp-up behavior. Outputs are namespaced by CBF mode so the two
runs never overwrite each other.
```bash
python run_iter.py -r 1 --cbf
python run_iter.py -r 1 --no-cbf
```

**3. Compare With vs. Without CBF:**
Produces trajectory-overlay, cross-track/heading-error, and CBF-intervention plots plus a
summary CSV under `results/comparison/`, ready for a mentor presentation or thesis report.
```bash
python compare_cbf.py --run 1
```

**4. Generate Training/Evaluation Plots:**
Parses the training loss CSVs and the per-mode inference comparison CSVs to generate MSE
convergence and "predicted vs. observed" graphs, saved into `results/`.
```bash
python plots_curves.py --mode with_cbf --run 1
```

**5. Plot All Driven Trajectories vs. Track Boundaries:**
```bash
python path_plot.py -n 10 --mode with_cbf
```

**6. Compile Demo Video:**
Assembles the saved third-person chase camera frames into an MP4 video playback.
```bash
python create_video.py -n 1
```

---

##  Known Limitations / Data Notes

- **Small dataset, noisy val/test percentages:** `run0_images` has only 468 frames total
  (400 train / 40 val / 28 test after the block split). Accuracy percentages on the val/test
  sets can swing by several points from a single epoch to the next purely from sample-size
  noise (each test frame is ~3.6% of the test set) -- read trends across several epochs, not
  single-epoch snapshots. Collecting more Autopilot data would tighten this considerably.
- **Heading-error (theta) label wrap-around (fixed):** a small fraction of frames in early
  data-collection runs recorded raw heading-error values near +/-360 degrees instead of
  being wrapped into [-180, 180] (the raw `current_yaw - closest_angle` difference in
  `run_iter.py` can cross the +/-pi boundary). Because training uses MSE loss, these
  frames produced periodic loss spikes roughly 300x the typical value, which dominated
  gradient updates and left the theta safety model barely better than a "predict zero"
  baseline. This is now fixed at the source (`run_iter.py` wraps `theta_save` before it's
  encoded into new frame filenames) and defensively in `train.py`'s dataset loader (any
  existing out-of-range values in already-collected data are wrapped at load time).
- **Cross-track error (X) "predicted vs. observed" plots:** `run_iter.py` logs 3 columns
  per frame to `x_comps.csv` -- `[prediction, MC-dropout uncertainty, ground_truth]`.
  Ground truth is column index 2, not column 1 (the uncertainty estimate, which is
  usually near zero) -- `plots_curves.py`/`plot_predictions.py` read the correct column.
- **CBF dynamic-model coefficient:** the relative-degree-3 lane-boundary barrier condition
  in `run_iter.py`'s `get_optimal_control()` had an inconsistent coefficient on the left
  lane boundary vs. the right; both sides now use the same (correct) `lambda_**3 * h`
  term, matching the standard `(d/dt + lambda)^3 h` expansion.
- **`controller2d.py`** implements a PID + Stanley baseline controller that is instantiated
  by `run_iter.py` but not currently used to drive the vehicle (steering comes from the NN
  + CBF, or CARLA's autopilot). Kept as a self-contained, correct reference implementation
  for a possible future non-NN baseline comparison.
- Offline retraining (`train.py`) only needs the collected images and runs on CPU; live
  CARLA inference (`run_iter.py`, and therefore `compare_cbf.py`'s inputs) needs a running
  CARLA server on a machine with the full simulator installed.