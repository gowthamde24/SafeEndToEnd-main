# Results (scaffold — fills in after training + evaluation)

This is the results-chapter draft. Sections and reproduction commands are set up now so nothing
gets missed once the `austin_map` dataset finishes collecting and a model is trained on it;
`[TODO: ...]` markers show exactly what's still pending. See `docs/methodology.md` for how each
number/plot is produced and `docs/references.bib` for citations.

---

## 1. Dataset

| | Value |
|---|---|
| Track | Austin (COTA), custom-built (see `docs/methodology.md` §3) |
| Total frames | [TODO: from `collect_dataset.py`'s final frame count] |
| Weather conditions | ClearNoon, CloudyNoon, WetNoon, ClearSunset, CloudySunset, WetSunset |
| Collection method | Expert autopilot (`RUN_NO=0`), randomized spawn-offset recovery perturbation |
| Train/val/test split | 80/10/10, contiguous-block (see `train.py`'s `split_dataset`, block-per-episode-folder) |

[TODO: once collection finishes, note total wall-clock time, number of episodes, and confirm the
final per-weather-condition frame breakdown for balance.]

## 2. Training Curves

Produced by `train.py` (loss/accuracy CSVs) and `plots_curves.py` (rendered plots).

**Reproduce:**
```bash
python train.py -r 0 --image-glob 'run0_ep*_images/*.png' --output-dir <wherever the data landed>
python plots_curves.py --mode with_cbf --run 0
```

- [TODO] Steering loss/accuracy convergence (train vs. val)
- [TODO] Curvature safety-model loss/accuracy convergence
- [TODO] Cross-track error (X) safety-model loss/accuracy convergence
- [TODO] Heading error (theta) safety-model loss/accuracy convergence
- [TODO] Combined/overall summary table (mirror the Phase 1 table in `README.md`'s Results
  section — train loss/acc, val loss/acc, per head)
- [TODO] Final held-out test set metrics (`test_metrics_run0.csv`)

**Comparison point**: Phase 1's 468-frame Town04 baseline had a 0.098/0.305 train/val loss gap
(overfitting, driven by dataset size — see `README.md`). Worth an explicit before/after
statement here: did the larger, more diverse `austin_map` dataset close that gap.

## 3. Closed-Loop Route / Total Track Image

Produced by `path_plot.py` — the full track boundary plus every driven trajectory overlaid.

**Reproduce:**
```bash
python run_iter.py -r 1 --track austin_map --cbf      # generates controller_output/with_cbf/trajectory_run1.txt
python run_iter.py -r 1 --track austin_map --no-cbf    # generates controller_output/without_cbf/trajectory_run1.txt
python path_plot.py -n <N> --mode with_cbf --track austin_map
python path_plot.py -n <N> --mode without_cbf --track austin_map
```

- [TODO] Full-track image: track boundary (real per-point width, not a fixed constant — see
  `docs/decisions.md`) with the complete driven route overlaid. This is the "total route image."
- [TODO] Confirm whether a full-lap episode is being run for this image, or a partial-lap
  composite — `run_iter.py`'s waypoint-following logic doesn't yet handle closed-loop wraparound
  (see `docs/methodology.md` §3, "Route-following caveat"). If a full lap is wanted for this
  figure specifically, that gap needs addressing first.

## 4. With vs. Without CBF Comparison

Produced by `compare_cbf.py` — this is the core "difference between with and without CBF"
deliverable.

**Reproduce:**
```bash
python run_iter.py -r <N> --track austin_map --cbf
python run_iter.py -r <N> --track austin_map --no-cbf
python compare_cbf.py --run <N> --track austin_map
```

Outputs land in `results/comparison/`:
- [TODO] `trajectory_overlay_run<N>.png` — both driven paths vs. real track boundaries
- [TODO] `cross_track_error_run<N>.png` — x(t), predicted vs. observed, both modes
- [TODO] `heading_error_run<N>.png` — theta(t), predicted vs. observed, both modes
- [TODO] `cbf_intervention_run<N>.png` — |steer_applied - steer_raw| over time (with-CBF only)
- [TODO] `summary_run<N>.csv` — lane violation count, mean/max |x|, CBF intervention frame count
  and correction magnitude, both modes

**Reporting checklist** (mirror and update the Phase 1 narrative in `README.md`'s Results
section, which was Town04-only):
- [TODO] Did the CBF reduce lane violations vs. baseline, or (as in the Phase 1 Town04 result)
  stay silent because the base policy didn't get close enough to the boundary to need it?
- [TODO] Peak deviation with vs. without CBF — Phase 1 found the CBF's one large intervention
  episode coincided with a *larger* peak deviation than doing nothing; check whether that holds
  on the more challenging Austin corners.
- [TODO] Number and magnitude of real interventions (>0.05 correction, per the `epsilon` in
  `compare_cbf.py`'s `summarize()`).
- [TODO] Given the CBF coefficient issues flagged in `docs/related_work.md` (missing factor of 2
  in the kinematic branch, wrong-order theta constraint) — decide whether this comparison runs
  against the *current* (as-is) CBF or a corrected one, and say so explicitly in the writeup so
  the comparison isn't misread as validating an unexamined implementation.

## 5. Comfort / Secondary Metrics

`get_statistics.py` (mean deviation from reference trajectory, total time per run) is
track-agnostic already, no changes needed.

- [TODO] Mean deviation from reference line, with vs. without CBF
- [TODO] Lap/episode completion time, with vs. without CBF
- [TODO] (If pursued) jerk/comfort metric — not currently computed by any script; would need new
  tooling if this ends up in the final report.

## 6. Figures Checklist (for the actual thesis document)

- [ ] Architecture diagram (Branch A / Branch B / CBF gate) — not yet drawn
- [ ] Austin track overview (aerial/plan view, e.g. `racelines/Austin_raceline.png` from the
  source repo, or a fresh render from `austin_waypoints.txt`)
- [ ] Training loss/accuracy curves (§2)
- [ ] Total route image (§3)
- [ ] With/without CBF comparison plots (§4)
- [ ] Summary tables (§2, §4, §5)

## 7. Citations

All sources this report draws on are in `docs/references.bib`. Key ones to make sure appear in
the actual thesis text (not just the bibliography): Kalaria et al. 2023 (architecture/CBF
reference), TUM racetrack-database (track data), CARLA (simulator), Gal & Ghahramani 2016
(MC-dropout), Xiao & Belta 2019 (HOCBF theory), Ross et al. 2011 (DAgger, for the recovery-data
framing).
