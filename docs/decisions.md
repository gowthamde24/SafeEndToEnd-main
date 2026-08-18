# Decision Log

Dated record of judgment calls made during development, with the reasoning behind each — kept so
"why did you choose X" has a real trail at defense time instead of being reconstructed from memory.

---

## 2026-08-17 — Force-skipped an anomalously stuck episode during overnight collection

**Context**: user asked for collection + training to run unattended overnight. Episode 11
(WetSunset, seed=11) retried for 3+ hours (~20 errors: a mix of "destroyed actor" and stall
watchdog triggers) without ever completing cleanly, vs. 15-30 min for a normal episode.

**Decision**: killed the stuck `run_iter.py` subprocess directly (`kill -TERM`) rather than
keep waiting. Checked `collect_dataset.py`'s loop first to confirm this was safe: it calls
`subprocess.run(cmd, check=False)` and unconditionally increments the episode counter afterward
regardless of exit status -- so killing a stuck attempt just makes it move on to the next episode
(a fresh seed/weather), it doesn't corrupt state or require any other cleanup. Confirmed working:
episode 12 started immediately and began writing frames normally.

**Why this one call**: with the user away overnight and a real target to hit, burning hours on
one statistically unlucky episode is worse than accepting whatever partial frames it left behind
and moving forward. This isn't a general policy change -- the retry/reseed/stall-watchdog system
described in earlier entries remains the default recovery path; this was a manual override for an
outlier that outran all of them combined.

---

## 2026-08-16 — Reversed the "train.py stays local-only" decision: made it SMB-aware after all

**Context**: user asked for collection to finish, training to run, and results ready by morning
(unattended overnight run). Earlier decision (see the SMB storage pipeline entry above) explicitly
kept `train.py` local-disk-only, reasoning that repeated random-access reads per epoch would be
slower over the network than local disk.

**Why reversed**: two facts changed the calculus. (1) Disk space: 50k images is ~22GB; this
sandbox has 15GB free -- the full dataset physically doesn't fit locally, so "local-only" would
require either discarding most of the collected data or manually staging/rotating a subset, both
worse than just reading over SMB. (2) No GPU here (`torch.cuda.is_available() == False`) --
training is CPU-bound, meaning epoch time is dominated by compute (ResNet forward/backward passes
for 4 heads per batch), not I/O. The network-read penalty that mattered for a GPU-fast training
loop is comparatively negligible against CPU-bound compute.

**Implementation**: mirrors the pattern already used for writing (`run_iter.py`) -- `is_unc_path()`
check, `smbclient.open_file()` + `Image.open(f); image.load()` (forcing the read before the file
handle closes, since `Image.open` is lazy) in `croppedDataset.__getitem__`, and a small `smb_glob()`
helper (Python's `glob.glob()` can't walk a UNC path at all) for the dataset listing. `DataLoader`
here uses the default `num_workers=0` (single-process), so no multi-process smbclient
session-sharing complexity to worry about -- left as-is deliberately, not something to "improve"
by adding workers without revisiting this.

**Verified against real data, not just synthetic**: ran `smb_glob()` + `croppedDataset.__getitem__`
against the actual in-progress collection on the share -- found 22,221 real frames, successfully
loaded and correctly parsed one into a labeled tensor.

---

## 2026-08-16 — Raised collection driving speed ~8 m/s -> ~21 m/s average

**Context**: user asked how fast the autopilot drives during collection. Checked the live log --
a steady ~8 m/s (29 km/h). Traced to CARLA's default 30 km/h speed limit assumption for OpenDRIVE
roads with no `<type><speed>` element (ours doesn't define one). Flagged as a real mismatch: the
project's framing (Kalaria et al.) is specifically about racing-line driving at cornering limits,
not a conservative cruise.

**Decision**: `traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, -200)` -- targets
roughly 3x the 30 km/h limit (~90 km/h ceiling on straights). Chose the traffic-manager API over
editing a speed limit into the `.xodr` because it's a one-line, immediately-testable change with
no track-regeneration step.

**Verified before applying, not just asserted**: live-tested for 35s (~500m, through a real
corner) with a collision sensor attached. Result: 0 collisions, speed ranged 8.7-24.4 m/s -- the
autopilot still slows appropriately for corners rather than being uniformly fast everywhere,
average 21.3 m/s. `-200` wasn't picked by feel; it's the value that was actually tested.

**Known side effect**: at ~21 m/s average a 200s episode now covers close to (sometimes past) a
full 5.5km lap, which makes the already-documented closed-loop-wraparound gap (see
`docs/methodology.md` §3, "Route-following caveat") more likely to actually get hit rather than
being a theoretical concern. Not fixed as part of this change -- noted for whenever it becomes a
real problem instead of a possible one.

**Batch 1 restarted from ep0** with the higher speed.

---

## 2026-08-16 — Lowered containment wall 1.5m -> 0.8m after user flagged unrealistic height

**Context**: after fixing the wall so it actually contains the vehicle (0.4m -> 1.5m, previous
entry), user pointed out a 1.5m wall doesn't look like a real racetrack -- real trackside barriers
(Armco/concrete) run roughly 0.8-1.2m, with taller catch fencing set back further, not one tall
wall hugging the racing line.

**Decision**: tested 1.0m and 0.8m with the same full-throttle-into-the-wall method used to
validate 1.5m. All three heights produced the *identical* ~5m stop distance -- the vehicle's
bumper contact with the wall's vertical face is what stops it, and extra height above ~0.8m wasn't
adding any containment. Set to 0.8m: realistic barrier height, empirically equal containment.

**Takeaway for future height changes**: if containment ever needs revisiting again, re-run this
same test (spawn vehicle, point it at the wall, full throttle for ~150-200 ticks, check final
distance from start against the true lane half-width) rather than guessing -- it's cheap and
already found the real threshold is well below what intuition suggested.

**Batch 1 restarted from ep0** with the 0.8m wall.

---

## 2026-08-16 — Disabled all scattered track props, not just the fence

**Context**: after removing `chainbarrier`, user flagged the remaining hay-bale stacks visible in
a screenshot as obstacles that also shouldn't be there.

**Decision**: stopped calling `spawn_track_scenery()` entirely (the function definition is kept,
just unused, in case scenery is revisited later) rather than removing individual prop types one
at a time. The track now has no scattered 3D props of any kind -- no haybale, no cones, no
background scenery. The low `wall_height=0.4` on `OpendriveGenerationParameters` stays, since
that's part of the generated road geometry (fixes the flat-blue-void problem) rather than a
discrete object placed on the track, and wasn't part of what was flagged.

**Why not just prune the problem prop**: `chainbarrier` alone had caused two independent real
bugs (cross-track chain linking, physics dragging). Rather than wait for the next prop-specific
issue to surface, removing all prop placement is the more conservative choice given collection
time is expensive (hours per batch) and every restart costs real progress.

**Batch 1 restarted from ep0** with scenery fully disabled.

---

## 2026-08-16 — Removed the chainbarrier fence entirely rather than reworking it

**Context**: after the physics fix (props pinned via `set_simulate_physics(False)`), user spotted
a chain visually stretching across the *middle* of the track, not just along an edge.

**Root cause**: `chainbarrier`'s chain segment appears to visually link to the nearest other
`chainbarrier` instance in the world, regardless of which side of the track placed it. At a
narrow point or tight bend, a post on the left edge can end up geometrically closer to a post on
the right edge than to its own neighbor on the same side, producing a chain that cuts straight
across the drivable surface.

**Decision**: remove `chainbarrier` from `spawn_track_scenery()` completely, per explicit
instruction, rather than attempt a workaround (e.g. wider spacing, alternating link-breaking
props). Two independent real problems from the same prop (this cross-track linking, and the
earlier physics-drag issue) were enough to drop it rather than keep patching it.

**What's left for the track edge**: the low `wall_height=0.4` boundary (already doing the actual
void-prevention job) plus hay-bale run-off stacks, corner cones, and sparse background props from
`spawn_track_scenery()`. No continuous fence/rail asset remains. If a stronger continuous edge
cue turns out to matter later, that's a fresh design decision, not a resurrection of chainbarrier.

**Batch 1 restarted from ep0** with the fence-free environment; live-verified 0 chainbarrier
actors in a fresh world (276 total props: haybale/cone/background only).

---

## 2026-08-16 — Made `compare_cbf.py`/`path_plot.py` track-aware; scaffolded results chapter + citations

**Context**: user specified the full defense deliverable up front — training curves, accuracies,
losses, a total-route image, and an explicit with-vs-without-CBF comparison, "clean and detailed,"
professional, reusable, with citations. Checked the existing evaluation scripts before assuming
anything needed building from scratch.

**Finding**: `compare_cbf.py` and `path_plot.py` (both from the Phase 1 Town04 work) hardcoded
`town04_waypoints.txt` and a fixed 7m track-boundary half-width. Run unmodified against
`austin_map` data, they would have silently plotted the wrong track shape and used the wrong lane
width for violation-counting — not a crash, a wrong-looking result that could have gone into the
thesis unnoticed.

**Decision**: added a `--track {town04,austin_map}` flag to both scripts, with a small local
`TRACK_CONFIGS` dict (deliberately not importing `run_iter.py`'s `TRACKS`, to avoid pulling
carla/torch into what are otherwise pure-numpy/matplotlib analysis scripts). For `austin_map`,
track-boundary width now comes from the real per-point `racetrack_source/Austin.csv` data
(11-27.6m, matching the actual track) instead of a single constant — Town04's behavior is
unchanged (still a fixed 7m/side, matching the original Phase 1 results so those numbers stay
comparable).

**Also created**: `docs/results.md` (results-chapter scaffold — every plot/table the defense
needs, mapped to the exact command that produces it, with `[TODO]` markers) and
`docs/references.bib` (BibTeX: Kalaria et al. 2023, TUM racetrack-database, Heilmeier et al. 2020,
CARLA, Gal & Ghahramani 2016 MC-dropout, Xiao & Belta 2019 HOCBF, Ross et al. 2011 DAgger) — so the
citation list exists before it's needed, not assembled under deadline pressure.

**Flagged explicitly in `docs/results.md`**: the CBF comparison should state up front whether it's
evaluating the *current* (as-is, with known coefficient issues — see the related_work.md entry
above) CBF implementation or a corrected one, so the eventual with/without-CBF numbers aren't
misread as validating an unexamined implementation.

---

## 2026-08-16 — Rename `--track austin` to `--track austin_map`; large dataset collects on it alone

**Decision**: renamed the `--track` identifier for the custom circuit from `austin` to
`austin_map`, and decided the 50k-frame large-dataset collection will run on it exclusively, not
split with Town04.

**Why**: Town04 is CARLA's stock built-in map; `austin_map` is the one actually built for this
project (`tools/track_to_opendrive.py` + `spawn_track_scenery()`), and the naming should make
that distinction obvious at the CLI rather than reading as two equivalent stock options. Given the
research is framed around racing-line driving at cornering limits (per Kalaria et al.), the
purpose-built racing circuit is also the more relevant target for the large collection effort —
Town04's existing 468-frame dataset stays as the legacy baseline rather than being expanded.

**Note**: earlier dated entries in this log and in `docs/experiments.md` reference `--track
austin` (the pre-rename identifier) — left as-is since they're an accurate record of what was
actually run at the time; only the current/future identifier is `austin_map`.

**Collection plan**: 50,000 frames total, `--track austin_map`, standard weather set, collected
in monitored batches of ~5k-10k frames rather than one long unattended run (session-reliability
concern for a 14-hour unattended process — see `docs/experiments.md` for actual progress).

---

## 2026-08-16 — Dataset scale-up: Town04-only, randomized spawn offset for recovery data

**Context**: teammate feedback flagged the 468-frame dataset (Train loss 0.098 / Val loss 0.305 —
overfitting) as critical, and asked for 50k-100k frames across diverse maps/weather/perturbations,
including DAgger-style recovery maneuvers.

**Decision**: scale up on Town04 only (no new towns at this stage) using CARLA weather presets,
route-position randomization, and randomized lateral/yaw spawn perturbation — not mid-drive
teleport perturbation.

**Why**: a randomized *spawn* offset lets CARLA's existing autopilot naturally steer back to lane
center, and that recovery arc becomes the DAgger-style training signal for free, using the
data-collection code path (`RUN_NO == 0`) that already exists. Mid-drive teleport would need new
autopilot-disengage/re-engage logic and multiple recovery events per episode, for a benefit that
didn't seem to justify the added complexity at this stage.

**Outcome**: implemented as `--weather`, `--randomize-spawn`, `--lateral-perturb-max`,
`--yaw-perturb-max-deg`, `--seed`, `--episode-tag` on `run_iter.py`, plus `collect_dataset.py` as
the multi-episode driver. Live-smoke-tested; caught and fixed a real bug in the process (see next
entry).

---

## 2026-08-16 — Spawn-collision handling: resample, don't retry the same point

**Context**: first live test of `--randomize-spawn` hit a spawn collision. `run_iter.py`'s
`main()` retries the whole episode function on any exception, using the *same* `args` (same
`--seed`) each time — so a deterministic collision at a given seed would retry forever.

**Decision**: switched from `world.spawn_actor` (raises on collision) to
`world.try_spawn_actor` (returns `None`) with up to 20 resample attempts, each drawing a *new*
random point from the same `rng` instance so it doesn't repeat the failed point.

**Why**: this was the only fix that doesn't touch the retry loop's general error-recovery
behavior (which is useful for transient CARLA/network errors) while still terminating instead of
hanging forever on a bad seed.

---

## 2026-08-16 — Storage: SMB via pure-Python client, not a kernel mount

**Context**: 50k-100k frames need ~22-90GB (measured ~0.9MB/collected-frame across image+video
folders); the dev sandbox has 14GB free. Teammate/user decided to store on the Windows CARLA
box's `E:` drive instead.

**Decision**: rejected a kernel `mount -t cifs` (no passwordless `sudo` available in the sandbox)
in favor of the pure-Python `smbprotocol`/`smbclient` library, installed via `pip` in the project
venv, writing directly over SMB2/3 with no root privileges needed at all.

**Why**: this was purely a constraint-driven pivot — the original mount-based plan would have been
simpler if `sudo` had been available. `smbclient`'s `open_file()` returning a file-like object
that `PIL.Image.save()` accepts directly meant the actual code change was small (a handful of
`os.makedirs`/`image.save(path)` calls routed through SMB-aware helpers when `--output-dir` is a
UNC path).

**Scope decision**: `train.py` was deliberately *not* made SMB-aware — training does many
random-access passes per epoch over the same images, which would be much slower reading
repeatedly over the network than from local/staged disk. Only the collection scripts
(`run_iter.py`, `collect_dataset.py`) write to SMB; training data should be copied/staged locally
first.

**Operational note**: force-killing (`kill -9`) a running `run_iter.py` process can leave the
CARLA server stuck in synchronous mode with no client ticking it — the next connection attempt
times out on `register_vehicle`/`set_synchronous_mode`. Fix: connect fresh and set
`synchronous_mode = False`, then retry. Documented here because it cost real debugging time once
already.

---

## 2026-08-16 — New racetrack from centerline data: generate OpenDRIVE directly, not via OSM/osm2odr

**Context**: teammate provided the TUMFTM/racetrack-database repo (centerline + track-width CSVs
for ~25 real F1/DTM circuits) as a source for a custom track, distinct from Town04.

**Decision**: generate a `.xodr` OpenDRIVE file directly from the centerline+width polyline
(piecewise-linear geometry, one `<geometry>` per point with explicit x/y/heading, self-linked
closed-loop road, per-point `<width>` breakpoints), loaded live via
`client.generate_opendrive_world()`. Rejected the alternative of geolocating the real circuit and
converting real OpenStreetMap road data via CARLA's `osm2odr`.

**Why**: the direct-from-centerline approach uses only data already in hand (no external
geocoding/Overpass-API dependency), and confirmed via search that no ready-made
TUMFTM-racetrack-database-to-CARLA converter exists either way, so both paths were "build it
yourself" — the simpler one was preferred. Track chosen: Austin (COTA), on the smaller-file-size
end of the available list, moderate corner complexity (18 segments with >15deg heading change out
of 1102).

**Outcome**: live-validated against the real CARLA server — road parses correctly (1102
waypoints generated, exactly matching input point count), autopilot drives it without collisions.

---

## 2026-08-16 — Track scenery: hay bales/barrels/cones, not a plain road

**Context**: after the Austin track loaded and drove correctly, question raised: is a bare road
mesh (no barriers, no scenery — `wall_height=0.0`, nothing else spawned) good enough for
vision-based training?

**Decision**: no — added `spawn_track_scenery()`, placing hay-bale/barrel boundary markers every
~40m along both track edges (using the same per-point width data the `.xodr` was built from),
traffic cones at the 18 sharper corners, and sparse background props (garden lamps, street signs)
further outside the barrier line for visual variety.

**Why**: this isn't cosmetic. Branch B is trained to predict cross-track error and curvature *from
the camera image*; a boundary-less track gives it almost no visual signal for "where the edge is."
Checked the CARLA install's actual `static.prop.*` blueprint library first (96 assets) — no
dedicated guardrail/fence/tree assets exist in this install, but hay bales and barrels are
historically accurate racetrack safety barriers, so the asset constraint and the thematic choice
lined up.

**Outcome**: live-validated — 368 props spawn in ~0.3s with zero spawn collisions, and a full
episode with both scenery and `--randomize-spawn` active ran cleanly (vehicle spawned successfully
on the first attempt, in the middle of the decorated section of track).

---

## 2026-08-16 — CBF: paper is a reference for the idea, not a formula to copy

**Context**: direct line-by-line comparison against Kalaria et al.'s equations (prompted by the
user sharing paper excerpts, then the full PDF) surfaced multiple real discrepancies between the
paper's derivation and the current code (see `docs/related_work.md` for the itemized list).

**Decision**: do not treat matching the paper's exact coefficients as the goal. Any future CBF
rework should be independently re-derived from *our* actual vehicle dynamics model, with the paper
used as motivation and a point of comparison — not copied wholesale.

**Status**: analysis complete, documented; no code changes made yet pending a decision on whether
to (a) add the paper's missing steering term to `v_perp_dot` and re-derive a correct degree-2
constraint, or (b) keep the current simplified dynamics and correctly re-derive the degree-3
chain. Both are legitimate; neither has been chosen yet.

---

## 2026-08-17 — Batch 2 collection: stopped after target was reached but CARLA server hung

**Context**: continuing the overnight run, episode 14 (WetNoon, seed=14) ran for ~2h15m across
many in-process retries (a much higher error rate than any prior episode — repeated "destroyed
actor" errors plus one stall-watchdog trigger). Checked the SMB share directly throughout (not
relying on the client's own accounting) and confirmed the 50,000-frame target was actually crossed
mid-episode, at 50,044 frames. Shortly after, the log showed a new error class never seen before:
`time-out of 60000ms while waiting for the simulator` — the tqdm progress bar froze for a full
minute with zero tick progress, unlike every prior recovery which kept advancing between retries.

**Diagnosis**: opened a completely fresh `carla.Client` connection (no client-side session poisoning
possible) — it also timed out on `get_server_version()`. This is a hung CARLA *server* process on
the Windows host, not a client-side retry-logic issue; nothing in `run_iter.py` can fix that. Checked
the SMB share (a separate service on the same host) in parallel — fully responsive, final count
50,429 frames across ep0-ep14 — so this wasn't a wider host outage, just the CARLA process itself.

**Decision**: killed the stuck `run_iter.py` (pid 90669) and its parent `collect_dataset.py` (pid
86227) rather than wait for a server that showed no sign of self-recovering. Justified because (a)
the frame target was already cleared before the hang (50,429 ≥ 50,000), so no further collection was
needed, and (b) I have no remote-console access to the Windows box to restart the CARLA server
myself — waiting indefinitely for it to come back on its own wasn't a productive use of unattended
time. Moved directly on to training with the 50,429-frame dataset.

**Final collection tally** (`\\192.168.56.1\CarlaData`, `run0_ep{0..14}_images/`):
ep0=3701, ep1=2891, ep2=3517, ep3=3409, ep4=2564, ep5=1717, ep6=871, ep7=3224, ep8=2381, ep9=1536,
ep10=871, ep11=7096, ep12=5374, ep13=3561, ep14=7716 — **total 50,429**, spanning all 6 weather
conditions in `STANDARD_WEATHER` and randomized-spawn recovery perturbation on every episode.

---

## 2026-08-17 — Verified training pipeline against Kalaria et al. for unit/formula errors

**Context**: with training running, re-checked the paper (full PDF) specifically for whether
`train.py`'s target encoding and `run_iter.py`'s decoding of those predictions for the CBF are
internally consistent -- i.e., not "does it match the paper's exact math" (already settled, see
the "CBF: paper is a reference for the idea" entry above) but "is our own round-trip correct."

**Checked and confirmed correct**:
- State variables predicted by the 3 safety heads (`curvature, x, theta`) exactly match the
  paper's `[x, θ, c]` (Sec 3.3) -- `v`/`v_perp`/`ω` are read directly from the CARLA sensor in
  both places, never predicted, matching the paper's rationale (those come from IMU/wheel
  encoders, not vision).
- Unit round-trip: `train.py` encodes targets as `curvature/0.005`, `x/1`, `theta/5` (train.py:193);
  `run_iter.py` decodes with the exact inverse, including converting theta back from degrees to
  radians before any `sin`/`cos` call in the CBF math (run_iter.py:1010-1012). Steering follows the
  same save/train/decode pattern. Traced every conversion line-by-line -- no mismatched units, no
  missing conversions.
- MC-dropout uncertainty (`N_ITERS` rollouts, mean + spread per output, run_iter.py:957-1008) is
  the same epistemic-uncertainty scheme the paper cites from Gal & Ghahramani (2016).

**One architectural divergence noted (not a bug)**: the paper describes a single state-prediction
network `M_st` with 3 outputs; our code uses three fully separate ResNet18 backbones (one per
`curvature`/`x`/`theta`), each with its own optimizer -- 3x the params/compute vs. the paper's
shared-backbone design, trading efficiency for zero cross-task gradient interference. Deliberate
choice already implicit in the existing code, just made explicit here; worth a line in the thesis
methodology section. Not changed -- training was already in progress and this doesn't affect
correctness, just efficiency/architecture.

**Outcome**: no formula or code errors found in the training pipeline. Training continues as-is.

---

## 2026-08-17 — Fixed x_factor normalization; stopped the CPU training run to apply it

**Context**: follow-up to the verification pass above. The user asked directly why `x_factor=1`
couldn't just be corrected. Re-examined: `curvature_factor=0.005` and `theta_factor=5` both
deliberately rescale their targets to roughly unit scale before training; `x_factor=1` applies no
scaling, leaving the cross-track target in raw meters -- almost certainly an oversight rather than
a deliberate choice, since it's inconsistent with the other two and produces MSE loss values
2-3 orders of magnitude larger for no functional reason (each head has its own optimizer, so this
never affected training *correctness* -- confirmed in the prior entry -- just made the printed
loss hard to compare against the other 3 heads, and unstandardized for the thesis write-up).

**Decision**: the user chose to stop the in-progress CPU training run (~95 min into epoch 0) rather
than let it finish on the unnormalized scale, and fix `x_factor` properly first. Set
`x_factor = 7` in both `train.py` and `run_iter.py` (kept in sync -- `run_iter.py` decodes
`model_safety_2`'s output by multiplying back by this same constant, so the two must always match).
Chose 7 to match the fixed `boundary_half_width=7` already used for town04 in `path_plot.py`/
`compare_cbf.py`, rather than inventing a new constant -- roughly half the ~14m average Austin
corridor width, bringing typical cross-track values to O(1) scale like the other two heads.

**Status**: code fixed and verified (`py_compile` clean on both files). Training itself is on hold
-- the user is discussing GPU access (an RTX 5080 on a company-owned Windows machine) with a
teammate before deciding whether to restart on CPU again or wait for that. No training is running
right now pending that decision.
