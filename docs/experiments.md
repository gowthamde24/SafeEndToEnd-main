# Experiment Log

Running record of what was actually run, what was observed, and where the output lives. For
narrative results write-ups (e.g. the Phase 1 training/CBF comparison), see `README.md`'s Results
section — this log is the finer-grained trail underneath it, especially for infrastructure
validation that doesn't produce a results-directory artifact of its own.

---

## 2026-08-16 — Austin track: OpenDRIVE load + driveability

**What**: `client.generate_opendrive_world()` loaded against the live CARLA server
(`192.168.56.1:2000`) with `tracks/austin.xodr`.

**Result**: parsed successfully — `world.get_map().generate_waypoints(5.0)` returned exactly 1102
waypoints (matching the 1102-point input centerline), 1 topology edge (single closed-loop road as
designed). Spawned the ego vehicle, enabled autopilot, ran 400 ticks (20s sim time): traveled
147m from spawn with zero collisions, steady ~8 m/s.

**Conclusion**: track geometry and lane width are valid and driveable. No further tuning needed
at this stage.

---

## 2026-08-16 — Full episode test: `--track austin`, expert-autopilot collection

**What**: `run_iter.py -r 0 --track austin --host 192.168.56.1 --port 2000 --episode-tag
smoketest`, first attempt.

**Result**: crashed — `local variable 'cmd_brake' referenced before assignment`. Root cause: a
latent pre-existing bug (`cmd_brake` wasn't pre-initialized before the main loop, unlike
`cmd_steer`/`cmd_throttle`), which only manifests if the wait-period-end frame doesn't land on a
`frame % 5 == 0` boundary — apparently never hit before on Town04, hit immediately on Austin.
Fixed (`cmd_brake = 0` added to the pre-loop initialization).

**Retest**: clean run, zero errors, 78 image + 79 video frames with correctly-formatted
filenames.

---

## 2026-08-16 — SMB output-dir end-to-end

**What**: `run_iter.py --output-dir '\\192.168.56.1\CarlaData'` with `SMB_USERNAME`/
`SMB_PASSWORD` env vars set, writing over SMB via the `smbclient` library.

**Result**: first attempt failed on DNS (`\\CHERRY\...` — sandbox can't resolve that hostname;
switched to the IP). Second attempt hit CARLA RPC timeouts (`register_vehicle`,
`set_synchronous_mode`) — traced to the server being left in synchronous mode with no client
ticking it, from a previous test's `kill -9`. Reset via a fresh `client.get_world()` +
`synchronous_mode = False`. Third attempt: clean — confirmed via direct `smbclient.listdir()`
that 28 image + 29 video frames landed on the share with correct filenames and non-trivial file
sizes (~280KB avg).

**Process note**: cleanup script was run once while the collection process was still alive,
which triggered its retry loop and left one stray frame behind — caught and cleaned up properly
on the second pass. Lesson recorded in `docs/decisions.md`: stop the process, confirm it's dead,
*then* clean up.

---

## 2026-08-16 — Track scenery spawn timing/collision check

**What**: `spawn_track_scenery()` called standalone against a freshly generated Austin world.

**Result**: 368 props spawned in 0.3s, 0 skipped due to collision.

**Follow-up**: full episode with both scenery and `--randomize-spawn --seed 7` active — vehicle
spawned successfully on the first attempt (route_idx=331, in the middle of the decorated section),
75 frames collected, zero errors.

---

## 2026-08-16 — Batch 1 collection: fence post physically dragged by the vehicle

**What**: user watching the live CARLA window during batch 1 (episode 1, CloudyNoon) spotted a
`chainbarrier` post attached to and dragging behind the ego vehicle by its chain.

**Root cause**: `static.prop.*` blueprints simulate physics by default; `spawn_track_scenery()`
wasn't disabling it. A sufficiently close pass let the vehicle's collision snag the chain and drag
the whole post along the track.

**Risk assessed**: beyond the visual glitch, a dragged prop could plausibly distort the ego
vehicle's actual physics/steering behavior during the frames it happened in -- the recorded
"expert" telemetry for those frames would then not represent normal driving. Both episodes
collected in this batch so far (ep0, ep1 -- ~1.7k frames) were discarded rather than risk
including subtly-corrupted data this early in collection.

**Fix**: `actor.set_simulate_physics(False)` on every prop immediately after spawn in
`spawn_track_scenery()`'s `try_spawn()` helper. Verified the call executes without error across
all 2480 props in a fresh spawn (timing regression: 4.3s vs. 2.2s before, still negligible against
~15min episodes).

**Batch 1 restarted from ep0** with the physics fix in place.

---

## 2026-08-16 — Batch 1 collection: vehicle drifted through the wall, autopilot parked in the void

**What**: user watching the live CARLA window reported the car sitting motionless. Frame
filenames confirmed it (`frame_83_..._0_0.png` and `frame_156_..._0_0.png` had byte-identical
steering/speed fields, i.e. genuinely no motion for 70+ saved frames, not just a filename
coincidence).

**Diagnosis**: `world.get_actors()` needed a `world.wait_for_tick()` first to get a non-stale
actor list from a fresh diagnostic connection (a second, unrelated caching quirk noted here so it
doesn't get re-debugged from scratch next time). Once synced: vehicle control showed
`throttle=0, brake=1.0` -- the autopilot itself had stopped the car, not an external obstruction
(consistent with no props being spawned at all in this run). Vehicle position was 14.7m from the
nearest centerline point, where the corridor half-width is only 7.2m -- i.e. the car was sitting
~7.5m past the (0.4m-tall) wall, off the paved surface entirely.

**Root cause**: `wall_height=0.4` is curb height, not a real barrier -- a car can drive straight
over it. At some point during autopilot driving the vehicle drifted off-line, crossed the low
wall unimpeded, ended up in the trackless void beyond it, and the autopilot (which relies on the
road's own waypoint graph) had nothing left to navigate against and stopped.

**Fix**: `wall_height` raised 0.4 -> 1.5m. Verified physically, not just visually: spawned a
vehicle, pointed it directly at the wall, and drove full-throttle straight at it for 150 ticks
(7.5s). It travelled only ~5m before being physically stopped -- short of even reaching the true
7.5m lane edge, let alone passing through. Confirms the wall now genuinely contains the vehicle
rather than just visually suggesting a boundary.

**Batch 1 restarted from ep0** (again) with the taller wall. Both partial episodes from this run
discarded -- an unknown stretch of frames in ep1 recorded a stationary, autopilot-abandoned
vehicle, which is not usable training data.

---

## 2026-08-16 — Same ep1 stall recurred at the identical coordinates with the 0.8m wall

**What**: after tuning the wall to a realistic 0.8m (previous entries), the same "car not moving"
symptom recurred in ep1 (seed=1, weather=CloudyNoon). Frame filenames again showed
byte-identical telemetry across 100+ frames.

**Diagnosis**: queried the live vehicle via `world.wait_for_tick()` + `get_actors()` -- stopped
at `(514.4, -331.7)`, the *exact same coordinates* (to 1 decimal place) as the earlier wall_height=0.4
incident. Since this is deterministic across two different wall heights (one of which was already
confirmed, via direct testing, to physically stop a vehicle driven straight at it), the wall
height was never the actual root cause -- something about this specific seed=1 randomized-spawn
draw (route_idx=137, lateral=0.21m, yaw_offset=9.07deg) deterministically leads the CARLA
traffic-manager autopilot into a state it can't recover from on this custom single-lane
closed-loop OpenDRIVE road. Root-causing the traffic-manager behavior itself was judged not worth
the time against collection progress; treated as a design constraint to route around instead.

**Fix (two-part, in `run_iter.py`)**:
1. **Stall watchdog**: new `STALL_SPEED_THRESHOLD`/`STALL_TIMEOUT_SECONDS` constants. If
   `abs(current_speed) < 0.3` m/s for >=10 consecutive seconds after the startup window, the main
   loop now raises `RuntimeError` instead of grinding through the rest of the ~200s episode
   recording a parked car.
2. **Seed-varying retries**: `main()`'s retry loop previously called `exec_waypoint_nav_demo(args)`
   identically on every attempt, so a deterministic stall (like this one) would raise, get caught,
   and retry into the *identical* doomed spawn point forever. `exec_waypoint_nav_demo` now takes
   `spawn_seed_offset` (`main()` passes its retry-attempt counter), and the randomize-spawn `rng`
   is seeded from `args.seed + spawn_seed_offset` -- each retry after a stall now draws a
   different spawn point instead of repeating the same one.

**Not yet done**: root-causing *why* the traffic manager gets stuck at this specific draw. If
stalls turn out to be common enough to matter for total collection throughput, that's the next
place to look (candidates: something about the self-referencing closed-loop road link confusing
route planning, or a specific yaw-offset range near this route index). For now the watchdog+reseed
combination converts a hard hang into a quick, self-recovering skip.

**Batch 1 restarted from ep0** (discarding the otherwise-clean 878-frame ep0 too, for consistency
with every other restart in this log -- not because ep0 itself was affected).

---

## 2026-08-16 — Batch 1 (post speed-increase) running log: stall/error frequency and a recurring hotspot

**Context**: after raising driving speed to ~21 m/s average (see `docs/decisions.md`), the
"destroyed actor" transient error recurred noticeably more often than before the speed change --
observed 9+ times across one collection session, vs. roughly 1 per full batch previously. User
was informed and chose to let collection continue as-is rather than pause to investigate, since
the watchdog+reseed mechanism (see 2026-08-16 stall entry above) recovers every time without
manual intervention.

**New finding**: the stall watchdog fired multiple times during this run, and one location --
`(680.3, 1.8)` -- triggered a stall **twice**, from different randomize-spawn seeds/attempts. This
confirms it's a genuine recurring trouble spot on the track (something about the road/wall
geometry or traffic-manager path planning at that specific point), not random per-seed noise.
Not investigated further -- the watchdog handles it in 10s regardless of cause -- but worth
knowing if a proper root-cause pass ever happens on the traffic-manager stalling behavior.

**Outcome**: despite the elevated retry rate, net collection progress remained steady --
observed roughly 900-1500 frames accumulating between check-ins throughout the session with no
stuck/blocked state at any point. The self-recovery design is working as intended even under a
higher error rate than originally anticipated.

---

## Open / not yet run

- Full-scale `collect_dataset.py` run (50k-100k frames) — code and storage pipeline ready, not
  yet launched (multi-hour unattended run, pending go-ahead).
- CBF re-derivation / fix (see `docs/decisions.md`, "CBF: paper is a reference..." entry) — not
  started.
- Closed-loop wraparound handling for full-lap Austin episodes — not needed at current episode
  lengths, not implemented.
