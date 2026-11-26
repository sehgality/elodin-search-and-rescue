# Search-and-Rescue Drone Simulation (Elodin)

A six-degree-of-freedom quadrotor search-and-rescue simulation built on [Elodin](https://elodin.systems), an ECS-based aerospace/robotics simulation SDK. A drone autonomously searches a defined area for a heat-emitting target using a simulated thermal camera, tracks the target once detected, and confirms the find by descending to a stable hover directly above it.

## Background

Inspired by my work on the Cal Poly × Northrop Grumman collaboration (NGCP), where our team is building an autonomous air vehicle for wildfire detection. This project is a quadcopter analog of that same kind of mission: autonomous area search, onboard thermal detection, and autonomous arrival at the target. It is built as a reproducible simulation so the mission logic can be exercised, fault-injected, and regression-tested without flying real hardware for every change.

## The mission, end to end

Each run plays out the same arc, with the only external input being *where* the heat source spawns:

1. **The world.** `sim.py` spawns the drone at the origin (airborne at z=2m) and a static heat-source entity somewhere inside a 15m-radius search disk. The heat source is an `el.Archetype` with world position but no physics. Its spawn position is chosen uniformly at random each run, or placed deterministically via `DEBUG_HEAT_X`/`DEBUG_HEAT_Y` env vars for repeatable testing. The drone does **not** know where it is.
2. **SEARCH.** The drone climbs to a cruise altitude of 8m and flies a 32-waypoint loiter circle (10m-radius circuit, patrolled nose-forward via the autopilot's `goto_xy`). All the while, the drone's downward-facing simulated thermal camera (`world.sensor_camera(..., effect="lwir", fps=30)`) is rendering 128×128 LWIR frames at 30 Hz, and the `post_step` hook in `main.py` reads each new frame via `ctx.read_msg("drone.thermal_cam", ...)`, thresholds the red channel (`WARM_THRESHOLD = 100`), and computes the centroid of warm pixels to produce a normalized `(bearing_x, bearing_y)` offset of the heat source from image center. That bearing, plus a "detected" flag, is written as a `target_detect` component.
3. **APPROACH.** The instant `target_detect[0] > 0.5` fires, the guidance state machine in `guidance.py` transitions out of SEARCH. The guidance layer feeds an estimated world `(x, y)` of the heat source as the target for `goto_xy`. That estimate comes from `autopilot.bearing_to_world_xy`, a pinhole-camera trig conversion that takes `(bearing, drone_altitude, camera_FOV=80°)` and spits out the world coordinates. The drone brakes first (a short velocity-damping window targeting its own current position to shed leftover search momentum), then tracks the live estimate. HOVER entry is gated by a combined bearing-magnitude-plus-low-speed check: centered within ±0.3 normalized bearing, airspeed under 0.5 m/s, sustained for ~1 second (300 ticks at 300 Hz). Crucially, bearing alone is not enough. The drone has to actually have stopped moving, which keeps it from committing to HOVER mid-flyover.
4. **HOVER / land.** When the centered-and-slow condition fires, the guidance layer freezes onto the current `target_estimate` as the hold point, cuts the target altitude from 8m down to 1m, and lets the autopilot settle. Mission complete fires when the drone has sustained HOVER mode at altitude 1m ± 0.2m with 3D speed below 0.3 m/s for at least 30 consecutive ticks. At that point `main.py` prints a `[MISSION COMPLETE]` line with the final offset from the real heat-source position and exits the sim via `os._exit(0)`.

The whole arc is built on Elodin's ECS model. Physics (`el.six_dof`), guidance, attitude control, sensing, and the thermal camera are all composed as independent systems (`@el.map`/`@el.system`) that Elodin runs against every matching entity automatically. Adding more drones means spawning more entities with the same components, not writing new per-drone code paths.

## What's in here

**Single-drone mission (reliable, demo-ready)**
- `sim.py`: world construction, entity spawning, heat-source placement, camera registration.
- `guidance.py`: SEARCH → APPROACH → HOVER state machine, waypoint selection, bearing-to-target-estimate conversion.
- `autopilot.py`: PD position/altitude control (`goto_xy`, `hold_altitude`), pinhole camera-bearing-to-world-position math (`bearing_to_world_xy`), ground-avoidance override.
- `control.py`: cascaded attitude and rate control (PID).
- `motors.py`, `sensors.py`, `mekf.py`, `telemetry.py`: rest of the flight stack. Motor mixing with per-motor tilt directions, IMU with configurable noise, MEKF attitude estimation, telemetry logging.
- `main.py`: entry point, mission-complete detection, CSV flight-telemetry logging.
- `flight_controller.py`: standalone scripted-mission interface (`go_to` / `hover` / `reached`), independent of the search-and-rescue state machine but sharing the same `autopilot.py` control laws. Lets you fly scripted waypoint-to-waypoint missions with a few lines of Python.

**Monte Carlo mutation-testing harness (`monte_carlo/`)**
- `mc_sim.py`: fault catalog (8 fault conditions). 3 real regressions this project found and fixed during development, 4 designed perturbations, 1 clean-flight control. Each fault is applied as a monkeypatch of one module-level constant. The one exception is a code-shape fault, which is a source-patched variant of `guidance.waypoint_guidance` extracted via `ast` from the real `guidance.py` and exec'd in guidance's own namespace with exactly one line swapped. See `FAULT_CATALOG.md` for the full list.
- `mc_main.py`: campaign entry point. Replaces the thermal-camera read with analytical FOV-cone detection so the guidance-layer faults can be exercised independently of render-server rate limits under parallel orchestration. The long comment above `_make_post_step()` explains why in detail, with the real-run evidence that forced this design.
- `campaign.toml`, `spec.toml`: 96 runs (12 trials × 8 fault conditions) with Latin-hypercube-sampled heat-source placement inside the search disk.
- `hooks/score.py`: 5 blind spec-based oracles (no oracle sees which fault was injected).
- `hooks/report.py`: per-fault pass-rate aggregation, CAUGHT/ESCAPED verdicts vs. the clean-flight control group's false-alarm rate.
- `test_mc_faults.py`: ~470-line unit test suite covering every fault's injection mechanics against the real unmodified `guidance.py`/`sensors.py`/`autopilot.py` sources (stubbed `elodin` module), plus hook math and synthetic detection geometry with a bearing-roundtrip property test.

**Reproducibility artifacts**
- `run_trials2.sh`: batch runner for reliability testing against randomized target placements.
- `cranelift_concatenate_bug_report.md`: full writeup of the SDK-level Cranelift crash blocking `N_DRONES>1`, including reproducer and MLIR-level analysis.

## Running it

```bash
# Single-drone search-and-rescue mission
elodin run main.py

# Standalone flight-controller demo (scripted waypoint-to-waypoint)
elodin run flight_controller.py

# Batch reliability test with randomized target placement
./run_trials2.sh 20

# Full Monte Carlo fault campaign (~60 seconds wall clock, 96 runs, 12 parallel workers)
elodin monte-carlo run monte_carlo/mc_main.py \
    --campaign monte_carlo/campaign.toml \
    --spec monte_carlo/spec.toml \
    --out dbs/mc-fault-campaign
cat dbs/mc-fault-campaign/fault_campaign_report.txt

# Unit tests
python3 monte_carlo/test_mc_faults.py
```

## Flight control architecture

The inner-loop attitude/rate/motor cascade (`control.py`, `motors.py`, `sensors.py`, `mekf.py`) is adapted from Elodin's own `examples/drone` template, which provides a proven cascaded PID attitude controller with INDI rate control, motor thrust modeling, IMU with noise, and MEKF attitude estimation. What this project adds on top of that template is the outer-loop position/altitude autopilot, the mission guidance state machine, the thermal camera integration, the standalone flight controller, and the Monte Carlo fault campaign.

**Added on top (project-specific):**
- **Position / altitude outer loop** (`autopilot.py`): PD control converting a target world position into a desired tilt (pitch/roll) and throttle correction, with a ground-avoidance override near the floor. It also contains `bearing_to_world_xy`, the pinhole-camera trig that converts a normalized bearing into a world coordinate given the drone's altitude and the camera's FOV. Both `guidance.py` and `flight_controller.py` call this layer directly, so one tuned control law sits behind both the SR mission and simple scripted missions, not two that can drift apart.
- **Mission guidance** (`guidance.py`): the SEARCH → APPROACH → HOVER state machine, loiter-pattern waypoint selection, HOVER-entry gating, and HOVER-lock-on logic described in the mission walkthrough above.
- **Thermal camera + detection** (`sim.py` registration, `main.py` post_step): simulated LWIR sensor camera attached to the drone, red-channel thresholding, pixel-centroid-to-bearing conversion, and `target_detect` component writeback.
- **Standalone flight controller** (`flight_controller.py`): a simpler `go_to` / `hover` / `reached` API for scripted waypoint missions, delegating to the same `autopilot.py` control laws so the two mission types can't drift apart.
- **Monte Carlo fault campaign** (`monte_carlo/`): the full mutation-testing harness described below.

**From Elodin's template (inner loop):**
- **Attitude control** (`control.py`): converts a desired attitude quaternion into body rates.
- **Rate control** (`control.py`): PID loop converting desired body rates into motor commands, run at 900 Hz (3× the 300 Hz outer-loop rate).
- **Motor mixing** (`motors.py`): maps commands to individual motor thrust/torque using per-motor tilt directions for a real Talon-quad-V2 airframe.
- **MEKF attitude estimation** (`mekf.py`): IMU-driven state estimation feeding the attitude loop.
- **Sensor simulation** (`sensors.py`): IMU gyro/accel with configurable noise, low-pass filtering, bias modeling.

## Monte Carlo fault campaign

The single-drone pipeline is validated by randomized real-engine trials (`run_trials2.sh`, confirmed working across 40+ runs). The MC harness in `monte_carlo/` goes further. It automatically flags whether a change to `guidance.py`/`sensors.py`/`autopilot.py` has broken something a reviewer would miss.

**Methodology (classic mutation testing applied to flight software):**
1. A **fault catalog** of 7 injected perturbations plus a clean control group. Each fault is a module-attribute monkeypatch. The one code-shape fault is a source-patched function variant pulled via `ast` from the real `guidance.py` and exec'd in guidance's own namespace with exactly one line swapped, so the diff is provably exactly one expression and not a hand-copy that could drift.
2. **5 blind spec-based oracles** in `hooks/score.py`: mission completion, final accuracy, altitude envelope, no-premature-hover, no-mode-thrashing. None of them read the `fault` field.
3. **96 runs** (12 LHS-dispersed trials × 8 fault conditions) in ~60 wall seconds across 12 parallel workers.
4. **CAUGHT/ESCAPED verdicts** per fault, compared against the clean-flight control group's false-alarm rate.

**Latest result: 3 of 7 injected faults caught, each by the right oracle:**
- `thermal_threshold_drift` → `oracle_mission_completes` dropped 92% → 42% (CAUGHT as expected).
- `hold_point_freeze`, a real regression originally found by hand from `flight_log.csv`, → `oracle_final_accuracy` dropped 92% → 58% (CAUGHT as expected).
- `imu_noise_spike` → `oracle_altitude_envelope` dropped 100% → 0% (CAUGHT by a different oracle than hypothesized, which is itself informative: IMU noise manifests here as a safety-envelope breach, not mode thrashing).

The 4 escaped faults (`camera_fov_mismatch`, `hover_entry_no_speed_check`, `tick_timing_unscaled`, `tilt_cap_relaxed`) all depend on sensor-noise amplification that analytical FOV detection doesn't reproduce. Clearly identified next-step work: either inject synthetic Gaussian jitter on the analytical bearing, or run a smaller sub-campaign under the real pixel-based thermal camera at `generate_real_time=True`.

## A genuine Elodin SDK bug, found along the way

Extending this project to a multi-drone swarm (`N_DRONES > 1`) hit a hard crash in Elodin's own Cranelift JIT backend. Not in anything in this repo. The architecture supports arbitrary N drones patrolling N non-overlapping zones of the shared loiter circle; in practice only `N_DRONES=1` runs today because of this upstream crash.

Full reproducer, debug-dump analysis, and the exact crash site in `lower_concatenate_nd` are in `cranelift_concatenate_bug_report.md`.

**What was ruled out before filing:**
- Our own system composition. I rebuilt from `out_sys = out_sys | sys` reuse to three fully independent `System` objects, and the same crash recurred with the same symbol name `inner_295`.
- A StableHLO-level shape mismatch. I scanned all 316 `stablehlo.concatenate` ops in the dumped MLIR; 0 shape inconsistencies.

**Narrowed down to:** `cranelift-mlir`'s own value materialization produces a `Vec<Value>` with fewer entries than the declared `TensorType` implies, likely from a `BroadcastInDim` `DenseSplat` fold not being expanded when a downstream concatenate indexes into it. Suggested a one-line debug assertion at the top of `lower_concatenate_nd`'s `flat_out` loop to pinpoint the upstream producer immediately rather than failing at an opaque flat index downstream.

To be filed upstream as a GitHub issue against `elodin-sys/elodin`.

## Debugging methodology

Every non-trivial change here (feature or fix) went through the same loop. Build a minimal unit-test harness that imports the real source files under a stubbed `elodin` module. Verify the behavior in isolation, which includes feeding the test the *broken* input first to confirm the test actually catches the failure and not just that it passes. Then validate against real multi-trial `elodin run` batches. Every bug I found inside Elodin itself was root-caused by reading Elodin's own Rust source (`libs/nox-py/src/`, `libs/monte-carlo/src/`, `libs/s10/src/`, `libs/cranelift-mlir/src/`) at the installed `v0.20.0` tag, not from guessing.

## Built with

[Elodin](https://github.com/elodin-sys/elodin): open-source aerospace/robotics simulation SDK.
