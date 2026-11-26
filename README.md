# Search-and-Rescue Drone Simulation (Elodin)

A six-DOF quadrotor search-and-rescue simulation built on [Elodin](https://elodin.systems), an ECS-based aerospace/robotics simulation SDK. A drone autonomously searches a defined area for a heat-emitting target using a simulated thermal camera, approaches it once detected, and confirms the find by holding a stable hover over the target.

The project started as a working example of using Elodin end-to-end for an autonomy problem; while building it I found an SDK-level Cranelift bug that blocks multi-drone runs and built a Monte Carlo mutation-testing harness on top of the single-drone pipeline that catches real regressions in the guidance stack.

## What's in here

**Single-drone mission (reliable, demo-ready)**
- `sim.py` — world construction, entity spawning, camera registration.
- `guidance.py` — SEARCH → APPROACH → HOVER state machine, waypoint selection.
- `autopilot.py` — PD position/altitude control, camera-bearing-to-world-position math, ground-avoidance override.
- `control.py` — cascaded attitude and rate control (PID).
- `motors.py`, `sensors.py`, `mekf.py`, `telemetry.py` — rest of the flight stack.
- `main.py` — entry point, mission-complete detection, CSV telemetry logging.
- `flight_controller.py` — standalone scripted-mission interface (`go_to` / `hover` / `reached`), independent of the full SR state machine but sharing the same `autopilot.py` control laws.

**Monte Carlo mutation-testing harness (`monte_carlo/`)**
- `mc_sim.py` — fault catalog (8 fault conditions: 3 real regressions this project found and fixed + 4 designed perturbations + 1 control). Each fault is applied as a module-attribute monkeypatch before the compiled system is built, so the real source files on disk are never touched. See `FAULT_CATALOG.md` for the full list.
- `mc_main.py` — campaign entry point. Replaces the thermal-camera read with analytical FOV detection so the guidance-layer faults can be exercised independently of render-server rate limits (see the long comment above `_make_post_step()` for why).
- `campaign.toml`, `spec.toml` — 96 runs (12 trials × 8 fault conditions) with LHS-dispersed heat-source placement.
- `hooks/score.py` — 5 blind spec-based oracles (no oracle sees which fault was injected).
- `hooks/report.py` — per-fault pass-rate aggregation, CAUGHT/ESCAPED verdicts vs. control.
- `test_mc_faults.py` — unit test suite covering every fault's injection mechanics against the real unmodified `guidance.py`/`sensors.py`/`autopilot.py` sources (stubbed `elodin` module), plus hook math and synthetic detection geometry.

**Reproducibility artifacts**
- `run_trials2.sh` — batch runner for reliability testing (confirmed ~90%+ success over 40+ real trials of the single-drone pipeline).
- `cranelift_concatenate_bug_report.md` — full writeup of the SDK-level Cranelift crash blocking `N_DRONES>1`.

## Running it

```bash
# single-drone search-and-rescue mission
elodin run main.py

# flight controller standalone demo (scripted waypoint-to-waypoint)
elodin run flight_controller.py

# batch reliability testing (randomized target placement)
./run_trials2.sh 20

# full Monte Carlo fault campaign (~60 seconds wall clock)
elodin monte-carlo run monte_carlo/mc_main.py \
    --campaign monte_carlo/campaign.toml \
    --spec monte_carlo/spec.toml \
    --out dbs/mc-fault-campaign
cat dbs/mc-fault-campaign/fault_campaign_report.txt

# unit tests
python3 monte_carlo/test_mc_faults.py
```

## Flight control architecture

A from-scratch simulation analog of how a real flight-control stack is structured — not a reimplementation of any specific firmware:

- **Position/altitude loop** (`autopilot.py`) — PD control converting a target world position into a desired tilt (pitch/roll) and throttle correction, with a ground-avoidance override near the floor. Both `guidance.py` and `flight_controller.py` call this layer directly; one tuned control law sits behind both the SR mission and simple scripted missions, not two that can drift apart.
- **Attitude control** (`control.py`) — converts a desired attitude into body rates.
- **Rate control** (`control.py`) — PID loop converting desired body rates into motor commands.
- **Motor mixing** (`motors.py`) — maps commands to individual motor thrust/torque with per-motor tilt directions.
- **MEKF attitude estimation** (`mekf.py`) — IMU-driven state estimation feeding the attitude loop.

## Monte Carlo fault campaign

The single-drone pipeline is validated by 40+ randomized real-engine trials (`run_trials2.sh`). The MC harness in `monte_carlo/` goes further: it automatically flags whether a change to `guidance.py`/`sensors.py`/`autopilot.py` has broken something a reviewer would miss.

**Methodology (classic mutation testing applied to flight software):**
1. A **fault catalog** of 7 injected perturbations + a clean control. Each fault is a module-attribute monkeypatch (or, for one code-shape fault, a source-patched function variant pulled via `ast` from the real `guidance.py` and exec'd in guidance's own namespace with one line swapped — so the diff is provably exactly one expression, not a hand-copy that could drift).
2. **5 blind spec-based oracles** in `hooks/score.py` — mission completion, final accuracy, altitude envelope, no-premature-hover, no-mode-thrashing. None of them read the `fault` field.
3. **96 runs** (12 LHS-dispersed trials × 8 fault conditions) in ~60 wall seconds across 12 parallel workers.
4. **CAUGHT/ESCAPED verdicts** per fault, compared against the clean-flight control group's false-alarm rate.

**Current result:** 3 of 7 faults caught, each with the right oracle:
- `thermal_threshold_drift` → `oracle_mission_completes` dropped 92%→42% (CAUGHT as expected)
- `hold_point_freeze` (a real regression originally found by hand from `flight_log.csv`) → `oracle_final_accuracy` dropped 92%→58% (CAUGHT as expected)
- `imu_noise_spike` → `oracle_altitude_envelope` dropped 100%→0% (CAUGHT by a different oracle than hypothesized — informative: IMU noise manifests as a safety-envelope breach, not mode thrashing)

The 4 escaped faults (`camera_fov_mismatch`, `hover_entry_no_speed_check`, `tick_timing_unscaled`, `tilt_cap_relaxed`) all depend on sensor-noise amplification that analytical FOV detection doesn't reproduce — clearly identified next-step work for the harness.

## A genuine Elodin SDK bug, found along the way

Extending this to a multi-drone swarm (`N_DRONES > 1`) hit a hard crash in Elodin's own Cranelift JIT backend — not in anything in this repo. Full writeup with reproducer, debug-dump analysis, and the exact crash site in `lower_concatenate_nd` is in `cranelift_concatenate_bug_report.md`, filed upstream as [elodin-sys/elodin#ISSUE_NUMBER](ISSUE_URL_HERE).

What was ruled out before filing:
- Our own system composition (rebuilt from `out_sys = out_sys | sys` reuse to three independent `System` objects — same crash, same symbol name `inner_295`).
- A StableHLO-level shape mismatch (scanned all 316 `stablehlo.concatenate` ops in the dumped MLIR; 0 shape inconsistencies).

Narrowed down to: `cranelift-mlir`'s own value materialization produces a `Vec<Value>` with fewer entries than the declared `TensorType` implies, likely from a `BroadcastInDim` `DenseSplat` fold not being expanded when a downstream concatenate indexes into it. Suggested a one-line debug assertion at the top of `lower_concatenate_nd`'s `flat_out` loop to pinpoint the upstream producer.

As a result, `N_DRONES=1` is fully reliable and every demo above works; `N_DRONES>1` reproduces the SDK-level crash until it's fixed upstream.

## Debugging methodology

Every non-trivial change here — feature or fix — went through the same loop: build a minimal unit-test harness that imports the real source files under a stubbed `elodin` module, verify the behavior in isolation (including feeding the test the *broken* input to confirm the test actually catches the failure, not just that it passes), then validate against real multi-trial `elodin run` batches. Every bug I found in Elodin itself was root-caused by reading Elodin's own Rust source (`libs/nox-py/src/`, `libs/monte-carlo/src/`, `libs/s10/src/`, `libs/cranelift-mlir/src/`) at the installed `v0.20.0` tag, not from guessing.

## Built with

[Elodin](https://github.com/elodin-sys/elodin) — open-source aerospace/robotics simulation SDK.
