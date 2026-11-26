# Fault catalog: Monte Carlo mutation testing for the wildfire detection sim

Same idea as a mutation-testing harness posted in the Elodin Discord: run
the real mission many times under a dispersion of initial conditions,
once clean and once per injected fault, and check every run against a
small set of **spec-based oracles that never see which fault (if any) was
injected.** A fault "escapes" if no oracle's pass rate drops relative to
the clean control group; it's "caught" if one does.

This harness is intentionally scoped small: **8 fault conditions (7 real
faults + a `none` baseline) x 12 dispersion samples x 5 oracles = 96
runs**, all single-drone (`N_DRONES=1`).

## Why single-drone only

The originally-discussed regression to re-inject -- the multi-drone
search-recovery bug where losing the target snapped to the globally
nearest waypoint instead of one in this drone's own patrol zone -- is a
**no-op at `N_DRONES=1`**: a single drone's zone covers the entire loiter
circle, so "globally nearest" and "zone-nearest" are identical regardless
of whether the zone restriction is applied. Exercising it for real needs
`N_DRONES>1`, which currently reproduces the live Cranelift JIT crash this
project found and filed upstream (see `cranelift_concatenate_bug_report.md`).
So that fault is cataloged (`search_recovery_global_snap` in
`mc_sim.FAULT_CATALOG`) but left as a documented no-op, not silently
dropped, with the one-line change needed to re-enable it once the SDK bug
is fixed. The 7 faults actually wired in below are all independently real,
single-drone-observable, and -- in 3 cases -- previously-fixed bugs from
this same codebase.

## How a fault is injected

Every fault except one is a **monkeypatch of a module-level constant** in
`guidance.py` or `sensors.py`, applied in `mc_sim.build()` before
`sim.world()`/`sim.system()` are called. Elodin traces/compiles `@el.map`
systems by reading whatever those module globals are at trace time, so
patching the attribute before the system is built bakes the faulty value
into that run's compiled system. **The real source files on disk are
never touched** -- only that one Python process's already-imported module
object is mutated, and only for the duration of that one Monte Carlo run.

The one exception, `hold_point_freeze`, is a genuine code-shape bug (which
*expression* computes a value), not a wrong constant, so it can't be a
one-line monkeypatch. Instead, `mc_sim._build_hold_point_freeze_variant()`
pulls `waypoint_guidance`'s own source text out of `guidance.py` (via
`ast`, not a hand-typed copy), asserts the one targeted line appears
exactly once, swaps it, and `exec`s the result in a copy of `guidance.py`'s
own namespace. If `guidance.py` is ever edited such that the targeted line
no longer matches exactly once, this raises immediately instead of
silently testing a stale copy of the function.

## The catalog

| fault | category | what it does | real bug? |
|---|---|---|---|
| `none` | control | no patch -- clean-flight baseline for false-alarm-rate tracking | -- |
| `hover_entry_no_speed_check` | guidance | `HOVER_ENTRY_MAX_SPEED_MPS` patched to `+inf`, so the speed gate on committing to HOVER becomes a no-op | **yes** -- confirmed from a real `flight_log.csv` run (see `guidance.py`'s own comment above `HOVER_ENTRY_MAX_SPEED_MPS`): centered_ticks hit threshold at ~2.4 m/s while distance to the heat source was *increasing* |
| `tick_timing_unscaled` | guidance | `CENTERED_TICKS_REQUIRED`/`LOST_TICKS_REQUIRED` reverted from 300/150 back to 30/15 | **yes** -- the sim runs at 300 Hz, not the 30 Hz the original comments assumed; found and scaled 10x (see `guidance.py`'s comment above `CENTERED_TICKS_REQUIRED`) |
| `hold_point_freeze` | guidance | HOVER lock-on point freezes to the drone's own position instead of the camera's target estimate | **yes** -- confirmed from a real `flight_log.csv` run: froze ~1m from (and still moving past) the actual heat source, then held that wrong point for the rest of the flight |
| `tilt_cap_relaxed` | guidance | `APPROACH_MAX_TILT_RAD` relaxed from 0.06 rad (tuned so the rigidly-mounted camera stays pointed near the target) to the generic 0.30 rad default | not a shipped bug -- a design parameter whose value matters |
| `thermal_threshold_drift` | sensor | thermal camera's detection threshold (`WARM_THRESHOLD`) drifts from 100 to 165 | simulated sensor/calibration fault |
| `camera_fov_mismatch` | sensor | guidance's believed camera FOV (`CAMERA_FOV_DEG`) drifts to 55 deg while the camera's true FOV stays 80 deg | simulated sensor/mounting fault |
| `imu_noise_spike` | sensor | gyro/accel noise covariance scaled 25x | simulated sensor/vibration fault |
| `search_recovery_global_snap` | guidance | **documented but not wired in** -- see "Why single-drone only" above | **yes**, but blocked on the Cranelift crash |

## The oracles (`monte_carlo/hooks/score.py`)

Every oracle is a pure spec check over signals `mc_main.py`'s `post_step`
measures (mode, position, velocity, altitude) and hands to
`el.monte_carlo.result()` -- none of them read `fault`.

1. **`oracle_mission_completes`** -- did the run settle into HOVER at the
   confirm altitude within the time budget?
2. **`oracle_final_accuracy`** -- is the final horizontal offset from the
   heat source under 2.0 m?
3. **`oracle_no_premature_hover`** -- never committed to HOVER while both
   fast (>1 m/s) and far (>3 m) from the target.
4. **`oracle_altitude_envelope`** -- altitude stayed within
   [0.5 m, 10 m] for the whole flight.
5. **`oracle_no_thrash`** -- fewer than 6 mode switches (SEARCH <-> APPROACH
   <-> HOVER) over the whole flight.

`monte_carlo/hooks/report.py` groups every run by `fault`, computes each
oracle's pass rate per fault and for the `none` control group, and -- for
the 7 wired-in faults -- reports a documented-hypothesis verdict
(`EXPECTED_ORACLE` in `report.py`) of CAUGHT (as expected, or by a
different oracle than expected) or ESCAPED.

## What's verified vs. not

Verified in `test_mc_faults.py`, against the REAL unmodified
`guidance.py`/`sensors.py`/`autopilot.py` (stubbed `elodin`/jax-adjacent
modules only, same discipline as the rest of this project):
constant-patch faults mutate exactly the attribute(s) they claim and
nothing else; `hold_point_freeze`'s targeted line matches exactly once in
the real source and the resulting variant function measurably produces a
different `hold_point` than the real function under identical inputs;
`hooks/score.py`'s oracle math and `hooks/report.py`'s pass-rate/verdict
math against synthetic fixtures.

NOT verified (no Elodin engine in the environment this was built in):
that `el.monte_carlo.params_spec`/`params`/`result` behave exactly as
assumed (confirmed by reading elodin-sys/elodin's own
`examples/monte-carlo` at the v0.20.0 tag, not by running it),
`campaign.toml`/`spec.toml`'s exact schema against the real
`elodin monte-carlo` CLI, or whether `Params.get()` reads a meta_sweep
column back as `"fault"` or `"meta.fault"` (`mc_sim.build()` tries both).
This will be the first real run -- see `monte_carlo/mc_main.py`'s
docstring for the one specific thing to check if a run picks up the wrong
fault.
