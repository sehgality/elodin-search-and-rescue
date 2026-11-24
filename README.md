# Search-and-Rescue Drone Simulation (Elodin)

A six-degree-of-freedom quadrotor search-and-rescue simulation built on [Elodin](https://elodin.systems), an ECS-based physics/robotics simulation SDK. A drone autonomously searches a defined area for a heat-emitting target using a simulated thermal camera, approaches it once detected, and confirms the find by holding a stable hover over the target.

## What it does

- **Search**: the drone flies a loiter pattern (a circular waypoint circuit) over a search area until its onboard thermal camera detects a heat signature.
- **Approach**: once a target is detected and centered in frame for a confirmation window, the drone breaks off its search pattern and flies toward it.
- **Hover-confirm**: on arrival, the drone holds a slow, stable hover directly over the target for a settle period before declaring mission complete.
- **Swarm mode** (`N_DRONES` env var): the search area's loiter circle can be split into N non-overlapping zones, with N drones patrolling their own zone in parallel instead of redundantly flying the same circuit. Each drone runs its own instance of the full guidance/control/physics stack, sharing one target.

Everything is built on Elodin's ECS model: physics (`el.six_dof`), guidance, attitude control, and sensing are all composed as independent systems (`@el.map`/`@el.system`) that Elodin runs against every matching entity automatically — adding more drones means spawning more entities with the same components, not writing new per-drone code paths.

## Project structure

- `sim.py` — world construction (entity spawning, camera registration, zone partitioning for swarm mode) and system composition.
- `guidance.py` — search/approach/hover state machine, waypoint selection, target detection logic.
- `control.py` — attitude and rate control (PID-based).
- `main.py` — entry point, mission-complete detection, CSV telemetry logging, CLI.

Run a single-drone mission:

```bash
elodin run main.py
```

Run a 3-drone swarm search (zone-partitioned):

```bash
N_DRONES=3 elodin run main.py
```

Run a batch of trials for reliability testing:

```bash
./run_trials2.sh 20   # 20 independent randomized-target trials
```

## Debugging and validation process

Every change in this project went through the same loop before being considered done: build a minimal unit-test harness (stubbing out Elodin/JAX modules where real hardware/engine behavior wasn't testable from a dev sandbox), verify the logic in isolation, then validate against real multi-trial `elodin run` batches with randomized target placement. Notably:

- A search-recovery bug where losing sight of the target mid-approach caused the drone to fly to the *globally* nearest waypoint rather than one within its own patrol area — fixed and verified against a jax-based test harness covering wraparound, zone-restriction, and multi-zone tiling cases.
- The multi-drone `post_step` mission logic was rewritten from a hardcoded single-drone implementation to a factory that tracks per-drone state independently, verified with mocked step-context harnesses before ever touching real hardware-adjacent code.

## A genuine Elodin SDK bug, found along the way

While extending this to a multi-drone swarm (`N_DRONES > 1`), we hit a hard crash in Elodin's own Cranelift JIT backend — not in anything in this repo. Full writeup, including the exact crash site in `lower_concatenate_nd`, what was ruled out (our own system composition, and a StableHLO-level shape mismatch — verified clean across all 316 concatenate ops in the failing compile), and what's narrowed down as the likely cause, is filed as [an issue against elodin-sys/elodin](ISSUE_LINK_HERE).

As a result, `N_DRONES=1` is fully reliable; `N_DRONES>1` reproduces the SDK-level crash until that's fixed upstream.

## Built with

[Elodin](https://github.com/elodin-sys/elodin) — an open-source aerospace/robotics simulation SDK.
