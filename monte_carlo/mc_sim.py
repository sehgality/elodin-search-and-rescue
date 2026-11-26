"""Monte Carlo mutation-testing harness for the wildfire detection sim.

Methodology (same idea as the mutation-testing project posted in the
Discord -- fault catalog x spec-based oracles that never see which fault
was injected x a clean-flight control group for false-alarm-rate):

  - FAULT_CATALOG below names every fault this harness knows how to inject.
    Each one is applied as a *monkeypatch of a module-level constant* (or,
    for the one fault that's a genuine code-shape bug rather than a wrong
    number, a full alternate function -- see hold_point_freeze below)
    in guidance.py / sensors.py, applied BEFORE sim.world()/sim.system()
    are called. Elodin traces/compiles @el.map functions by reading
    whatever those module globals are at trace time, so patching the
    module attribute before the system is built bakes the faulty value
    into that run's compiled system -- the real guidance.py/sensors.py
    source files on disk are never touched.
  - Everything else (world construction, spawn, camera, control cascade)
    is the REAL sim.world()/sim.system() from this project, unmodified --
    the only thing that differs between runs is the one patched
    constant/function named by `fault`.
  - post_step tracks oracle-relevant signals over the whole flight and
    hands them to el.monte_carlo.result() on the last tick. The actual
    pass/fail oracle checks live in hooks/score.py, NOT here -- this file
    only measures, it never decides pass/fail, so the oracles are
    genuinely blind to which fault (if any) was injected.

Three of the eight catalog entries below are REGRESSIONS: real bugs this
project found, fixed, and documented in guidance.py's own comments
(confirmed against an actual flight_log.csv run in two cases) -- cataloging
them here means a future change that reintroduces one of them should get
caught by the oracles in hooks/score.py, not just by eyeballing a replay.

NOT included: the original multi-drone search-recovery bug (losing the
target snaps to the globally-nearest waypoint instead of one in this
drone's own zone) -- that bug is a no-op with N_DRONES=1 (a single drone's
zone IS the whole circle, so "global nearest" and "zone-nearest" are
identical) and this harness deliberately stays N_DRONES=1 throughout,
because N_DRONES>1 reproduces the live Cranelift compiler crash filed
upstream. It's still worth cataloging as DOCUMENTED-BUT-BLOCKED (see
FAULT_CATALOG["search_recovery_global_snap"]) so it's not forgotten, and
re-enabling it is a one-line addition to this harness once that SDK bug
is fixed and N_DRONES>1 is viable again.
"""

import os
import sys
import typing as ty

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import elodin as el
import autopilot
import guidance
import sensors
import sim
from config import Config, Control, Frame

# --- dispersion parameters (continuous, sampled by spec.toml's
# [monte_carlo.variables] -- these replace sim.py's own pure
# np.random.uniform heat-source placement with an MC-seeded, reproducible
# equivalent of the exact same math) ---
PARAMS = el.monte_carlo.params_spec(
    heat_angle_rad=el.monte_carlo.Param(float, default=0.0, min=0.0, max=2 * np.pi),
    heat_radius_frac=el.monte_carlo.Param(float, default=0.5, min=0.0, max=1.0),
)

# --- fault catalog -----------------------------------------------------
# key -> (category, one-line description, patch_fn). patch_fn takes no
# args and mutates module globals in place; it runs once per run, before
# sim.world()/sim.system() are called in build() below.

ACCURACY_BOUND_M = 2.0          # oracle_final_accuracy threshold
THRASH_LIMIT = 6                # oracle_no_thrash threshold
PREMATURE_HOVER_SPEED_MPS = 1.0  # oracle_no_premature_hover speed threshold
PREMATURE_HOVER_DIST_M = 3.0     # oracle_no_premature_hover distance threshold


def _patch_none():
    pass  # baseline / control -- no patch applied


def _patch_hover_entry_no_speed_check():
    # Regression: removes the speed gate on committing to HOVER. Real bug,
    # confirmed via an actual flight_log.csv run (see guidance.py's own
    # comment above HOVER_ENTRY_MAX_SPEED_MPS) -- centered_ticks hit its
    # threshold while the drone was moving ~2.4 m/s and distance to the
    # heat source was INCREASING. Setting the speed cap to +inf makes
    # "speed < HOVER_ENTRY_MAX_SPEED_MPS" always true, i.e. a no-op gate.
    guidance.HOVER_ENTRY_MAX_SPEED_MPS = float("inf")


def _patch_tick_timing_unscaled():
    # Regression: the sim runs at 300 Hz, not the 30 Hz the original
    # comments assumed -- CENTERED_TICKS_REQUIRED/LOST_TICKS_REQUIRED were
    # found firing 10x too fast and were scaled up 10x to fix it (see
    # guidance.py's comment above CENTERED_TICKS_REQUIRED). This fault
    # reverts that scaling.
    guidance.CENTERED_TICKS_REQUIRED = 30
    guidance.LOST_TICKS_REQUIRED = 15


def _patch_tilt_cap_relaxed():
    # Not a bug this project shipped -- a design parameter whose value
    # matters. APPROACH_MAX_TILT_RAD (0.06 rad) exists specifically to
    # keep the rigidly-mounted (no gimbal) camera pointed near the target
    # during tracking; this fault relaxes it back to the generic default
    # tilt cap (autopilot.MAX_TILT_RAD, 0.30 rad) to see whether the
    # oracles notice the camera losing lock under aggressive tracking.
    guidance.APPROACH_MAX_TILT_RAD = autopilot.MAX_TILT_RAD


def _patch_thermal_threshold_drift():
    # Sensor fault: the thermal camera's detection threshold (main.py's
    # WARM_THRESHOLD) drifts high, as if the camera's calibration shifted
    # or the target's apparent heat signature weakened. mc_main_post_step
    # below reads this from MC_STATE rather than importing main.py (main.py
    # runs its whole pipeline at import time, so it can't be imported as a
    # library -- see the module docstring in hooks/score.py for why this
    # harness has its own thin post_step instead).
    MC_STATE["warm_threshold"] = 165


def _patch_camera_fov_mismatch():
    # Sensor fault: guidance.py's belief about the camera's FOV (used to
    # convert a bearing into a world position in autopilot.bearing_to_world_xy)
    # drifts away from the camera's TRUE fov (hardcoded in sim.py's
    # world.sensor_camera(..., fov=80.0, ...), left untouched here) --
    # models a camera lens/mount miscalibration between flight software's
    # assumption and the physical sensor.
    guidance.CAMERA_FOV_DEG = 55.0


def _patch_imu_noise_spike():
    # Sensor fault: gyro/accel noise covariance spikes (e.g. vibration,
    # a degraded IMU) -- scales up the noise fed into the gyro/accel LPF
    # chain and, downstream, the MEKF attitude estimate.
    sensors.gyro_noise.noise_covariance *= 25.0
    sensors.accel_noise.noise_covariance *= 25.0


def _patch_hold_point_freeze():
    # Regression: freezes the HOVER lock-on point to the drone's own
    # current position instead of the camera's target estimate. Real bug,
    # confirmed from an actual flight_log.csv run (see guidance.py's
    # comment above the real `new_hold_point` line) -- a run froze
    # hold_point to the drone's own position while it was ~1m away from
    # (and still moving past) the actual heat source, then held that
    # wrong point for the rest of the flight. This is a code-shape bug,
    # not a wrong constant, so it needs a full alternate function rather
    # than a one-line patch -- see _hold_point_freeze_variant() and its
    # diff-guard test in test_mc_faults.py, which asserts this variant's
    # source differs from the real guidance.waypoint_guidance by exactly
    # the one intended expression.
    guidance.waypoint_guidance = _build_hold_point_freeze_variant()


HOLD_POINT_FREEZE_OLD_LINE = (
    "    new_hold_point = jnp.where(enter_hover, target_estimate, hold_point)"
)
HOLD_POINT_FREEZE_NEW_LINE = (
    "    new_hold_point = jnp.where(enter_hover, pos_xy, hold_point)"
    "  # FAULT (hold_point_freeze): should be target_estimate, see mc_sim.py"
)


def _get_function_source_by_ast(module, func_name: str) -> str:
    """Extracts func_name's exact source text (decorators excluded) from
    module's own .py FILE via ast, not via inspect.getsource() on the
    live (possibly @el.map-wrapped, possibly not plain-function) runtime
    object -- this only depends on the source file on disk, so it works
    regardless of what el.map's real decorator does to the object it
    returns."""
    import ast

    with open(module.__file__) as f:
        full_src = f.read()
    tree = ast.parse(full_src, filename=module.__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(full_src, node)
    raise RuntimeError(f"couldn't find def {func_name}(...) in {module.__file__}")


def _build_hold_point_freeze_variant():
    """Builds an @el.map system that is the REAL guidance.waypoint_guidance
    with exactly one line swapped -- not a hand-transcribed copy. It pulls
    waypoint_guidance's actual source text out of guidance.py (via ast, so
    it doesn't depend on how @el.map wraps the live object), asserts the
    targeted line appears in it EXACTLY once, replaces that one line, and
    execs the result in a copy of guidance's own module namespace (so
    every bare name -- CAMERA_FOV_DEG, WAYPOINTS, Config, autopilot, jnp,
    el, ... -- resolves exactly as it does in the real function). If
    guidance.py is ever edited such that the targeted line no longer
    matches exactly once, this raises immediately instead of silently
    drifting from the real function -- see test_mc_faults.py section 3.
    """
    src = _get_function_source_by_ast(guidance, "waypoint_guidance")

    count = src.count(HOLD_POINT_FREEZE_OLD_LINE)
    if count != 1:
        raise RuntimeError(
            f"expected the new_hold_point line to appear exactly once in "
            f"guidance.waypoint_guidance's source, found {count} -- "
            f"guidance.py has changed since this fault was written; update "
            f"HOLD_POINT_FREEZE_OLD_LINE in mc_sim.py"
        )
    patched_src = src.replace(HOLD_POINT_FREEZE_OLD_LINE, HOLD_POINT_FREEZE_NEW_LINE)
    patched_src = patched_src.replace(
        "def waypoint_guidance(", "def waypoint_guidance_hold_point_freeze(", 1
    )
    patched_src = "@el.map\n" + patched_src

    namespace = dict(vars(guidance))
    exec(compile(patched_src, "<mc_sim hold_point_freeze variant of guidance.waypoint_guidance>", "exec"), namespace)
    return namespace["waypoint_guidance_hold_point_freeze"]


FAULT_CATALOG: dict[str, tuple[str, str, ty.Callable[[], None]]] = {
    "none": ("control", "baseline -- no fault injected", _patch_none),
    "hover_entry_no_speed_check": (
        "guidance", "REGRESSION: HOVER entry speed gate disabled", _patch_hover_entry_no_speed_check,
    ),
    "tick_timing_unscaled": (
        "guidance", "REGRESSION: centered/lost-tick thresholds reverted to the wrong (30Hz-assumed) scale", _patch_tick_timing_unscaled,
    ),
    "hold_point_freeze": (
        "guidance", "REGRESSION: HOVER lock-on point freezes to own position, not the target estimate", _patch_hold_point_freeze,
    ),
    "tilt_cap_relaxed": (
        "guidance", "APPROACH tilt cap relaxed to the generic default (camera can swing off target)", _patch_tilt_cap_relaxed,
    ),
    "thermal_threshold_drift": (
        "sensor", "thermal camera detection threshold drifts high", _patch_thermal_threshold_drift,
    ),
    "camera_fov_mismatch": (
        "sensor", "guidance's believed camera FOV diverges from the camera's true FOV", _patch_camera_fov_mismatch,
    ),
    "imu_noise_spike": (
        "sensor", "gyro/accel noise covariance spikes 25x", _patch_imu_noise_spike,
    ),
    # Documented, not wired in -- see module docstring.
    "search_recovery_global_snap": (
        "guidance",
        "DOCUMENTED BUT BLOCKED: only observable with N_DRONES>1, which reproduces "
        "the live Cranelift crash (see cranelift_concatenate_bug_report.md). "
        "Re-enable once that upstream bug is fixed: patch guidance.waypoint_guidance "
        "to drop the `dists_to_wp = jnp.where(in_zone, dists_to_wp, jnp.inf)` zone "
        "restriction, and run this harness with N_DRONES=3+.",
        _patch_none,  # intentionally a no-op; see docstring
    ),
}

MC_STATE: dict[str, float] = {"warm_threshold": 100}  # read by the post_step detection logic


def apply_fault(fault: str) -> None:
    if fault not in FAULT_CATALOG:
        raise ValueError(f"unknown fault {fault!r}; known: {sorted(FAULT_CATALOG)}")
    _, _, patch_fn = FAULT_CATALOG[fault]
    patch_fn()


def resolve_fault(params: "el.monte_carlo.Params") -> str:
    """The ONE place that reads the fault out of a run's Params -- both
    build() below and mc_main.py call this, so they can't silently
    disagree on which key resolved.

    ROOT CAUSE FOUND (read directly from elodin's own PyO3 source,
    libs/nox-py/src/monte_carlo.rs, at the installed v0.20.0 tag): a
    [meta_sweep] value is NEVER reachable via `Params.get()`, under any
    key spelling. `sample.py` (libs/nox-py/python/elodin/monte_carlo/
    sample.py) writes it into plan.csv as a `meta.<key>` column; the Rust
    orchestrator's `read_plan()` strips that prefix and carries it as
    `PlanRow.meta["<key>"]`; and `write_run_context()` serializes that into
    the per-run context JSON as a SEPARATE `"meta"` map, sibling to
    `"params"` -- not merged into it. The native `PyParams.get(key,
    default)` method (see `impl PyParams` in monte_carlo.rs) only ever
    looks in `self.params` (sampled/[monte_carlo.variables] values); it
    never consults `self.meta` at all. The only way to reach a meta_sweep
    value from Python is the separate `params.meta` property, a plain
    dict built straight from that same map. So every one of the three
    spellings tried earlier ("meta.fault", "fault", "meta_fault") via
    `params.get(...)` was always going to return the default -- the key
    name was never the bug, the accessor was.
    """
    meta = params.meta
    fault = meta.get("fault", "none") if isinstance(meta, dict) else "none"
    if not fault:
        fault = "none"
    if os.environ.get("MC_DEBUG") == "1":
        print(f"[mc_debug] resolve_fault: params.meta={meta!r} -> fault={fault!r}", flush=True)
    return fault


def _configure_global() -> None:
    """Builds and sets the global Config, exactly matching main.py's
    TALON_QUAD_CONFIG -- sim.world()/sim.system() both read Config.GLOBAL
    (motor layout, gains, sim rate, ...) and raise ValueError("No global
    config set") if nothing has called .set_as_global() first (confirmed
    by a real run). main.py can't be imported to reuse its
    TALON_QUAD_CONFIG directly -- it isn't a library, it runs the whole
    simulation at import time (world()/world.run() at module scope) -- so
    this is a deliberate, commented duplicate of that one Config literal,
    same as flight_controller.py's own _configure() already does.
    """
    q_tilt_right = el.Quaternion.from_axis_angle(axis=np.array([1.0, 0.0, 0.0]), angle=np.deg2rad(5.0))
    q_tilt_left = el.Quaternion.from_axis_angle(axis=np.array([1.0, 0.0, 0.0]), angle=np.deg2rad(-5.0))
    q_tilt_back = el.Quaternion.from_axis_angle(axis=np.array([0.0, 1.0, 0.0]), angle=np.deg2rad(-3.75))
    up = np.array([0.0, 0.0, 1.0])
    motor_thrust_directions = jnp.array(
        [
            q_tilt_right @ up,
            (q_tilt_left * q_tilt_back) @ up,
            q_tilt_left @ up,
            (q_tilt_right * q_tilt_back) @ up,
        ]
    )
    Config(
        control=Control(
            rate_pid_gains=np.array([[0.32, 0.32, 1.10], [0.05, 0.05, 0.08], [0.12, 0.08, 0.01]]),
            angle_p_gains=np.array([4.0, 4.0, 1.0]),
            motor_thrust_exponent=0.833,
            motor_thrust_hover=0.689,
            attitude_control_input_tc=0.2,
            pilot_yaw_rate_tc=0.25,
        ),
        drone_glb="talon-quad-v2.glb",
        mass=2.586,
        inertia_diagonal=np.array([0.0854, 0.1149, 0.1604]),
        start_pos=np.array([0.0, 0.0, 2.0]),
        start_euler_angles=np.array([0.0, 0.0, 0.0]),
        motor_positions=np.array(
            [
                [0.26, -0.26, 0.26, -0.26],
                [-0.2075, 0.2015, 0.2075, -0.2015],
                [-0.0215, 0.0215, -0.0215, 0.0215],
            ]
        ).T,
        motor_thrust_directions=np.array(motor_thrust_directions),
        motor_thrust_curve_path="./motor_thrust_curve.csv",
        simulation_rate=300.0,
        frame=Frame.QUAD_X,
        fast_loop_time_step=(1.0 / 900.0),
        simulation_time=90.0,
        sensor_noise=True,
    ).set_as_global()


def build(params: "el.monte_carlo.Params") -> tuple["el.World", "el.System"]:
    _configure_global()

    fault = resolve_fault(params)
    apply_fault(fault)

    angle = params.get("heat_angle_rad", 0.0)
    radius_frac = params.get("heat_radius_frac", 0.5)
    center = np.asarray(guidance.SEARCH_AREA_CENTER)
    r = guidance.SEARCH_AREA_RADIUS_M * np.sqrt(radius_frac)
    heat_xy = center + r * np.array([np.cos(angle), np.sin(angle)])
    os.environ["DEBUG_STATIC"] = "1"
    os.environ["DEBUG_HEAT_X"] = str(heat_xy[0])
    os.environ["DEBUG_HEAT_Y"] = str(heat_xy[1])
    os.environ["DEBUG_HEAT_Z"] = "0.0"
    # sim.py reads N_DRONES from the environment only once, at import time
    # (it's a module-level constant, not re-read inside world()) -- setting
    # the env var here would be too late. Patch the already-imported
    # module's constant directly instead. See module docstring: this
    # harness is single-drone-only.
    sim.N_DRONES = 1

    world, drone_names = sim.world()
    system = sim.system()
    return world, system
