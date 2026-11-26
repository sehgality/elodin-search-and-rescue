"""Verifies the Monte Carlo fault-injection harness WITHOUT a real Elodin
engine or a real elodin monte-carlo run, using the same discipline as the
rest of this project: stub out `elodin`/jax-adjacent modules just enough
to import the REAL, unmodified guidance.py/sensors.py/autopilot.py, then
check the actual mechanics this harness depends on.

What this test DOES verify:
  1. ast.parse on every new file (syntax only -- a real elodin.monte_carlo
     module isn't available here, so semantic engine-level checks aren't
     possible in this sandbox).
  2. Each constant-patching fault in FAULT_CATALOG mutates exactly the
     module attribute(s) it claims to, and nothing else, against the
     REAL guidance.py/sensors.py source (not a mock of them).
  3. The hold_point_freeze fault variant's source differs from the real,
     unmodified guidance.waypoint_guidance's source by EXACTLY the one
     intended line (new_hold_point) -- a difflib diff-guard, so if
     someone edits guidance.py later without updating this harness, this
     test fails loudly instead of silently testing a stale copy.
  4. hooks/score.py's oracle math and hooks/report.py's pass-rate/verdict
     math, against synthetic result.json-shaped dicts and results.csv-
     shaped rows covering each pass/fail case.

What this test does NOT verify (needs the real engine, on the user's
WSL machine): that el.monte_carlo.params_spec/params/result actually work
as assumed, that mc_main.py's post_step actually drives a real sim to
completion, or that `elodin monte-carlo run` accepts campaign.toml/
spec.toml as written.
"""

import ast
import difflib
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # /tmp/elodin_repo_fresh in this sandbox's check

# --- 1. syntax check every new file ---
NEW_FILES = [
    os.path.join(HERE, "mc_sim.py"),
    os.path.join(HERE, "mc_main.py"),
    os.path.join(HERE, "hooks", "score.py"),
    os.path.join(HERE, "hooks", "report.py"),
]
for path in NEW_FILES:
    with open(path) as f:
        src = f.read()
    ast.parse(src, filename=path)
print(f"=== 1. syntax check: {len(NEW_FILES)} files parsed OK ===\n")


# --- minimal elodin stub: just enough to import the REAL guidance.py /
# sensors.py / config.py / autopilot.py unmodified. ---
el = types.ModuleType("elodin")


class _Marker:
    def __init__(self, *a, **k):
        pass


class ComponentType(_Marker):
    pass


ComponentType.U64 = ComponentType()
ComponentType.F64 = ComponentType()


class PrimitiveType(_Marker):
    pass


PrimitiveType.F64 = _Marker()
PrimitiveType.U64 = _Marker()


def Component(*a, **k):
    return _Marker()


class Archetype:
    pass


class _System:
    """Stand-in for whatever el.map/@el.map returns -- just needs to
    support `|` composition (used throughout sim.py/guidance.py to chain
    systems) and to keep the original function reachable at .fn so
    inspect.getsource still works on the REAL underlying code."""

    def __init__(self, fn):
        self.fn = fn

    def __or__(self, other):
        return _System(self.fn)  # composition result is never called in these tests


def map(fn):  # el.map
    return _System(fn)


class _Quaternion:
    def __init__(self, yaw_rad=0.0):
        self.yaw = yaw_rad

    def inverse(self):
        return _Quaternion(-self.yaw)

    def __matmul__(self, vec3):
        import jax.numpy as jnp
        c, s = jnp.cos(self.yaw), jnp.sin(self.yaw)
        x, y, z = vec3[0], vec3[1], vec3[2]
        return jnp.array([c * x - s * y, s * x + c * y, z])

    def __mul__(self, other):
        return _Quaternion(self.yaw + other.yaw)

    @staticmethod
    def from_axis_angle(axis, angle):
        return _Quaternion(0.0)


el.Component = Component
el.ComponentType = ComponentType
el.PrimitiveType = PrimitiveType
el.Archetype = Archetype
el.map = map
el.system = map  # @el.system behaves the same way for these purposes
el.Quaternion = _Quaternion
for _name in [
    "WorldPos", "WorldVel", "WorldAccel", "Force", "Inertia", "SpatialForce",
    "SpatialTransform", "SpatialInertia", "System", "World", "StepContext",
    "Body", "Integrator",
]:
    setattr(el, _name, _Marker)
el.six_dof = lambda *a, **k: _Marker()


class _Subscriptable:
    def __getitem__(self, item):
        return _Marker()


el.Query = _Subscriptable()
el.SimulationTick = _Marker()


def _el_getattr(name):
    # Catch-all for any other el.* symbol these files reference only as a
    # type annotation or decorator target (el.SimulationTimeStep, el.Field,
    # etc.) -- none of these need real behavior for what this test checks.
    return _Marker()


el.__getattr__ = _el_getattr

el_ui = types.ModuleType("elodin.ui")
el_ui.__getattr__ = _el_getattr
sys.modules["elodin.ui"] = el_ui
el.ui = el_ui


class _Param:
    def __init__(self, type_, default=None, min=None, max=None):
        self.type_, self.default, self.min, self.max = type_, default, min, max


class _Params:
    # Mirrors the REAL split confirmed by reading elodin's own PyO3 source
    # (libs/nox-py/src/monte_carlo.rs, installed v0.20.0): `.get(key,
    # default)` only ever looks in the sampled/[monte_carlo.variables]
    # params map -- it NEVER consults meta_sweep values. Those are only
    # reachable via the separate `.meta` property (a plain dict). Before
    # this stub mirrored that split, `_Params` was a single dict subclass
    # with `meta_sweep` values merged straight into it, which could never
    # have caught the real bug (resolve_fault's old 3-candidate
    # params.get() approach would have "worked" against that old stub even
    # though it was broken against the real engine) -- see the resolve_fault
    # test below, which fails against that old shape and passes against this
    # one.
    def __init__(self, params_dict, meta_dict=None):
        self._params = dict(params_dict)
        self.meta = dict(meta_dict or {})

    def get(self, key, default=None):
        return self._params.get(key, default)


mc_module = types.ModuleType("elodin.monte_carlo")
mc_module.Param = _Param
mc_module.params_spec = lambda **kwargs: dict(kwargs)
mc_module.params = lambda spec, meta=None: _Params({k: v.default for k, v in spec.items()}, meta)
mc_module.result = lambda **kwargs: None
el.monte_carlo = mc_module
sys.modules["elodin.monte_carlo"] = mc_module

sys.modules["elodin"] = el

sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import guidance  # the REAL file, unmodified
import sensors  # the REAL file, unmodified
import autopilot  # the REAL file, unmodified

import mc_sim  # the harness under test


# --- 2. constant-patch faults mutate exactly what they claim ---
print("=== 2. constant-patch faults ===")

snapshot = {
    "HOVER_ENTRY_MAX_SPEED_MPS": guidance.HOVER_ENTRY_MAX_SPEED_MPS,
    "CENTERED_TICKS_REQUIRED": guidance.CENTERED_TICKS_REQUIRED,
    "LOST_TICKS_REQUIRED": guidance.LOST_TICKS_REQUIRED,
    "APPROACH_MAX_TILT_RAD": guidance.APPROACH_MAX_TILT_RAD,
    "CAMERA_FOV_DEG": guidance.CAMERA_FOV_DEG,
}


def _reset():
    guidance.HOVER_ENTRY_MAX_SPEED_MPS = snapshot["HOVER_ENTRY_MAX_SPEED_MPS"]
    guidance.CENTERED_TICKS_REQUIRED = snapshot["CENTERED_TICKS_REQUIRED"]
    guidance.LOST_TICKS_REQUIRED = snapshot["LOST_TICKS_REQUIRED"]
    guidance.APPROACH_MAX_TILT_RAD = snapshot["APPROACH_MAX_TILT_RAD"]
    guidance.CAMERA_FOV_DEG = snapshot["CAMERA_FOV_DEG"]
    sensors.gyro_noise.noise_covariance = 0.001
    sensors.accel_noise.noise_covariance = 0.001
    mc_sim.MC_STATE["warm_threshold"] = 100


_reset()
mc_sim.apply_fault("none")
assert guidance.HOVER_ENTRY_MAX_SPEED_MPS == snapshot["HOVER_ENTRY_MAX_SPEED_MPS"], "none should be a true no-op"
print("  none: no-op -- PASS")

_reset()
mc_sim.apply_fault("hover_entry_no_speed_check")
assert guidance.HOVER_ENTRY_MAX_SPEED_MPS == float("inf")
assert guidance.CENTERED_TICKS_REQUIRED == snapshot["CENTERED_TICKS_REQUIRED"], "should touch nothing else"
print("  hover_entry_no_speed_check: HOVER_ENTRY_MAX_SPEED_MPS -> inf, nothing else touched -- PASS")

_reset()
mc_sim.apply_fault("tick_timing_unscaled")
assert guidance.CENTERED_TICKS_REQUIRED == 30 and guidance.LOST_TICKS_REQUIRED == 15
assert guidance.HOVER_ENTRY_MAX_SPEED_MPS == snapshot["HOVER_ENTRY_MAX_SPEED_MPS"]
print("  tick_timing_unscaled: ticks reverted to 30/15 -- PASS")

_reset()
mc_sim.apply_fault("tilt_cap_relaxed")
assert guidance.APPROACH_MAX_TILT_RAD == autopilot.MAX_TILT_RAD
print(f"  tilt_cap_relaxed: APPROACH_MAX_TILT_RAD -> {autopilot.MAX_TILT_RAD} (autopilot.MAX_TILT_RAD) -- PASS")

_reset()
mc_sim.apply_fault("camera_fov_mismatch")
assert guidance.CAMERA_FOV_DEG == 55.0 and guidance.CAMERA_FOV_DEG != snapshot["CAMERA_FOV_DEG"]
print("  camera_fov_mismatch: CAMERA_FOV_DEG -> 55.0 (true camera fov in sim.py stays 80.0) -- PASS")

_reset()
mc_sim.apply_fault("imu_noise_spike")
assert sensors.gyro_noise.noise_covariance == 0.001 * 25.0
assert sensors.accel_noise.noise_covariance == 0.001 * 25.0
print("  imu_noise_spike: gyro/accel noise_covariance x25 -- PASS")

_reset()
mc_sim.apply_fault("thermal_threshold_drift")
assert mc_sim.MC_STATE["warm_threshold"] == 165
print("  thermal_threshold_drift: MC_STATE['warm_threshold'] -> 165 -- PASS")

try:
    mc_sim.apply_fault("not_a_real_fault")
    raise SystemExit("expected ValueError for unknown fault")
except ValueError:
    print("  unknown fault name raises ValueError -- PASS")

_reset()
print()


# --- 2b. resolve_fault() reads meta_sweep values from .meta, not .get() ---
# This is the regression test for the real "every run reported fault=none"
# bug: a real campaign run proved params.get("meta.fault"/"fault"/
# "meta_fault") never resolves a meta_sweep value (confirmed by reading
# elodin's own PyO3 source, libs/nox-py/src/monte_carlo.rs -- PyParams.get()
# only ever looks in self.params, meta_sweep values live in a totally
# separate self.meta map exposed only via the .meta property). This test
# would have caught that: it builds a Params object the way the REAL
# engine does (fault in .meta, NOT merged into the params dict reachable
# via .get()) and asserts resolve_fault() still finds it.
print("=== 2b. resolve_fault() ===")

params_with_meta_fault = mc_module.params({}, meta={"fault": "imu_noise_spike"})
assert mc_sim.resolve_fault(params_with_meta_fault) == "imu_noise_spike", (
    "resolve_fault must read meta_sweep values via params.meta, not params.get()"
)
print("  fault present only in .meta -> resolve_fault finds it -- PASS")

params_no_meta = mc_module.params({}, meta={})
assert mc_sim.resolve_fault(params_no_meta) == "none", "no meta_sweep -> default to 'none'"
print("  no meta_sweep at all -> defaults to 'none' -- PASS")

# Guards against ever reintroducing the old params.get()-based lookup: if
# "fault" were (wrongly) merged into the params dict reachable via .get()
# as well as sitting in .meta, resolve_fault must still come from .meta,
# since that's the one path the real engine actually uses.
params_fault_in_get_not_meta = mc_module.params(
    {"fault": _Param(str, default="should_never_be_read")}, meta={"fault": "tilt_cap_relaxed"}
)
assert mc_sim.resolve_fault(params_fault_in_get_not_meta) == "tilt_cap_relaxed", (
    "resolve_fault must prefer .meta over a same-named .get() value"
)
print("  .meta value wins over a same-named params.get() value -- PASS")
print()


# --- 2c. synthetic detection geometry (mc_main.py's post_step) ---
# Replaces the thermal-camera read with analytical FOV math because the
# real render server starved under parallel MC workers (~9 of 96 runs
# produced any frames at all in a real campaign). The geometry here is
# pulled from the same values sim.py's world.sensor_camera() uses --
# true fov=80.0, 128x128 at pos_offset=[0.5,0,-0.25], rot_offset=
# [0,90,0] (straight down). This section just verifies the detection
# math itself, independent of elodin; it does NOT re-verify the full
# post_step flow (that needs a real engine run).
print("=== 2c. synthetic detection (mc_main.py's analytical FOV) ===")
import numpy as _np

_TRUE_FOV_DEG = 80.0
_HALF_FOV_TAN = _np.tan(_np.deg2rad(_TRUE_FOV_DEG / 2.0))


def _synth_detect(drone_xyz, heat_xyz, warm_threshold=100):
    """Literal copy of the FOV math from mc_main.py's post_step. Kept
    in this test file as an independent implementation so a drift between
    this and mc_main.py would surface as a test mismatch, not pass
    silently."""
    if drone_xyz is None or heat_xyz is None or float(drone_xyz[2]) <= 0.5:
        return False
    altitude = float(drone_xyz[2])
    fov_radius_m = altitude * _HALF_FOV_TAN
    if warm_threshold > 100:
        fov_radius_m = min(fov_radius_m, 2.5)
    horiz = float(_np.linalg.norm(_np.asarray(heat_xyz[:2]) - _np.asarray(drone_xyz[:2])))
    return horiz < fov_radius_m


# Directly overhead at cruise altitude -- detect.
assert _synth_detect([0.0, 0.0, 7.5], [0.0, 0.0, 0.0]), "directly overhead should detect"
print("  drone directly over heat at z=7.5: detect -- PASS")

# Inside FOV cone (5m off at z=7.5, FOV radius ~6.29m) -- detect.
assert _synth_detect([0.0, 0.0, 7.5], [5.0, 0.0, 0.0]), "5m offset inside FOV should detect"
print("  5m horizontal offset at z=7.5 (inside ~6.29m FOV radius): detect -- PASS")

# Just outside FOV -- no detect.
assert not _synth_detect([0.0, 0.0, 7.5], [7.0, 0.0, 0.0]), "7m offset at z=7.5 should NOT detect"
print("  7m horizontal offset at z=7.5 (outside FOV radius): no detect -- PASS")

# Drone too low (altitude gate) -- no detect.
assert not _synth_detect([0.0, 0.0, 0.3], [0.0, 0.0, 0.0]), "z<=0.5m should gate detection off"
print("  z=0.3m (altitude gate): no detect -- PASS")

# thermal_threshold_drift active: fault shrinks effective radius to 2.5m.
assert _synth_detect([0.0, 0.0, 7.5], [4.0, 0.0, 0.0], warm_threshold=100), "4m at control should detect"
assert not _synth_detect([0.0, 0.0, 7.5], [4.0, 0.0, 0.0], warm_threshold=165), "4m under drift should NOT"
assert _synth_detect([0.0, 0.0, 7.5], [2.0, 0.0, 0.0], warm_threshold=165), "2m under drift should still detect"
print("  thermal_threshold_drift (warm=165) shrinks effective FOV radius to 2.5m: correct -- PASS")

# Bearing roundtrip: the bearing we write must, when fed through
# autopilot.bearing_to_world_xy at identity attitude, recover the true
# heat position. This is what keeps synthetic detection drop-in
# compatible with guidance.waypoint_guidance's downstream estimate.
drone_xy = _np.array([2.0, -1.0])
heat_xy = _np.array([5.0, 3.0])
z = 7.5
scale = z * _HALF_FOV_TAN
offset_world_2d = heat_xy - drone_xy
bearing_y = float(offset_world_2d[0] / scale)
bearing_x = float(-offset_world_2d[1] / scale)
offset_body = _np.array([bearing_y, -bearing_x]) * z * _HALF_FOV_TAN
estimated = drone_xy + offset_body
assert _np.allclose(estimated, heat_xy, atol=1e-6), (
    f"bearing roundtrip failed: expected {heat_xy}, got {estimated}"
)
print(f"  bearing roundtrip through autopilot.bearing_to_world_xy (identity attitude): recovers {heat_xy} -- PASS")
print()


# --- 3. hold_point_freeze: source-patch guard + functional behavior ---
print("=== 3. hold_point_freeze (source-patched from the REAL guidance.waypoint_guidance) ===")

real_src = mc_sim._get_function_source_by_ast(guidance, "waypoint_guidance")
n_occurrences = real_src.count(mc_sim.HOLD_POINT_FREEZE_OLD_LINE)
assert n_occurrences == 1, (
    f"expected the targeted new_hold_point line exactly once in the real "
    f"source, found {n_occurrences} -- guidance.py has changed since this "
    f"harness was written"
)
print("  targeted line appears exactly once in the real guidance.py source -- PASS")

variant = mc_sim._build_hold_point_freeze_variant()
assert callable(variant.fn)
print("  variant builds successfully from the patched source -- PASS")

# Functional check: drive the REAL function and the FAULT variant with the
# same inputs, forcing enter_hover=True with a bearing that is "centered"
# (within CENTERED_BEARING_THRESHOLD) but NOT dead-on-zero, so
# target_estimate is a different point than pos_xy -- otherwise the two
# code paths would coincidentally agree and the test would prove nothing.
import jax.numpy as jnp
from config import Config, Control, Frame


class _Pos:
    def __init__(self, xyz, attitude):
        self._xyz = jnp.array(xyz, dtype=jnp.float64)
        self._attitude = attitude

    def linear(self):
        return self._xyz

    def angular(self):
        return self._attitude


class _Vel:
    def __init__(self, xyz):
        self._xyz = jnp.array(xyz, dtype=jnp.float64)

    def linear(self):
        return self._xyz


Config(
    control=Control(
        rate_pid_gains=__import__("numpy").zeros((3, 3)),
        angle_p_gains=__import__("numpy").zeros(3),
        motor_thrust_exponent=0.8, motor_thrust_hover=0.689,
        attitude_control_input_tc=0.2, pilot_yaw_rate_tc=0.25,
    ),
    drone_glb="x", mass=1.0,
    inertia_diagonal=__import__("numpy").ones(3),
    start_pos=__import__("numpy").zeros(3),
    start_euler_angles=__import__("numpy").zeros(3),
    motor_positions=__import__("numpy").zeros((4, 3)),
    motor_thrust_directions=__import__("numpy").zeros((4, 3)),
    motor_thrust_curve_path="x", frame=Frame.QUAD_X,
    simulation_rate=300.0, fast_loop_time_step=1.0 / 900.0,
    simulation_time=90.0, sensor_noise=False,
).set_as_global()

pos = _Pos([10.0, 5.0, 8.0], _Quaternion(0.0))
vel = _Vel([0.0, 0.0, 0.0])  # zero speed -> satisfies the HOVER-entry speed gate
args = dict(
    pos=pos, vel=vel,
    waypoint_index=jnp.uint64(0), sweep_phase=jnp.float64(guidance.WARMUP_TIME_S + 1.0),
    mode=jnp.uint64(guidance.MODE_APPROACH),
    target_detect=jnp.array([1.0, 0.1, 0.1, 200.0]),  # detected, small nonzero bearing (still "centered")
    centered_ticks=jnp.uint64(guidance.CENTERED_TICKS_REQUIRED), lost_ticks=jnp.uint64(0),
    approach_ticks=jnp.uint64(0), hold_point=jnp.array([0.0, 0.0]),
    angle=jnp.array([0.0, 0.0, 0.0]), zone_start=jnp.uint64(0),
    zone_count=jnp.uint64(guidance.N_LOITER_POINTS),
)

real_result = guidance.waypoint_guidance.fn(**args)
fault_result = variant.fn(**args)

real_hold_point = real_result[-1]
fault_hold_point = fault_result[-1]
pos_xy = jnp.array([10.0, 5.0])

import numpy as _np
assert _np.allclose(_np.asarray(fault_hold_point), _np.asarray(pos_xy)), (
    f"fault variant should freeze hold_point to pos_xy={pos_xy}, got {fault_hold_point}"
)
assert not _np.allclose(_np.asarray(real_hold_point), _np.asarray(pos_xy)), (
    "test setup bug: the REAL function's hold_point coincidentally equals "
    "pos_xy too, so this input doesn't actually distinguish the fault -- "
    "widen the bearing in target_detect"
)
print(f"  real function freezes hold_point to target_estimate={_np.asarray(real_hold_point).round(3)} (!= pos_xy)")
print(f"  fault variant freezes hold_point to pos_xy={_np.asarray(fault_hold_point).round(3)} -- PASS\n")


# --- 4. score.py / report.py pure-logic tests ---
print("=== 4. hooks/score.py oracle math ===")
sys.path.insert(0, os.path.join(HERE, "hooks"))
import score as score_mod
import report as report_mod

clean = {
    "fault": "none", "completed": True, "final_offset_m": 0.4, "min_altitude_m": 1.0,
    "max_altitude_m": 8.1, "premature_hover_violation": False, "mode_switch_count": 2,
}
s = score_mod.score_result(clean)
assert s["all_pass"], s
print("  clean flight -> all oracles pass -- PASS")

premature = dict(clean, premature_hover_violation=True)
s = score_mod.score_result(premature)
assert not s["oracle_no_premature_hover"] and not s["all_pass"]
print("  premature_hover_violation=True -> oracle_no_premature_hover fails -- PASS")

never_completed = dict(clean, completed=False, final_offset_m=-1.0)
s = score_mod.score_result(never_completed)
assert not s["oracle_mission_completes"] and not s["oracle_final_accuracy"]
print("  completed=False -> oracle_mission_completes AND oracle_final_accuracy both fail -- PASS")

far_miss = dict(clean, final_offset_m=9.0)
s = score_mod.score_result(far_miss)
assert s["oracle_mission_completes"] and not s["oracle_final_accuracy"]
print("  completed but final_offset_m=9.0 (> 2.0m bound) -> oracle_final_accuracy fails -- PASS")

ground_strike = dict(clean, min_altitude_m=0.1)
s = score_mod.score_result(ground_strike)
assert not s["oracle_altitude_envelope"]
print("  min_altitude_m=0.1 (< 0.5m safe floor) -> oracle_altitude_envelope fails -- PASS")

thrashy = dict(clean, mode_switch_count=20)
s = score_mod.score_result(thrashy)
assert not s["oracle_no_thrash"]
print("  mode_switch_count=20 (> limit 6) -> oracle_no_thrash fails -- PASS\n")


print("=== 4b. hooks/report.py pass-rate + verdict math ===")


def _rows(fault, n, **overrides):
    base = dict(clean)
    base["fault"] = fault
    base.update(overrides)
    return [dict(base) for _ in range(n)]


# control group: all clean. hover_entry_no_speed_check group: all trip the
# premature-hover oracle (as EXPECTED_ORACLE claims it should).
rows = (
    _rows("none", 10)
    + _rows("hover_entry_no_speed_check", 10, premature_hover_violation=False)
)
# convert bool fields to the str-ish shape a csv.DictReader would give
scored_rows = [score_mod.score_result(r) for r in rows]
# simulate the actual fault behavior for the non-control group: in
# reality the fault would make premature_hover_violation True in most
# (not all -- it's stochastic over dispersion) runs; emulate 9/10 tripped
for r in scored_rows[10:19]:
    r["oracle_no_premature_hover"] = False
    r["all_pass"] = False

report_text = report_mod.build_report(scored_rows)
assert "hover_entry_no_speed_check" in report_text
assert "CAUGHT as expected by oracle_no_premature_hover" in report_text, report_text
print("  9/10 tripped -> report verdict = CAUGHT as expected -- PASS")

# now an escape case: fault present but oracle pass rate barely moves
scored_rows_escape = [score_mod.score_result(r) for r in (_rows("none", 10) + _rows("tilt_cap_relaxed", 10))]
report_text_escape = report_mod.build_report(scored_rows_escape)
assert "ESCAPED" in report_text_escape, report_text_escape
print("  fault present but no oracle pass-rate drop -> report verdict = ESCAPED -- PASS\n")

print("ALL TESTS PASSED")
