"""
Mission layer: search -> track -> land state machine.

This file decides WHAT point (and what altitude) the drone should aim for
and WHEN to switch between modes. The actual flying -- "how do I get
from here to that point/altitude" -- lives entirely in autopilot.py's
goto_xy and hold_altitude, the same two functions every mode below calls.
Altitude used to be its own file (altitude.py); folded in here since it
was just another piece of mission logic (which altitude does each mode
want?) sitting on top of the same autopilot, not a separate concern.
"""

import os
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
from control import AngleDesired, ThrottleCommand
from dataclasses import dataclass, field
from config import Config

import autopilot

# --- search area: the heat source spawns somewhere in here, but we don't
# know exactly where -- that's the point. sim.py picks a random point
# inside this disk each run.
SEARCH_AREA_CENTER = jnp.array([0.0, 0.0])
SEARCH_AREA_RADIUS_M = 15.0

LOITER_CENTER = SEARCH_AREA_CENTER
LOITER_RADIUS_M = float(os.environ.get("SEARCH_RADIUS_M", "10.0"))  # patrol loop radius -- stays inside the search area; override with SEARCH_RADIUS_M env var
N_LOITER_POINTS = 32

_loiter_angles = jnp.linspace(0.0, 2 * jnp.pi, N_LOITER_POINTS, endpoint=False)
WAYPOINTS = LOITER_CENTER + LOITER_RADIUS_M * jnp.stack(
    [jnp.cos(_loiter_angles), jnp.sin(_loiter_angles)], axis=-1
)
CAPTURE_RADIUS_M = 1.5  # how close counts as "reached this waypoint"

CAMERA_FOV_DEG = 80.0  # must match sim.py's world.sensor_camera(..., fov=...) -- used to convert the camera's bearing into a real-world offset

# The camera is rigidly mounted (no gimbal) -- any tilt during APPROACH
# tilts its view cone off-vertical and can swing the target out of
# frame. Cap tilt tighter here than SEARCH/HOVER's default (0.30 rad):
# slower close-in, but it keeps the camera pointed near the target
# instead of corkscrewing it out of view on every correction.
APPROACH_MAX_TILT_RAD = 0.06  # ~7 degrees, vs. the default ~17

# Right when APPROACH starts, the drone is still carrying whatever
# sideways velocity it had on the SEARCH loiter -- that's random in
# direction and magnitude every run, which is why some runs track
# straight in and others overshoot the target and lose the camera
# frame. Brake first, then track: for a short window right after
# entering APPROACH, aim goto_xy at the drone's OWN current position
# (pure velocity-damping, no steering) instead of the live target
# estimate, so every APPROACH episode starts tracking from roughly the
# same near-stationary condition instead of a random leftover velocity.
BRAKE_VELOCITY_THRESHOLD_MPS = 0.5  # once speed drops below this, stop braking and start tracking the live estimate
BRAKE_MAX_TICKS = 150  # ~0.5s safety fallback -- don't brake forever if speed never drops below the threshold

# "Centered" (below) is purely a bearing check -- at altitude, a wide-FOV
# camera can read as centered with real, meter-scale position error, so
# without also requiring actual low speed, the drone can satisfy
# CENTERED_TICKS_REQUIRED while still moving fast and commit to HOVER
# mid-flyover. Confirmed from a real flight_log.csv run: centered_ticks
# hit its threshold while the drone was moving at ~2.4 m/s and its
# distance to the heat source was increasing, not decreasing.
HOVER_ENTRY_MAX_SPEED_MPS = 0.5

# --- altitude targets per mode (was altitude.py) ---
TARGET_ALTITUDE_M = 8.0  # cruise altitude while searching/tracking, meters
HOVER_ALTITUDE_M = 1.0  # descend to this once locked on -- the "1m above the heat source" confirm altitude
ALT_SWEEP_AMPLITUDE_M = 0.0  # no real gimbal to nod the camera -- kept as a documented, currently-disabled option (bob SEARCH altitude instead) rather than deleted
ALT_SWEEP_PERIOD_S = 6.0

# --- startup transient guard ---
# On the first ~1s of a run, sensors/estimators haven't converged yet --
# control laws can see garbage for a tick or two and command something
# aggressive. Hold a neutral command (and freeze mode) until sweep_phase
# (repurposed as elapsed sim time) clears this threshold.
WARMUP_TIME_S = 1.0

WaypointIndex = ty.Annotated[
    jax.Array,
    el.Component("waypoint_index", el.ComponentType.U64),
]

# Camera points straight down -- translation alone sweeps the ground
# under the flight path, so no yaw sweep is needed during SEARCH.
SEARCH_YAW_RATE_RAD_S = 0.0

SweepPhase = ty.Annotated[
    jax.Array,
    el.Component("sweep_phase", el.ComponentType.F64),
]

# [0] = detected this tick (1.0/0.0)
# [1] = bearing_x (normalized -1..1)
# [2] = bearing_y (normalized -1..1)
# [3] = max_red
TargetDetect = ty.Annotated[
    jax.Array, el.Component("target_detect", el.ComponentType(el.PrimitiveType.F64, (4,)))
]

# --- search -> approach(track) -> hover(land) state machine ---
MODE_SEARCH = 0
MODE_APPROACH = 1
MODE_HOVER = 2

CENTERED_BEARING_THRESHOLD = 0.3  # "close enough to center" instead of near dead-on
# These three were commented as if the sim ran at ~30Hz (so "30 ticks" =
# 1 second). It actually runs at 300Hz -- confirmed from `elodin run`'s
# own per-run summary ("simulation steps ... at 300 Hz sim") -- so every
# value below was firing 10x faster than its own comment intended.
# Scaled up 10x to match what the comments always said they should be.
CENTERED_TICKS_REQUIRED = 300  # ~1s of sustained centering before committing to HOVER
LOST_TICKS_REQUIRED = 150  # ~0.5s of sustained *miss* before giving up and going back to SEARCH
MAX_APPROACH_TICKS = 3000  # ~10s hard cap -- if tracking hasn't resolved to HOVER or lost the target by then, force back to SEARCH

# rate-limit how fast the commanded angle can change, regardless of mode
ANGLE_SLEW_ALPHA = 0.08

Mode = ty.Annotated[
    jax.Array,
    el.Component("guidance_mode", el.ComponentType.U64),
]
CenteredTicks = ty.Annotated[
    jax.Array,
    el.Component("centered_ticks", el.ComponentType.U64),
]
LostTicks = ty.Annotated[
    jax.Array,
    el.Component("lost_ticks", el.ComponentType.U64),
]
ApproachTicks = ty.Annotated[
    jax.Array,
    el.Component("approach_ticks", el.ComponentType.U64),
]
HoldPoint = ty.Annotated[
    jax.Array,
    el.Component("hold_point", el.ComponentType(el.PrimitiveType.F64, (2,))),
]
# Multi-drone swarm search: each drone is restricted to its own arc of the
# shared WAYPOINTS loiter circle, so N drones can split one search area
# into N non-overlapping patrol zones instead of all flying the same
# circuit. ZoneStart/ZoneCount describe that arc as [zone_start,
# zone_start+zone_count) (indices into WAYPOINTS, wrapping mod
# N_LOITER_POINTS). Defaults (0, N_LOITER_POINTS) cover the WHOLE circle,
# so a single-drone run is unaffected unless sim.py explicitly assigns a
# smaller zone when spawning.
ZoneStart = ty.Annotated[
    jax.Array,
    el.Component("zone_start", el.ComponentType.U64),
]
ZoneCount = ty.Annotated[
    jax.Array,
    el.Component("zone_count", el.ComponentType.U64),
]
_DEBUG_STATIC_MODE = os.environ.get("DEBUG_STATIC") == "1"
_DEBUG_SPIN_MODE = os.environ.get("DEBUG_SPIN") == "1"


@dataclass
class Guidance(el.Archetype):
    waypoint_index: WaypointIndex = field(default_factory=lambda: jnp.uint64(0))
    sweep_phase: SweepPhase = field(default_factory=lambda: jnp.float64(0.0))
    mode: Mode = field(default_factory=lambda: jnp.uint64(2) if _DEBUG_STATIC_MODE else jnp.uint64(0))
    centered_ticks: CenteredTicks = field(default_factory=lambda: jnp.uint64(0))
    lost_ticks: LostTicks = field(default_factory=lambda: jnp.uint64(0))
    approach_ticks: ApproachTicks = field(default_factory=lambda: jnp.uint64(0))
    hold_point: HoldPoint = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    throttle_command: ThrottleCommand = field(default_factory=lambda: jnp.float64(0.0))
    zone_start: ZoneStart = field(default_factory=lambda: jnp.uint64(0))
    zone_count: ZoneCount = field(default_factory=lambda: jnp.uint64(N_LOITER_POINTS))


@el.map
def waypoint_guidance(
    pos: el.WorldPos,
    vel: el.WorldVel,
    waypoint_index: WaypointIndex,
    sweep_phase: SweepPhase,
    mode: Mode,
    target_detect: TargetDetect,
    centered_ticks: CenteredTicks,
    lost_ticks: LostTicks,
    approach_ticks: ApproachTicks,
    hold_point: HoldPoint,
    angle: AngleDesired,
    zone_start: ZoneStart,
    zone_count: ZoneCount,
) -> tuple[AngleDesired, ThrottleCommand, WaypointIndex, SweepPhase, Mode, CenteredTicks, LostTicks, ApproachTicks, HoldPoint]:
    pos_xy = pos.linear()[:2]
    z = pos.linear()[2]
    vel_xy = vel.linear()[:2]
    z_vel = vel.linear()[2]
    attitude = pos.angular()

    detected = target_detect[0] > 0.5
    bearing_xy = target_detect[1:3]
    bearing_mag = jnp.linalg.norm(bearing_xy)
    centered = detected & (bearing_mag < CENTERED_BEARING_THRESHOLD)

    # Computed early -- needed both for the HOVER-entry decision right
    # below and for APPROACH tracking further down: where the camera
    # actually thinks the target is, and how fast the drone is moving.
    target_estimate = autopilot.bearing_to_world_xy(pos_xy, z, bearing_xy, attitude, CAMERA_FOV_DEG)
    speed = jnp.linalg.norm(vel_xy)

    is_search = mode == MODE_SEARCH
    is_approach = mode == MODE_APPROACH

    # --- mode transitions (take effect next tick) ---
    enter_approach = is_search & detected
    # decay by 1 on a miss instead of hard-resetting to 0 -- a single
    # noisy/flickery frame shouldn't wipe out a near-complete streak
    new_centered_ticks = jnp.where(
        is_approach & centered,
        centered_ticks + 1,
        jnp.where(
            is_approach,
            jnp.where(centered_ticks > 0, centered_ticks - 1, jnp.uint64(0)),
            centered_ticks,
        ),
    )
    new_lost_ticks = jnp.where(
        is_approach & jnp.logical_not(detected),
        lost_ticks + 1,
        jnp.where(is_approach, jnp.uint64(0), lost_ticks),
    )
    enter_hover = (
        is_approach
        & (new_centered_ticks >= CENTERED_TICKS_REQUIRED)
        & (speed < HOVER_ENTRY_MAX_SPEED_MPS)
    )
    new_approach_ticks = jnp.where(is_approach, approach_ticks + 1, jnp.uint64(0))
    approach_timeout = is_approach & (new_approach_ticks >= MAX_APPROACH_TICKS)
    lose_target = is_approach & ((new_lost_ticks >= LOST_TICKS_REQUIRED) | approach_timeout)

    new_mode = jnp.where(
        enter_hover,
        jnp.uint64(MODE_HOVER),
        jnp.where(
            enter_approach,
            jnp.uint64(MODE_APPROACH),
            jnp.where(lose_target, jnp.uint64(MODE_SEARCH), mode),
        ),
    )
    # Freeze onto where the camera says the target IS (target_estimate),
    # not wherever the drone itself happens to be standing right now --
    # pos_xy was the bug: a real run froze hold_point to the drone's own
    # position while it was ~1m away from (and still moving past) the
    # actual heat source, then held that wrong point for the rest of
    # the flight.
    new_hold_point = jnp.where(enter_hover, target_estimate, hold_point)

    # === SEARCH: fly this entity's own ARC of the shared loiter circle,
    # looking for a detection. waypoint_index is a LOCAL index (0 ..
    # zone_count-1) within this entity's zone, not an absolute index into
    # WAYPOINTS -- that's what lets N drones split one circle into N
    # non-overlapping patrol zones (multi-drone swarm search) while every
    # entity still runs this exact same function (ECS: one system, many
    # entities, no per-entity branching needed). A single-drone run gets
    # zone_start=0, zone_count=N_LOITER_POINTS (the Guidance archetype's
    # defaults), which covers the whole circle -- identical behavior to
    # before zones existed.
    abs_index = (zone_start + waypoint_index) % WAYPOINTS.shape[0]
    target_wp = WAYPOINTS[abs_index]
    reached = autopilot.arrived_xy(pos_xy, target_wp, CAPTURE_RADIUS_M)
    # On giving up a chase (lose_target, computed above -- just dropped
    # back to SEARCH), resume the loiter from whichever point WITHIN THIS
    # ENTITY'S OWN ZONE is nearest to where the drone ACTUALLY is right
    # now, not wherever the fixed circuit was left off before the chase,
    # and never a point outside this entity's assigned zone -- that would
    # have it drift into another drone's patrol area. Without the
    # zone-restriction (the `jnp.where(in_zone, ...)` below), a drone
    # could snap to the globally-nearest point on the ENTIRE circle, which
    # might belong to a different drone's arc entirely.
    all_offsets = (jnp.arange(WAYPOINTS.shape[0]) - zone_start) % WAYPOINTS.shape[0]
    in_zone = all_offsets < zone_count
    dists_to_wp = jnp.linalg.norm(WAYPOINTS - pos_xy[None, :], axis=1)
    dists_to_wp = jnp.where(in_zone, dists_to_wp, jnp.inf)
    nearest_abs_index = jnp.argmin(dists_to_wp)
    nearest_local_index = ((nearest_abs_index - zone_start) % WAYPOINTS.shape[0]).astype(jnp.uint64)
    next_index = jnp.where(
        lose_target,
        nearest_local_index,
        jnp.where(reached, (waypoint_index + 1) % zone_count, waypoint_index),
    )
    pitch_search, roll_search = autopilot.goto_xy(pos_xy, vel_xy, target_wp, attitude)

    # sweep_phase is repurposed as an elapsed-sim-time accumulator, used
    # below as the startup-transient warmup gate.
    new_phase = sweep_phase + Config.GLOBAL.dt
    warmed_up = sweep_phase >= WARMUP_TIME_S

    # === APPROACH (tracking): brake first, then track. Right after
    # entering APPROACH, aim at the drone's OWN current position (pure
    # velocity-damping -- error_world is ~0, so this is just -KD_VEL*vel,
    # unambiguous deceleration, no steering conflict) until speed drops
    # below BRAKE_VELOCITY_THRESHOLD_MPS or BRAKE_MAX_TICKS runs out.
    # Only then switch to tracking the live camera-estimated position --
    # the SAME goto_xy call SEARCH just used, just aimed at a
    # continuously-updating estimate instead of a fixed waypoint.
    # "Center the frame" falls out of this for free: once the estimated
    # target position converges on the drone's actual position, the
    # bearing that produced it is necessarily near zero.
    # target_estimate / speed already computed earlier (needed for the
    # HOVER-entry decision above too) -- not recomputed here.
    is_braking = is_approach & (approach_ticks < BRAKE_MAX_TICKS) & (speed >= BRAKE_VELOCITY_THRESHOLD_MPS)
    # Braking gets FULL tilt authority, not the tight APPROACH cap --
    # pure braking (target == own position) has no steering-direction
    # risk, it's always decelerating straight against current velocity,
    # so there's no reason to hold back. Only tracking (once braking is
    # done) needs the gentle cap, since that's where a tilt can distort
    # commanded direction or swing the camera off a target it's still
    # trying to center.
    brake_pitch, brake_roll = autopilot.goto_xy(pos_xy, vel_xy, pos_xy, attitude, max_tilt=autopilot.MAX_TILT_RAD)
    track_pitch, track_roll = autopilot.goto_xy(
        pos_xy, vel_xy, target_estimate, attitude, max_tilt=APPROACH_MAX_TILT_RAD
    )
    pitch_approach = jnp.where(is_braking, brake_pitch, track_pitch)
    roll_approach = jnp.where(is_braking, brake_roll, track_roll)
    yaw_approach = 0.0  # straight-down camera -- pure translation, no yaw needed

    # === HOVER (landing): fly to and hold the frozen lock-on point while
    # descending to the confirm altitude below ===
    pitch_hover, roll_hover = autopilot.goto_xy(pos_xy, vel_xy, new_hold_point, attitude)

    pitch_raw = jnp.where(
        is_search, pitch_search, jnp.where(is_approach, pitch_approach, pitch_hover)
    )
    roll_raw = jnp.where(
        is_search, roll_search, jnp.where(is_approach, roll_approach, roll_hover)
    )
    yaw_raw = jnp.where(is_search | _DEBUG_SPIN_MODE, SEARCH_YAW_RATE_RAD_S, jnp.where(is_approach, yaw_approach, 0.0))

    raw_angle = jnp.array([roll_raw, pitch_raw, yaw_raw])
    # during the startup warmup window, ignore whatever the control laws
    # computed and just command level/no-yaw instead
    raw_angle = jnp.where(warmed_up, raw_angle, jnp.zeros(3, dtype=jnp.float64))
    safe_mode = jnp.where(warmed_up, new_mode, mode)

    # slew-rate limit: blend toward the raw command instead of jumping to
    # it, so a mode switch can't yank the commanded angle instantly
    new_angle = angle + ANGLE_SLEW_ALPHA * (raw_angle - angle)

    # --- altitude: decide the target for the current mode, hand it to
    # the autopilot (was altitude.py's whole job) ---
    search_altitude = TARGET_ALTITUDE_M + ALT_SWEEP_AMPLITUDE_M * jnp.sin(
        2 * jnp.pi * sweep_phase / ALT_SWEEP_PERIOD_S
    )
    target_altitude = jnp.where(
        mode == MODE_HOVER,
        HOVER_ALTITUDE_M,
        jnp.where(mode == MODE_SEARCH, search_altitude, TARGET_ALTITUDE_M),
    )
    hover_thrust = Config.GLOBAL.control.motor_thrust_hover
    throttle = autopilot.hold_altitude(z, z_vel, target_altitude, hover_thrust)

    return (
        new_angle,
        throttle,
        next_index,
        new_phase,
        safe_mode,
        new_centered_ticks,
        new_lost_ticks,
        new_approach_ticks,
        new_hold_point,
    )


# --- debug scaffold ---
ThermalDebug = ty.Annotated[
    jax.Array, el.Component("thermal_debug", el.ComponentType(el.PrimitiveType.F64, (4,)))
]


@dataclass
class ThermalDebugArchetype(el.Archetype):
    thermal_debug: ThermalDebug = field(default_factory=lambda: jnp.zeros(4, dtype=jnp.float64))


@dataclass
class TargetDetectArchetype(el.Archetype):
    target_detect: TargetDetect = field(default_factory=lambda: jnp.zeros(4, dtype=jnp.float64))
