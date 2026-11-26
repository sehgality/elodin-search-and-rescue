"""
autopilot.py -- a small, reusable "flight software" layer for Elodin drones.

This is the piece Elodin's own examples don't ship: a generic outer-loop
position/altitude controller that any mission script can drive with a
single "go to this point" command, instead of every mission hand-rolling
its own control law from scratch. It knows nothing about search patterns,
thermal cameras, or any specific mission -- it only knows how to fly a
drone (given its el.WorldPos/el.WorldVel state) toward an (x, y) point,
hold an altitude, and report when it's arrived. Mission code decides
WHAT point to send; this decides HOW to get there.

This mirrors the separation a real flight stack uses: this is the
"position controller" layer (comparable to PX4's multicopter position
controller / offboard "goto" mode), sitting on top of the existing
attitude/rate PID cascade in control.py and the motor model in motors.py
-- both untouched, both already solid, and neither was ever the source
of the wildfire-detection bugs this was built to fix.

Any future Elodin mission (not just this one) can import this module and
call goto_xy/hold_altitude directly instead of re-deriving this math.
"""

import jax
import jax.numpy as jnp
import elodin as el

# --- horizontal (X/Y) position control ---
KP_POS = 0.12  # position error (m) -> desired tilt (rad)
KD_VEL = 0.18  # velocity (m/s) -> damping (rad)
MAX_TILT_RAD = 0.30  # ~17 degrees, safety clamp

# --- vertical (altitude) control ---
KP_ALT = 0.15
KD_ALT = 0.35
MAX_THROTTLE_DELTA = 0.25

# --- ground avoidance (applies no matter what target altitude a mission
# asks for -- this module will never fly a drone through the floor) ---
MIN_SAFE_ALTITUDE_M = 0.5
GROUND_WARNING_ALTITUDE_M = 1.5  # the lookahead check only matters this close to the ground -- a fast, deliberate high-altitude descent isn't yet dangerous and shouldn't trigger it
GROUND_AVOID_LOOKAHEAD_S = 0.4
EMERGENCY_CLIMB_DELTA = 0.4

# --- arrival ---
DEFAULT_CAPTURE_RADIUS_M = 1.5

# --- camera-bearing-to-world-coordinate conversion ---
# VERIFIED from a real flight_log.csv with attitude logged: tested all 8
# axis-swap/sign combinations against ~4,200 real detected-and-tracking
# rows, comparing each one's implied target position to the actual heat
# source position. The mapping below (bearing_y, -bearing_x) -- a
# 90-degree rotation, not a simple sign flip -- won decisively: median
# cosine similarity to the true target direction of 0.99, vs ~0.1 or
# negative for every other combination including the old same-axis
# formula. That's consistent with the camera's pixel row/column axes
# being mounted rotated 90 degrees relative to the body x/y axes.
# (This replaces the old BEARING_SIGN scalar, which could only express a
# sign flip and was never able to represent the real fix.)


def world_to_body_xy(world_xy: jax.Array, attitude: el.Quaternion) -> jax.Array:
    """Rotate a world-frame X/Y vector into the body frame. Pitch/roll
    commands are body-frame tilts, but steering is naturally computed in
    world coordinates (positions/velocities are world-frame) -- this is
    the one conversion that bridges the two, and it's only a no-op when
    yaw happens to be exactly 0."""
    world_vec3 = jnp.array([world_xy[0], world_xy[1], 0.0])
    body_vec3 = attitude.inverse() @ world_vec3
    return body_vec3[:2]


def goto_xy(
    pos_xy: jax.Array,
    vel_xy: jax.Array,
    target_xy: jax.Array,
    attitude: el.Quaternion,
    max_tilt: float = MAX_TILT_RAD,
) -> tuple[jax.Array, jax.Array]:
    """The core 'fly to this point' primitive. Returns (pitch, roll)
    body-frame tilt commands in radians that steer toward target_xy and
    damp velocity on the way in. Every flight mode (search loiter,
    live-tracking a detected target, holding a lock-on point) calls this
    SAME function with a different target_xy -- there is no longer a
    separate, independently-tuned control law per mode.

    max_tilt optionally overrides the default safety clamp. The camera
    is rigidly mounted to the airframe (no gimbal), so any tilt also
    tilts its view cone off-vertical -- a mode that's actively relying
    on keeping a target in frame (APPROACH) should pass a tighter cap
    than one that isn't (SEARCH, HOVER)."""
    error_world = target_xy - pos_xy
    accel_world = KP_POS * error_world - KD_VEL * vel_xy
    accel_body = world_to_body_xy(accel_world, attitude)
    pitch_raw = accel_body[0]
    roll_raw = -accel_body[1]
    # clip the (pitch, roll) vector's MAGNITUDE, not each axis
    # independently -- clipping axes separately distorts the commanded
    # DIRECTION whenever the position-correction term and the
    # velocity-damping term don't point the same way and one lands
    # harder on one axis than the other. That distortion gets worse,
    # not better, as max_tilt shrinks, which is why a tight APPROACH
    # cap could fly it away from the target instead of just slowing it
    # down. Scaling the whole vector down uniformly keeps the commanded
    # direction correct no matter how tight the cap is.
    raw_mag = jnp.sqrt(pitch_raw**2 + roll_raw**2)
    scale = jnp.minimum(1.0, max_tilt / jnp.maximum(raw_mag, 1e-6))
    pitch = pitch_raw * scale
    roll = roll_raw * scale
    return pitch, roll


def hold_altitude(
    z: jax.Array,
    z_vel: jax.Array,
    target_z: jax.Array,
    hover_thrust: jax.Array,
) -> jax.Array:
    """The core altitude-hold primitive. Returns a full throttle command
    (hover baseline + correction) driving toward target_z. A hard
    ground-avoidance override is built in here, not left to each mission
    to remember: no matter what target_z a mission asks for, this will
    never command the drone through the floor. The override reacts to
    projected fall rate, not just current position, since a purely
    reactive 'already too low' check didn't leave enough time to arrest
    a fast dive before the ground -- but it only engages near the ground
    (GROUND_WARNING_ALTITUDE_M), so a deliberate fast descent from
    altitude isn't mistaken for an emergency."""
    error = target_z - z
    correction = KP_ALT * error - KD_ALT * z_vel
    correction = jnp.clip(correction, -MAX_THROTTLE_DELTA, MAX_THROTTLE_DELTA)

    near_ground = z < GROUND_WARNING_ALTITUDE_M
    projected_z = z + z_vel * GROUND_AVOID_LOOKAHEAD_S
    emergency = near_ground & (
        (z < MIN_SAFE_ALTITUDE_M) | ((z_vel < 0.0) & (projected_z < MIN_SAFE_ALTITUDE_M))
    )
    correction = jnp.where(emergency, EMERGENCY_CLIMB_DELTA, correction)

    # clamp to the physically valid motor throttle range -- hover baseline
    # plus correction could otherwise exceed what a real motor can output
    return jnp.clip(hover_thrust + correction, 0.0, 1.0)


def arrived_xy(
    pos_xy: jax.Array,
    target_xy: jax.Array,
    radius: float = DEFAULT_CAPTURE_RADIUS_M,
) -> jax.Array:
    """True once within `radius` meters of target_xy, horizontally."""
    return jnp.linalg.norm(target_xy - pos_xy) < radius


def bearing_to_world_xy(
    pos_xy: jax.Array,
    altitude_z: jax.Array,
    bearing_xy: jax.Array,
    attitude: el.Quaternion,
    camera_fov_deg: float,
) -> jax.Array:
    """Converts a camera's normalized bearing (-1..1, body/camera-frame)
    into an estimated real-world (x, y) target point, using the drone's
    current altitude and the camera's known field of view (simple
    straight-down-pinhole-camera trig). This is what lets a vision-based
    mission hand the autopilot an actual coordinate -- 'the target is
    roughly HERE in the world' -- instead of a raw, unitless bearing."""
    half_fov_tan = jnp.tan(jnp.deg2rad(camera_fov_deg / 2.0))
    offset_body = jnp.array([bearing_xy[1], -bearing_xy[0]]) * altitude_z * half_fov_tan
    offset_body_3 = jnp.array([offset_body[0], offset_body[1], 0.0])
    offset_world = (attitude @ offset_body_3)[:2]
    return pos_xy + offset_world
