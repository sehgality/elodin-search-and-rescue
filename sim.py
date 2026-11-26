import typing as ty
from dataclasses import dataclass, field

import control
import guidance
import elodin as el
import jax
import jax.numpy as jnp
import mekf
import motors
import numpy as np
import params
import sensors
import telemetry
from config import Config
from schematic import build as build_schematic

BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={
            "priority": 200,
            "element_names": "τx,τy,τz,x,y,z",
        },
    ),
]
BodyDrag = ty.Annotated[
    jax.Array,
    el.Component(
        "body_drag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

Thrust = ty.Annotated[
    jax.Array,
    el.Component(
        "thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 98},
    ),
]
Torque = ty.Annotated[
    jax.Array,
    el.Component(
        "torque",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 97},
    ),
]


@dataclass
class Drone(el.Archetype):
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))


@dataclass
class StaticMarker(el.Archetype):
    """A static visual marker with no physics (no inertia, force, velocity)."""

    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


@el.map
def motor_thrust_response(
    pwm: motors.MotorPwm, prev_thrust: Thrust, prev_torque: Torque, prev_rpm: motors.MotorRpm
) -> tuple[Thrust, Torque, motors.MotorRpm]:
    dt = Config.GLOBAL.fast_loop_time_step
    pwm_ref, thrust_ref, torque_ref, rpm_ref = Config.GLOBAL.thrust_curve()
    _, _, yaw_factor, _ = Config.GLOBAL.frame.motor_matrix
    thrust_constant = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, thrust_ref, rcond=None)[0][0]
    torque_constant = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, torque_ref, rcond=None)[0][0]

    alpha = dt / (dt + params.MOT_TIME_CONST)
    rpm = jnp.interp(pwm, pwm_ref, rpm_ref)
    rpm = prev_rpm + alpha * (rpm - prev_rpm)

    thrust = rpm**2 * thrust_constant
    torque = rpm**2 * torque_constant * yaw_factor
    return thrust, torque, rpm


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    config = Config.GLOBAL
    thrust_dir = config.motor_thrust_directions
    torque_dir = config.motor_torque_axes
    body_thrust = el.SpatialForce(linear=jnp.sum(thrust_dir * thrust[:, None], axis=0))
    yaw_torque = el.SpatialForce(torque=jnp.sum(thrust_dir * torque[:, None], axis=0))
    # additional torque from differential thrust:
    pitch_roll_torque = el.SpatialForce(torque=jnp.sum(torque_dir * thrust[:, None], axis=0))
    return body_thrust + yaw_torque + pitch_roll_torque


@el.map
def drag(v: el.WorldVel) -> BodyDrag:
    rel_v = -v.linear()
    rel_v_norm = jnp.linalg.norm(rel_v)
    return 0.2 * 0.5 * rel_v * rel_v_norm


@el.map
def apply_body_forces(
    thrust: BodyThrust, drag: BodyDrag, pos: el.WorldPos, f: el.Force
) -> el.Force:
    return f + el.SpatialForce(linear=drag) + pos.angular() @ thrust


@el.map
def gravity(inertia: el.Inertia, f: el.Force) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


def _random_heat_source_xy() -> np.ndarray:
    """Pick a uniformly random point inside the search disk -- we're not
    supposed to know exactly where the target is, that's the point of the
    search pattern."""
    center = np.asarray(guidance.SEARCH_AREA_CENTER)
    radius = guidance.SEARCH_AREA_RADIUS_M
    angle = np.random.uniform(0.0, 2 * np.pi)
    r = radius * np.sqrt(np.random.uniform(0.0, 1.0))
    return center + r * np.array([np.cos(angle), np.sin(angle)])


import os

# Multi-drone swarm search: N_DRONES=1 (default) is the exact original
# single-drone behavior. N_DRONES>1 spawns that many drones, each with its
# own camera, splitting the shared loiter circle (guidance.WAYPOINTS) into
# N roughly-equal, non-overlapping zones (see guidance.py's ZoneStart/
# ZoneCount) so they patrol different parts of the search area in parallel
# instead of all flying the same circuit redundantly. They share ONE heat
# source -- this models "a fleet searching for one target faster together",
# not "N independent searches."
N_DRONES = int(os.environ.get("N_DRONES", "1"))


def _zone_bounds(n_drones: int, n_points: int = guidance.N_LOITER_POINTS) -> tuple[list[int], list[int]]:
    """Split n_points waypoints into n_drones contiguous, non-overlapping
    zones, as equal as possible (any remainder spread across the first
    few zones). Returns (zone_starts, zone_counts), each length n_drones."""
    base = n_points // n_drones
    rem = n_points % n_drones
    starts, counts = [], []
    cursor = 0
    for i in range(n_drones):
        count = base + (1 if i < rem else 0)
        starts.append(cursor)
        counts.append(count)
        cursor += count
    return starts, counts


def world() -> tuple[el.World, list[str]]:
    world = el.World()
    zone_starts, zone_counts = _zone_bounds(N_DRONES)
    drone_names: list[str] = []

    start_z = float(np.asarray(Config.GLOBAL.start_pos)[2])

    for i in range(N_DRONES):
        name = "drone" if N_DRONES == 1 else f"drone_{i}"
        drone_names.append(name)

        if N_DRONES == 1:
            # Exact original single-drone spawn transform -- unchanged.
            spawn_pos = Config.GLOBAL.spatial_transform
        else:
            # Start each drone near the middle of its own patrol zone
            # (on the loiter circle) instead of all stacked at the same
            # point -- avoids spawning drones on top of each other and
            # gets each one into its own zone quickly. start_euler_angles
            # is [0,0,0] for both configs in main.py, so identity
            # rotation here matches what Config.GLOBAL.spatial_transform
            # would have given a single drone.
            zone_mid_angle = (
                2 * np.pi * (zone_starts[i] + zone_counts[i] / 2.0) / guidance.N_LOITER_POINTS
            )
            center = np.asarray(guidance.LOITER_CENTER)
            start_xy = center + guidance.LOITER_RADIUS_M * np.array(
                [np.cos(zone_mid_angle), np.sin(zone_mid_angle)]
            )
            spawn_pos = el.SpatialTransform(
                linear=jnp.array([start_xy[0], start_xy[1], start_z])
            )

        drone = world.spawn(
            [
                el.Body(
                    world_pos=spawn_pos,
                    inertia=Config.GLOBAL.spatial_inertia,
                ),
                Drone(),
                motors.Motors(),
                sensors.IMU(),
                control.AttitudeController(),
                guidance.Guidance(
                    zone_start=jnp.uint64(zone_starts[i]),
                    zone_count=jnp.uint64(zone_counts[i]),
                ),
                guidance.ThermalDebugArchetype(),
                guidance.TargetDetectArchetype(),
                mekf.MEKF(),
                telemetry.Telemetry(),
            ],
            name=name,
        )

        world.sensor_camera(
            entity=drone,
            name="thermal_cam",
            width=128,
            height=128,
            fov=80.0,  # opened back up partway -- 65 fixed airframe grazing but created a too-large near-field blind spot directly under the drone
            near=0.05,
            far=80.0,
            pos_offset=[0.5, 0.0, -0.25],
            rot_offset=[0.0, 90.0, 0.0],  # steeper pitch
            format="rgba",
            effect="lwir",
            effect_params={"palette": "white_hot"},
            fps=30.0,
            create_frustum=True,
            frustums_color=[1.0, 0.0, 0.0, 1.0],
            projection_color=[0.0, 1.0, 0.0, 1.0],
        )

        if N_DRONES == 1:
            # Unchanged from the original single-drone path. NOT yet
            # extended to N>1: build_schematic()/the "drone.kdl" file are
            # written assuming a single entity literally named "drone",
            # and we haven't verified schematic.py supports binding the
            # same schematic to multiple differently-named entities. For
            # a swarm run, skip it for now (see the else-branch note) --
            # CSV telemetry + the existing orientation-triad rendering
            # (confirmed working previously) are enough to verify
            # behavior; the real mesh schematic for N>1 drones is a
            # follow-up, not blocking on the first swarm test.
            world.schematic(
                build_schematic(drone_glb=Config.GLOBAL.drone_glb),
                "drone.kdl",
            )
        elif i == 0:
            print(
                "[sim] N_DRONES>1: skipping world.schematic() for now (untested with "
                "multiple named entities) -- orientation triads + CSV telemetry still "
                "work for verification.",
                flush=True,
            )

    # Milestone 2: a static "heat source" target for the thermal camera to
    # find, shared by every drone in the swarm. StaticMarker has no
    # inertia/force/velocity, so gravity/drag/etc. simply don't apply to
    # it -- it sits at its spawn position forever.
    if os.environ.get("DEBUG_STATIC") == "1":
        heat_source_xy = np.array([
            float(os.environ.get("DEBUG_HEAT_X", "3.3")),
            float(os.environ.get("DEBUG_HEAT_Y", "0.0")),
        ])
        heat_source_z = float(os.environ.get("DEBUG_HEAT_Z", "0.0"))
    else:
        heat_source_xy = _random_heat_source_xy()
        heat_source_z = 0.0
    heat_source = world.spawn(
        StaticMarker(
            world_pos=el.SpatialTransform(
                linear=jnp.array([heat_source_xy[0], heat_source_xy[1], heat_source_z + 0.5])  # small real-world height so it's not glued to the exact ground plane
            ),
        ),
        name="heat_source",
    )
    world.thermal_tag(heat_source, temperature_c=800.0, emissivity=0.95)

    return world, drone_names


def system(only_rate_control: bool = False) -> el.System:
    non_effectors = (
        guidance.waypoint_guidance
        | control.update_target_attitude
        | control.attitude_control
        | control.rate_pid_state
        | control.rate_control
        | motors.output
    )
    if only_rate_control:
        non_effectors = (
            control.rate_flight_plan | control.rate_pid_state | control.rate_control | motors.output
        )

    INNER_RUN_COUNT = round(Config.GLOBAL.dt / Config.GLOBAL.fast_loop_time_step)
    assert INNER_RUN_COUNT == 3

    def _fast_block() -> el.System:
        # Built fresh on every call -- no System object is piped into
        # itself or reused across the chain below. This replaces the old
        # inner_loop() helper, which took ONE system object and did
        # `out_sys = out_sys | system` in a loop (self-composing the exact
        # same Python object 3x). That self-composition pattern doesn't
        # appear anywhere in Elodin's own examples (many-agents, n-body),
        # and the Cranelift panic we hit under N_DRONES=3 named the symbol
        # "inner_295" -- matching this code's own `inner` variable -- so
        # this rewrite is a targeted experiment to see whether avoiding
        # same-object reuse sidesteps the compiler bug.
        effectors = gravity | drag | motor_thrust_response | body_thrust | apply_body_forces
        return (
            el.six_dof(
                Config.GLOBAL.fast_loop_time_step,
                effectors,
                integrator=el.Integrator.SemiImplicit,
            )
            | sensors.imu
            | telemetry.telemetry
        )

    inner = _fast_block() | _fast_block() | _fast_block()
    return non_effectors | inner
