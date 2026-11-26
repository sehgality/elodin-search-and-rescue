"""Reusable position controller for the search-and-rescue Elodin drone.

Put this file next to sim.py and import FlightController and run_mission.
Coordinates are local ENU: x east, y north, z up, in meters. The drone
starts airborne at z=2 m; this model has no ground-contact/landing system.

Example mission (in another Python file):

    from flight_controller import run_mission

    def mission(drone):
        if drone.time < 10:
            drone.go_to(0, 0, 5)
        else:
            drone.go_to(5, 3, 5)

    run_mission(mission, duration=30)

The mission callback runs periodically after the simulation advances. It must
return promptly; do not wait or sleep inside it. Commands persist between calls.
This is a simulation controller using true simulated position and velocity,
not PX4, MAVLink, a real-aircraft controller, or a landing implementation.
"""

from dataclasses import dataclass, field
import math
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

import autopilot
import control
import mekf
import motors
import sensors
import sim
import telemetry
import util
from config import Config, Control, Frame


Target = ty.Annotated[
    jax.Array,
    el.Component("flight_target", el.ComponentType(el.PrimitiveType.F64, (3,)),
                 metadata={"external_control": "true", "element_names": "east,north,up"}),
]


@dataclass
class FlightTarget(el.Archetype):
    target: Target = field(default_factory=lambda: jnp.array([0.0, 0.0, 2.0]))


# altitude.py was folded into guidance.py/control.py a while back (see
# control.py's own comment above its ThrottleCommand definition -- that's
# where the component type lives now, not in a standalone altitude
# module). guidance.py's state machine carries throttle_command as a
# field on its own Guidance archetype, but this file doesn't use
# guidance.Guidance (it's a simpler standalone controller, independent of
# the search/approach/hover mission logic), so it needs its own minimal
# archetype to hold the component -- this replaces the old altitude.Throttle().
@dataclass
class Throttle(el.Archetype):
    throttle_command: control.ThrottleCommand = field(default_factory=lambda: jnp.float64(0.0))


@el.map
def position_control(pos: el.WorldPos, vel: el.WorldVel, target: Target,
                     angle: control.AngleDesired) -> tuple[control.AngleDesired, control.ThrottleCommand]:
    # Delegates to autopilot.py instead of re-deriving the same control law
    # inline. This used to duplicate autopilot's gains (0.12/0.18 position,
    # 0.15/0.35/0.25 altitude, +-0.30 tilt) with two real behavior gaps:
    # it clipped pitch/roll independently instead of clamping the combined
    # tilt vector's magnitude (autopilot.py's docstring explains why that's
    # wrong -- it can distort the commanded direction and fly away from the
    # target under a tight cap), and it had no ground-avoidance override at
    # all. Routing through autopilot.goto_xy/hold_altitude fixes both and
    # means there's one tuned control law, not two that can drift apart.
    position = pos.linear()
    velocity = vel.linear()
    pitch, roll = autopilot.goto_xy(position[:2], velocity[:2], target[:2], pos.angular())
    desired = jnp.array([roll, pitch, 0.0])
    throttle = autopilot.hold_altitude(
        position[2], velocity[2], target[2], Config.GLOBAL.control.motor_thrust_hover
    )
    return angle + 0.08 * (desired - angle), throttle


class FlightController:
    """Command interface available to a mission callback during simulation."""

    def __init__(self, initial_position=(0.0, 0.0, 2.0)):
        self._target = np.asarray(initial_position, dtype=np.float64).copy()
        self.position = self._target.copy()
        self.time = 0.0

    @property
    def target(self):
        return self._target.copy()

    def go_to(self, x, y, altitude):
        """Fly toward an ENU position in meters and hold it until changed."""
        target = np.asarray([x, y, altitude], dtype=np.float64)
        if not np.isfinite(target).all() or target[2] < 0.5:
            raise ValueError("Coordinates must be finite and altitude at least 0.5 m")
        self._target = target

    def set_altitude(self, altitude):
        """Change altitude while keeping the current horizontal target."""
        self.go_to(self._target[0], self._target[1], altitude)

    def hover(self, altitude=None):
        """Hold the current horizontal position, optionally at a new altitude."""
        self.go_to(self.position[0], self.position[1],
                   self.position[2] if altitude is None else altitude)

    def reached(self, tolerance=0.5):
        """Whether the current position is within tolerance meters of target."""
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        return bool(np.linalg.norm(self.position - self._target) <= tolerance)


def _configure():
    q_right = el.Quaternion.from_axis_angle(axis=np.array([1., 0., 0.]), angle=np.deg2rad(5.))
    q_left = el.Quaternion.from_axis_angle(axis=np.array([1., 0., 0.]), angle=np.deg2rad(-5.))
    q_back = el.Quaternion.from_axis_angle(axis=np.array([0., 1., 0.]), angle=np.deg2rad(-3.75))
    up = np.array([0., 0., 1.])
    Config(
        control=Control(rate_pid_gains=np.array([[.32, .32, 1.10], [.05, .05, .08], [.12, .08, .01]]),
                        angle_p_gains=np.array([4., 4., 1.]), motor_thrust_exponent=.833,
                        motor_thrust_hover=.689, attitude_control_input_tc=.2, pilot_yaw_rate_tc=.25),
        drone_glb="talon-quad-v2.glb", mass=2.586,
        inertia_diagonal=np.array([.0854, .1149, .1604]), start_pos=np.array([0., 0., 2.]),
        start_euler_angles=np.zeros(3),
        motor_positions=np.array([[.26, -.26, .26, -.26], [-.2075, .2015, .2075, -.2015],
                                  [-.0215, .0215, -.0215, .0215]]).T,
        motor_thrust_directions=np.array([q_right @ up, (q_left * q_back) @ up,
                                          q_left @ up, (q_right * q_back) @ up]),
        motor_thrust_curve_path="./motor_thrust_curve.csv", simulation_rate=300.,
        frame=Frame.QUAD_X, fast_loop_time_step=1./900., simulation_time=90., sensor_noise=True,
    ).set_as_global()


def _fast_block() -> el.System:
    # sim.py no longer has an inner_loop() helper to call -- it was removed
    # during the Cranelift-crash investigation (self-composing the SAME
    # system object 3x was one of two hypotheses for the N_DRONES>1 crash;
    # see cranelift_concatenate_bug_report.md). sim.py's own system()
    # replaced it with this exact pattern -- build a fresh system object on
    # every call so nothing gets self-composed -- and this file follows
    # suit rather than calling the no-longer-existing sim.inner_loop().
    effectors = sim.gravity | sim.drag | sim.motor_thrust_response | sim.body_thrust | sim.apply_body_forces
    return (
        el.six_dof(Config.GLOBAL.fast_loop_time_step, effectors, integrator=el.Integrator.SemiImplicit)
        | sensors.imu
        | telemetry.telemetry
    )


def run_mission(mission, duration=30.0, callback_rate=30.0, realtime=True):
    """Run mission(drone) against the project's Talon simulation.

    Run from the search-and-rescue project directory. Mission state can live in
    a callable object or closure. The callback sees drone.time and position.
    """
    if not callable(mission) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Provide a mission callback and a positive finite duration")
    if not math.isfinite(callback_rate) or not 0 < callback_rate <= 300:
        raise ValueError("callback_rate must be between 0 and 300 Hz")
    _configure()
    w = el.World()
    w.spawn([el.Body(world_pos=Config.GLOBAL.spatial_transform, inertia=Config.GLOBAL.spatial_inertia),
             sim.Drone(), motors.Motors(), sensors.IMU(), control.AttitudeController(),
             FlightTarget(), Throttle(), mekf.MEKF(), telemetry.Telemetry()], name="drone")
    w.schematic('''
        coordinate frame=ENU
        timeline follow_latest=#true
        viewport name="Drone" pos="drone.world_pos.translate_world(-6,-6,4)" look_at="drone.world_pos" show_grid=#true
        object_3d drone.world_pos { glb path="talon-quad-v2.glb" }
    ''', "flight_controller.kdl")
    inner = _fast_block() | _fast_block() | _fast_block()
    system = (position_control | control.update_target_attitude | control.attitude_control |
              control.rate_pid_state | control.rate_control | motors.output | inner)
    drone = FlightController()
    period = Config.GLOBAL.simulation_rate / callback_rate
    next_callback = [0.0]

    def post_step(tick, ctx):
        if tick < next_callback[0]:
            return
        next_callback[0] += period
        pose = ctx.read_component("drone.world_pos")
        if pose is None:
            return
        drone.position = np.asarray(pose, dtype=np.float64).reshape(-1)[-3:].copy()
        drone.time = tick / Config.GLOBAL.simulation_rate
        mission(drone)
        ctx.write_component("drone.flight_target", drone.target)

    w.run(system, simulation_rate=Config.GLOBAL.simulation_rate,
          telemetry_rate=Config.GLOBAL.simulation_rate / 3, generate_real_time=realtime,
          max_ticks=int(duration * Config.GLOBAL.simulation_rate), interactive=False, post_step=post_step)


if __name__ == "__main__":
    last_log = [-5.0]
    def demo(drone):
        drone.go_to(0, 0, 5) if drone.time < 10 else drone.go_to(5, 3, 5)
        if drone.time - last_log[0] >= 5:
            last_log[0] = drone.time
            print(f"t={drone.time:.1f}s position={drone.position.round(2)} target={drone.target}", flush=True)
    run_mission(demo)
