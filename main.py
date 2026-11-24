#!/usr/bin/env uv run

import sys

import guidance
import elodin as el
import jax.numpy as jnp
import numpy as np
import polars as pl
import util
from config import Config, Control, Frame
from sim import system, world

EDU_450_CONFIG = Config(
    control=Control(
        rate_pid_gains=np.array(
            [
                [0.40, 0.40, 2.50],  # P
                [0.02, 0.02, 0.02],  # I
                [0.05, 0.05, 0.01],  # D
            ]
        ),
        angle_p_gains=np.array([4.0, 4.0, 1.0]),
        motor_thrust_exponent=0.65,
        motor_thrust_hover=0.35,
        attitude_control_input_tc=0.1,
        pilot_yaw_rate_tc=0.1,
    ),
    drone_glb="edu-450-v2-drone.glb",
    mass=1.0,
    inertia_diagonal=np.array([0.1, 0.1, 0.2]),
    start_pos=np.array([0.0, 0.0, 2.0]),
    start_euler_angles=np.array([0.0, 0.0, 0.0]),
    motor_positions=util.motor_positions(np.pi * np.array([0.25, -0.75, 0.75, -0.25]), 0.24),
    motor_thrust_directions=np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    motor_thrust_curve_path="./motor_thrust_curve.csv",
    simulation_rate=300.0,
    frame=Frame.QUAD_X,
    fast_loop_time_step=(1.0 / 900.0),
    simulation_time=90.0,
    sensor_noise=True,
)

# Motor configuration: Quad-X
# Motor 1: Front right motor, counter-clockwise
# Motor 2: Front left motor, clockwise
# Motor 3: Rear right motor, clockwise
# Motor 4: Rear left motor, counter-clockwise
#  (CW) 3 1 (CCW)
#        X
# (CCW) 2 4 (CW)

q_tilt_right = el.Quaternion.from_axis_angle(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=np.deg2rad(5.0),
)
q_tilt_left = el.Quaternion.from_axis_angle(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=np.deg2rad(-5.0),
)
q_tilt_back = el.Quaternion.from_axis_angle(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=np.deg2rad(-3.75),
)
up = np.array([0.0, 0.0, 1.0])
motor_thrust_directions = jnp.array(
    [
        q_tilt_right @ up,
        (q_tilt_left * q_tilt_back) @ up,
        q_tilt_left @ up,
        (q_tilt_right * q_tilt_back) @ up,
    ]
)

# Assume CG is at the center of the quad motor positions
TALON_QUAD_CONFIG = Config(
    control=Control(
        rate_pid_gains=np.array(
            [
                [0.32, 0.32, 1.10],  # P
                [0.05, 0.05, 0.08],  # I
                [0.12, 0.08, 0.01],  # D
            ]
        ),
        angle_p_gains=np.array([4.0, 4.0, 1.0]),
        motor_thrust_exponent=0.833,
        motor_thrust_hover=0.689,
        attitude_control_input_tc=0.2,  # soft
        pilot_yaw_rate_tc=0.25,  # soft
    ),
    drone_glb="talon-quad-v2.glb",
    mass=2.586,
    inertia_diagonal=np.array([0.0854, 0.1149, 0.1604]),
    start_pos=np.array([0.0, 0.0, 2.0]),
    start_euler_angles=np.array([0.0, 0.0, 0.0]),
    motor_positions=np.array(
        [
            [0.26, -0.26, 0.26, -0.26],  # X position for each motor
            [-0.2075, 0.2015, 0.2075, -0.2015],  # Y position for each motor
            [-0.0215, 0.0215, -0.0215, 0.0215],  # Z position for each motor
        ]
    ).T,
    motor_thrust_directions=np.array(motor_thrust_directions),
    motor_thrust_curve_path="./motor_thrust_curve.csv",
    simulation_rate=300.0,
    frame=Frame.QUAD_X,
    fast_loop_time_step=(1.0 / 900.0),
    simulation_time=90.0,
    sensor_noise=True,
)
TALON_QUAD_CONFIG.set_as_global()
# EDU_450_CONFIG.set_as_global()

import os
import pathlib

# --- thermal target detection (milestone 2, step 1: detect + log only) ---
THERMAL_WIDTH = 128
THERMAL_HEIGHT = 128
THERMAL_FOV_DEG = 90.0
THERMAL_LATENCY_US = 33_333  # ~1 frame at 30fps
WARM_THRESHOLD = 100  # 0-255; lowered for real lwir white_hot render (was tuned against fake 'normal' RGB mesh color)
DEBUG_THERMAL = os.environ.get("DEBUG_THERMAL", "0") == "1"
DEBUG_EVERY = int(os.environ.get("DEBUG_THERMAL_EVERY", "5"))
LOG_CSV = os.environ.get("LOG_CSV", "0") == "1"  # writes a full per-tick CSV -- see this patch's docstring
# Default path is ABSOLUTE, anchored to this script's own folder -- elodin
# run may execute this process with a different working directory than the
# one you launched it from, so a bare relative filename can silently land
# somewhere else with no error at all.
try:
    _SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = pathlib.Path.cwd()
LOG_CSV_PATH = os.environ.get("LOG_CSV_PATH", str(_SCRIPT_DIR / "flight_log.csv"))
if LOG_CSV:
    print(f"[csv] LOG_CSV=1 -- will write to {LOG_CSV_PATH}", flush=True)


CAM_HEIGHT = 128  # must match sim.py's world.sensor_camera(..., height=...)
CAM_WIDTH = 128   # must match sim.py's world.sensor_camera(..., width=...)

CONFIRM_ALTITUDE_M = guidance.HOVER_ALTITUDE_M  # success = descended to and settled at this altitude
CONFIRM_ALTITUDE_TOLERANCE_M = 0.2
HOVER_SETTLE_TICKS_REQUIRED = 30  # ~1s of sustained hover at the confirm altitude, not just one tick passing through it on the way down
HOVER_SETTLE_MAX_SPEED_MPS = 0.3  # must have actually stopped moving (3D speed, not just vertical) -- without this, a slow pass THROUGH the altitude band while still descending could rack up settled ticks and trigger shutdown before it's really landed

_MODE_NAMES = {0: "SEARCH", 1: "APPROACH", 2: "HOVER"}
CAMERA_LATENCY_US = 33_000  # ~1 frame at 30fps

_csv_file = [None]  # lazily opened on first write so LOG_CSV=0 never touches disk


def _open_csv_if_needed():
    if not LOG_CSV or _csv_file[0] is not None:
        return
    _csv_file[0] = open(LOG_CSV_PATH, "w", buffering=1)
    print(f"[csv] opened {LOG_CSV_PATH} for writing (cwd={os.getcwd()})", flush=True)
    _csv_file[0].write(
        "tick,drone,mode,drone_x,drone_y,drone_z,vel_x,vel_y,vel_z,"
        "quat_w,quat_x,quat_y,quat_z,"
        "heat_x,heat_y,dist,detected,bearing_x,bearing_y,max_red,mask_px,"
        "centered_ticks,lost_ticks,approach_ticks,hold_x,hold_y\n"
    )


def _n(v):
    return "" if v is None else f"{float(v):.4f}"


def _log_csv_row(tick, drone_name, mode_name, drone_xyz, drone_vel_xyz, drone_quat_wxyz,
                  heat_xyz, dist, detected, bearing_x=None, bearing_y=None, max_red=None,
                  mask_count=None, centered_ticks=None, lost_ticks=None, approach_ticks=None,
                  hold_xy=None):
    if not LOG_CSV:
        return
    _open_csv_if_needed()
    dx, dy, dz = drone_xyz if drone_xyz is not None else (None, None, None)
    vx, vy, vz = drone_vel_xyz if drone_vel_xyz is not None else (None, None, None)
    qw, qx, qy, qz = drone_quat_wxyz if drone_quat_wxyz is not None else (None, None, None, None)
    hx, hy = (heat_xyz[0], heat_xyz[1]) if heat_xyz is not None else (None, None)
    hpx, hpy = (hold_xy[0], hold_xy[1]) if hold_xy is not None else (None, None)
    _csv_file[0].write(
        f"{tick},{drone_name},{mode_name},"
        f"{_n(dx)},{_n(dy)},{_n(dz)},{_n(vx)},{_n(vy)},{_n(vz)},"
        f"{_n(qw)},{_n(qx)},{_n(qy)},{_n(qz)},"
        f"{_n(hx)},{_n(hy)},{_n(dist)},{int(detected)},"
        f"{_n(bearing_x)},{_n(bearing_y)},{_n(max_red)},"
        f"{'' if mask_count is None else mask_count},"
        f"{'' if centered_ticks is None else centered_ticks},"
        f"{'' if lost_ticks is None else lost_ticks},"
        f"{'' if approach_ticks is None else approach_ticks},"
        f"{_n(hpx)},{_n(hpy)}\n"
    )


def _state_line(tick, drone_name, mode_name, drone_xyz, heat_xyz, dist, hold_xy, detected,
                 bearing_x=None, bearing_y=None, max_red=None, mask_count=None,
                 centered_ticks=None, lost_ticks=None, approach_ticks=None, note=""):
    dxyz_s = f"({drone_xyz[0]:.2f},{drone_xyz[1]:.2f},alt={drone_xyz[2]:.2f})" if drone_xyz is not None else "?"
    hxy_s = f"({heat_xyz[0]:.2f},{heat_xyz[1]:.2f})" if heat_xyz is not None else "?"
    hold_s = f"({hold_xy[0]:.2f},{hold_xy[1]:.2f})" if hold_xy is not None else "?"
    bearing_s = f"({bearing_x:.2f},{bearing_y:.2f})" if bearing_x is not None else "?"
    max_red_s = f"{max_red:.1f}" if max_red is not None else "?"
    return (
        f"[state:{drone_name}] tick={tick} mode={mode_name} "
        f"drone_xyz={dxyz_s} heat_xy={hxy_s} dist={dist} "
        f"bearing={bearing_s} R_max={max_red_s} mask_px={mask_count} "
        f"centered_ticks={centered_ticks} lost_ticks={lost_ticks} approach_ticks={approach_ticks} "
        f"hold_xy={hold_s} -- {'DETECTED' if detected else 'no detect'} {note}"
    )


def make_thermal_detect_post_step(drone_names: list[str]):
    """Builds the post_step hook for this specific set of drones. Every
    drone gets its OWN mode-transition tracking and hover-settle-ticks
    counter (a dict keyed by name) instead of the single shared module-
    level state the single-drone version used -- otherwise one drone's
    mode changes would stomp on another's tracking. Mission-complete is
    "first drone to lock on and land wins" -- the whole sim ends (same
    hard os._exit as before) the moment ANY one drone satisfies the
    settle condition, the others' progress notwithstanding.
    """
    last_mode = {name: 0 for name in drone_names}
    hover_settled_ticks = {name: 0 for name in drone_names}

    def thermal_detect_post_step(tick: int, ctx: el.StepContext) -> None:
        heat_pos_arr = ctx.read_component("heat_source.world_pos")
        heat_xyz = np.asarray(heat_pos_arr).reshape(-1)[-3:] if heat_pos_arr is not None else None

        for name in drone_names:
            mode_arr = ctx.read_component(f"{name}.guidance_mode")
            current_mode = last_mode[name]
            if mode_arr is not None:
                current_mode = int(np.asarray(mode_arr).reshape(-1)[0])
                if current_mode != last_mode[name]:
                    print(
                        f"[guidance:{name}] tick {tick}: mode "
                        f"{_MODE_NAMES.get(last_mode[name], last_mode[name])} -> "
                        f"{_MODE_NAMES.get(current_mode, current_mode)}",
                        flush=True,
                    )
                    last_mode[name] = current_mode
            mode_name = _MODE_NAMES.get(last_mode[name], last_mode[name])

            dist = None
            drone_xyz = None
            drone_vel_xyz = None
            drone_quat_wxyz = None
            try:
                drone_pos_arr = ctx.read_component(f"{name}.world_pos")
                drone_vel_arr = ctx.read_component(f"{name}.world_vel")
                if drone_pos_arr is not None and heat_xyz is not None:
                    drone_pos_flat = np.asarray(drone_pos_arr).reshape(-1)
                    drone_xyz = drone_pos_flat[-3:]
                    dist = float(np.linalg.norm(drone_xyz[:2] - heat_xyz[:2]))
                    if tick == 0:
                        print(f"[debug:{name}] world_pos raw length={drone_pos_flat.size} (expect 7: quat[4]+xyz[3])", flush=True)
                    if drone_pos_flat.size >= 7:
                        drone_quat_wxyz = drone_pos_flat[:4]
                if drone_vel_arr is not None:
                    drone_vel_xyz = np.asarray(drone_vel_arr).reshape(-1)[-3:]
            except Exception as e:
                if tick == 0:
                    print(f"[debug:{name}] couldn't read drone/heat position: {e}", flush=True)

            # --- mission-complete check: same rule as the single-drone
            # version (settled in HOVER at the confirm altitude, actually
            # stopped moving, for a sustained stretch), just tracked per
            # drone now. First drone to satisfy it ends the whole sim.
            drone_speed_3d = float(np.linalg.norm(drone_vel_xyz)) if drone_vel_xyz is not None else None
            if (
                current_mode == 2
                and drone_xyz is not None
                and abs(float(drone_xyz[2]) - CONFIRM_ALTITUDE_M) < CONFIRM_ALTITUDE_TOLERANCE_M
                and drone_speed_3d is not None
                and drone_speed_3d < HOVER_SETTLE_MAX_SPEED_MPS
            ):
                hover_settled_ticks[name] += 1
            elif current_mode == 2:
                hover_settled_ticks[name] = max(0, hover_settled_ticks[name] - 1)
            else:
                hover_settled_ticks[name] = 0

            if current_mode == 2 and hover_settled_ticks[name] >= HOVER_SETTLE_TICKS_REQUIRED:
                dist_s = f"{dist:.2f}m" if dist is not None else "?"
                print(
                    f"\n[MISSION COMPLETE] tick {tick}: {name} settled in HOVER at "
                    f"({drone_xyz[0]:.2f}, {drone_xyz[1]:.2f}, alt={drone_xyz[2]:.2f}), "
                    f"horizontal offset from heat source={dist_s} -- ending simulation\n",
                    flush=True,
                )
                # sys.exit(0) raises SystemExit, which elodin's Rust simulation
                # loop just catches and logs as a "post_step error" warning
                # instead of actually stopping -- os._exit is a hard OS-level
                # kill that nothing on the Rust side can catch.
                os._exit(0)

            centered_ticks = lost_ticks = approach_ticks = None
            hold_xy = None
            if DEBUG_THERMAL or LOG_CSV:
                try:
                    ct_arr = ctx.read_component(f"{name}.centered_ticks")
                    lt_arr = ctx.read_component(f"{name}.lost_ticks")
                    at_arr = ctx.read_component(f"{name}.approach_ticks")
                    hp_arr = ctx.read_component(f"{name}.hold_point")
                    if ct_arr is not None:
                        centered_ticks = int(np.asarray(ct_arr).reshape(-1)[0])
                    if lt_arr is not None:
                        lost_ticks = int(np.asarray(lt_arr).reshape(-1)[0])
                    if at_arr is not None:
                        approach_ticks = int(np.asarray(at_arr).reshape(-1)[0])
                    if hp_arr is not None:
                        hold_xy = np.asarray(hp_arr).reshape(-1)[-2:]
                except Exception as e:
                    if tick == 0:
                        print(f"[debug:{name}] couldn't read extra guidance state: {e}", flush=True)

            frame = ctx.read_msg(f"{name}.thermal_cam", timestamp=ctx.timestamp - CAMERA_LATENCY_US)
            if frame is None:
                frame = ctx.read_msg(f"{name}.thermal_cam")  # fall back to "latest" during warmup

            if frame is None:
                ctx.write_component(f"{name}.target_detect", np.zeros(4, dtype=np.float64))
                _log_csv_row(tick, name, mode_name, drone_xyz, drone_vel_xyz, drone_quat_wxyz,
                             heat_xyz, dist, False, centered_ticks=centered_ticks,
                             lost_ticks=lost_ticks, approach_ticks=approach_ticks, hold_xy=hold_xy)
                if DEBUG_THERMAL and tick % DEBUG_EVERY == 0:
                    print(_state_line(tick, name, mode_name, drone_xyz, heat_xyz, dist, hold_xy,
                                       False, centered_ticks=centered_ticks, lost_ticks=lost_ticks,
                                       approach_ticks=approach_ticks, note="(NO FRAME)"), flush=True)
                continue

            flat = np.asarray(frame)
            expected_size = CAM_HEIGHT * CAM_WIDTH * 4
            assert flat.size == expected_size, (
                f"[{name}] thermal_cam frame size {flat.size} != expected {expected_size} "
                f"({CAM_HEIGHT}x{CAM_WIDTH}x4) — camera config changed in sim.py?"
            )

            arr = flat.reshape(CAM_HEIGHT, CAM_WIDTH, 4)
            arr = np.flipud(arr)  # camera buffer appears to be stored bottom-up -- flip so row 0 = top, matching normal image/bearing convention
            red = arr[..., 0].astype(np.float64)
            max_red = float(red.max())
            mask = red > WARM_THRESHOLD
            mask_count = int(mask.sum())

            if not mask.any():
                ctx.write_component(f"{name}.target_detect", np.array([0.0, 0.0, 0.0, max_red], dtype=np.float64))
                _log_csv_row(tick, name, mode_name, drone_xyz, drone_vel_xyz, drone_quat_wxyz,
                             heat_xyz, dist, False, max_red=max_red, mask_count=mask_count,
                             centered_ticks=centered_ticks, lost_ticks=lost_ticks,
                             approach_ticks=approach_ticks, hold_xy=hold_xy)
                if DEBUG_THERMAL and tick % DEBUG_EVERY == 0:
                    print(_state_line(tick, name, mode_name, drone_xyz, heat_xyz, dist, hold_xy,
                                       False, max_red=max_red, mask_count=mask_count,
                                       centered_ticks=centered_ticks, lost_ticks=lost_ticks,
                                       approach_ticks=approach_ticks), flush=True)
                continue

            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())
            h, w = red.shape[:2]
            bearing_x = (cx - w / 2.0) / (w / 2.0)
            bearing_y = (cy - h / 2.0) / (h / 2.0)

            ctx.write_component(
                f"{name}.target_detect",
                np.array([1.0, bearing_x, bearing_y, max_red], dtype=np.float64),
            )

            _log_csv_row(tick, name, mode_name, drone_xyz, drone_vel_xyz, drone_quat_wxyz,
                         heat_xyz, dist, True, bearing_x=bearing_x, bearing_y=bearing_y,
                         max_red=max_red, mask_count=mask_count, centered_ticks=centered_ticks,
                         lost_ticks=lost_ticks, approach_ticks=approach_ticks, hold_xy=hold_xy)
            if DEBUG_THERMAL and tick % DEBUG_EVERY == 0:
                print(_state_line(tick, name, mode_name, drone_xyz, heat_xyz, dist, hold_xy, True,
                                   bearing_x=bearing_x, bearing_y=bearing_y, max_red=max_red,
                                   mask_count=mask_count, centered_ticks=centered_ticks,
                                   lost_ticks=lost_ticks, approach_ticks=approach_ticks), flush=True)

    return thermal_detect_post_step


args = sys.argv[1:]
world, drone_names = world()
post_step_fn = make_thermal_detect_post_step(drone_names)
if "--telemetry" in args:
    exec = world.build(system(only_rate_control=True))
    exec.run(Config.GLOBAL.total_sim_ticks)
    df = exec.history(["drone.body_ang_vel", "drone.motor_ang_vel"])
    print(df)
    df = df.select(
        pl.col("drone.body_ang_vel").arr.get(0).alias("body_ang_vel_x"),
        pl.col("drone.body_ang_vel").arr.get(1).alias("body_ang_vel_y"),
        pl.col("drone.body_ang_vel").arr.get(2).alias("body_ang_vel_z"),
        pl.col("drone.motor_ang_vel").arr.get(0).alias("motor_ang_vel_1"),
        pl.col("drone.motor_ang_vel").arr.get(1).alias("motor_ang_vel_2"),
        pl.col("drone.motor_ang_vel").arr.get(2).alias("motor_ang_vel_3"),
        pl.col("drone.motor_ang_vel").arr.get(3).alias("motor_ang_vel_4"),
    )
    df.write_csv("telemetry.csv")
else:
    world.run(
        system(),
        simulation_rate=Config.GLOBAL.simulation_rate,
        telemetry_rate=Config.GLOBAL.simulation_rate / 3.0,
        generate_real_time=True,
        max_ticks=int(Config.GLOBAL.total_sim_ticks or 1200),
        interactive=False,
        post_step=post_step_fn,
    )
