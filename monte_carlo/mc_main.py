#!/usr/bin/env uv run
"""Monte Carlo campaign entry point. Run via:

    elodin monte-carlo run monte_carlo/mc_main.py \\
        --campaign monte_carlo/campaign.toml \\
        --spec monte_carlo/spec.toml \\
        --out dbs/mc-fault-campaign

DETECTION IS SYNTHETIC HERE, NOT FROM THE REAL THERMAL CAMERA. See
_synthetic_detect() below for the full rationale with real-run evidence.
Short version: in a real 96-run campaign only ~9 of 96 workers' render
servers kept up well enough to emit camera frames at all (the render
server is wall-clock-paced at 30fps per camera, and 12 parallel workers
starve each other's render pipelines even with generate_real_time=True).
So the mission oracles' pass rates were noise-floored by camera dropouts
rather than reflecting the guidance faults we actually care about. The
REAL thermal camera path stays unchanged in main.py's single-drone run
(that's where the CV demo lives); this Monte Carlo harness bypasses the
render server entirely with a ground-truth FOV check, so the guidance
and control pipeline is exercised WITHOUT the rendering noise. The two
"sensor" faults are preserved -- see FAULT_CATALOG[...] in mc_sim.py and
_synthetic_detect() below for how they map onto the analytical path.
"""

import os
import sys

import numpy as np

# `elodin monte-carlo run` executes this file from a working directory
# that doesn't have the project root (search-and-rescue/, where
# guidance.py etc. live) on sys.path -- confirmed from a real run:
# "ModuleNotFoundError: No module named 'guidance'" at the `import
# guidance` line below, even though mc_sim.py (imported right after)
# does this same sys.path fix for ITSELF. The fix has to happen here too,
# and before `import guidance`, not after.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import elodin as el
import guidance
from mc_sim import PARAMS, MC_STATE, build, resolve_fault, PREMATURE_HOVER_SPEED_MPS, PREMATURE_HOVER_DIST_M

params = el.monte_carlo.params(PARAMS)
# resolve_fault() is the one place this is looked up (see mc_sim.py) --
# build() below calls it again internally for the actual fault
# application, so this and that can't silently disagree on which key
# resolved.
fault = resolve_fault(params)
world, system = build(params)

MAX_TICKS = int(60.0 * 300.0)  # 60s at 300Hz -- generous; most runs resolve well before this

CAM_HEIGHT = 128
CAM_WIDTH = 128
CAMERA_LATENCY_US = 33_000
CONFIRM_ALTITUDE_M = guidance.HOVER_ALTITUDE_M
CONFIRM_ALTITUDE_TOLERANCE_M = 0.2
HOVER_SETTLE_TICKS_REQUIRED = 30
HOVER_SETTLE_MAX_SPEED_MPS = 0.3


MC_DEBUG = os.environ.get("MC_DEBUG") == "1"


def _make_post_step():
    state = {
        "last_mode": 0,
        "hover_settled_ticks": 0,
        "mode_switch_count": 0,
        "completed": False,
        "complete_tick": -1,
        "min_altitude_m": float("inf"),
        "max_altitude_m": float("-inf"),
        "premature_hover_violation": False,
        "hover_entries": 0,
        "final_offset_m": None,
        "final_tick": 0,
        # Added after the first real campaign run came back 0/96
        # completions with no visible error -- these cost nothing and let
        # us tell "detection never fired" apart from "detection fired but
        # tracking/hover never converged" from result.json alone, without
        # needing a rerun with MC_DEBUG just to find out.
        "ever_detected": False,
        "first_frame_seen": False,
        "first_frame_size": -1,
        "first_heat_xy": None,
        "ticks_in_search": 0,
        "ticks_in_approach": 0,
        "ticks_in_hover": 0,
        # Call counter for debug printing -- see the MC_DEBUG block below.
        # post_step runs at the TELEMETRY rate (100 Hz), not the 300 Hz
        # simulation_rate, confirmed from a real run: "simulation cycles"
        # in elodin's own summary (post_step call count) was exactly 1/3
        # of "simulation steps" (raw 300 Hz ticks), and ticks_in_search
        # capped at exactly 6000 = 60s*100Hz across every single campaign
        # run regardless of fault/placement. The `tick` argument IS the
        # real underlying 300 Hz simulation tick, but which ticks get a
        # post_step call at all is decided by elodin's telemetry sampling,
        # not by a clean tick%3==0 stride -- a first debug run (`MC_DEBUG=1
        # elodin run monte_carlo/mc_main.py`) printed zero of the `tick in
        # (0, 300, 3000, 9000, 15000)` lines despite 100% completion of the
        # engine run (17,898 of 18,000 ticks logged, "simulation cycles:
        # 5,966"), meaning none of those 5 exact tick values ever happened
        # to be one elodin actually called post_step on. Counting calls
        # instead of matching tick VALUES is the fix.
        "_debug_calls": 0,
    }

    def post_step(tick: int, ctx: "el.StepContext") -> None:
        state["final_tick"] = tick
        heat_pos_arr = ctx.read_component("heat_source.world_pos")
        heat_xyz = np.asarray(heat_pos_arr).reshape(-1)[-3:] if heat_pos_arr is not None else None

        mode_arr = ctx.read_component("drone.guidance_mode")
        current_mode = state["last_mode"]
        if mode_arr is not None:
            current_mode = int(np.asarray(mode_arr).reshape(-1)[0])
            if current_mode != state["last_mode"]:
                state["mode_switch_count"] += 1
                if current_mode == guidance.MODE_HOVER:
                    state["hover_entries"] += 1
                state["last_mode"] = current_mode
        if current_mode == guidance.MODE_SEARCH:
            state["ticks_in_search"] += 1
        elif current_mode == guidance.MODE_APPROACH:
            state["ticks_in_approach"] += 1
        elif current_mode == guidance.MODE_HOVER:
            state["ticks_in_hover"] += 1

        if heat_xyz is not None and state["first_heat_xy"] is None:
            state["first_heat_xy"] = (float(heat_xyz[0]), float(heat_xyz[1]))

        drone_xyz = None
        drone_vel_xyz = None
        drone_pos_arr = ctx.read_component("drone.world_pos")
        drone_vel_arr = ctx.read_component("drone.world_vel")
        dist = None
        if drone_pos_arr is not None:
            drone_xyz = np.asarray(drone_pos_arr).reshape(-1)[-3:]
            state["min_altitude_m"] = min(state["min_altitude_m"], float(drone_xyz[2]))
            state["max_altitude_m"] = max(state["max_altitude_m"], float(drone_xyz[2]))
            if heat_xyz is not None:
                dist = float(np.linalg.norm(drone_xyz[:2] - heat_xyz[:2]))
                state["final_offset_m"] = dist
        if drone_vel_arr is not None:
            drone_vel_xyz = np.asarray(drone_vel_arr).reshape(-1)[-3:]

        # --- oracle signal: premature HOVER commit (far and/or fast) ---
        if (
            current_mode == guidance.MODE_HOVER
            and dist is not None
            and drone_vel_xyz is not None
        ):
            speed_3d = float(np.linalg.norm(drone_vel_xyz))
            if speed_3d > PREMATURE_HOVER_SPEED_MPS and dist > PREMATURE_HOVER_DIST_M:
                state["premature_hover_violation"] = True

        # --- mission-complete check (same rule as main.py's single-drone
        # path) ---
        drone_speed_3d = float(np.linalg.norm(drone_vel_xyz)) if drone_vel_xyz is not None else None
        if (
            current_mode == guidance.MODE_HOVER
            and drone_xyz is not None
            and abs(float(drone_xyz[2]) - CONFIRM_ALTITUDE_M) < CONFIRM_ALTITUDE_TOLERANCE_M
            and drone_speed_3d is not None
            and drone_speed_3d < HOVER_SETTLE_MAX_SPEED_MPS
        ):
            state["hover_settled_ticks"] += 1
        elif current_mode == guidance.MODE_HOVER:
            state["hover_settled_ticks"] = max(0, state["hover_settled_ticks"] - 1)
        else:
            state["hover_settled_ticks"] = 0

        if not state["completed"] and current_mode == guidance.MODE_HOVER and state["hover_settled_ticks"] >= HOVER_SETTLE_TICKS_REQUIRED:
            state["completed"] = True
            state["complete_tick"] = tick

        # --- SYNTHETIC analytical detection (replaces reading the real
        # drone.thermal_cam message). The real camera path is proven
        # working in main.py; this harness uses analytical FOV math
        # instead, based on real-run evidence that only ~9 of 96 parallel
        # Monte Carlo workers' render servers emitted frames at all
        # (render server is wall-clock-paced at 30fps per camera, and 12
        # concurrent workers starve each other's rendering).
        # Geometry: sim.py mounts the camera on each drone pointing
        # STRAIGHT DOWN (rot_offset=[0,90,0], pos_offset=[0.5,0,-0.25])
        # with fov=80.0 and width=height=128. For a level-flying drone
        # that's a pinhole model whose ground footprint radius =
        # altitude * tan(fov/2). The invariant we assert: the heat
        # source is "detected" iff its horizontal offset from the drone
        # falls within that FOV cone from the drone's current altitude,
        # AND the drone is actually above it (altitude > 0). We write
        # drone.target_detect in the same (detected, bearing_x, bearing_y,
        # max_red) shape guidance.waypoint_guidance reads, with bearings
        # matching autopilot.bearing_to_world_xy's inverse so the
        # downstream estimate still lands on the real heat_xy.
        detected_this_tick = False
        bearing_x = 0.0
        bearing_y = 0.0
        synthetic_strength = 0.0
        if drone_xyz is not None and heat_xyz is not None and float(drone_xyz[2]) > 0.5:
            altitude = float(drone_xyz[2])
            # Camera's TRUE FOV, straight from sim.py -- deliberately
            # NOT guidance.CAMERA_FOV_DEG, which the camera_fov_mismatch
            # fault patches to simulate the AUTOPILOT believing a
            # different FOV than the camera actually has. Detection uses
            # physical truth (what the camera really sees); downstream
            # guidance's bearing_to_world_xy conversion uses its
            # (possibly-mismatched) belief, so that fault still
            # propagates where it should: through the autopilot's wrong
            # interpretation of the correct bearing.
            true_fov_deg = 80.0
            half_fov_tan = np.tan(np.deg2rad(true_fov_deg / 2.0))
            fov_radius_m = altitude * half_fov_tan
            # thermal_threshold_drift was originally "raise
            # WARM_THRESHOLD from 100 to 165", i.e. the heat source
            # reads as dimmer, which reduces the pixel mask area and
            # effectively requires the drone to be nearly on top of
            # the target before the camera triggers. Analytical
            # equivalent: shrink the effective FOV radius so only runs
            # that pass nearly directly over the source trigger.
            # MC_STATE["warm_threshold"] stays 100 in the control and
            # bumps to 165 for the fault; the 2.5m cap below is the
            # radius at which the raised-threshold camera's hot-pixel
            # mask is still large enough to center on in the real run.
            if MC_STATE["warm_threshold"] > 100:
                fov_radius_m = min(fov_radius_m, 2.5)
            offset_world_2d = np.asarray(heat_xyz[:2]) - np.asarray(drone_xyz[:2])
            horiz_dist = float(np.linalg.norm(offset_world_2d))
            if horiz_dist < fov_radius_m:
                detected_this_tick = True
                state["ever_detected"] = True
                # Invert autopilot.bearing_to_world_xy: given world
                # offset (heat - drone), find the bearing that would
                # project back to this world point.
                #   offset_body = inv(attitude) @ [offset_world, 0]
                # then bearing_to_world_xy builds offset_body as
                # [bearing_y, -bearing_x] * altitude * half_fov_tan,
                # so bearing_y = offset_body[0] / (altitude * tan)
                # and  bearing_x = -offset_body[1] / (altitude * tan).
                # For a roughly-level drone (SEARCH altitude, no sharp
                # tilts) attitude is close to identity, so using the
                # world offset directly is a tight approximation --
                # and even where it's off, guidance's own bearing-to-
                # world conversion applies the same rotation again, so
                # the two mostly cancel. The analytical path
                # DELIBERATELY doesn't model attitude rotation: that
                # keeps the thing we're testing (guidance state
                # machine under fault injection) distinct from
                # camera-geometry noise.
                scale = altitude * half_fov_tan
                bearing_y = float(offset_world_2d[0] / scale)
                bearing_x = float(-offset_world_2d[1] / scale)
                # Clamp to the [-1, 1] band bearing_to_world_xy
                # expects; geometry puts us inside already since we
                # passed the FOV check, but float math near the edge
                # shouldn't push a stale 1.000001 downstream.
                bearing_x = max(-1.0, min(1.0, bearing_x))
                bearing_y = max(-1.0, min(1.0, bearing_y))
                # Synthetic "strength" field (mirrors the real
                # camera's max_red, kept so downstream code and
                # result.json fields match shape). 150.0 is just a
                # sentinel > guidance.py's own detection-is-real
                # check uses -- target_detect[0] > 0.5 is the real
                # gate.
                synthetic_strength = 150.0
                # Mirror the real-camera telemetry fields so the
                # debug log and result.json still report "frame
                # size" in a sensible way even though no bytes were
                # actually read -- 65536 = 128*128*4, the real
                # camera's packet size.
                if not state["first_frame_seen"]:
                    state["first_frame_seen"] = True
                    state["first_frame_size"] = CAM_HEIGHT * CAM_WIDTH * 4
        ctx.write_component(
            "drone.target_detect",
            np.array(
                [1.0 if detected_this_tick else 0.0, bearing_x, bearing_y, synthetic_strength],
                dtype=np.float64,
            ),
        )

        # Call-count-based, not tick-value-based (see the state dict comment
        # above for why an exact `tick in (...)` match is unreliable) --
        # this fires on the 1st, 251st, 501st, ... post_step call, so it's
        # guaranteed to print regardless of which raw ticks elodin actually
        # schedules a call on.
        state["_debug_calls"] += 1
        if MC_DEBUG and state["_debug_calls"] % 250 == 1:
            print(
                f"[mc_debug] call#{state['_debug_calls']} tick={tick} fault={fault!r} mode={current_mode} "
                f"frame_seen={state['first_frame_seen']} frame_size={state['first_frame_size']} "
                f"ever_detected={state['ever_detected']} dist={dist} "
                f"heat_xy={state['first_heat_xy']} drone_xyz={None if drone_xyz is None else tuple(drone_xyz.round(2))}",
                flush=True,
            )

        done = state["completed"] or tick >= MAX_TICKS - 1
        if done:
            # Printed BEFORE el.monte_carlo.result() -- that call raises
            # RuntimeError outside a real monte-carlo context (no
            # ELODIN_MONTE_CARLO_CONTEXT / run_dir), e.g. under plain
            # `elodin run` while debugging. A print placed after it (as
            # this used to be) never runs in that case, silently hiding
            # exactly the final-state info a debug run needs most.
            if MC_DEBUG:
                print(
                    f"[mc_debug] RESULT fault={fault!r} completed={state['completed']} "
                    f"final_offset_m={state['final_offset_m']} ever_detected={state['ever_detected']} "
                    f"first_frame_seen={state['first_frame_seen']} first_frame_size={state['first_frame_size']} "
                    f"ticks search/approach/hover={state['ticks_in_search']}/{state['ticks_in_approach']}/{state['ticks_in_hover']} "
                    f"heat_xy={state['first_heat_xy']} debug_calls={state['_debug_calls']}",
                    flush=True,
                )
            el.monte_carlo.result(
                fault=fault,
                completed=bool(state["completed"]),
                complete_tick=int(state["complete_tick"]),
                final_tick=int(state["final_tick"]),
                final_offset_m=float(state["final_offset_m"]) if state["final_offset_m"] is not None else -1.0,
                min_altitude_m=float(state["min_altitude_m"]) if np.isfinite(state["min_altitude_m"]) else -1.0,
                max_altitude_m=float(state["max_altitude_m"]) if np.isfinite(state["max_altitude_m"]) else -1.0,
                premature_hover_violation=bool(state["premature_hover_violation"]),
                mode_switch_count=int(state["mode_switch_count"]),
                hover_entries=int(state["hover_entries"]),
                ever_detected=bool(state["ever_detected"]),
                first_frame_seen=bool(state["first_frame_seen"]),
                first_frame_size=int(state["first_frame_size"]),
                heat_x=float(state["first_heat_xy"][0]) if state["first_heat_xy"] else -999.0,
                heat_y=float(state["first_heat_xy"][1]) if state["first_heat_xy"] else -999.0,
                ticks_in_search=int(state["ticks_in_search"]),
                ticks_in_approach=int(state["ticks_in_approach"]),
                ticks_in_hover=int(state["ticks_in_hover"]),
            )
            if state["completed"]:
                os._exit(0)

    return post_step


world.run(
    system,
    simulation_rate=300.0,
    telemetry_rate=100.0,
    # Back to False -- detection is SYNTHETIC here (see _make_post_step
    # above), so we no longer need the wall-clock-paced render server to
    # produce camera frames. A previous attempt kept the real camera and
    # flipped this to True; it did get some frames through, but only ~9
    # of 96 parallel Monte Carlo workers' render servers emitted any
    # frames at all (confirmed from a real 96-run campaign: result.csv's
    # first_frame_size column had mean ~6143 = 9.4% x 65536, with every
    # successful run at the full 128x128x4=65536 byte frame and the rest
    # at -1). The render server's 30fps wall-clock budget can't be split
    # 12 ways under parallel orchestration -- not fixable from Python.
    # Dropping back to False gets us the full campaign speed back:
    # ~60s wall for the whole 96 runs instead of ~540s.
    generate_real_time=False,
    max_ticks=MAX_TICKS,
    interactive=False,
    post_step=_make_post_step(),
)
