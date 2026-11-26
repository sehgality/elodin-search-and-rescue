"""Per-run spec-based oracles. post_run(ctx) is called once per completed
Monte Carlo run; it reads that run's result.json (written by
el.monte_carlo.result() in mc_main.py's post_step) and returns a dict of
oracle verdicts.

Every oracle here is a pure spec check against the measured signals in
result.json -- NONE of them look at `fault` to decide pass/fail (the
"blind oracle" half of mutation testing). `fault` is carried through only
so hooks/report.py can group pass rates by fault after the fact.

Exact `ctx` shape (what attributes/methods it exposes) is taken on faith
from the pattern described in elodin-sys/elodin's examples/monte-carlo
hooks (post_run(ctx) reading ctx.run_dir + "/result.json") -- the real
attribute name might differ slightly; if this hook errors on your first
campaign run, print(dir(ctx)) once and adjust the lookup below.
"""

import json
import os

# Tightened from 2.0m to 1.0m after the first clean campaign (campaign5)
# proved analytical synthetic detection reaches 0.22m best-case accuracy
# on control runs, with 92% pass rate at 2.0m. A wide 2.0m band means
# subtle guidance-bias faults (hold_point_freeze snapping onto a 1-1.5m
# off point, camera_fov_mismatch systematically biasing target_estimate
# by ~38% of altitude, tilt_cap_relaxed oscillating through the target)
# all land inside the band and go undetected. 1.0m is still wider than
# the real thermal camera's expected pointing precision but narrow
# enough that subtle guidance-layer divergences push a run over the
# edge, which is what the oracle's job actually is.
ACCURACY_BOUND_M = 1.0
THRASH_LIMIT = 6
MIN_SAFE_ALTITUDE_M = 0.5
MAX_SAFE_ALTITUDE_M = 10.0  # TARGET_ALTITUDE_M (8.0) + margin


def _load_result(ctx) -> dict:
    run_dir = getattr(ctx, "run_dir", None) or getattr(ctx, "output_dir", None)
    path = os.path.join(run_dir, "result.json")
    with open(path) as f:
        return json.load(f)


def score_result(result: dict) -> dict:
    """Pure function (no ctx) so this is unit-testable without a real
    Monte Carlo run -- see test_mc_faults.py."""
    oracle_mission_completes = bool(result.get("completed", False))

    final_offset = result.get("final_offset_m", -1.0)
    oracle_final_accuracy = (
        oracle_mission_completes and 0.0 <= final_offset <= ACCURACY_BOUND_M
    )

    oracle_no_premature_hover = not result.get("premature_hover_violation", False)

    min_alt = result.get("min_altitude_m", -1.0)
    max_alt = result.get("max_altitude_m", -1.0)
    oracle_altitude_envelope = (
        min_alt >= MIN_SAFE_ALTITUDE_M and max_alt <= MAX_SAFE_ALTITUDE_M
    ) if min_alt >= 0 and max_alt >= 0 else False

    oracle_no_thrash = result.get("mode_switch_count", 10**9) <= THRASH_LIMIT

    all_pass = all(
        [
            oracle_mission_completes,
            oracle_final_accuracy,
            oracle_no_premature_hover,
            oracle_altitude_envelope,
            oracle_no_thrash,
        ]
    )

    return {
        "fault": result.get("fault", "unknown"),
        "oracle_mission_completes": oracle_mission_completes,
        "oracle_final_accuracy": oracle_final_accuracy,
        "oracle_no_premature_hover": oracle_no_premature_hover,
        "oracle_altitude_envelope": oracle_altitude_envelope,
        "oracle_no_thrash": oracle_no_thrash,
        "all_pass": all_pass,
        "final_offset_m": final_offset,
        "complete_tick": result.get("complete_tick", -1),
        "mode_switch_count": result.get("mode_switch_count", None),
        # Passed through so results.csv itself answers "did detection ever
        # fire / where did tracking actually get stuck" without having to
        # open every run's individual result.json by hand -- results.csv's
        # columns are exactly whatever this function returns (elodin's own
        # write_results_csv only ever surfaces a hook's own scalar output),
        # so anything from el.monte_carlo.result() that isn't listed here
        # never reaches the CSV at all.
        "completed": result.get("completed", False),
        "final_tick": result.get("final_tick", -1),
        "min_altitude_m": result.get("min_altitude_m", -1.0),
        "max_altitude_m": result.get("max_altitude_m", -1.0),
        "premature_hover_violation": result.get("premature_hover_violation", False),
        "hover_entries": result.get("hover_entries", 0),
        "ever_detected": result.get("ever_detected", False),
        "first_frame_seen": result.get("first_frame_seen", False),
        "first_frame_size": result.get("first_frame_size", -1),
        "heat_x": result.get("heat_x", -999.0),
        "heat_y": result.get("heat_y", -999.0),
        "ticks_in_search": result.get("ticks_in_search", 0),
        "ticks_in_approach": result.get("ticks_in_approach", 0),
        "ticks_in_hover": result.get("ticks_in_hover", 0),
    }


def post_run(ctx):
    result = _load_result(ctx)
    scores = score_result(result)
    return scores
