"""post_campaign(ctx): reads the campaign's results.csv (each row = one
run, with the columns hooks/score.py's post_run() returned merged in,
per elodin's own examples/monte-carlo/hooks/report.py pattern) and writes
a text report: per-fault oracle pass rates, a false-alarm check on the
"none" control group, and a must_catch/escaped verdict for each real
fault against the ONE oracle it's expected to trip (EXPECTED_ORACLE
below -- a documented hypothesis about which oracle should notice each
fault, not something auto-derived; a fault tripping a DIFFERENT oracle
than expected is still informative and is reported as "caught (by a
different oracle)" rather than silently counted as escaped).

Exact attribute ctx exposes (out_dir vs output_dir, etc.) is taken on
faith from the one real example read -- adjust _results_csv_path() if the
first real run errors here.
"""

import csv
import os

ORACLE_COLUMNS = [
    "oracle_mission_completes",
    "oracle_final_accuracy",
    "oracle_no_premature_hover",
    "oracle_altitude_envelope",
    "oracle_no_thrash",
]

# Documented hypothesis, not derived from the data: which oracle each
# real fault is expected to trip. See mc_sim.py's FAULT_CATALOG for what
# each fault actually does.
EXPECTED_ORACLE = {
    "hover_entry_no_speed_check": "oracle_no_premature_hover",
    "tick_timing_unscaled": "oracle_no_premature_hover",
    "hold_point_freeze": "oracle_final_accuracy",
    "tilt_cap_relaxed": "oracle_final_accuracy",
    "thermal_threshold_drift": "oracle_mission_completes",
    "camera_fov_mismatch": "oracle_final_accuracy",
    "imu_noise_spike": "oracle_no_thrash",
}


def _results_csv_path(ctx) -> str:
    out_dir = getattr(ctx, "out_dir", None) or getattr(ctx, "output_dir", None)
    return os.path.join(out_dir, "results.csv")


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def build_report(rows: list[dict]) -> str:
    """Pure function over parsed CSV rows -- unit-testable without a real
    campaign run. See test_mc_faults.py."""
    by_fault: dict[str, list[dict]] = {}
    for row in rows:
        by_fault.setdefault(row.get("fault", "unknown"), []).append(row)

    lines = []
    lines.append("=== Monte Carlo fault campaign report ===")
    lines.append(f"{len(rows)} total runs across {len(by_fault)} fault conditions\n")

    # --- false-alarm check on the control group ---
    control = by_fault.get("none", [])
    lines.append(f"-- control group ('none', {len(control)} runs) --")
    control_pass_rate = {}
    for oracle in ORACLE_COLUMNS:
        n_pass = sum(1 for r in control if _to_bool(r.get(oracle)))
        rate = n_pass / len(control) if control else float("nan")
        control_pass_rate[oracle] = rate
        flag = "" if rate >= 0.9 or not control else "  <-- false-alarm-prone oracle or pre-existing bug"
        lines.append(f"  {oracle}: {n_pass}/{len(control)} pass ({rate:.0%}){flag}")
    lines.append("")

    # --- per-fault oracle pass rates + must_catch verdict ---
    for fault, fault_rows in sorted(by_fault.items()):
        if fault == "none":
            continue
        lines.append(f"-- {fault} ({len(fault_rows)} runs) --")
        pass_rates = {}
        for oracle in ORACLE_COLUMNS:
            n_pass = sum(1 for r in fault_rows if _to_bool(r.get(oracle)))
            rate = n_pass / len(fault_rows) if fault_rows else float("nan")
            pass_rates[oracle] = rate
            lines.append(f"  {oracle}: {n_pass}/{len(fault_rows)} pass ({rate:.0%})")

        expected = EXPECTED_ORACLE.get(fault)
        if expected:
            expected_rate = pass_rates[expected]
            control_rate = control_pass_rate.get(expected, float("nan"))
            dropped = control and (control_rate - expected_rate) >= 0.3
            # also check whether ANY other oracle caught it, in case the
            # fault tripped something other than the hypothesis
            other_caught = [
                o for o in ORACLE_COLUMNS
                if o != expected and pass_rates[o] < 0.7 and control_pass_rate.get(o, 1.0) >= 0.9
            ]
            if dropped:
                verdict = f"CAUGHT as expected by {expected} ({control_rate:.0%} -> {expected_rate:.0%})"
            elif other_caught:
                verdict = f"CAUGHT, but by {other_caught} instead of the expected {expected}"
            else:
                verdict = f"ESCAPED -- no oracle's pass rate dropped meaningfully vs. control"
            lines.append(f"  verdict: {verdict}")
        lines.append("")

    return "\n".join(lines)


def post_campaign(ctx):
    path = _results_csv_path(ctx)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    report = build_report(rows)
    out_dir = getattr(ctx, "out_dir", None) or getattr(ctx, "output_dir", None)
    report_path = os.path.join(out_dir, "fault_campaign_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    return {"report_path": report_path}
