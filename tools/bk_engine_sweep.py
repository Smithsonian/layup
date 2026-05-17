"""Engine-sweep harness: run both engine='cartesian' and engine='bk_native'
on every case in a diagnostic scan dataset, write a CSV with per-case
metrics, and print a population-level summary table.

Each case is seeded with the truth state at epoch, so this is a "given
a perfect starting point, how does each LM behave on the data" comparison
rather than an IOD-recovery test.  The data themselves carry the diagnostic
scan's pre-baked Gaussian noise (sigma_arcsec is read from each JSON).

Expected input format: a directory containing one .json file per case,
each with at least the following keys::

    {
      "population": "...",
      "arc_length_days": ...,
      "epoch_jd_tdb": ...,
      "sigma_arcsec": ...,
      "truth_state_at_epoch": [x, y, z, vx, vy, vz],
      "observations": [
        {"ra": <deg>, "dec": <deg>, "jd_tdb": ...,
         "observer_state_AU": [x, y, z], ...},
        ...
      ]
    }

The diagnostic-scan dataset shipping with the project lives at
``~/Dropbox/claude_layup/diagnostic/scan/truth/`` (98 cases, 7
populations x 14 arc lengths).

Usage::

    python tools/bk_engine_sweep.py --scan-dir ~/path/to/truth/ \\
        --cache-dir ~/Library/Caches/layup \\
        --output bk_engine_sweep.csv

Defaults:
    --scan-dir   ~/Dropbox/claude_layup/diagnostic/scan/truth
    --cache-dir  ~/Library/Caches/layup
    --output     bk_engine_sweep.csv (in the cwd)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import List

import numpy as np

from layup.orbitfit import _MU_SUN
from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    run_bk_native_fit,
    run_from_vector_with_initial_guess,
)

DEFAULT_SCAN_DIR = "~/Dropbox/claude_layup/diagnostic/scan/truth"
DEFAULT_CACHE_DIR = "~/Library/Caches/layup"
DEFAULT_OUTPUT = "bk_engine_sweep.csv"


# ----------------------------------------------------------------------
# Case loading / observation construction
# ----------------------------------------------------------------------


def load_case(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_observations(case: dict) -> list:
    """Construct layup Observations from a case dict's observation list.

    `observer_state_AU` is treated as position-only (velocity zero); the
    layup fit pipeline only uses observer velocity for second-order
    corrections that don't affect the chain-rule comparison here.
    """
    sigma_arcsec = float(case["sigma_arcsec"])
    sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)
    obs_list = []
    for o in case["observations"]:
        pos = list(o["observer_state_AU"])
        vel = [0.0, 0.0, 0.0]
        obs = Observation.from_astrometry(
            ra=np.deg2rad(o["ra"]),
            dec=np.deg2rad(o["dec"]),
            epoch=float(o["jd_tdb"]),
            observer_position=pos,
            observer_velocity=vel,
        )
        obs.ra_unc = sigma_rad
        obs.dec_unc = sigma_rad
        obs_list.append(obs)
    return obs_list


def truth_seed(case: dict) -> FitResult:
    fit = FitResult()
    fit.state = list(map(float, case["truth_state_at_epoch"]))
    fit.epoch = float(case["epoch_jd_tdb"])
    return fit


# ----------------------------------------------------------------------
# Per-case sweep
# ----------------------------------------------------------------------


def sweep_one(ephem, case_path: Path) -> dict:
    case = load_case(case_path)
    obs = build_observations(case)
    seed = truth_seed(case)
    truth = np.asarray(case["truth_state_at_epoch"])

    cart = run_from_vector_with_initial_guess(ephem, seed, obs)
    bk = run_bk_native_fit(ephem, seed, obs, _MU_SUN)

    r_helio = float(np.linalg.norm(truth[:3]))
    cart_drift = float(np.linalg.norm(np.asarray(cart.state)[:3] - truth[:3]))
    bk_drift = float(np.linalg.norm(np.asarray(bk.state)[:3] - truth[:3]))

    return {
        "case": case_path.stem,
        "population": case["population"],
        "arc_days": float(case["arc_length_days"]),
        "n_obs": len(case["observations"]),
        "r_helio_AU": r_helio,
        "cart_flag": int(cart.flag),
        "cart_niter": int(cart.niter),
        "cart_csq": float(cart.csq),
        "cart_drift_AU": cart_drift,
        "bk_flag": int(bk.flag),
        "bk_niter": int(bk.niter),
        "bk_csq": float(bk.csq),
        "bk_drift_AU": bk_drift,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {path}")


def _fmt_drift(d: float) -> str:
    if d < 1e-6:
        return f"{d:.2e}"
    return f"{d:.4f}"


def print_summary(rows: List[dict]) -> None:
    print()
    print("=" * 96)
    print(
        f"{'Case':<40s}  {'r_helio':>8s}  {'arc':>6s}  "
        f"{'cart_drift':>12s}  {'bk_drift':>12s}  {'cart_it':>7s}  {'bk_it':>5s}"
    )
    print("=" * 96)
    pop_summary: dict = {}
    for row in rows:
        pop = row["population"]
        s = pop_summary.setdefault(
            pop,
            {
                "n": 0,
                "bk_better": 0,
                "cart_better": 0,
                "tie": 0,
                "both_failed": 0,
                "only_cart_failed": 0,
                "only_bk_failed": 0,
                "cart_total_iter": 0,
                "bk_total_iter": 0,
            },
        )
        s["n"] += 1
        cf = row["cart_flag"]
        bf = row["bk_flag"]
        if cf != 0 and bf != 0:
            s["both_failed"] += 1
        elif cf != 0:
            s["only_cart_failed"] += 1
        elif bf != 0:
            s["only_bk_failed"] += 1
        elif row["bk_drift_AU"] < row["cart_drift_AU"]:
            s["bk_better"] += 1
        elif row["cart_drift_AU"] < row["bk_drift_AU"]:
            s["cart_better"] += 1
        else:
            s["tie"] += 1
        if cf == 0:
            s["cart_total_iter"] += row["cart_niter"]
        if bf == 0:
            s["bk_total_iter"] += row["bk_niter"]

        print(
            f"{row['case']:<40s}  {row['r_helio_AU']:>8.2f}  "
            f"{row['arc_days']:>6.2f}  "
            f"{_fmt_drift(row['cart_drift_AU']):>12s}  "
            f"{_fmt_drift(row['bk_drift_AU']):>12s}  "
            f"{row['cart_niter']:>7d}  {row['bk_niter']:>5d}"
        )

    print()
    print("=" * 102)
    print(f"Per-population summary ({len(rows)} cases total)")
    print("=" * 102)
    header = (
        f"{'Population':<20s}  {'n':>3s}  {'BK win':>7s}  {'Cart win':>9s}  "
        f"{'cart fail':>10s}  {'bk fail':>8s}  {'both fail':>10s}  "
        f"{'mean cart it':>13s}  {'mean bk it':>11s}"
    )
    print(header)
    print("-" * len(header))
    total = {
        "n": 0,
        "bk_better": 0,
        "cart_better": 0,
        "tie": 0,
        "only_cart_failed": 0,
        "only_bk_failed": 0,
        "both_failed": 0,
        "cart_total_iter": 0,
        "bk_total_iter": 0,
    }
    for pop in sorted(pop_summary):
        s = pop_summary[pop]
        for k in total:
            total[k] += s[k]
        cart_succ = max(s["n"] - s["only_cart_failed"] - s["both_failed"], 0)
        bk_succ = max(s["n"] - s["only_bk_failed"] - s["both_failed"], 0)
        cart_mean_it = s["cart_total_iter"] / cart_succ if cart_succ else float("nan")
        bk_mean_it = s["bk_total_iter"] / bk_succ if bk_succ else float("nan")
        print(
            f"{pop:<20s}  {s['n']:>3d}  {s['bk_better']:>7d}  "
            f"{s['cart_better']:>9d}  {s['only_cart_failed']:>10d}  "
            f"{s['only_bk_failed']:>8d}  {s['both_failed']:>10d}  "
            f"{cart_mean_it:>13.1f}  {bk_mean_it:>11.1f}"
        )
    print("-" * len(header))
    cart_succ = max(total["n"] - total["only_cart_failed"] - total["both_failed"], 0)
    bk_succ = max(total["n"] - total["only_bk_failed"] - total["both_failed"], 0)
    cart_mean = total["cart_total_iter"] / cart_succ if cart_succ else float("nan")
    bk_mean = total["bk_total_iter"] / bk_succ if bk_succ else float("nan")
    print(
        f"{'TOTAL':<20s}  {total['n']:>3d}  {total['bk_better']:>7d}  "
        f"{total['cart_better']:>9d}  {total['only_cart_failed']:>10d}  "
        f"{total['only_bk_failed']:>8d}  {total['both_failed']:>10d}  "
        f"{cart_mean:>13.1f}  {bk_mean:>11.1f}"
    )

    # Drift / iter ratios across cases where both engines succeed.
    both_ok = [r for r in rows if r["cart_flag"] == 0 and r["bk_flag"] == 0]
    if both_ok:
        ratios = [r["bk_drift_AU"] / r["cart_drift_AU"] for r in both_ok if r["cart_drift_AU"] > 1e-9]
        iter_ratios = [r["bk_niter"] / max(r["cart_niter"], 1) for r in both_ok]
        print()
        print(f"Across {len(both_ok)} cases where both engines succeed:")
        if ratios:
            print(
                f"  drift ratio  (BK / Cart):  median={statistics.median(ratios):.3f}, "
                f"mean={statistics.mean(ratios):.3f}"
            )
        print(
            f"  iter ratio   (BK / Cart):  median={statistics.median(iter_ratios):.3f}, "
            f"mean={statistics.mean(iter_ratios):.3f}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--scan-dir",
        default=DEFAULT_SCAN_DIR,
        help=f"Directory of diagnostic-scan .json cases (default: {DEFAULT_SCAN_DIR})",
    )
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"layup ephemeris cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    scan_dir = Path(args.scan_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    output = Path(args.output).expanduser()

    if not scan_dir.is_dir():
        print(f"ERROR: diagnostic scan not found at {scan_dir}", file=sys.stderr)
        return 1
    if not cache_dir.is_dir():
        print(f"ERROR: ephem cache not found at {cache_dir}", file=sys.stderr)
        return 1

    ephem = get_ephem(str(cache_dir))
    case_paths = sorted(scan_dir.glob("*.json"))
    print(f"Running engine sweep on {len(case_paths)} cases from {scan_dir}...")

    rows = []
    for i, path in enumerate(case_paths, start=1):
        try:
            row = sweep_one(ephem, path)
        except Exception as exc:  # noqa: BLE001 -- want full coverage
            print(
                f"  [{i}/{len(case_paths)}] {path.stem}: raised " f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        rows.append(row)
        if i % 10 == 0:
            print(f"  ...{i}/{len(case_paths)} done")

    write_csv(rows, output)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
