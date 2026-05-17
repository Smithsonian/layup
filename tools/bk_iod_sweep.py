"""IOD-sweep harness: compare BK-IOD vs Gauss-IOD on a diagnostic
scan, both as raw IOD output and as a seed for the BK LM fit.

For each case we record:
  * IOD output drift from truth (Gauss picks its first viable root;
    BK-IOD's single output)
  * Final LM-converged drift from truth and iteration count, starting
    from the IOD output (Gauss uses the same "try each root in turn"
    fallback as orbitfit.do_fit)
  * Success/failure flags

The point is to see whether BK-IOD is at least as good a seed as
Gauss for the BK LM, and whether one or the other expands the regime
where the full fit converges to truth.

Usage::

    python tools/bk_iod_sweep.py --scan-dir <dir> --output <csv>

Defaults:
    --scan-dir   ~/Dropbox/claude_layup/diagnostic/scan/truth
    --cache-dir  ~/Library/Caches/layup
    --output     bk_iod_sweep.csv  (in the cwd)
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
    gauss,
    get_ephem,
    run_bk_iod,
    run_bk_native_fit,
)

DEFAULT_SCAN_DIR = "~/Dropbox/claude_layup/diagnostic/scan/truth"
DEFAULT_CACHE_DIR = "~/Library/Caches/layup"
DEFAULT_OUTPUT = "bk_iod_sweep.csv"

# Constants for the Gauss call -- match orbitfit.do_gauss_iod's call site.
_GMTOTAL = 0.0002963092748799319
_AU_M = 149597870700
_SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / _AU_M


# ----------------------------------------------------------------------
# Case loading and observation construction
# ----------------------------------------------------------------------


def load_case(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_observations(case: dict) -> list:
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


# ----------------------------------------------------------------------
# IOD wrappers
# ----------------------------------------------------------------------


def gauss_iod(observations: list) -> list:
    """Run Gauss IOD using the first, middle, last observations.
    Returns a list of FitResult candidates (Gauss roots); empty if no
    viable solutions."""
    if len(observations) < 3:
        return []
    idx0 = 0
    idx1 = len(observations) // 2
    idx2 = len(observations) - 1
    return gauss(
        _GMTOTAL,
        observations[idx0],
        observations[idx1],
        observations[idx2],
        0.0001,
        _SPEED_OF_LIGHT,
    )


def bk_iod(observations: list, epoch: float) -> FitResult:
    return run_bk_iod(observations, epoch, _MU_SUN)


# ----------------------------------------------------------------------
# Per-case sweep
# ----------------------------------------------------------------------


def _drift_AU(state, truth):
    return float(np.linalg.norm(np.asarray(state)[:3] - np.asarray(truth)[:3]))


def _try_lm(ephem, seed: FitResult, observations: list, mu: float) -> FitResult:
    """Run BK LM from a seed.  Returns the FitResult."""
    return run_bk_native_fit(ephem, seed, observations, mu)


def sweep_one(ephem, case_path: Path) -> dict:
    case = load_case(case_path)
    obs = build_observations(case)
    truth = np.asarray(case["truth_state_at_epoch"])
    epoch = float(case["epoch_jd_tdb"])
    r_helio = float(np.linalg.norm(truth[:3]))

    # --- Gauss IOD: take its candidate roots ---
    gauss_solns = gauss_iod(obs)
    # Default values when Gauss returns nothing usable.
    g_iod_drift = float("nan")
    g_lm_drift = float("nan")
    g_lm_niter = -1
    g_lm_flag = -1
    g_n_roots = len(gauss_solns)

    # Use only the FIRST Gauss root.  orbitfit.do_fit also starts with
    # solns[0]; the fall-back to roots 1, 2 only fires when the first
    # root's LM fails.  Running LM on all 3 roots for every case made
    # the sweep effectively unusable (pathological seeds from spurious
    # Gauss roots burn iter_max=100 with no convergence).  This way we
    # measure the typical do_fit experience.  Note: Gauss returns flag
    # in an uninitialized state, so we ignore soln.flag and only check
    # the LM's flag afterwards.
    if gauss_solns:
        soln = gauss_solns[0]
        g_iod_drift = _drift_AU(soln.state, truth)
        lm_res = _try_lm(ephem, soln, obs, _MU_SUN)
        if lm_res.flag == 0:
            g_lm_drift = _drift_AU(lm_res.state, truth)
        g_lm_niter = lm_res.niter
        g_lm_flag = lm_res.flag

    # --- BK IOD ---
    bk_iod_result = bk_iod(obs, epoch)
    b_iod_drift = float("nan")
    b_lm_drift = float("nan")
    b_lm_niter = -1
    b_lm_flag = -1
    if bk_iod_result.flag == 0:
        b_iod_drift = _drift_AU(bk_iod_result.state, truth)
        bk_lm_res = _try_lm(ephem, bk_iod_result, obs, _MU_SUN)
        b_lm_niter = bk_lm_res.niter
        b_lm_flag = bk_lm_res.flag
        if bk_lm_res.flag == 0:
            b_lm_drift = _drift_AU(bk_lm_res.state, truth)

    return {
        "case": case_path.stem,
        "population": case["population"],
        "arc_days": float(case["arc_length_days"]),
        "n_obs": len(obs),
        "r_helio_AU": r_helio,
        # Gauss
        "gauss_n_roots": g_n_roots,
        "gauss_iod_drift_AU": g_iod_drift,
        "gauss_lm_drift_AU": g_lm_drift,
        "gauss_lm_niter": g_lm_niter,
        "gauss_lm_flag": g_lm_flag,
        # BK
        "bk_iod_flag": int(bk_iod_result.flag),
        "bk_iod_drift_AU": b_iod_drift,
        "bk_lm_drift_AU": b_lm_drift,
        "bk_lm_niter": b_lm_niter,
        "bk_lm_flag": b_lm_flag,
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


def _fmt_drift(d):
    if d is None or (isinstance(d, float) and (np.isnan(d) or not np.isfinite(d))):
        return "--"
    if d < 1e-6:
        return f"{d:.2e}"
    return f"{d:.4f}"


def _converged_drift(row, key):
    """Returns the LM-converged drift, or NaN if LM didn't converge."""
    flag = row[f"{key}_lm_flag"]
    if flag != 0:
        return float("nan")
    return row[f"{key}_lm_drift_AU"]


def print_summary(rows: List[dict]) -> None:
    print()
    print("=" * 110)
    print(
        f"{'Case':<40s}  {'r/AU':>6s}  {'arc':>5s}  "
        f"{'g_iod':>10s}  {'g_lm':>10s}  {'g_it':>4s}  "
        f"{'b_iod':>10s}  {'b_lm':>10s}  {'b_it':>4s}"
    )
    print("=" * 110)
    pop_summary: dict = {}
    for row in rows:
        pop = row["population"]
        s = pop_summary.setdefault(
            pop,
            {
                "n": 0,
                "gauss_lm_ok": 0,
                "bk_lm_ok": 0,
                "both_ok": 0,
                "bk_better": 0,
                "gauss_better": 0,
                "bk_only": 0,
                "gauss_only": 0,
                "neither": 0,
                "gauss_total_iter": 0,
                "bk_total_iter": 0,
            },
        )
        s["n"] += 1
        g_ok = row["gauss_lm_flag"] == 0
        b_ok = row["bk_lm_flag"] == 0
        if g_ok:
            s["gauss_lm_ok"] += 1
            s["gauss_total_iter"] += row["gauss_lm_niter"]
        if b_ok:
            s["bk_lm_ok"] += 1
            s["bk_total_iter"] += row["bk_lm_niter"]
        if g_ok and b_ok:
            s["both_ok"] += 1
            gd = row["gauss_lm_drift_AU"]
            bd = row["bk_lm_drift_AU"]
            if bd < gd:
                s["bk_better"] += 1
            elif gd < bd:
                s["gauss_better"] += 1
        elif b_ok and not g_ok:
            s["bk_only"] += 1
        elif g_ok and not b_ok:
            s["gauss_only"] += 1
        else:
            s["neither"] += 1

        print(
            f"{row['case']:<40s}  {row['r_helio_AU']:>6.1f}  "
            f"{row['arc_days']:>5.2f}  "
            f"{_fmt_drift(row['gauss_iod_drift_AU']):>10s}  "
            f"{_fmt_drift(row['gauss_lm_drift_AU']):>10s}  "
            f"{row['gauss_lm_niter']:>4d}  "
            f"{_fmt_drift(row['bk_iod_drift_AU']):>10s}  "
            f"{_fmt_drift(row['bk_lm_drift_AU']):>10s}  "
            f"{row['bk_lm_niter']:>4d}"
        )

    print()
    print("=" * 115)
    print(f"Per-population summary ({len(rows)} cases total)")
    print("=" * 115)
    header = (
        f"{'Population':<20s}  {'n':>3s}  {'G->LM ok':>8s}  {'B->LM ok':>8s}  "
        f"{'BK win':>7s}  {'G win':>5s}  {'BK only':>7s}  {'G only':>6s}  {'neither':>7s}  "
        f"{'mean g_it':>10s}  {'mean b_it':>10s}"
    )
    print(header)
    print("-" * len(header))
    total = {
        k: 0
        for k in (
            "n",
            "gauss_lm_ok",
            "bk_lm_ok",
            "bk_better",
            "gauss_better",
            "bk_only",
            "gauss_only",
            "neither",
            "gauss_total_iter",
            "bk_total_iter",
        )
    }
    for pop in sorted(pop_summary):
        s = pop_summary[pop]
        for k in total:
            total[k] += s[k]
        g_mean_it = s["gauss_total_iter"] / s["gauss_lm_ok"] if s["gauss_lm_ok"] else float("nan")
        b_mean_it = s["bk_total_iter"] / s["bk_lm_ok"] if s["bk_lm_ok"] else float("nan")
        print(
            f"{pop:<20s}  {s['n']:>3d}  {s['gauss_lm_ok']:>8d}  {s['bk_lm_ok']:>8d}  "
            f"{s['bk_better']:>7d}  {s['gauss_better']:>5d}  {s['bk_only']:>7d}  "
            f"{s['gauss_only']:>6d}  {s['neither']:>7d}  "
            f"{g_mean_it:>10.1f}  {b_mean_it:>10.1f}"
        )
    print("-" * len(header))
    g_mean = total["gauss_total_iter"] / total["gauss_lm_ok"] if total["gauss_lm_ok"] else float("nan")
    b_mean = total["bk_total_iter"] / total["bk_lm_ok"] if total["bk_lm_ok"] else float("nan")
    print(
        f"{'TOTAL':<20s}  {total['n']:>3d}  {total['gauss_lm_ok']:>8d}  {total['bk_lm_ok']:>8d}  "
        f"{total['bk_better']:>7d}  {total['gauss_better']:>5d}  {total['bk_only']:>7d}  "
        f"{total['gauss_only']:>6d}  {total['neither']:>7d}  "
        f"{g_mean:>10.1f}  {b_mean:>10.1f}"
    )

    # Drift-ratio + iter-ratio across cases where both LMs succeed.
    both_ok = [r for r in rows if r["gauss_lm_flag"] == 0 and r["bk_lm_flag"] == 0]
    if both_ok:
        ratios = []
        for r in both_ok:
            gd = r["gauss_lm_drift_AU"]
            bd = r["bk_lm_drift_AU"]
            if gd > 1e-9:
                ratios.append(bd / gd)
        iter_ratios = [r["bk_lm_niter"] / max(r["gauss_lm_niter"], 1) for r in both_ok]
        print()
        print(f"Across {len(both_ok)} cases where BOTH IOD->LM pipelines converge:")
        if ratios:
            print(
                f"  drift ratio (BK_LM / Gauss_LM):  "
                f"median={statistics.median(ratios):.3f}, mean={statistics.mean(ratios):.3f}"
            )
        print(
            f"  iter ratio  (BK_LM / Gauss_LM):  "
            f"median={statistics.median(iter_ratios):.3f}, mean={statistics.mean(iter_ratios):.3f}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scan-dir", default=DEFAULT_SCAN_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
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
    print(f"Running IOD sweep on {len(case_paths)} cases from {scan_dir}...")

    rows = []
    for i, path in enumerate(case_paths, start=1):
        try:
            row = sweep_one(ephem, path)
        except Exception as exc:  # noqa: BLE001
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
