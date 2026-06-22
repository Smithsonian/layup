"""Shared environment guards + data loaders for the BK engine test suites.

Provides two things the BK suites depend on:

1. ``requires_ephem`` -- skip marker active only when the ASSIST
   ephemeris is present in layup's pooch cache. The cache path is
   platform-dependent, so derive it via ``pooch.os_cache("layup")`` the
   same way the library does rather than hardcoding it.

2. ``requires_diagnostic`` plus loaders for the diagnostic-scan truth
   set, shipped in-repo as ``tests/data/bk_scan_truth.json`` (see below).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pooch
import pytest

# ---------------------------------------------------------------------------
# ASSIST ephemeris guard
# ---------------------------------------------------------------------------

EPHEM_CACHE = Path(pooch.os_cache("layup"))
EPHEM_PLANETS = EPHEM_CACHE / "linux_p1550p2650.440"
EPHEM_SMALLBODIES = EPHEM_CACHE / "sb441-n16.bsp"
EPHEM_AVAILABLE = EPHEM_PLANETS.exists() and EPHEM_SMALLBODIES.exists()

requires_ephem = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=(
        f"ASSIST ephemeris missing at {EPHEM_CACHE} "
        f"(expected {EPHEM_PLANETS.name} + {EPHEM_SMALLBODIES.name}); "
        "run `layup bootstrap` to download it."
    ),
)

# ---------------------------------------------------------------------------
# Diagnostic-scan truth set (shipped in-repo)
# ---------------------------------------------------------------------------

# 98 ASSIST-integrated cases in one JSON file keyed by case stem. Each case
# carries only the fields the BK suites read: sigma_arcsec, epoch_jd_tdb,
# truth_state_at_epoch, and per-observation {ra, dec, jd_tdb, observer_state_AU}.
DIAGNOSTIC_SCAN = Path(__file__).resolve().parents[1] / "data" / "bk_scan_truth.json"
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_file()


@functools.lru_cache(maxsize=1)
def _diagnostic_cases() -> dict:
    """Load and cache the consolidated diagnostic-scan truth set."""
    with open(DIAGNOSTIC_SCAN) as f:
        return json.load(f)


requires_diagnostic = pytest.mark.skipif(
    not DIAGNOSTIC_AVAILABLE,
    reason=f"Diagnostic-scan truth set missing at {DIAGNOSTIC_SCAN}.",
)


def load_diagnostic_case(name: str) -> dict:
    """Load a diagnostic-scan case by stem (e.g. 'classical_42AU_arc_007.00d')."""
    return _diagnostic_cases()[name]


def diagnostic_case_names() -> list[str]:
    """Sorted stems of every diagnostic-scan case shipped in-repo."""
    return sorted(_diagnostic_cases())
