"""Shared environment guards + data loaders for the BK engine test suites.

Centralizes two things that the BK tests (Layers 2 and 3, plus the
BK-IOD and iod='auto' suites that stack on top) used to each hardcode:

1. **The ASSIST ephemeris location.**  Layup resolves its cache with
   ``pooch.os_cache("layup")`` -- ``~/Library/Caches/layup`` on macOS,
   ``~/.cache/layup`` on Linux.  The tests previously hardcoded the
   macOS path, so on the Linux CI legs ``EPHEM_AVAILABLE`` was always
   False and every BK suite skipped even though ``layup bootstrap`` had
   downloaded the files.  Deriving the path from ``pooch`` the same way
   the library does keeps the guard correct on every platform.

2. **The diagnostic-scan truth set.**  These ASSIST-integrated truth
   cases used to live at a personal absolute path
   (``~/Dropbox/claude_layup/diagnostic/scan/truth``) that existed on no
   CI runner and no other contributor's machine, so the Layer-3 suites
   skipped everywhere.  The set is small (98 cases, ~756 KB) and now
   ships in-repo under ``tests/data/bk_scan_truth/``.
"""

from __future__ import annotations

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

DIAGNOSTIC_SCAN = Path(__file__).resolve().parents[1] / "data" / "bk_scan_truth"
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_dir() and any(DIAGNOSTIC_SCAN.glob("*.json"))

requires_diagnostic = pytest.mark.skipif(
    not DIAGNOSTIC_AVAILABLE,
    reason=f"Diagnostic-scan truth set missing at {DIAGNOSTIC_SCAN}.",
)


def load_diagnostic_case(name: str) -> dict:
    """Load a diagnostic-scan case by stem (e.g. 'classical_42AU_arc_007.00d')."""
    with open(DIAGNOSTIC_SCAN / f"{name}.json") as f:
        return json.load(f)


def diagnostic_case_names() -> list[str]:
    """Sorted stems of every diagnostic-scan case shipped in-repo."""
    return sorted(p.stem for p in DIAGNOSTIC_SCAN.glob("*.json"))
