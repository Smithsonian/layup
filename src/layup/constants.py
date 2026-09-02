"""Constants for layup, defined once.

Before this module these values were re-declared in several places
(``orbitfit.py``, ``predict.py``, ``iod.py``,
``utilities/data_processing_utilities.py``). The duplicates happened to agree
numerically, but independent definitions are a drift hazard -- e.g.
``SPEED_OF_LIGHT`` was written two different ways (via metres in some modules,
via kilometres in another) that only coincidentally produced the same value.
Importing everything from here gives a single, citable source of truth.

Units follow layup's internal convention: distances in astronomical units (au),
times in days, so GM values are in au^3/day^2 and the speed of light is in
au/day.

The orbit-fit status values at the end are here for the same reason: they are an
output format, read by callers as well as written by the fitter, and they were
previously scattered as bare integers across ``orbitfit.py``.
"""

from __future__ import annotations

# Astronomical unit in metres -- exact, by the IAU 2012 definition (Resolution
# B2): 1 au = 149_597_870_700 m.
AU_M = 149597870700
AU_KM = AU_M / 1000.0

# Speed of light in vacuum -- exact, by the SI definition: 299_792_458 m/s.
C_M_PER_S = 299792458.0
# ...expressed in layup's au/day (~173.144633 au/day).
SPEED_OF_LIGHT = C_M_PER_S * 86400.0 / AU_M

# Heliocentric gravitational parameter GM_sun in au^3/day^2. This is the square
# of the Gaussian gravitational constant k = 0.01720209895, i.e. GM_sun = k^2.
MU_SUN = 0.00029591220828559104

# Total gravitational parameter of the solar system (Sun + planets) in
# au^3/day^2, used as the central GM for barycentric two-body initial orbit
# determination (Gauss's method) where the reference point is the solar-system
# barycentre rather than the Sun.
GMtotal = 0.0002963092748799319


# ---------------------------------------------------------------------------
# Orbit-fit status
#
# ``flag`` is the single-value summary an orbit fit reports. It is 0 if and only
# if the fit converged and passed every check; the values below name the ways it
# can be non-zero. The columns further down report the same information as
# independent facts, which is what a caller should filter on when it needs to
# know *which* check failed.
# ---------------------------------------------------------------------------

FLAG_NOT_ATTEMPTED = -1  # placeholder for a row that was never fit
FLAG_CONVERGED = 0  # converged, and every check passed
FLAG_DID_NOT_CONVERGE = 1  # the differential correction did not converge
FLAG_CSQ_TOO_LARGE = 2  # converged; reduced chi-square above threshold
FLAG_NO_ROOT_CONVERGED = 3  # candidates were produced, none converged on the primary interval
FLAG_BUILDUP_FAILED = 4  # primary interval converged; the incremental build-up did not
FLAG_NO_SOLUTION = 5  # no initial-orbit candidates and no usable fallback seed
FLAG_DEGENERATE_COV = 6  # converged; covariance degenerate, or a variance non-positive
FLAG_PRIOR_NOT_POSITIVE_DEFINITE = 7  # incremental: prior covariance ill-posed, so a full refit
FLAG_INCREMENTAL_NO_FULL_OBS = 8  # incremental update with no full observation set to refit from
FLAG_IMPLAUSIBLE_ORBIT = 9  # converged; hyperbolic excess speed implausibly large

# Hyperbolic excess speed, in km/s, above which a converged fit is reported as
# physically impossible.
#
# A short arc can converge with an excellent reduced chi-square onto a state no
# real object could occupy, because the arc does not constrain the velocity. The
# chi-square check cannot catch this: it is anti-correlated with the failure,
# since the less the object moves across the arc the better the fit.
#
# Excess speed is the one statement available without assuming the object is
# bound. Layup is expected to fit genuine interstellar objects, and those are
# unbound and fast -- 3I/ATLAS arrives at about 59 km/s -- so boundedness itself
# is never grounds for rejection, and only a speed far above any plausible
# arrival speed is evidence of a bad fit rather than an unusual object. The
# default is about 3.4x the fastest interstellar object yet observed; measured
# against long-arc truth orbits it rejected no correct fit, and every fit it
# rejected was wrong.
#
# The value means what it says: raise it to accept faster orbits, and set it low
# to reject anything near-unbound. To switch the check off, set it far above any
# achievable speed.
MAX_EXCESS_SPEED_KM_S = 200.0

# km/s expressed in layup's au/day.
KM_S_IN_AU_PER_DAY = 86400.0 / AU_KM

# How far the fitting pipeline got before it stopped.
STAGE_NOT_ATTEMPTED = 0
STAGE_NO_CANDIDATES = 1  # initial orbit determination produced nothing usable
STAGE_PRIMARY = 2  # reached the fit over the primary interval
STAGE_BUILDUP = 3  # reached the incremental build-up to all observations
STAGE_COMPLETE = 4  # fit the full observation set
STAGE_INCREMENTAL = 5  # sequential-update bookkeeping rather than a fresh fit

# The fit outcome as independent facts, one output column each. Named for their
# polarity: each ``failed_*`` is 1 when the fit failed that check, so a clean fit
# is zero across all of them, matching ``flag == FLAG_CONVERGED``. A ``passed_*``
# convention would make a never-attempted fit (all zero) indistinguishable from
# one that failed everything.
OUTCOME_COLUMNS = ("converged", "stage", "failed_csq", "failed_cov", "failed_physical")

# Which check each of the fitter's own post-convergence verdicts reports as
# failed. Both are set *after* the Levenberg-Marquardt loop converges, so each
# means "converged, then rejected".
CXX_GATE_FLAGS = {FLAG_CSQ_TOO_LARGE: "failed_csq", FLAG_DEGENERATE_COV: "failed_cov"}

# The flags that mean the differential correction reached a solution.
CONVERGED_FLAGS = (FLAG_CONVERGED, FLAG_CSQ_TOO_LARGE, FLAG_DEGENERATE_COV)

# The flags for which no chi-square exists to report: one where no fit was ever
# run, and one where no initial-orbit candidate was found to score. Every other
# flag carries a chi-square -- including the stage markers, since ``do_fit``
# returns the lowest-chi-square candidate it tried.
NO_CSQ_FLAGS = (FLAG_NOT_ATTEMPTED, FLAG_NO_SOLUTION)
