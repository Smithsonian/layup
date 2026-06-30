"""Physical constants for layup, defined once.

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
