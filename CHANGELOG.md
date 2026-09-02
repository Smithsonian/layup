# Changelog

All notable changes to Layup are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are set from git tags via `setuptools_scm`; there is no version string in
the source tree to edit.

## [Unreleased]

First public release. Layup determines orbits of solar system objects at the scale
of the Rubin Observatory's Legacy Survey of Space and Time, and reports a full state
covariance with every fit.

### Observation types

- Optical astrometry, read from the MPC 80-column format and from ADES in four
  serializations (CSV, PSV, XML, HDF5).
- Shift-and-stack measurements: sky position together with sky-motion rates, fit as
  data rather than converted into synthetic tracklets.
- Radar range and Doppler, with a two-leg light-time model (the station taken at the
  transmit time on the up leg).
- Observations from space-based platforms, with spacecraft positions resolved through
  JPL Horizons and cached.

All four share the same fitting machinery: they differ in the number and kind of
residual rows they contribute, not in the code path they take.

### Orbit determination

- Initial orbit determination by Gauss's method and by a Bernstein-Khushalani linear
  fit for distant objects, with automatic selection (`iod="auto"`) and a multi-root
  picker that screens Gauss's candidate roots before committing to one.
- Differential correction by Levenberg-Marquardt in either of two six-parameter
  parameterizations: a barycentric equatorial Cartesian state, or a
  Bernstein-Khushalani basis with a bound-orbit energy prior.
- Non-gravitational accelerations via the Marsden A1/A2/A3 model, with a configurable
  radial dependence spanning asteroidal and cometary laws, and amplitudes fittable per
  apparition.
- Incremental (sequential) orbit determination, folding new observations into existing
  solutions at full-fit accuracy for roughly an order of magnitude less cost.

### Covariance

- Every fit returns a full state covariance, propagated consistently through element
  and frame conversions (by JAX automatic differentiation) and through ephemeris
  predictions (by variational particles integrated in REBOUND/ASSIST).

### Ephemerides and prediction

- Ephemeris-quality integrations through ASSIST and REBOUND, using the IAS15
  integrator.
- Predictions that integrate once and interpolate to many epochs and observatory
  locations, propagating any non-gravitational acceleration the fit solved for, so
  a predicted position and its uncertainty ellipse are consistent with the orbit
  that produced them.

### Interfaces

- A command-line interface and a Python API, with parallel execution across objects.
- `layup bootstrap` to fetch the required ephemeris and reference data, and a bundled
  demo dataset (`layup demo`).

### Validation

- Cross-validated against JPL Horizons across main-belt asteroids, trans-Neptunian
  objects, interstellar (hyperbolic) objects, and radar-observed near-Earth objects.
- Demonstrated at scale by re-fitting the full Minor Planet Center catalog.

[Unreleased]: https://github.com/Smithsonian/layup/commits/main
