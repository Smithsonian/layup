---
title: 'Layup: Orbit Fitting at LSST Scale'
tags:
  - Python
  - C++
  - astronomy
  - solar system
authors:
  - name: Matthew J. Holman
    orcid:  0000-0002-1139-4880
    affiliation: "1"
    corresponding: true
  - name: Megan E. Schwamb
    orcid: 0000-0003-4365-1455
    affiliation: "2"
  - name: Kevin J. Napier
    orcid: 0000-0003-4827-5049
    affiliation: "1"
  - name: Pedro H. Bernardinelli
    orcid:  0000-0003-0743-9422
    affiliation: "3,13"
  - name: Ryan R. Lyttle
    orcid:  0009-0007-8602-2954
    affiliation: "2"
  - name: Joseph Murtagh
    orcid: 0000-0001-9505-1131
    affiliation: "2"
  - name: Adam Wilson
    affiliation: "2"
  - name: Hanno Rein
    orcid: 0000-0003-1927-731X
    affiliation: "9,10,11,12"
  - name: Drew Oldag
    orcid:  0000-0001-6984-8411
    affiliation: "3,7"
  - name: Maxine West
    orcid: 0009-0003-3171-3118
    affiliation: "3,7"
  - name: Wilson Beebe
    orcid: 0009-0003-1791-8707
    affiliation: "3"
  - name: Mario Jurić
    orcid:  0000-0003-1996-9252
    affiliation: "3"
  - name: Siegfried Eggl
    orcid:  0000-0002-1398-6302
    affiliation: "4,5,6"
  - name: Rahil Makadia
    orcid: 0000-0001-9265-2230
    affiliation: "4"
  - name: Joachim Moeyens
    orcid: 0000-0001-5820-3925
    affiliation: "8,3"
  - name: Colin Orion Chandler
    orcid: 0000-0001-7335-1715
    affiliation: "3,7"
  - name: Thomas R. Ruch
    orcid: 0000-0003-0403-0891
    affiliation: "14"
  - name: Carrie E. Holt
    orcid: 0000-0002-4043-6445
    affiliation: "15"
affiliations:
 - name: Center for Astrophysics | Harvard & Smithsonian, 60 Garden St., MS 51, Cambridge, MA 02138, USA
   index: 1
 - name: Astrophysics Research Centre, School of Mathematics and Physics, Queen’s University Belfast, Belfast, BT7 1NN, UK
   index: 2
 - name: DiRAC Institute and the Department of Astronomy, University of Washington, 3910 15th Ave NE, Seattle, WA 98195, USA
   index: 3
 - name: Department of Aerospace Engineering, Grainger College of Engineering, University of Illinois at Urbana-Champaign,Urbana, IL 61801, USA
   index: 4
 - name: Department of Astronomy, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA
   index: 5
 - name: National Center for Supercomputing Applications, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA
   index: 6
 - name: LSST Interdisciplinary Network for Collaboration and Computing Frameworks, 933 N. Cherry Avenue, Tucson, AZ 8572, USA
   index: 7
 - name: Asteroid Institute, 20 Sunnyside Ave., Suite 427, Mill Valley, CA 94941, USA
   index: 8
 - name: Department of Physical and Environmental Sciences, University of Toronto at Scarborough, Toronto, Ontario, M1C 1A4, Canada
   index: 9
 - name: Department of Astronomy and Astrophysics, University of Toronto, Toronto, Ontario, M5S 3H4, Canada
   index: 10
 - name: Department of Computer Science, University of Toronto, 40 St. George Street, Toronto, Ontario, M5S 2E4, Canada
   index: 11
 - name: Department of Physics, University of Toronto, Toronto, Ontario, M5S 3H4, Canada
   index: 12
 - name: Departamento de Astronomia, Instituto de Astronomia, Geofísica e Ciências Atmosféricas, Universidade de São Paulo, 05508-090, São Paulo, SP, Brazil
   index: 13
 - name: University of Michigan, Ann Arbor, MI 48109, USA
   index: 14
 - name: Las Cumbres Observatory, 6740 Cortona Drive, Suite 102, Goleta, CA 93117, USA
   index: 15

date: 11 July 2026
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-journal: Astronomical Journal

# this raises the left sidebar to prevent it from overflowing
latex:
  before-metadata: '\vspace{-3.1cm}'
---

# Summary

The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) is under way [@lsstsciencebook2009; @ivezic2019; @bianco2022]. The LSST is expected to raise the number of known solar system objects in the Minor Planet Center's catalogs to roughly 127,000 near-Earth objects (NEOs), 5.1 million main-belt asteroids (MBAs), 1200–2000 Centaurs, and 37,000 trans-Neptunian objects (TNOs) — a four- to nine-fold increase over the currently known populations [@kurlander2025; @murtagh2025]. The solar system community needs efficient tools to maximize the scientific yield of the survey. We present `Layup`, an open-source package for orbit determination at LSST scale that serves as a companion to the `Sorcha` survey simulator [@merritt2025; @holman2025]. `Layup` is built on REBOUND/ASSIST for ephemeris-quality numerical integrations, with a C++ engine and a Python command-line interface and API. It can ingest astrometry in obs80 and ADES formats, as well as radar range and Doppler observations, streak (position + rate) measurements from shift-and-stack surveys, and space-based observations. `Layup` provides two orbit parameterizations — a 6-parameter Cartesian state and a Bernstein-Khushalani basis (distance-scaled parameters in a local tangent-plane reference frame) [@bernstein2000] — and can fit both gravitational and non-gravitational accelerations. Gauss and Bernstein-Khushalani initial orbit determination (IOD) are included, and additional IOD methods can be added easily. Every `Layup` fit reports a full state covariance, which it propagates through element and frame conversions and through ephemeris predictions to support attribution and linking.


# Statement of Need

The LSST [@ivezic2019] is expected to discover ~5 million new small bodies, an order of magnitude more objects than are known today in nearly all of the solar system's small body reservoirs. Fitting orbits for this enormous data set is essential to LSST solar system science. Discovery and orbital classification are the top priorities in the LSST Solar System Science Collaboration's (SSSC's) Roadmap [@schwamb2019], but there is no orbit fitting package that can support the needs of the planetary community in the Rubin era. Three widely used open-source packages exist --- Find_orb [@findorb], OpenOrb [@granvik2009], and OrbFit [@orbfit] --- but each has limitations.  None is written for Python-based, LSST-scale pipelines.  Neither OrbFit nor OpenOrb matched JPL Horizons in detailed comparisons [@giorgini1996; @chernyavskaya2021], and none handles the bound-to-unbound transition, i.e., interstellar objects [@chernyavskaya2021].  A more recent open-source package, GRSS [@makadia2025], provides small-body propagation and orbit determination in Python with a C++ core, but is oriented toward planetary defense --- high-fidelity trajectories and impact monitoring for individual objects --- rather than the LSST-scale survey processing that `Layup` targets.

Some of the most exciting science from Rubin involves the results from shifting and stacking numerous exposures with KBMOD [@whidden2019; @smotherman2021] or heliostack [@napier2026]. The source detections from shift-and-stack routines are the combination of a position (RA/Dec) and corresponding rates. However, no orbit fitting routines use that combination as their primary input. As a result, people resort to synthesizing tracklets from the shift-and-stack sources. This not only necessitates an additional processing step but can introduce correlated astrometric errors.

The Minor Planet Center (MPC) fits orbits using all available observations of the object reported to the MPC. Detailed population studies require orbits fit from solely LSST data provided at data release (DR), and many key software utilities currently being developed from the SSSC's Software Roadmap [@schwamb2019] assume an orbit fit has already been generated using LSST only data. No public orbit fitting code is suitable for fitting DR-only data, and the MPC software is not public.

The `Layup` orbit fitting package fills this need.


# Functionality

The `Layup` orbit fitting package is built on the ASSIST small body integration package [@holman2023], which itself uses REBOUND's framework [@rein2012] and its IAS15 integrator [@rein2015]. ASSIST includes all the terms in the equations of motion, and its results match JPL Horizons predictions to high precision.
Orbit fitting is essentially the process of minimizing the chi-square or log-likelihood function between the set of observed sky-plane positions and those predicted by an ephemeris model. ASSIST provides the partial derivatives of the observables with respect to the orbit parameters, to support the minimization process.

`Layup` can ingest and fit optical astrometry, shift-and-stack observations, radar range and Doppler (two-leg light time) measurements, and observations from space-based platforms.

Fitting begins with initial orbit determination (IOD), which produces a preliminary orbit from a short arc of observations as a starting point for the full fit. `Layup` includes two IOD methods: Gauss's method and a Bernstein-Khushalani linear fit for distant objects, with automatic selection.  Additional IOD methods can be easily incorporated.

Starting from this initial estimate, `Layup` differentially corrects the orbit with a full least-squares fit to all of the observations, using either of two parameterizations: a 6-parameter barycentric, equatorial, Cartesian state, or a Bernstein-Khushalani basis that also supports energy-constrained 5-parameter fits. Both share the same internal numerical-integration and observation-modeling framework, written in C++. In addition, `Layup` supports incremental (sequential) orbit determination, which incorporates new observations into existing orbital solutions, reproducing the accuracy of full fits in an order of magnitude less time.  This is crucial for LSST, which delivers new astrometry nightly for millions of objects.

The fits can include terms for non-gravitational accelerations, via the Marsden A1/A2/A3 model [@marsden1973].  The radial dependence can be configured to span both asteroidal and cometary laws.  The amplitudes can also be fit per apparition.

Every fit returns a full 6×6 covariance, propagated consistently through ephemeris predictions, using variational particles in REBOUND/ASSIST.  We use JAX-based automatic differentiation for the Jacobians of the orbital-element and frame conversions.  This enables rigorous uncertainty ellipses, which in turn support attribution and linking.


<!-- CAPABILITIES SPEC (2026-07-23, verified vs ~/layup-419; stripped by the build).
Meg's review: a few sentences on capabilities beyond orbit fitting.  Facts only.
1. PREDICT (already below): ASSIST integrates once, interpolates to many epochs/sites.
2. CONVERT: Cartesian (CART/BCART/BCART_EQ), cometary (COM/BCOM), Keplerian (KEP/BKEP),
   helio+bary, both ways -- and carries the covariance through each conversion (the real
   point; a format list undersells it).
3. COMET ORIGINAL ORBIT (Hanno's `layup comet`; the "1/a_0" from Meg; not yet in paper):
   integrates each long-period comet to 250 au and reports the osculating orbit there --
   original (backward) or future (forward).  Cols: inv_ao=1/a_0 (--code-format scales to
   1e-6 AU^-1), ao, e_ao, d_ao.  Hook: 1/a_0 ~0 is the bound-to-unbound boundary the
   Statement of Need already sells.
   CAVEATS: d_ao is the ~250 au evaluation distance, NOT an uncertainty (no error bar on
   1/a_0).  Elements are heliocentric (primary=Sun) but labeled ao_barycentric -- do not
   write "barycentric" pending layup#447.
Deferred, unrelated: the 5-parameter/energy-prior BK line -> layup#445.
-->
Ephemeris predictions with `Layup` are highly efficient, using ASSIST's ability to integrate once and interpolate to many epochs and observatory locations. `Layup` also includes utilities for orbital element conversions. Finally, `Layup` includes an extensive command-line interface and a Python API.


# Validation

We cross-validate `Layup`'s results against those from JPL Horizons across a wide range of objects: MBAs, TNOs, interstellar objects, and radar-observed NEOs.  `Layup`'s solutions agree with JPL Horizons to within the fit uncertainties.  We further demonstrate `Layup`'s accuracy and throughput by re-fitting the full MPC catalog of over 1.5 million objects, reproducing the reference orbits of the well-observed numbered objects to about one part in $10^{8}$.


# Acknowledgements

M.J.H. and M.E.S. acknowledge support from the LSST Discovery Alliance (LSST-DA) through LINCC Frameworks Incubator grants 2025-SFF-LFI-10-Holman and 2025-SFF-LFI-11-Schwamb. LINCC Frameworks is supported by Schmidt Sciences, a philanthropic initiative founded by Eric and Wendy Schmidt, as part of the Virtual Institute of Astrophysics (VIA). M.E.S. acknowledges support in part from UK Science and Technology Facilities Council (STFC) grants ST/V000691/1 and ST/X001253/1. M.J. and P.H.B. acknowledge the support from the University of Washington College of Arts and Sciences, Department of Astronomy, and the DiRAC (Data-intensive Research in Astrophysics and Cosmology) Institute. The DiRAC Institute is supported through generous gifts from the Charles and Lisa Simonyi Fund for Arts and Sciences and the Washington Research Foundation. H. R. acknowledges support by the Natural Sciences and Engineering Research Council (NSERC) Discovery Grant RGPIN-2020-04513. M.J. wishes to acknowledge the support of the Washington Research Foundation Data Science Term Chair fund, and the University of Washington Provost's Initiative in Data-Intensive Discovery. J. Murtagh acknowledges support from the Department for the Economy (DfE) Northern Ireland postgraduate studentship scheme. S.E. acknowledges support from the National Science Foundation through the following awards: Collaborative Research: SWIFT-SAT: Minimizing Science Impact on LSST and Observatories Worldwide through Accurate Predictions of Satellite Position and Optical Brightness NSF Award Number: 2332736 and Collaborative Research: Rubin Rocks: Enabling near-Earth asteroid science with LSST NSF Award Number: 2307570. R.R.L. was supported by the UK STFC grant ST/V506990/1. A. Wilson's studentship is funded under STFC grant UKRI177. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

This work was also supported via the Preparing for Astrophysics with LSST Program, funded by the Heising Simons Foundation through grant 2021-2975, and administered by Las Cumbres Observatory.

This work was supported in part by the LSST Discovery Alliance Enabling Science grants program, the B612 Foundation, the University of Washington's DiRAC Institute, the Planetary Society, Karman+, and Adler Planetarium through generous support of the LSST Solar System Readiness Sprints.

This research has made use of NASA’s Astrophysics Data System Bibliographic Services. This research has made use of data and/or services provided by the International Astronomical Union's Minor Planet Center. The SPICE Resource files used in this work are described in [@acton1996; @acton2018]. This work made use of Astropy (http://www.astropy.org), a community-developed core Python package and an ecosystem of tools and resources for astronomy [@astropy2013; @astropy2018; @astropy2022].

This material or work is supported in part by the National Science Foundation through Cooperative Agreement AST-1258333 and Cooperative Support Agreement AST1836783 managed by the Association of Universities for Research in Astronomy (AURA), and the Department of Energy under Contract No. DE-AC02-76SF00515 with the SLAC National Accelerator Laboratory managed by Stanford University.

# AI Usage Disclosure

Portions of the `Layup` software and this manuscript were prepared with the assistance of OpenAI's ChatGPT (GPT-4, via the web interface) and Anthropic's Claude Opus models (including Claude Opus 4.8, via the Claude Code command-line assistant). In the software, AI assistance was used for some code conversion (from Rust to C++), code implementation, refactoring, test scaffolding, debugging, and code review. In this paper, the prose was written by the human authors, and AI assistance was used for copy-editing and proofreading, for drafting document structure, and for reference verification. All AI-assisted outputs were reviewed, edited, and validated by the human authors — including via the test suite, continuous integration, and cross-validation against JPL Horizons — and the human authors made all core design and scientific decisions. The authors take full responsibility for the accuracy, originality, licensing, and integrity of the software and this manuscript.

# References

