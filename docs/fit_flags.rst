Orbit fit status flags
========================================================================================

Every fit that ``layup orbitfit`` produces carries an integer ``flag`` column
reporting how the fit ended. The values are listed below, together with the part
of the code that sets each one.

``flag = 0`` means the differential correction converged and the result passed
the quality gates that are currently applied. It does not assert that the orbit
is physically plausible; see :ref:`flag-groups` below.

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - Flag
     - Meaning
   * - ``-1``
     - Not attempted. A placeholder written for rows that were never fit.
   * - ``0``
     - Converged and accepted.
   * - ``1``
     - Did not converge. This is the initial value, left in place when nothing
       better is reached.
   * - ``2``
     - Converged, but the chi-square per degree of freedom is above threshold.
       A quality gate applied to a fit that did converge.
   * - ``3``
     - Initial orbit determination (IOD) produced candidates, but none of them
       converged on the primary interval. The least-bad attempt, by chi-square,
       is returned so that the caller has something to inspect.
   * - ``4``
     - The primary interval converged, but the incremental build-up to the full
       set of observations failed partway. The flag is set at the segment where
       it broke.
   * - ``5``
     - No solution at all. Either the IOD returned no candidates and
       ``iod='auto'`` was not in use, so there was no fallback, or ``'auto'``
       produced no Gauss roots and no usable Bernstein-Khushalani seed.
   * - ``6``
     - Converged, but weakly constrained: the covariance is degenerate, or a
       variance is non-positive.
   * - ``7``
     - Incremental update only. The prior covariance was not positive-definite,
       so the information update is ill-posed and the driver falls back to a
       full refit over all observations.
   * - ``8``
     - Incremental update applied without a full set of observations supplied
       for the refit. Bookkeeping rather than a fit failure.

.. _flag-groups:

What the groups mean
----------------------------------------------------------------------------------------

The values are an enumeration, not a severity ordering. They fall into four
groups:

Converged and accepted
   ``0``.

Converged, but rejected by a gate
   ``2`` and ``6``. The differential correction reached a solution, which was
   then rejected — on chi-square per degree of freedom, or on a degenerate
   covariance.

Never converged
   ``1``, ``3``, ``4``, ``5`` and ``7``. These distinguish where along the
   initial orbit determination and incremental-fit path the attempt stopped.

Bookkeeping
   ``-1`` and ``8``. Neither reports on the quality of a fit.

.. note::

   The gates behind ``flag = 2`` and ``flag = 6`` are statistical, not physical.
   A fit can converge, sit comfortably inside both gates, and still describe an
   orbit that is not physically plausible — for example one implying an
   absolute magnitude far fainter than any object the survey could have
   detected. Short arcs are the usual case. Treat ``flag = 0`` as "the
   estimator succeeded", not as "the orbit is real".
