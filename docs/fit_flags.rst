Orbit fit status
========================================================================================

Every fit that ``layup orbitfit`` produces reports its outcome in seven columns: a
single-value summary, ``flag``, a plain ``accepted`` predicate, and five columns
that report the individual facts behind them.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ``accepted``
     - ``1`` if the fit is usable: it converged and passed every check. The same
       thing ``flag == 0`` says, as a predicate you can filter on without knowing
       the flag values.
   * - ``flag``
     - Summary. ``0`` if and only if the fit converged and passed every check.
   * - ``converged``
     - ``1`` if the differential correction reached a solution. A fit can converge
       and still be rejected, so this is not the same as ``flag == 0``.
   * - ``stage``
     - How far the fitting pipeline got before it stopped. See below.
   * - ``failed_csq``
     - ``1`` if the fit converged but its chi-square per degree of freedom is above
       the acceptance threshold.
   * - ``failed_cov``
     - ``1`` if the fit converged but its covariance is degenerate, or a variance is
       non-positive. This check runs **only when non-gravitational parameters are
       being fitted**, where it detects a non-gravitational term that has become
       collinear with the state. A gravity-only fit is never marked here, so a
       column of zeros does not mean the covariances were examined and found
       sound.
   * - ``failed_physical``
     - ``1`` if the fit converged but describes an orbit no real object could occupy:
       its hyperbolic excess speed is implausibly large.

The three ``failed_*`` columns are named for their polarity: each is ``1`` when the
fit *failed* that check. A clean fit is therefore zero across all of them, which
matches ``flag == 0``.

Which column should I filter on?
----------------------------------------------------------------------------------------

**For orbits you intend to use, filter on** ``accepted == 1``, or equivalently
``flag == 0``. Both mean the fit converged and every check passed, and ``flag == 0``
is what the rest of ``layup`` uses internally.

Read the other columns when you need to know *why* something was rejected — for
triage, for diagnosing a survey's failure modes, or to accept fits that failed a
particular check on purpose.

.. warning::

   **Do not filter on** ``flag == 2`` **or** ``flag == 6`` **to count rejections.**
   They under-report. ``flag`` records a single outcome, and where a fit stopped
   takes precedence over why it was rejected: a candidate that converged and was
   then rejected on chi-square is reported as ``3`` or ``4`` if the pipeline went on
   to run out of candidates or fail its build-up. The ``failed_csq`` and
   ``failed_cov`` columns do not have this problem — they record the fitter's own
   verdict before any of that happens. Count rejections there.

Values of ``flag``
----------------------------------------------------------------------------------------

The values are an enumeration, not a severity ordering.

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - Flag
     - Meaning
   * - ``-1``
     - Not attempted. A placeholder written for rows that were never fit.
   * - ``0``
     - Converged, and passed every check.
   * - ``1``
     - The differential correction did not converge.
   * - ``2``
     - Converged; chi-square per degree of freedom above threshold. See the warning
       above before counting these.
   * - ``3``
     - Initial orbit determination produced candidate orbits, but none of them
       converged on the primary interval. The least-bad attempt, by chi-square, is
       returned so the caller has something to inspect.
   * - ``4``
     - The primary interval converged, but the incremental build-up to the full
       observation set failed partway.
   * - ``5``
     - No solution at all: initial orbit determination produced no candidates and no
       fallback seed was usable.
   * - ``6``
     - Converged; covariance degenerate, or a variance non-positive. See the warning
       above before counting these.
   * - ``7``
     - Incremental update only. The prior covariance was not positive-definite, so
       the information update is ill-posed and the driver falls back to a full refit.
   * - ``8``
     - Incremental update applied without a full observation set supplied for the
       refit. Bookkeeping rather than a fit failure.
   * - ``9``
     - Converged, but the orbit is not physically possible. See the note below.

Values of ``stage``
----------------------------------------------------------------------------------------

How far the pipeline got. Unlike ``flag``, this says nothing about whether the
result was accepted.

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - Stage
     - Meaning
   * - ``0``
     - Not attempted.
   * - ``1``
     - Initial orbit determination produced nothing usable, so nothing was fit.
   * - ``2``
     - Reached the fit over the primary interval.
   * - ``3``
     - Reached the incremental build-up to all observations.
   * - ``4``
     - Fit the full observation set.
   * - ``5``
     - Sequential-update bookkeeping rather than a fresh fit.

.. note::

   **The physical check is a floor, not a guarantee.** The checks behind
   ``failed_csq`` and ``failed_cov`` are statistical: they ask whether the estimator
   behaved, not whether the answer is possible. A short arc can converge with an
   excellent chi-square onto a state no object could occupy, because the arc does
   not constrain the velocity — and chi-square cannot catch it, since the less the
   object moves across the arc the better the fit.

   ``failed_physical`` catches the extreme case, on hyperbolic excess speed. The
   threshold is deliberately generous: layup is expected to fit genuine interstellar
   objects, which are unbound and fast — 3I/ATLAS arrives at about 59 km/s — so being
   unbound is never itself grounds for rejection, and the default sits well above any
   plausible arrival speed. It will therefore accept short-arc orbits that are
   implausible without being impossible. Screen short arcs on your own criteria as
   well. The threshold is ``MAX_EXCESS_SPEED_KM_S`` in ``layup.constants``; lower it
   to reject anything near-unbound, or set it far above any achievable speed to
   switch the check off.

The values are defined once, in ``layup.constants``, as ``FLAG_*``, ``STAGE_*`` and
``OUTCOME_COLUMNS``.
