# Contributing to Layup

Thanks for your interest in Layup. Contributions are welcome — bug reports, fixes,
documentation, and new capability alike.

Layup is an orbit-determination package for the Rubin Observatory era, so a good
fraction of contributions involve orbital dynamics as much as software. Both kinds of
expertise are useful, and you do not need both.

## Reporting a bug

Open an issue at https://github.com/Smithsonian/layup/issues. The most useful reports
include:

- what you ran (the command line or the Python call, with the arguments);
- the input data, or enough of it to reproduce — a handful of observations is usually
  plenty, and a designation plus the observatory codes and epochs is often enough;
- what you expected and what you got, including the `flag` value if a fit returned one;
- your platform, Python version, and the versions of `layup`, `assist`, `rebound` and
  `sorcha`.

If a fit is slow or fails to converge rather than crashing, say so explicitly — those
are usually a different class of problem from an exception, and the orbital geometry
(heliocentric distance, arc length, number of observations) is the part we will ask
about first.

## Asking a question

Questions about how to use Layup are welcome as issues. If something was hard to work
out, that is usually a documentation bug and worth reporting as one.

## Contributing code

1. **Open an issue first** for anything beyond a small fix, so the approach can be
   discussed before you spend time on it.
2. **Fork and branch.** Branch from `main`.
3. **Install for development**, following the README:
   ```
   git clone --recursive https://github.com/Smithsonian/layup.git
   cd layup
   pip install -e ".[dev]"
   ```
   Layup builds a C++ extension, so a compiler is required; the `--recursive` clone
   brings in the `assist`, `eigen` and `rebound` submodules.
4. **Run the tests**: `pytest`. Please add tests for what you change.
5. **Run the formatter**: the project uses `black` (pinned in the `dev` extra) via
   `pre-commit`. Install the hooks once with `pre-commit install`.
6. **Open a pull request** against `main`, describing what changed and why.

### A note on numerical changes

Layup is a numerical package, and changes to the fitting, covariance, or observation
models need evidence, not just passing tests — the existing tests will happily pass
against a subtly wrong Jacobian. If you change a partial derivative, a covariance
propagation, or an observation model, please say in the pull request how you checked
it: a finite-difference comparison against the analytic form, a cross-check against
JPL Horizons, or a Monte-Carlo test are all good. Several real bugs in Layup's history
were found exactly this way, and the tests did not catch any of them.

### Review

Pull requests need a review from someone other than the author before merging. Once a
pull request is approved, the submitter merges it.

## Code of Conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Contributions are licensed under the MIT License, the same terms as the rest of the
project. See [LICENSE](LICENSE).
