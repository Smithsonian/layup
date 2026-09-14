"""``conv_frac`` must reach the fitter from layup's own entry points (#477).

The C++ fit takes a scaled convergence tolerance, but nothing in ``src/layup``
passed it: ``run_from_vector_with_initial_guess`` was called at five sites and
none forwarded the argument, so the parameter shipped with a default a caller
could receive but never change. Same shape as #515, where
``primary_id_column_name`` existed on ``incremental_orbitfit`` and was dropped
before ``_observations_for_update``.

A signature cannot pin this on its own: a parameter that is ACCEPTED but not
FORWARDED is indistinguishable from one that is forwarded and inert, because
both leave the answer unchanged. So two properties are asserted --

* ``conv_frac=0.0`` reproduces the default exactly, on every recorded field, so
  the threading is a no-op at the shipped default;
* every call to the C++ fit inside ``orbitfit.py`` forwards ``conv_frac``, and
  every public entry point on the way down accepts it.

The second is checked statically, against the module source. Two runtime
alternatives were tried first and neither can work here: a spy on
``run_from_vector_with_initial_guess`` is invisible through ``orbitfit``, which
dispatches each object into a multiprocessing pool, and an assertion that a
loose tolerance lowers ``niter`` cannot fire, because the staged pipeline's
final full-data fit is warm-started from the converged primary-interval fit and
already reports ``niter == 0`` on well-behaved objects. The static form has the
property that matters for a regression test: adding a sixth call site without
the argument fails it.
"""

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import layup.orbitfit as orbitfit_mod
from layup.orbitfit import orbitfit
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

FIT_CALL = "run_from_vector_with_initial_guess"
# The entry points a caller can reach, and which therefore have to expose the knob.
PUBLIC_ENTRY_POINTS = ("orbitfit", "sequential_update", "incremental_orbitfit")


def _fit_calls():
    tree = ast.parse(Path(inspect.getsourcefile(orbitfit_mod)).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == FIT_CALL
    ]


def test_every_fit_call_forwards_conv_frac():
    """Adding a call site without the argument must fail here."""
    calls = _fit_calls()
    # Guard against a vacuous pass if the scan or the call name ever changes.
    assert len(calls) >= 5, f"expected at least 5 {FIT_CALL} calls, found {len(calls)}"
    missing = [c.lineno for c in calls if not any(kw.arg == "conv_frac" for kw in c.keywords)]
    assert not missing, f"{FIT_CALL} called without conv_frac at line(s) {missing}"


@pytest.mark.parametrize("name", PUBLIC_ENTRY_POINTS)
def test_entry_point_exposes_conv_frac(name):
    """A caller must be able to set it, and the shipped default must be off."""
    params = inspect.signature(getattr(orbitfit_mod, name)).parameters
    assert "conv_frac" in params, f"{name}() does not expose conv_frac"
    assert params["conv_frac"].default == 0.0, f"{name}() defaults conv_frac to something other than 0.0"


def test_python_default_matches_the_compiled_default():
    """Threading is a no-op only while the Python and C++ defaults agree.

    Forwarding the parameter means every call now passes a value explicitly, so
    a Python default that drifted from the C++ one would silently OVERRIDE it
    for every caller -- the failure mode the threading itself introduces, and
    the one no behavioural test below can see, since both arms would move
    together. This caught a stale build on 2026-09-14: the tree was at the
    revert while the installed extension still carried the 3e-5 default.
    """
    doc = orbitfit_mod.run_from_vector_with_initial_guess.__doc__ or ""
    m = re.search(r"conv_frac[^,)]*?=\s*([0-9eE.+-]+)", doc)
    assert m, f"conv_frac not found in the compiled signature: {doc.splitlines()[:1]}"
    cpp_default = float(m.group(1))
    py_default = inspect.signature(orbitfit_mod.orbitfit).parameters["conv_frac"].default
    assert py_default == cpp_default, (
        f"layup's Python default ({py_default}) and the compiled C++ default ({cpp_default}) "
        "disagree; threading would silently override the C++ value for every caller"
    )


def test_zero_conv_frac_runs_the_whole_pipeline_unchanged():
    """End-to-end plumbing check: an explicit 0.0 fits, and agrees field for field.

    ⚠️ Weaker than it looks, and deliberately named for what it does: once the
    parameter is threaded, the default path ALSO passes 0.0, so this compares
    two identical inputs. It still earns its place -- it is the only test here
    that executes every layer, so a wrong keyword name anywhere in the chain
    raises rather than passing a static scan. The real no-op evidence is the
    cross-build measurement in the PR body, not this test.
    """
    data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    data = np.sort(data, order="obsTime", kind="mergesort")

    default = orbitfit(data, cache_dir=None)
    assert default[0]["flag"] == 0, "baseline fit did not converge; fixture is unusable"
    explicit = orbitfit(data, cache_dir=None, conv_frac=0.0)

    for field in default.dtype.names:
        a, b = default[0][field], explicit[0][field]
        if isinstance(a, (float, np.floating)) and np.isnan(a):
            assert np.isnan(b), f"{field}: {a!r} vs {b!r}"
        else:
            assert a == b, f"{field}: {a!r} vs {b!r}"
