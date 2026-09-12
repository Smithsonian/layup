import pytest

import layup

VERBS = ("orbitfit", "predict", "convert", "comet", "unpack")
FUNCTIONS = ("visualize_notebook",)


def test_verbs_are_callable_from_the_package() -> None:
    """layup.orbitfit(), layup.predict() and the rest, as the command line names them."""
    for name in VERBS + FUNCTIONS:
        assert callable(getattr(layup, name)), name


def test_verbs_keep_their_module_contents() -> None:
    """Calling the verb runs it; the module's own names still resolve through it."""
    assert callable(layup.orbitfit.do_fit)
    assert callable(layup.convert.convert)
    from layup import orbitfit

    assert callable(orbitfit.create_empty_result)


def test_verbs_are_advertised() -> None:
    for name in VERBS + FUNCTIONS:
        assert name in layup.__all__
        assert name in dir(layup)


def test_importing_the_package_stays_cheap() -> None:
    """Nothing is imported until a verb is asked for, so `import layup` pulls in
    no JAX, no ASSIST and no compiled extension."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", "import layup, sys; print('layup.orbitfit' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "False"


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        layup.not_a_verb
