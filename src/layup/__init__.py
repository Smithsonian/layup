import sys as _sys
import types as _types
import warnings as _warnings

# Silence a cosmetic AstropyDeprecationWarning emitted at import time by sbpy
# (sbpy/_astropy_init.py builds a deprecated astropy TestRunner). sbpy is a
# transitive dependency of layup, so the warning is pure noise on the layup CLI
# and in notebooks. The filter is narrow -- it matches only the TestRunner
# message, so unrelated astropy deprecation warnings are unaffected -- and is
# installed here, before any layup submodule imports sbpy. See issue #376.
_warnings.filterwarnings("ignore", message=r".*TestRunner.*")

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


# The verbs are reachable at package level -- layup.orbitfit(), layup.predict()
# -- so that a script calls them by the same names the command line uses.
#
# Each of these names is already a submodule, and layup.orbitfit has long meant
# the MODULE: tests reach do_fit and _select_nongrav_auto through it, and 92
# places import siblings as `from layup.convert import convert`. Rebinding the
# name to the function would break all of that. So the module is made callable
# instead, and answers to both: layup.orbitfit(...) runs the fit, while
# layup.orbitfit.do_fit still resolves.
#
# The lookup lives in __getattribute__ rather than in a PEP 562 __getattr__
# because importing any verb imports its siblings -- layup.orbitfit pulls in
# layup.convert -- and that binds the submodule here under the same name, so a
# __getattr__ hook would simply never be consulted for it.
_VERBS = {
    "orbitfit": "layup.orbitfit",
    "predict": "layup.predict",
    "convert": "layup.convert",
    "comet": "layup.comet",
    "unpack": "layup.unpack",
}

# visualize_notebook does not share its name with a module, so it needs none of this.
_FUNCTIONS = {"visualize_notebook": "layup.visualize"}

__all__ = sorted({**_VERBS, **_FUNCTIONS})


class _CallableVerb(_types.ModuleType):
    """A verb module that can also be called, delegating to its own function."""

    def __call__(self, *args, **kwargs):
        name = self.__name__.rpartition(".")[2]
        return getattr(self, name)(*args, **kwargs)


class _LayupPackage(_types.ModuleType):
    def __getattribute__(self, name):
        if name in _VERBS or name in _FUNCTIONS:
            import importlib

            if name in _FUNCTIONS:
                return getattr(importlib.import_module(_FUNCTIONS[name]), name)
            module = importlib.import_module(_VERBS[name])
            if type(module) is not _CallableVerb:
                module.__class__ = _CallableVerb
            return module
        return super().__getattribute__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(__all__))


_sys.modules[__name__].__class__ = _LayupPackage
