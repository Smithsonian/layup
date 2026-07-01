"""Cross-verb CLI consistency guards.

These lock down conventions that the ``layup`` verbs must share so the
load -> fit -> convert/predict/visualize workflow chains without surprises:

* every orbit-consuming verb defaults ``-pid`` to ``provID`` (what orbitfit and
  predict write), so a fresh fit output flows on with no extra flag;
* every ephemeris-using verb accepts the ``--ar`` cache-directory flag.

They parse each verb's argument parser with ``execute`` stubbed out, so nothing
is actually run (no ephemeris, no network).
"""

import importlib
import sys

import pytest

# verb -> the minimal positional args its parser needs to succeed
_POSITIONALS = {
    "convert": ["in.csv", "KEP"],
    "comet": ["in.csv"],
    "predict": ["in.csv"],
    "unpack": ["in.csv"],
    "orbitfit": ["in.csv", "ADES_csv"],
    "visualize": ["in.csv"],
    "bootstrap": [],
}


def _parse(verb, extra, monkeypatch):
    """Parse a verb's CLI with ``execute`` stubbed; return the parsed args."""
    mod = importlib.import_module(f"layup_cmdline.{verb}")
    captured = {}
    monkeypatch.setattr(mod, "execute", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", [f"layup-{verb}"] + _POSITIONALS[verb] + extra)
    mod.main()
    return captured["args"]


@pytest.mark.parametrize("verb", ["convert", "comet", "predict", "unpack", "orbitfit", "visualize"])
def test_pid_default_is_provid(verb, monkeypatch):
    """All orbit-consuming verbs default the primary-id column to provID."""
    args = _parse(verb, [], monkeypatch)
    assert (
        args.primary_id_column_name == "provID"
    ), f"{verb} defaults -pid to {args.primary_id_column_name!r}, breaking chain consistency"


@pytest.mark.parametrize("verb", ["convert", "comet", "predict", "orbitfit", "visualize"])
def test_ar_flag_accepted(verb, monkeypatch):
    """Every ephemeris-using verb accepts the unified --ar cache-dir flag."""
    args = _parse(verb, ["--ar", "/tmp/layup_ar"], monkeypatch)
    assert args.ar_data_file_path == "/tmp/layup_ar"


def test_bootstrap_accepts_ar_flag(monkeypatch):
    """bootstrap accepts --ar as an alias for its download directory."""
    args = _parse("bootstrap", ["--ar", "/tmp/layup_ar"], monkeypatch)
    assert args.cache == "/tmp/layup_ar"


@pytest.mark.parametrize("verb", ["convert", "comet", "predict", "orbitfit", "bootstrap"])
def test_config_flag_spellings(verb, monkeypatch):
    """Both --conf and --config resolve everywhere (unified config flag)."""
    for spelling in ("--conf", "--config"):
        args = _parse(verb, [spelling, "cfg.ini"], monkeypatch)
        assert args.config == "cfg.ini", f"{verb} rejected {spelling}"
