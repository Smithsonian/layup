"""Tests for the default cache location (issue #448).

layup writes ~1.6 GB of ephemeris and reference data to a cache directory. The
default lives under the user's home directory, which is the wrong partition on
a cluster. LAYUP_CACHE_DIR moves the default without threading `cache_dir`
through every call.
"""

import pooch
import pytest

from layup.utilities.cache_location import LAYUP_CACHE_ENV_VAR, default_cache_dir


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Each test starts with the variable unset, whatever the developer's shell has."""
    monkeypatch.delenv(LAYUP_CACHE_ENV_VAR, raising=False)


def test_default_is_the_pooch_os_cache_when_unset():
    """With nothing set, the default is exactly what layup used before #448."""
    assert str(default_cache_dir()) == str(pooch.os_cache("layup"))


def test_env_var_overrides_the_default(monkeypatch, tmp_path):
    target = tmp_path / "shared" / "layup-cache"
    monkeypatch.setenv(LAYUP_CACHE_ENV_VAR, str(target))
    assert default_cache_dir() == target


def test_env_var_expands_a_leading_tilde(monkeypatch):
    monkeypatch.setenv(LAYUP_CACHE_ENV_VAR, "~/layup-data")
    result = default_cache_dir()
    assert "~" not in str(result)
    assert result.is_absolute()


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_env_var_falls_back_to_the_default(monkeypatch, blank):
    """An empty or whitespace-only value is a shell accident, not a request to
    write into the current directory."""
    monkeypatch.setenv(LAYUP_CACHE_ENV_VAR, blank)
    assert str(default_cache_dir()) == str(pooch.os_cache("layup"))


def test_the_directory_is_not_created_as_a_side_effect(monkeypatch, tmp_path):
    """Asking where the cache is must not make one. pooch creates it on download."""
    target = tmp_path / "not-yet"
    monkeypatch.setenv(LAYUP_CACHE_ENV_VAR, str(target))
    default_cache_dir()
    assert not target.exists()
