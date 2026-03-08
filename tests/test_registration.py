"""Tests for jSnow17 plugin registration."""

import pytest


def test_register_function_exists():
    """jsnow17 module should have a register() function."""
    import jsnow17
    assert hasattr(jsnow17, 'register')
    assert callable(jsnow17.register)


def test_entry_point_discoverable():
    """The snow17 entry point should be discoverable."""
    from importlib.metadata import entry_points
    eps = entry_points(group='symfluence.plugins')
    names = [ep.name for ep in eps]
    assert 'snow17' in names


def test_register_creates_config_adapter():
    """Calling register() should register SNOW17 config adapter."""
    import jsnow17
    jsnow17.register()

    from symfluence.core.registries import R
    assert 'SNOW17' in R.config_adapters
