# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Snow-17 Temperature Index Snow Model — Standalone Plugin Package.

A standalone JAX/NumPy dual-backend implementation of the Anderson (1973, 2006)
temperature-index model for snow accumulation and ablation.

Designed for coupling with any rainfall-runoff model (XAJ, SAC-SMA, GR4J, etc.) via:
- Functional API: ``snow17_step()``, ``snow17_simulate()`` for lax.scan coupling
- BMI-like class: ``Snow17BMI`` for imperative coupling

Usage:
    from jsnow17 import Snow17BMI, snow17_simulate

    # BMI coupling
    snow = Snow17BMI(params={'SCF': 1.1}, latitude=51.17)
    snow.initialize()
    rain_plus_melt = snow.update(precip, temp, doy)

    # Functional coupling (JAX-compatible)
    rpm, state = snow17_simulate(precip, temp, doy, params, lat=51.17)

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.
"""

from typing import TYPE_CHECKING

_LAZY_IMPORTS = {
    # Configuration
    'Snow17Config': ('.config', 'Snow17Config'),
    'Snow17ConfigAdapter': ('.config', 'Snow17ConfigAdapter'),

    # BMI interface
    'Snow17BMI': ('.bmi', 'Snow17BMI'),

    # Parameters
    'Snow17State': ('.parameters', 'Snow17State'),
    'Snow17Params': ('.parameters', 'Snow17Params'),
    'SNOW17_PARAM_NAMES': ('.parameters', 'SNOW17_PARAM_NAMES'),
    'SNOW17_PARAM_BOUNDS': ('.parameters', 'SNOW17_PARAM_BOUNDS'),
    'SNOW17_DEFAULTS': ('.parameters', 'SNOW17_DEFAULTS'),
    'DEFAULT_ADC': ('.parameters', 'DEFAULT_ADC'),

    # Core model
    'snow17_step': ('.model', 'snow17_step'),
    'snow17_simulate': ('.model', 'snow17_simulate'),
    'snow17_simulate_jax': ('.model', 'snow17_simulate_jax'),
    'snow17_simulate_numpy': ('.model', 'snow17_simulate_numpy'),
    'seasonal_melt_factor': ('.model', 'seasonal_melt_factor'),
    'HAS_JAX': ('.model', 'HAS_JAX'),
    'create_initial_state': ('.model', 'create_initial_state'),
}


def __getattr__(name: str):
    """Lazy import handler for Snow-17 module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + ['register']


def register() -> None:
    """Register Snow-17 components with symfluence plugin registry."""
    from symfluence.core.registry import model_manifest
    from .config import Snow17ConfigAdapter
    model_manifest("SNOW17", config_adapter=Snow17ConfigAdapter)

    # Contribute Snow-17's calibration bounds to symfluence's catalogue.
    #
    # Snow-17 predates the register_model_bounds seam, so symfluence carried a
    # get_snow17_bounds() entry as a compatibility shim -- a change to these
    # bounds needed a FRAMEWORK release. Registering here makes this package
    # the owner; get_model_bounds('SNOW17') resolves what we register, ahead of
    # the built-in entry.
    #
    # PXADJ IS DELIBERATELY EXCLUDED. SNOW17_PARAM_BOUNDS carries 11 entries;
    # the 10 below are the ones symfluence already serves, value for value, so
    # adopting the seam changes no calibration result. PXADJ was added after
    # v0.2.0 and including it here would silently give every "calibrate all
    # parameters" run an extra knob -- a bounds change that deserves its own
    # release note rather than arriving inside a no-op migration. Add it in a
    # deliberate change when PXADJ is meant to be calibrated, and note that
    # jsacsma composes its own set from this dict, so its parameter-count tests
    # move at the same time.
    from symfluence.core.calibration.parameters import ParameterInfo, register_model_bounds

    from .parameters import SNOW17_PARAM_BOUNDS

    _CATALOGUE_NAMES = (
        'SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF',
        'MBASE', 'TIPM', 'UADJ', 'PLWHC', 'DAYGM',
    )
    register_model_bounds(
        "SNOW17",
        params={
            name: ParameterInfo(
                float(SNOW17_PARAM_BOUNDS[name][0]),
                float(SNOW17_PARAM_BOUNDS[name][1]),
                description=f"Snow-17 {name}",
            )
            for name in _CATALOGUE_NAMES
        },
        names=list(_CATALOGUE_NAMES),
    )


if TYPE_CHECKING:
    from .bmi import Snow17BMI
    from .config import Snow17Config, Snow17ConfigAdapter
    from .model import (
        HAS_JAX,
        create_initial_state,
        seasonal_melt_factor,
        snow17_simulate,
        snow17_simulate_jax,
        snow17_simulate_numpy,
        snow17_step,
    )
    from .parameters import (
        DEFAULT_ADC,
        SNOW17_DEFAULTS,
        SNOW17_PARAM_BOUNDS,
        SNOW17_PARAM_NAMES,
        Snow17Params,
        Snow17State,
    )


__all__ = [
    'Snow17Config', 'Snow17ConfigAdapter',
    'Snow17BMI',
    'Snow17State', 'Snow17Params',
    'SNOW17_PARAM_NAMES', 'SNOW17_PARAM_BOUNDS', 'SNOW17_DEFAULTS', 'DEFAULT_ADC',
    'snow17_step', 'snow17_simulate', 'snow17_simulate_jax', 'snow17_simulate_numpy',
    'seasonal_melt_factor', 'HAS_JAX', 'create_initial_state',
    'register',
]
