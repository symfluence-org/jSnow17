# jSnow17

[![PyPI version](https://img.shields.io/pypi/v/jsnow17.svg)](https://pypi.org/project/jsnow17/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Dual-backend (JAX + NumPy) implementation of the Snow-17 temperature-index snow accumulation and ablation model, with functional and BMI interfaces.

Part of the SYMFLUENCE JAX-native model family — self-contained packages that
run standalone (NumPy fallback, no JAX required) and register automatically with
[SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE) when installed alongside it.

## Features

- **Differentiable**: automatic differentiation through the full simulation (JAX)
- **Fast**: JIT compilation via `lax.scan`; `vmap` for ensembles; GPU-capable
- **Dependency-light**: pure-NumPy fallback when JAX is not installed
- **Plugin architecture**: auto-registers with SYMFLUENCE via entry points

## Installation

```bash
pip install jsnow17          # NumPy backend
pip install 'jsnow17[jax]'    # with JAX (differentiable, JIT)
```

## Quickstart

```python
from jsnow17 import Snow17BMI, snow17_simulate

# functional interface (JAX-compatible, differentiable)
rain_plus_melt, state = snow17_simulate(precip, temp, doy, params, lat=51.17)

# BMI interface for model coupling
snow = Snow17BMI(params={"SCF": 1.1}, latitude=51.17)
snow.initialize()
rpm = snow.update(precip_t, temp_t, doy_t)
```

## Use with SYMFLUENCE

Within SYMFLUENCE, jSnow17 is the snow component coupled to rainfall–runoff packages such as jSACSMA and jXAJ; it is installed automatically as their dependency and is not selected as a standalone model.

## Model structure

Snow-17 (Anderson, 1973, 2006) with all branching expressed through smooth
`where` operations for JAX differentiability:

1. Rain/snow partitioning via `PXTEMP` with a 2 °C mixed-phase transition
2. Snowfall gauge-undercatch correction (`SCF`)
3. Hemisphere-aware seasonal melt-factor sinusoid
4. Non-rain and rain-on-snow melt
5. Heat-deficit tracking (TIPM-weighted antecedent temperature)
6. Liquid-water routing with `PLWHC` capacity and refreezing
7. Areal depletion curve
8. Daily ground melt (`DAYGM`)

11 calibration parameters (`jsnow17.parameters.PARAM_BOUNDS`).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## How to cite

If you use jSnow17 in your research, please cite the SYMFLUENCE companion papers,
which describe the design of the JAX-native model family (registry integration,
differentiability, and the calibration experiments they enable):

> Eythorsson, D., et al. (2026). The registry as social contract: Architectural patterns
> for community hydrological modeling. *Water Resources Research* (submitted).
>
> Eythorsson, D., et al. (2026). From configuration to prediction: Multi-model,
> multi-basin experiments with SYMFLUENCE. *Water Resources Research* (submitted).

Citation metadata for this package is provided in [`CITATION.cff`](CITATION.cff);
a version-specific DOI is minted via Zenodo for each GitHub release.
<!-- After the first Zenodo release, add the concept-DOI badge here. -->

## References

- Anderson, E. A. (1973). National Weather Service River Forecast System — Snow Accumulation and Ablation Model. NOAA Technical Memorandum NWS HYDRO-17.
- Anderson, E. A. (2006). Snow Accumulation and Ablation Model — SNOW-17. NOAA Technical Report NWS HYDRO-17.

## License

Apache-2.0. See [LICENSE](LICENSE).
