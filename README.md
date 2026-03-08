# jSnow17

Snow-17 temperature-index snow model extracted from [SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE).

Anderson (1973, 2006) dual-backend (JAX + NumPy) implementation for snow accumulation and ablation.

## Installation

```bash
pip install jsnow17
```

## Usage

```python
from jsnow17 import Snow17BMI, snow17_simulate

# BMI coupling
snow = Snow17BMI(params={'SCF': 1.1}, latitude=51.17)
snow.initialize()
rain_plus_melt = snow.update(precip, temp, doy)

# Functional coupling (JAX-compatible)
rpm, state = snow17_simulate(precip, temp, doy, params, lat=51.17)
```

## License

Apache-2.0
