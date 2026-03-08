"""Tests for standalone Snow-17 model."""

import numpy as np
import pytest

from jsnow17.model import (
    HAS_JAX,
    create_initial_state,
    seasonal_melt_factor,
    snow17_simulate,
    snow17_simulate_numpy,
    snow17_step,
)
from jsnow17.parameters import (
    DEFAULT_ADC,
    SNOW17_DEFAULTS,
    SNOW17_PARAM_BOUNDS,
    Snow17Params,
    Snow17State,
    params_dict_to_namedtuple,
)


def _synthetic_snow_forcing(n_days=730, lat=51.0):
    """Create synthetic forcing with cold winters and warm summers."""
    t = np.arange(n_days)
    # Temperature: cold winters (-15C), warm summers (+20C)
    temp = 5.0 + 15.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    # Precipitation: ~3 mm/day year-round
    rng = np.random.default_rng(42)
    precip = rng.exponential(3.0, n_days)
    # Day of year
    doy = (t % 365) + 1
    return precip, temp, doy


class TestSeasonalMeltFactor:
    """Test seasonal melt factor computation."""

    def test_bounds(self):
        """Melt factor should stay between MFMIN and MFMAX."""
        mfmax, mfmin = 1.5, 0.3
        for doy in range(1, 366):
            mf = seasonal_melt_factor(np.float64(doy), mfmax, mfmin, lat=45.0, xp=np)
            assert float(mfmin) <= float(mf) + 1e-10, f"mf={mf} < MFMIN at doy={doy}"
            assert float(mf) <= float(mfmax) + 1e-10, f"mf={mf} > MFMAX at doy={doy}"

    def test_northern_hemisphere_peak(self):
        """Melt factor should peak near Jun 21 (doy~172) in NH."""
        mfmax, mfmin = 2.0, 0.5
        mf_jun = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=45.0, xp=np)
        mf_dec = seasonal_melt_factor(np.float64(355), mfmax, mfmin, lat=45.0, xp=np)
        assert float(mf_jun) > float(mf_dec)

    def test_southern_hemisphere_reversed(self):
        """Southern hemisphere should have reversed seasonality."""
        mfmax, mfmin = 2.0, 0.5
        mf_jun_nh = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=45.0, xp=np)
        mf_jun_sh = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=-45.0, xp=np)
        # Jun in SH should be near min, in NH near max
        assert float(mf_jun_nh) > float(mf_jun_sh)


class TestSnow17StepNumpy:
    """Test single Snow-17 timestep with NumPy."""

    def test_cold_accumulation(self):
        """Snowfall at cold temp should increase w_i."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = create_initial_state(use_jax=False)
        new_state, outflow = snow17_step(
            np.float64(10.0), np.float64(-10.0), 1.0,
            state, params, np.float64(15), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # All precip should become snow (temp well below PXTEMP)
        assert float(new_state.w_i) > 0.0
        # No outflow when snow is accumulating from cold
        assert float(outflow) < 1.0  # Very little or no outflow

    def test_warm_melt(self):
        """Warm temperature should melt snow and produce outflow."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        # Start with substantial snowpack
        state = Snow17State(
            w_i=np.float64(100.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(100.0),
        )
        new_state, outflow = snow17_step(
            np.float64(5.0), np.float64(10.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Should produce outflow from melt + rain
        assert float(outflow) > 0.0
        # SWE should decrease
        total_swe = float(new_state.w_i) + float(new_state.w_q)
        assert total_swe < 100.0

    def test_rain_on_snow(self):
        """Rain on warm snow should produce extra melt."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = Snow17State(
            w_i=np.float64(50.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(50.0),
        )
        # Heavy rain at warm temp
        _, outflow_ros = snow17_step(
            np.float64(15.0), np.float64(8.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Light rain at same temp
        _, outflow_light = snow17_step(
            np.float64(0.1), np.float64(8.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Rain-on-snow should produce more outflow
        assert float(outflow_ros) > float(outflow_light)


class TestSnow17SimulateNumpy:
    """Test full Snow-17 simulation with NumPy backend."""

    def test_basic_simulation(self):
        """Model should produce non-negative output."""
        precip, temp, doy = _synthetic_snow_forcing()
        rpm, state = snow17_simulate(precip, temp, doy, use_jax=False)

        assert len(rpm) == len(precip)
        assert np.all(rpm >= 0.0)
        assert np.all(np.isfinite(rpm))

    def test_mass_conservation(self):
        """Total output should not exceed total input."""
        precip, temp, doy = _synthetic_snow_forcing(n_days=1095)
        rpm, final_state = snow17_simulate(precip, temp, doy, use_jax=False)

        total_rpm = np.sum(rpm)
        total_precip = np.sum(precip)
        remaining_swe = float(final_state.w_i) + float(final_state.w_q)

        # Output + remaining SWE should not exceed input
        assert total_rpm + remaining_swe <= total_precip * 1.5  # Some SCF correction allowed


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestSnow17JAX:
    """Test JAX backend simulation."""

    def test_basic_simulation(self):
        """JAX simulation should produce valid output."""
        import jax.numpy as jnp

        precip, temp, doy = _synthetic_snow_forcing()
        rpm, state = snow17_simulate(
            jnp.array(precip), jnp.array(temp), jnp.array(doy), use_jax=True,
        )

        rpm_np = np.array(rpm)
        assert len(rpm_np) == len(precip)
        assert np.all(rpm_np >= 0.0)
        assert np.all(np.isfinite(rpm_np))

    def test_backend_equivalence(self):
        """JAX and NumPy backends should produce equivalent results."""
        import jax.numpy as jnp

        precip, temp, doy = _synthetic_snow_forcing(n_days=365)

        rpm_np, _ = snow17_simulate(precip, temp, doy, use_jax=False)
        rpm_jax, _ = snow17_simulate(
            jnp.array(precip), jnp.array(temp), jnp.array(doy), use_jax=True,
        )

        np.testing.assert_allclose(
            np.array(rpm_jax), rpm_np,
            atol=1e-4, rtol=1e-4,
            err_msg="JAX and NumPy backends diverge",
        )
