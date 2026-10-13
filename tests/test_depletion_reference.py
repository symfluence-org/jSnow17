"""Hand-derived Anderson (2006) pp. 13, 15–18, 32–34 reference cases."""

import numpy as np
import pytest

from jsnow17.depletion import accumulate_history, effective_snow_area
from jsnow17.model import seasonal_melt_factor, snow17_step
from jsnow17.parameters import (
    DEFAULT_ADC,
    SNOW17_DEFAULTS,
    Snow17State,
    params_dict_to_namedtuple,
)


def partial():
    return Snow17State(100.0, 0.0, 0.0, 0.0, -5.0, 600.0)


def test_fresh_snow_full_cover_and_quarter_melt_return_path():
    s = accumulate_history(partial(), 10.0, 1.0, 600.0)
    s = s._replace(w_i=110.0)
    assert s.w_ns == 100 and s.w_100 == 107.5
    assert effective_snow_area(s, 600.0) == 1.0
    assert effective_snow_area(s._replace(w_i=107.5), 600.0) == 1.0
    # Halfway through the last 75% of fresh-layer ablation.
    assert effective_snow_area(s._replace(w_i=103.75), 600.0) == pytest.approx(7 / 12)
    assert effective_snow_area(s._replace(w_i=100.0), 600.0) == pytest.approx(1 / 6)


def test_fresh_snow_threshold_scales_with_hours():
    for dt in [1.0, 0.25, 0.125]:
        assert not accumulate_history(partial(), 0.19 * 24 * dt, dt, 600.0).fresh_active
        assert accumulate_history(partial(), 0.21 * 24 * dt, dt, 600.0).fresh_active


def test_repeat_snow_and_new_accumulation_reset():
    s = accumulate_history(partial(), 10.0, 1.0, 600.0)._replace(w_i=110.0)
    s = accumulate_history(s, 3.0, 1.0, 600.0)
    assert s.w_ns == 100 and s.w_100 == 109.75
    renewed = accumulate_history(partial(), 210.0, 1.0, 600.0)
    assert renewed.swe == 310 and not renewed.fresh_active


def test_cold_fresh_snow_step_recovers_partial_cover():
    p = params_dict_to_namedtuple(dict(SNOW17_DEFAULTS, DAYGM=0.0), False)
    s, q = snow17_step(10.0, -5.0, 1.0, partial(), p, 300, 64.0, 600.0)
    assert effective_snow_area(s, 600.0) == 1.0
    assert s.w_i == 110.0 and q == 0.0


def test_empty_pack_and_high_latitude_winter():
    assert DEFAULT_ADC[0] == 0.05
    assert (
        effective_snow_area(Snow17State(0.0, 0.0, 0.0, 0.0, 0.0, 600.0), 600.0) == 0.0
    )
    assert seasonal_melt_factor(15, 2.0, 0.3, 64.0) == 0.3
    assert seasonal_melt_factor(65, 2.0, 0.3, 64.0) == 0.3


@pytest.mark.parametrize("rain", [1.0, 30.0])
def test_melt_equations_5_and_6(rain):
    # Fully covered, ripe pack: no holding/refreezing/ground-melt confounders.
    p = params_dict_to_namedtuple(
        dict(
            SNOW17_DEFAULTS,
            NMF=0.0,
            TIPM=0.0,
            PLWHC=0.0,
            DAYGM=0.0,
            MFMAX=1.0,
            MFMIN=1.0,
        ),
        False,
    )
    s = Snow17State(1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    updated, q = snow17_step(rain, 5.0, 0.25, s, p, 172, 45.0, 100.0)
    if rain > 1.5:
        expected = (
            6.12e-10 * 6 * (278.0**4 - 273.0**4)
            + 8.5
            * 0.04
            * (
                0.9 * 2.7489e8 * np.exp(-4278.63 / 247.792)
                - 6.11
                + 0.00057 * 1013.25 * 5
            )
            + 0.0125 * rain * 5
        )
    else:
        expected = 5.0 + 0.0125 * rain * 5
    assert 1000.0 - updated.w_i == pytest.approx(expected, abs=1e-10)
    assert q == pytest.approx(rain + expected, abs=1e-10)


def test_antecedent_temperature_time_scaling():
    p = params_dict_to_namedtuple(dict(SNOW17_DEFAULTS, DAYGM=0.0), False)
    s = Snow17State(1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    whole, _ = snow17_step(0.0, -5.0, 0.25, s, p, 172)
    half, _ = snow17_step(0.0, -5.0, 0.125, s, p, 172)
    twice, _ = snow17_step(0.0, -5.0, 0.125, half, p, 172)
    assert whole.ati == pytest.approx(-0.5)
    assert twice.ati == pytest.approx(whole.ati)


def test_backend_parity_and_water_conservation_with_history():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    p = params_dict_to_namedtuple(dict(SNOW17_DEFAULTS, DAYGM=0.0), False)
    state = partial()
    step = jax.jit(
        lambda s, pr, t: snow17_step(pr, t, 0.125, s, p, 172, 64.0, 600.0, xp=jnp)
    )
    for pr, t in [(2.0, -5.0), (0.0, 5.0), (20.0, 10.0), (0.0, 20.0), (4.0, -3.0)] * 10:
        before = state.w_i + state.w_q
        new, q = snow17_step(pr, t, 0.125, state, p, 172, 64.0, 600.0)
        jstate, jq = step(state, pr, t)
        np.testing.assert_allclose(
            np.array(jstate[:-1], float), np.array(new[:-1], float), atol=1e-9
        )
        assert bool(jstate.fresh_active) == bool(new.fresh_active)
        assert float(jq) == pytest.approx(q, abs=1e-9)
        # SCF=PXADJ=1, so effective input equals original precipitation.
        assert pr - q - (new.w_i + new.w_q - before) == pytest.approx(0.0, abs=1e-10)
        state = new
