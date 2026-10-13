# SPDX-License-Identifier: Apache-2.0
"""Effective melt-area history from Anderson (2006), Snow17 pp. 32–34.

This is not a geometric/optical snow-cover diagnostic. In particular it must
not be interpreted as exposed glacier ice area without an additional model.
"""

import numpy as np

from .parameters import DEFAULT_ADC


def effective_snow_area(state, si=100.0, adc=None, xp=np):
    """Effective area including a persistent fresh-snow return path."""
    if adc is None:
        adc = xp.asarray(DEFAULT_ADC)
    pack = state.w_i + state.w_q
    reference = xp.maximum(xp.minimum(si, xp.maximum(state.swe, pack)), 1e-12)
    base = xp.interp(xp.clip(pack / reference, 0, 1), xp.linspace(0, 1, 11), adc)
    ramp = state.a_ns + (1 - state.a_ns) * xp.clip(
        (pack - state.w_ns) / xp.maximum(state.w_100 - state.w_ns, 1e-12), 0, 1
    )
    area = xp.where(
        state.fresh_active & (pack > state.w_ns), xp.maximum(base, ramp), base
    )
    return xp.where(pack > 0, xp.where(xp.asarray(si) > 0, area, 1.0), 0.0)


def accumulate_history(state, snowfall, dt, si, adc=None, xp=np):
    """Update Wmax/Wns/Ans/W100 before melt, without changing water stores."""
    pack = state.w_i + state.w_q
    area = effective_snow_area(state, si, adc, xp)
    continuing = state.fresh_active & (pack > state.w_ns)
    onset = (
        (pack > 0)
        & (area < 1.0)
        & (snowfall > 0.2 * 24 * dt)
        & xp.logical_not(continuing)
    )
    wns = xp.where(onset, pack, state.w_ns)
    ans = xp.where(onset, area, state.a_ns)
    w100 = xp.where(
        onset,
        pack + 0.75 * snowfall,
        xp.where(continuing, state.w_100 + 0.75 * snowfall, state.w_100),
    )
    active = continuing | onset
    total = pack + snowfall
    peak = xp.maximum(state.swe, total)
    renewed = active & (wns > 0) & (total >= 3 * wns)
    peak = xp.where(renewed, total, peak)
    return state._replace(
        swe=peak,
        w_ns=wns,
        a_ns=ans,
        w_100=w100,
        fresh_active=active & xp.logical_not(renewed),
    )
