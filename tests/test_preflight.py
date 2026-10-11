"""Guide §6 cost pre-flight and the $15 ceiling (decision 13). Keyless."""

from __future__ import annotations

import pytest

import rab.constants as C
from rab import km
from rab.resume import CostMeter, PreflightAbort, preflight, recorded_calls_prior
from rab.types import load_events

DS = ("openrouter", C.AGENT_MODEL)


def usage(prompt=6000, out=600, cached=4000):
    return {"input_tokens": prompt, "output_tokens": out, "cache_read_input_tokens": cached}


def meter_with(n, **u):
    m = CostMeter(calls_per_continuation=4.0)
    for _ in range(n):
        m.add(*DS, usage(**u))
    return m


def test_nothing_before_twenty_calls():
    assert preflight(meter_with(19)) is None


def test_zero_cache_reads_on_all_twenty_warn_and_do_not_abort(capsys):
    v = preflight(meter_with(20, cached=0))
    assert v["zero_cache"] and not v["abort"] and not v["reasons"]
    assert v["warnings"] == ["cache_read_input_tokens is 0 on all of the first 20 calls"]
    assert "warning: cache_read_input_tokens is 0" in capsys.readouterr().err


def test_projection_above_the_ceiling_aborts():
    with pytest.raises(PreflightAbort, match="exceeds"):
        preflight(meter_with(20, prompt=200_000, out=4000, cached=1000))


def test_healthy_run_passes_and_is_evaluated_once():
    m = meter_with(20)
    v = preflight(m)
    assert not v["abort"] and v["projected_usd"] < C.CASH_CEILING_USD
    m.add(*DS, usage(cached=0))
    assert preflight(m) is v


def test_disabled_preflight_never_aborts():
    v = preflight(meter_with(20, prompt=200_000, out=4000, cached=1000), enabled=False)
    assert not v["abort"] and v["reasons"]


def test_prices_match_km_spend_formula(km_root):
    from analysis.spend import _price_one, load_prices

    ev = load_events()[0]
    u = km.load_record(ev.run_id, ev.position)["usage"]
    cell = {"runs": 1, "in": u["input_tokens"], "out": u["output_tokens"], "cached": u["cache_read_input_tokens"]}
    km_usd = _price_one("openrouter", C.AGENT_MODEL, cell, load_prices(), [], "agent")["usd"]
    assert round(CostMeter().price(*DS, u), 4) == km_usd > 0


def test_ceiling_counts_cash_not_credit():
    m = CostMeter()
    m.add("anthropic", "claude-sonnet-5", {"input_tokens": 10_000_000, "output_tokens": 0})
    assert m.credit_usd > C.CASH_CEILING_USD and not m.stop_reached
    m.add(*DS, {"input_tokens": 120_000_000, "output_tokens": 0, "cache_read_input_tokens": 0})
    assert m.stop_reached and m.cash_usd >= C.CASH_CEILING_USD


def test_recorded_calls_prior_is_positive(km_root):
    prior = recorded_calls_prior(load_events())
    assert 1.0 < prior < 30.0


def test_all_failed_calls_are_named_as_failures_not_as_cache_misses():
    m = CostMeter(calls_per_continuation=4.0)
    for _ in range(20):
        m.add(*DS, {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}, ok=False)
    v = preflight(m)
    assert not v["abort"] and v["warnings"] == [
        "all of the first 20 agent calls failed (API errors; see results/provider.log)"]
