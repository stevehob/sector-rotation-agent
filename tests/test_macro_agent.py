"""
test_macro_agent.py

Tests for the macro agent (src/sector_rotation_agent/macro_agent.py).

Status when written:
  * MacroAgent.run is a STUB (raises NotImplementedError). These tests are written
    against the documented contract and the module is marked xfail(strict=False)
    below, so the suite stays green now and each test flips to XPASS once the body
    is implemented -- your signal to delete the marker. (Same pattern as
    test_synthesize.py / test_historical_analogs.py.)

The agent calls an async data client, so run() is a coroutine; the tests drive it
with asyncio.run(...) rather than pulling in a pytest-async plugin. Every external
collaborator is a fake injected at construction -- there is no FRED, no vector
store, and no LLM here, which is the payoff of the agent's dependency-injection
design: the orchestration is exercised in full, offline.
"""
from __future__ import annotations

import asyncio

import pytest

import sector_rotation_agent.macro_agent as ma
import sector_rotation_agent.constants as const
from sector_rotation_agent.classify_regime_tot import (
    BranchResult,
    MacroSnapshot,
    RegimeHypothesis,
)

# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #
def _fred_payload(include_all: bool = True) -> dict:
    """A get_macro_indicators payload in the documented shape.

    Gives every direct indicator one observation; CPIAUCSL and PCEPI get 13 monthly
    points each (so YoY cpi_inflation / pce_inflation can be derived). When
    include_all is False, one required series (UNRATE) is omitted -- the
    incomplete-data failure case.
    """
    latest = {
        "FEDFUNDS": 4.50, "UNRATE": 4.3, "T10Y2Y": 0.15,
        "A191RL1Q225SBEA": 2.0, "MANEMP": 12950.0, "USALOLITOAASTSAM": 100.5,
    }
    series: dict[str, dict] = {}
    for code, value in latest.items():
        if not include_all and code == "UNRATE":
            continue
        series[code] = {
            "last_updated": "2026-06-01", "stale": False,
            "observations": [{"date": "2026-06-01", "value": value}],
        }
    # CPIAUCSL: 13 monthly points, ~3% YoY (315 -> 324.45)
    cpi = [{"date": f"2025-{m:02d}-01", "value": 315.0 + m * 0.75} for m in range(1, 13)]
    cpi.append({"date": "2026-01-01", "value": 324.45})
    series["CPIAUCSL"] = {"last_updated": "2026-06-01", "stale": False, "observations": cpi}
    # PCEPI: 13 monthly points, 2.5% YoY (120.0 -> 123.0) -- a price-index LEVEL, so
    # the agent must derive YoY just like CPI (regression guard for the bug where the
    # raw ~125 level leaked into the snapshot and flattened cosine similarity).
    pce = [{"date": f"2025-{m:02d}-01", "value": 120.0 + (m - 1) * 0.25} for m in range(1, 13)]
    pce.append({"date": "2026-01-01", "value": 123.0})
    series["PCEPI"] = {"last_updated": "2026-06-01", "stale": False, "observations": pce}
    return {"retrieved_at": "2026-06-01T00:00:00Z", "series": series}


class FakeFredClient:
    """Satisfies macro_agent.MacroDataClient with a canned payload."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def get_macro_indicators(self, series_ids, start_date, end_date=None) -> dict:
        return self._payload


def _fake_generate(snapshot: MacroSnapshot, n: int) -> list[RegimeHypothesis]:
    # Two distinct candidates so selection has something to choose between.
    return [
        RegimeHypothesis(const.Regime.MID_CYCLE, "expansion", 0.6),
        RegimeHypothesis(const.Regime.LATE_CYCLE, "slowing", 0.4),
    ][:n]


def _fake_find(snapshot: MacroSnapshot, n: int, regime_filter=None) -> list[dict]:
    return [
        {
            "date": "2017-05", "similarity": 0.85,
            "regime": (regime_filter.value if regime_filter else "mid_cycle"),
            "subsequent_sector_returns": {t: 0.02 for t in const.SECTOR_ETFS_LIST},
        }
    ]


def _fake_score(hypothesis, analogs, snapshot, *, current_momentum) -> BranchResult:
    # MID_CYCLE wins by a clear margin (no tie -> not low_confidence).
    support = 0.8 if hypothesis.regime is const.Regime.MID_CYCLE else 0.4
    return BranchResult(
        hypothesis=hypothesis,
        analog_similarity=support,
        signal_consistency=support,
        support_score=support,
    )


def _make_agent(payload: dict) -> "ma.MacroAgent":
    return ma.MacroAgent(
        FakeFredClient(payload),
        generate_hypotheses=_fake_generate,     #type: ignore
        find_historical_analogs=_fake_find,     #type: ignore
        score_branch=_fake_score,
    )


_MOMENTUM = {t: 0.0 for t in const.SECTOR_ETFS_LIST}


# --------------------------------------------------------------------------- #
# Valid: full orchestration returns a well-formed MacroResult
# --------------------------------------------------------------------------- #
def test_macro_agent_run_classifies_regime():
    agent = _make_agent(_fred_payload(include_all=True))

    result = asyncio.run(agent.run(as_of="2026-06-01", current_momentum=_MOMENTUM))

    assert isinstance(result, ma.MacroResult)
    assert result.regime in set(const.Regime)             # a real regime was chosen
    assert result.regime is const.Regime.MID_CYCLE        # the best-supported branch
    # snapshot carries the full indicator vector, keyed by INDICATOR_KEYS
    assert set(result.snapshot.indicators) >= set(const.INDICATOR_KEYS)
    # cpi_inflation was derived as a YoY percent (~3), not passed through as an index
    assert 1.0 < result.snapshot.indicators["cpi_inflation"] < 6.0
    # pce_inflation likewise: a derived YoY percent, not the raw PCEPI index level
    assert 1.0 < result.snapshot.indicators["pce_inflation"] < 6.0
    assert isinstance(result.analogs, list) and result.analogs
    assert isinstance(result.low_confidence, bool)


# --------------------------------------------------------------------------- #
# Invalid: incomplete FRED data must fail loudly, not emit a NaN snapshot
# --------------------------------------------------------------------------- #
def test_macro_agent_incomplete_fred_data_raises():
    agent = _make_agent(_fred_payload(include_all=False))  # UNRATE missing

    with pytest.raises(ValueError):
        asyncio.run(agent.run(as_of="2026-06-01", current_momentum=_MOMENTUM))


# --------------------------------------------------------------------------- #
# Series history retained for the audit layer's statistical checker
# --------------------------------------------------------------------------- #
def test_macro_agent_retains_series_history():
    """MacroResult carries per-indicator history (each in its OWN units) so the audit
    layer's statistical checker can run over it. Direct keys pass their raw FRED
    observations through; cpi_inflation and pce_inflation carry DERIVED YoY series --
    each last point equals the snapshot value and is a small percent, not the raw
    index level (~324 CPIAUCSL, ~123 PCEPI) the distributional checks would otherwise
    compare against."""
    agent = _make_agent(_fred_payload(include_all=True))

    result = asyncio.run(agent.run(as_of="2026-06-01", current_momentum=_MOMENTUM))

    assert set(result.series_history) >= set(const.INDICATOR_KEYS)
    # direct key: the raw observation carried through
    assert result.series_history["fed_funds_rate"][-1]["value"] == 4.50
    # cpi_inflation: derived YoY series, last point matches the snapshot value
    cpi_hist = result.series_history["cpi_inflation"]
    assert cpi_hist[-1]["value"] == pytest.approx(result.snapshot.indicators["cpi_inflation"])
    assert cpi_hist[-1]["value"] < 50.0    # a YoY percent, not an index level
    # pce_inflation: likewise a derived YoY series (regression guard for the
    # raw-index-level bug), last point matches the snapshot value
    pce_hist = result.series_history["pce_inflation"]
    assert pce_hist[-1]["value"] == pytest.approx(result.snapshot.indicators["pce_inflation"])
    assert pce_hist[-1]["value"] < 50.0
