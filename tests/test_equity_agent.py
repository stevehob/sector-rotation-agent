"""
test_equity_agent.py

Tests for the equity agent (src/sector_rotation_agent/equity_agent.py).

Status when written:
  * EquityAgent.run is a STUB (raises NotImplementedError). Written against the
    documented contract and marked xfail(strict=False) below, so the suite stays
    green now and flips to XPASS once implemented -- your signal to delete the
    marker.

run() is async (it calls the async yfinance MCP client), so the tests drive it
with asyncio.run(...). The client is faked, so these run fully offline. The whole
point of EquityResult is that its `equity_data` is the exact shape
compute_sector_score consumes and its `current_momentum` is what the macro agent's
ToT scorer needs -- the valid test checks both contracts.
"""
from __future__ import annotations

import asyncio

import pytest

import sector_rotation_agent.equity_agent as ea
import sector_rotation_agent.constants as const

# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #
class FakeYFinClient:
    """Satisfies equity_agent.EquityDataClient with canned per-sector payloads."""

    async def get_sector_performance(self, tickers, period="5y", metrics=None, as_of=None) -> dict:
        return {
            "retrieved_at": "2026-06-01T00:00:00Z", "period": period,
            "sectors": {
                t: {
                    "return_12m": 0.10, "momentum_6m": 0.05, "price_to_earnings": 20.0,
                    "momentum_6m_history": [
                        {"date": "2026-04-01", "value": 0.03},
                        {"date": "2026-05-01", "value": 0.04},
                        {"date": "2026-06-01", "value": 0.05},  # last point == momentum_6m
                    ],
                }
                for t in tickers
            },
        }

    async def get_sector_valuations(self, tickers) -> dict:
        return {
            "retrieved_at": "2026-06-01T00:00:00Z",
            "sectors": {t: {"price_to_earnings": 20.0, "pb": 3.0, "ps": 2.5} for t in tickers},  # was trailing_pe
        }


# --------------------------------------------------------------------------- #
# Valid: produces equity_data + current_momentum for the whole universe
# --------------------------------------------------------------------------- #
def test_equity_agent_run_scores_all_sectors():
    agent = ea.EquityAgent(FakeYFinClient())

    result = asyncio.run(agent.run())

    assert isinstance(result, ea.EquityResult)
    # equity_data covers every sector and is shaped for compute_sector_score
    assert set(result.equity_data) == set(const.SECTOR_ETFS_LIST)
    for row in result.equity_data.values():
        assert "momentum" in row and "valuation" in row
    # current_momentum covers every sector and is the {ticker: float} the ToT needs
    assert set(result.current_momentum) == set(const.SECTOR_ETFS_LIST)
    assert all(isinstance(v, float) for v in result.current_momentum.values())


# --------------------------------------------------------------------------- #
# Invalid: an empty ticker list is a caller error, not a silent empty result
# --------------------------------------------------------------------------- #
def test_equity_agent_empty_tickers_raises():
    agent = ea.EquityAgent(FakeYFinClient())

    with pytest.raises(ValueError):
        asyncio.run(agent.run(tickers=()))


# --------------------------------------------------------------------------- #
# Momentum history retained for the audit layer's statistical checker
# --------------------------------------------------------------------------- #
def test_equity_agent_retains_momentum_history():
    """EquityResult carries per-sector momentum history (from the performance payload)
    so the audit layer can run check_statistical_anomaly over it; the last point
    matches the momentum the scorer uses."""
    agent = ea.EquityAgent(FakeYFinClient())

    result = asyncio.run(agent.run())

    assert set(result.series_history) == set(const.SECTOR_ETFS_LIST)
    for tkr in const.SECTOR_ETFS_LIST:
        hist = result.series_history[tkr]
        assert hist and hist[-1]["value"] == result.equity_data[tkr]["momentum"]
