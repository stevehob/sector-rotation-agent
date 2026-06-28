"""
test_yfin_server.py

Tests for the Yahoo Finance MCP server
(src/sector_rotation_agent/mcp_servers/yfin_server.py).

Two tiers:

  * **Unit tests** (always run) — exercise the pure helpers and the tool
    functions with ``yfinance`` monkeypatched. Fast, deterministic, offline.
    These are where the real logic (return math, reciprocal inversion, NaN
    handling, error capture) is pinned.
  * **Integration tests** (only when ``TEST_MODE == "Integration"``) — call the
    tools for real, both over MCP stdio and in-process, and validate that the
    live payload matches the contract the client and downstream agents rely on.

The server's tool functions are decorated with ``@server.tool(...)`` but FastMCP
leaves them directly callable, so the unit tests invoke them as plain functions.
``get_sector_valuations`` and ``get_sector_performance`` are synchronous; only
``list_available_sectors`` is async.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import sector_rotation_agent.mcp_servers.yfin_server as yfin_server
from sector_rotation_agent.mcp_servers.yfin_server import (
    _clean,
    _ratios_from_equity_holdings,
    get_sector_performance,
    get_sector_valuations,
    list_available_sectors,
)

# Gate for the live tests, mirroring the FRED suite.
INTEGRATION_ONLY = pytest.mark.skipif(
    os.getenv("TEST_MODE") != "Integration",
    reason="Hits Yahoo Finance; Integration only",
)

SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "sector_rotation_agent" / "mcp_servers" / "yfin_server.py"
)

VALUATION_KEYS = (
    "price_to_earnings",
    "price_to_book",
    "price_to_sales",
    "price_to_cashflow",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sector_etfs() -> dict[str, str]:
    return {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLU": "Utilities",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
    }


@pytest.fixture
def fake_close_download() -> pd.DataFrame:
    """A ``yf.download()`` stand-in: 13 monthly closes for two tickers.

    XLK compounds at +10%/month, so every derived figure is a clean power of
    1.1 (return_1m == 0.1, momentum_6m == 1.1**6 - 1, ...). XLF is flat, so
    every figure is 0.0. Columns are a MultiIndex with a top-level ``Close`` —
    exactly the shape ``yf.download`` returns for a *list* of tickers, which is
    what the server's ``'Close' in sector_data.columns`` guard expects.
    """
    dates = pd.date_range("2024-01-31", periods=13, freq="ME")
    close = pd.DataFrame(
        {"XLK": [100 * (1.1 ** i) for i in range(13)], "XLF": [100.0] * 13},
        index=dates,
    )
    return pd.concat({"Close": close}, axis=1)


@pytest.fixture
def equity_holdings_xlk() -> pd.DataFrame:
    """``funds_data.equity_holdings`` as yfinance actually returns it.

    The Price/* rows are stored as *reciprocals* (earnings/price-style yields),
    the fund's own value is the first column, and the Category Average column is
    frequently ``<NA>``. The server inverts these into conventional ratios.
    """
    return pd.DataFrame(
        {
            "XLK": [1 / 34.5, 1 / 10.4, 1 / 8.27, 1 / 25.9, pd.NA],
            "Category Average": [pd.NA] * 5,
        },
        index=[
            "Price/Earnings",
            "Price/Book",
            "Price/Sales",
            "Price/Cashflow",
            "Median Market Cap",
        ],
    )


# --------------------------------------------------------------------------- #
# Unit tests — helpers (offline, always run)
# --------------------------------------------------------------------------- #
def test_clean_coerces_to_json_safe_scalar():
    assert _clean(None) is None
    assert _clean(float("nan")) is None
    assert _clean(pd.NA) is None
    assert _clean("not a number") is None
    # A non-scalar (truth value is ambiguous) must degrade to None, not raise.
    assert _clean(pd.Series([1, 2])) is None
    assert _clean(5) == 5.0
    cleaned = _clean(3.5)
    assert isinstance(cleaned, float) and cleaned == 3.5


def test_ratios_inverts_yfinance_reciprocals(equity_holdings_xlk):
    ratios = _ratios_from_equity_holdings(equity_holdings_xlk)
    assert ratios["price_to_earnings"] == pytest.approx(34.5)
    assert ratios["price_to_book"] == pytest.approx(10.4)
    assert ratios["price_to_sales"] == pytest.approx(8.27)
    assert ratios["price_to_cashflow"] == pytest.approx(25.9)


def test_ratios_empty_or_none_returns_empty_dict():
    assert _ratios_from_equity_holdings(None) == {}
    assert _ratios_from_equity_holdings(pd.DataFrame()) == {}


def test_ratios_zero_and_missing_label_become_none():
    eh = pd.DataFrame(
        {"X": [0.0, 1 / 10.0]},
        index=["Price/Earnings", "Price/Book"],
    )
    ratios = _ratios_from_equity_holdings(eh)
    assert ratios["price_to_earnings"] is None      # 0 -> None, no divide-by-zero
    assert ratios["price_to_book"] == pytest.approx(10.0)
    assert ratios["price_to_sales"] is None         # label absent -> None
    assert ratios["price_to_cashflow"] is None


# --------------------------------------------------------------------------- #
# Unit tests — get_sector_valuations (offline, always run)
# --------------------------------------------------------------------------- #
def test_valuations_builds_envelope_and_inverts(monkeypatch, equity_holdings_xlk):
    def fake_ticker(symbol):
        eh = equity_holdings_xlk if symbol == "XLK" else None
        return SimpleNamespace(funds_data=SimpleNamespace(equity_holdings=eh))

    monkeypatch.setattr(yfin_server.yf, "Ticker", fake_ticker)

    out = get_sector_valuations(["XLK", "XLV"])

    assert set(out) == {"retrieved_at", "sectors"}
    retrieved = datetime.fromisoformat(out["retrieved_at"])
    assert retrieved.tzinfo is not None             # tz-aware for the audit layer
    assert out["sectors"]["XLK"]["price_to_earnings"] == pytest.approx(34.5)
    assert out["sectors"]["XLV"] == {}              # eh is None -> no ratios, no crash
    json.dumps(out)                                 # must be wire-serializable


def test_valuations_captures_per_ticker_errors(monkeypatch):
    def boom(symbol):
        raise RuntimeError("network down")

    monkeypatch.setattr(yfin_server.yf, "Ticker", boom)

    out = get_sector_valuations(["XLE"])
    assert "error" in out["sectors"]["XLE"]
    assert "network down" in out["sectors"]["XLE"]["error"]
    json.dumps(out)


def test_valuations_empty_list_returns_bare_dict():
    # Mirrors the FRED server's empty-input convention: a bare {} short-circuit.
    assert get_sector_valuations([]) == {}


# --------------------------------------------------------------------------- #
# Unit tests — get_sector_performance (offline, always run)
# --------------------------------------------------------------------------- #
def test_performance_computes_returns_and_momentum(monkeypatch, fake_close_download):
    monkeypatch.setattr(yfin_server.yf, "download", lambda *a, **k: fake_close_download)

    out = get_sector_performance(
        ["XLK", "XLF"],
        period="1y",
        metrics=["returns", "momentum_6m", "momentum_12m"],
    )

    assert set(out) >= {"retrieved_at", "period", "sectors"}
    assert out["period"] == "1y"
    datetime.fromisoformat(out["retrieved_at"])

    xlk = out["sectors"]["XLK"]
    assert xlk["return_1m"] == pytest.approx(0.1)
    assert xlk["return_3m"] == pytest.approx(1.1 ** 3 - 1)
    assert xlk["return_12m"] == pytest.approx(1.1 ** 12 - 1)
    assert xlk["momentum_6m"] == pytest.approx(1.1 ** 6 - 1)
    assert xlk["momentum_12m"] == pytest.approx(1.1 ** 12 - 1)
    # every scalar metric is a finite float; momentum_6m_history is a series, checked below
    scalars = {k: v for k, v in xlk.items() if k != "momentum_6m_history"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in scalars.values())
    # momentum_6m_history: rolling 6-month momentum, last point == the scalar momentum_6m
    hist = xlk["momentum_6m_history"]
    assert isinstance(hist, list) and hist
    assert all(isinstance(p["value"], float) and math.isfinite(p["value"]) for p in hist)
    assert hist[-1]["value"] == pytest.approx(xlk["momentum_6m"])

    xlf = out["sectors"]["XLF"]
    assert xlf["return_1m"] == pytest.approx(0.0)
    assert xlf["momentum_6m"] == pytest.approx(0.0)
    json.dumps(out)


def test_performance_marks_unknown_ticker_unavailable(monkeypatch, fake_close_download):
    monkeypatch.setattr(yfin_server.yf, "download", lambda *a, **k: fake_close_download)

    out = get_sector_performance(["XLK", "ZZZZ"], metrics=["returns"])
    assert out["sectors"]["ZZZZ"] == {"error": "data unavailable"}
    assert "return_1m" in out["sectors"]["XLK"]


@pytest.mark.parametrize("bad_return", [pd.DataFrame(), None])
def test_performance_download_failure_returns_error(monkeypatch, bad_return):
    # No 'Close' column (or not even a DataFrame) -> graceful error envelope.
    monkeypatch.setattr(yfin_server.yf, "download", lambda *a, **k: bad_return)

    out = get_sector_performance(["XLK"])
    assert "error" in out
    assert "sectors" not in out
    json.dumps(out)


# --------------------------------------------------------------------------- #
# Unit test — list_available_sectors (offline, always run)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_list_available_sectors_returns_full_map():
    out = await list_available_sectors()
    assert out["XLK"] == "Technology"
    assert len(out) == 11


# --------------------------------------------------------------------------- #
# Integration — over MCP stdio (Integration only)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@INTEGRATION_ONLY
async def test_valuations_over_mcp(sector_etfs):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "get_sector_valuations", {"tickers": list(sector_etfs)}
            )
            assert not res.isError
            payload = json.loads(res.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]

    assert set(payload) == {"retrieved_at", "sectors"}
    retrieved = datetime.fromisoformat(payload["retrieved_at"])
    assert retrieved.tzinfo is not None
    assert timedelta(0) <= datetime.now(timezone.utc) - retrieved < timedelta(minutes=2)

    assert set(payload["sectors"]) == set(sector_etfs)
    for ratios in payload["sectors"].values():
        assert isinstance(ratios, dict)
        if not ratios or "error" in ratios:
            continue
        for key in VALUATION_KEYS:
            assert key in ratios
            val = ratios[key]
            assert val is None or (isinstance(val, (int, float)) and math.isfinite(val))


@pytest.mark.asyncio
@INTEGRATION_ONLY
async def test_performance_over_mcp():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "get_sector_performance",
                {"tickers": ["XLK", "XLF"], "period": "2y",
                 "metrics": ["returns", "momentum_6m"]},
            )
            assert not res.isError
            payload = json.loads(res.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]

    assert set(payload) >= {"retrieved_at", "period", "sectors"}
    assert payload["period"] == "2y"
    assert set(payload["sectors"]) == {"XLK", "XLF"}
    for data in payload["sectors"].values():
        assert isinstance(data, dict)
        if "error" in data:
            continue
        for key, val in data.items():
            if key == "momentum_6m_history":          # a rolling series, not a scalar
                assert isinstance(val, list)
                for point in val:
                    assert {"date", "value"} <= set(point)
                    pv = point["value"]
                    assert pv is None or (isinstance(pv, (int, float)) and math.isfinite(pv))
                continue
            assert val is None or (isinstance(val, (int, float)) and math.isfinite(val))


# --------------------------------------------------------------------------- #
# Integration — in-process direct calls (Integration only)
# --------------------------------------------------------------------------- #
@INTEGRATION_ONLY
def test_valuations_direct():
    out = get_sector_valuations(["XLK", "XLF"])
    assert set(out) == {"retrieved_at", "sectors"}
    assert set(out["sectors"]) == {"XLK", "XLF"}
    json.dumps(out)


@INTEGRATION_ONLY
def test_performance_direct():
    out = get_sector_performance(["XLK", "XLF"], period="1y", metrics=["returns", "momentum_6m"])
    assert "sectors" in out
    assert set(out["sectors"]) == {"XLK", "XLF"}
    json.dumps(out)
