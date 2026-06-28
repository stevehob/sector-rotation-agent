"""
test_yfin_client.py

Tests for the Yahoo Finance MCP *client* (src/sector_rotation_agent/yfin_query.py).

Two tiers:

  * **Unit tests** (always run) — inject a fake MCP session so we can assert
    exactly which tool the client calls, which arguments it forwards, and how it
    parses the reply, with no subprocess and no network. This is where the
    client's request/response contract is pinned.
  * **Integration tests** (only when ``TEST_MODE == "Integration"``) — drive the
    real server end to end.

NOTE: the unit tests assert the *correct* contract (the argument key the server
actually expects is ``tickers``, and ``get_sector_valuations`` is a method on the
client). If ``yfin_query.py`` still forwards ``sectors`` or defines
``get_sector_valuations`` at module scope, the corresponding tests will fail —
by design; they pin those bugs.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp import types

from sector_rotation_agent.yfin_query import YFinMCPClient

INTEGRATION_ONLY = pytest.mark.skipif(
    os.getenv("TEST_MODE") != "Integration",
    reason="Drives the real yfin MCP server; Integration only",
)

VALUATION_KEYS = (
    "price_to_earnings",
    "price_to_book",
    "price_to_sales",
    "price_to_cashflow",
)


@pytest.fixture
def server_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src" / "sector_rotation_agent" / "mcp_servers" / "yfin_server.py"
    )


# --------------------------------------------------------------------------- #
# Fake MCP session — lets the unit tests run offline, with no subprocess
# --------------------------------------------------------------------------- #
class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeToolResult:
    """Shaped like mcp.types.CallToolResult: JSON lives in content[0].text."""
    def __init__(self, payload: dict):
        self.content = [_FakeContent(json.dumps(payload))]
        self.isError = False


class _FakeSession:
    """Stand-in for an initialized MCP ClientSession.

    Records every ``call_tool`` invocation so a test can assert the tool name and
    arguments the client forwarded, and returns a canned, server-shaped reply.
    """
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name, tool_input):
        self.calls.append((tool_name, tool_input))
        return _FakeToolResult(self._payload)


def _client_with_fake_session(server_path, payload):
    """Build a client WITHOUT connecting, then inject a fake session.

    ``YFinMCPClient.__init__`` only constructs ``StdioServerParameters`` — no
    subprocess is spawned until ``connect()`` — so swapping in a fake session
    lets us exercise the request-building and response-parsing logic in
    isolation. Returns (client, fake_session).
    """
    client = YFinMCPClient(server_path=server_path)
    fake = _FakeSession(payload)
    # bypass connect(); session() now returns the fake
    client._session = fake          # pyright: ignore[reportAttributeAccessIssue]
    return client, fake


# --------------------------------------------------------------------------- #
# Unit tests — request/response contract (offline, always run)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_performance_forwards_tickers_and_parses(server_path):
    payload = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "period": "1y",
        "sectors": {"XLK": {"return_1m": 0.03, "momentum_6m": 0.11}},
    }
    client, fake = _client_with_fake_session(server_path, payload)

    out = await client.get_sector_performance(["XLK", "XLF"], period="1y", metrics=["returns"])

    assert out == payload                       # parsed from content[0].text
    assert len(fake.calls) == 1
    name, args = fake.calls[0]
    assert name == "get_sector_performance"
    # The server's tool parameter is `tickers` — the client must forward that key.
    assert args == {"tickers": ["XLK", "XLF"], "period": "1y", "metrics": ["returns"], "as_of": None}


@pytest.mark.asyncio
async def test_performance_uses_documented_defaults(server_path):
    client, fake = _client_with_fake_session(server_path, {"sectors": {}})

    await client.get_sector_performance(["XLK"])

    _, args = fake.calls[0]
    assert args == {"tickers": ["XLK"], "period": "5y", "metrics": None, "as_of": None}


@pytest.mark.asyncio
async def test_valuations_forwards_tickers_and_parses(server_path):
    payload = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sectors": {"XLK": {"price_to_earnings": 34.5}},
    }
    client, fake = _client_with_fake_session(server_path, payload)

    out = await client.get_sector_valuations(["XLK"])

    assert out == payload
    assert len(fake.calls) == 1
    name, args = fake.calls[0]
    assert name == "get_sector_valuations"
    assert args == {"tickers": ["XLK"]}


@pytest.mark.asyncio
async def test_session_guard_raises_before_connect(server_path):
    # No injected session and no connect() -> session() must refuse, not crash oddly.
    # Uses get_sector_performance (already a real method) so this isolates the
    # base-class session guard from the valuations-method bug noted in the docstring.
    client = YFinMCPClient(server_path=server_path)
    with pytest.raises(ConnectionError):
        await client.get_sector_performance(["XLK"])


# --------------------------------------------------------------------------- #
# Integration — tools list (needs the server subprocess; no network/keys)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tools_list(server_path):
    # sys.executable is used by the base client so the subprocess resolves to the venv.
    async with YFinMCPClient(server_path=server_path) as client:
        result = await client.list_tools()

    assert all(isinstance(t, types.Tool) for t in result)
    names = {t.name for t in result}
    assert {"get_sector_valuations", "get_sector_performance", "list_available_sectors"} <= names


# --------------------------------------------------------------------------- #
# Integration — end to end against the real server (Integration only)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@INTEGRATION_ONLY
async def test_get_sector_valuations_integration(server_path):
    async with YFinMCPClient(server_path=server_path) as client:
        result = await client.get_sector_valuations(["XLK", "XLF"])

    assert set(result) == {"retrieved_at", "sectors"}
    retrieved = datetime.fromisoformat(result["retrieved_at"])
    assert retrieved.tzinfo is not None
    assert timedelta(0) <= datetime.now(timezone.utc) - retrieved < timedelta(minutes=2)

    assert set(result["sectors"]) == {"XLK", "XLF"}
    for ratios in result["sectors"].values():
        assert isinstance(ratios, dict)
        if not ratios or "error" in ratios:
            continue
        for key in VALUATION_KEYS:
            assert key in ratios
            val = ratios[key]
            assert val is None or (isinstance(val, (int, float)) and math.isfinite(val))


@pytest.mark.asyncio
@INTEGRATION_ONLY
async def test_get_sector_performance_integration(server_path):
    async with YFinMCPClient(server_path=server_path) as client:
        result = await client.get_sector_performance(
            ["XLK", "XLF"], period="2y", metrics=["returns", "momentum_6m"]
        )

    assert set(result) >= {"retrieved_at", "period", "sectors"}
    assert result["period"] == "2y"
    assert set(result["sectors"]) == {"XLK", "XLF"}
    for data in result["sectors"].values():
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


@pytest.mark.asyncio
@INTEGRATION_ONLY
async def test_get_sector_valuations_empty_integration(server_path):
    async with YFinMCPClient(server_path=server_path) as client:
        result = await client.get_sector_valuations([])
    assert result == {}
