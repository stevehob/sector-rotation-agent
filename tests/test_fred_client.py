import pytest
import dotenv
import os
from pathlib import Path
from mcp import types
from datetime import datetime, timedelta, timezone
import math
from sector_rotation_agent.fred_query import FredMCPClient

dotenv.load_dotenv()

@pytest.fixture
def server_path():
    # get the path for the fred_server.py
    return Path(__file__).resolve().parents[1] / "src" / "sector_rotation_agent" / "mcp_servers" / "fred_server.py"


@pytest.mark.asyncio
async def test_tools_list(server_path):
    # execute uses sys.executable so that I'm sure it resolves to the venv version of python
    async with FredMCPClient(server_path=server_path) as client:

        result = await client.list_tools()
    
    # check that it's type of Tool
    assert all(isinstance(t, types.Tool) for t in result)

    # convert the keys to a list of names and check that the name is correct
    names = {t.name for t in result}
    assert "get_macro_indicators" in names and "list_available_series" in names


  
@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration", reason="Skipping long-running tests (only run during Integration)")
async def test_get_macro_indicators(server_path):
    series1 = "FEDFUNDS"
    series2 = "CPIAUCSL"
    series3 = "PCEPI"
    series4 = "UNRATE"
    series5 = "T10Y2Y"
    series6 = "A191RL1Q225SBEA"
    series7 = "MANEMP"
    series8 = "USALOLITOAASTSAM"
    start_date = "2020-01-01"
    async with FredMCPClient(server_path=server_path) as client:

        result = await client.get_macro_indicators(
            [series1, series2, series3, series4, series5, series6, series7, series8],
            start_date
            )
        
    # check the return object
    assert isinstance(result, dict)
    assert "retrieved_at" in result and "series" in result
    datetime.fromisoformat(result["retrieved_at"])          # valid ISO timestamp, else raises
    assert isinstance(result["series"], dict)

    # check that the result time is within the last 5 seconds
    now = datetime.now(timezone.utc)
    delta = now - datetime.fromisoformat(result["retrieved_at"])
    assert timedelta(0) <= delta <= timedelta(seconds=5)

    # check that we get series data
    assert isinstance(result["series"], dict)

    # check that the requested series came back
    assert set(result["series"]) == {series1, series2, series3, series4, series5, series6, series7, series8}
    
    # per-series structure + invariants
    for name, payload in result["series"].items():
        assert payload.keys() >= {"latest_observation", "stale", "observations"}
        assert isinstance(payload["stale"], bool)

        obs = payload["observations"]
        assert isinstance(obs, list) and len(obs) > 0       # a live series should have data

        prev = None
        for o in obs:
            assert set(o) == {"date", "value"}
            d = datetime.fromisoformat(o["date"])               # valid YYYY-MM-DD
            assert d >= datetime.fromisoformat(start_date)           # respects start_date
            assert isinstance(o["value"], (int, float)) and math.isfinite(o["value"])  # no NaN/inf
            if prev is not None:
                assert d >= prev                            # FRED returns ascending
            prev = d

        assert payload["latest_observation"] == obs[-1]["date"]   # last_updated == final obs

@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration",
                    reason="Hits the FRED MCP server; Integration only")
async def test_get_macro_indicators_shape(server_path):
    requested = ["PCEPI", "A191RL1Q225SBEA"]
    start_date = "2020-01-01"

    async with FredMCPClient(server_path=server_path) as client:
        result = await client.get_macro_indicators(requested, start_date)

    assert isinstance(result, dict)
    assert "retrieved_at" in result and "series" in result
    assert isinstance(result["series"], dict)

    retrieved = datetime.fromisoformat(result["retrieved_at"])
    assert retrieved.tzinfo is not None
    age = datetime.now(timezone.utc) - retrieved
    assert timedelta(0) <= age < timedelta(minutes=1)

    assert set(result["series"]) == set(requested)

    for name, payload in result["series"].items():
        assert payload.keys() >= {"latest_observation", "stale", "observations"}
        assert isinstance(payload["stale"], bool)
        obs = payload["observations"]
        assert isinstance(obs, list) and len(obs) > 0
        prev = None
        for o in obs:
            assert set(o) == {"date", "value"}
            d = datetime.fromisoformat(o["date"])
            assert d >= datetime.fromisoformat(start_date)
            assert isinstance(o["value"], (int, float)) and math.isfinite(o["value"])
            if prev is not None:
                assert d >= prev
            prev = d
        assert payload["latest_observation"] == obs[-1]["date"]


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration",
                    reason="Hits the FRED MCP server; Integration only")
async def test_get_macro_indicators_empty_series(server_path):
    async with FredMCPClient(server_path=server_path) as client:
        result = await client.get_macro_indicators([], "2020-01-01")
    assert result == {}


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration",
                    reason="Hits the FRED MCP server; Integration only")
async def test_get_macro_indicators_unsupported_series(server_path):
    async with FredMCPClient(server_path=server_path) as client:
        result = await client.get_macro_indicators(["PCEPI", "not_a_series"], "2020-01-01")
    assert "PCEPI" in result["series"]
    assert "not_a_series" not in result["series"]