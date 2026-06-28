from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import pytest
import json
import sys
import os
from pathlib import Path
import dotenv

from sector_rotation_agent.mcp_servers.fred_server import get_macro_indicators

dotenv.load_dotenv()

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def series_ids() -> dict[str, str]:
    """An expansionary tape: tech/comm/industrials strong, defensives weak."""
    return {
    'fed_funds_rate':    'FEDFUNDS',
    'cpi':               'CPIAUCSL',
    'pce_inflation':     'PCEPI',
    'unemployment':      'UNRATE',
    'yield_spread_10_2': 'T10Y2Y',      # 10yr minus 2yr Treasury
    'gdp_growth':        'A191RL1Q225SBEA',  # real GDP, quarterly
    'ism_pmi':           'MANEMP',      # manufacturing employment as PMI proxy
    'leading_index':     'USALOLITOAASTSAM',  # OECD composite leading indicator (US, amplitude-adjusted)
}

@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration", reason="Skipping long-running tests (only run during Integration)")
async def test_all_series(series_ids):
    
    start_date="2000-01-01"
    end_date=None
    
    # get the path for the fred_server.py
    server = Path(__file__).resolve().parents[1] / "src" / "sector_rotation_agent" / "mcp_servers" / "fred_server.py"
    # execute uses sys.executable so that I'm sure it resolves to the venv version of python
    params = StdioServerParameters(command=sys.executable, args=[str(server)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("get_macro_indicators",
                    {"series_ids": list(series_ids.values()), "start_date": start_date, "end_date": end_date})
            # check something came back
            assert not res.isError
            # check the return payload
            payload = json.loads(res.content[0].text) # pyright: ignore[reportAttributeAccessIssue] # (or res.structuredContent, if populated)  res.content[0].text
            assert "FEDFUNDS" in payload["series"]
            # just check one series (the first one - via an iterator over the list (instead of hard-code the check)
            obs = payload["series"][next(iter(series_ids.values()))]["observations"]
            assert len(obs) > 0
            assert "date" in obs[0] and "value" in obs[0]
            # TODO: should check that the rest of the series came back as well


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration", reason="Skipping long-running tests (only run during Integration)")
async def test_get_macro_indicators_direct(series_ids):
    it = iter(series_ids.values())
    series1 = next(it)
    series2 = next(it)
    result = await get_macro_indicators([series1, series2], "2020-01-01")
    assert "retrieved_at" in result
    assert "series" in result
    obs = result["series"][series1]["observations"]
    assert len(obs) > 0
    assert "date" in obs[0] and "value" in obs[0]
