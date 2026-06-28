"""
yfin_query.py

Agent tool: sector ETF performance retrieval from Yahoo Finance (spec Section 5,
`get_sector_performance`). Consumed by the equity agent.

IMPLEMENTATION PLAN — MCP client/server
----------------------------------------
This module is the CLIENT side of the tool. The actual market-data access
(yfinance, return/momentum/valuation computation) runs in a separate MCP
*server* process; this wrapper forwards the agent's request and returns the
structured result. Note yfinance is an unofficial endpoint, so the server is the
right place to pin the library version and handle intermittent failures.

Server: src/sector_rotation_agent/mcp_servers/yfin_server.py

"""
from __future__ import annotations
from pathlib import Path
import json
import logging
import time

from sector_rotation_agent.base_query import MCPClientBase

logger = logging.getLogger(__name__)

class YFinMCPClient(MCPClientBase):
    def __init__(self, server_path: str | Path):
        # connect the base object to the correct MCP Server
        # TODO: add error checking to make sure it's yfin_server.py
        super().__init__(server_path)

    async def get_sector_performance(
            self,
            tickers: list[str],
            period: str = "5y",
            metrics: list[str] | None = None,
            as_of: str | None = None,
        ) -> dict:
        """
        Retrieve sector ETF performance via the Yahoo Finance MCP server.

        Expected return shape (downstream code is built against this contract):

            {
            "retrieved_at": "<ISO timestamp>",
            "period": "5y",
            "sectors": {
                "XLK": {
                "return_1m": 0.03, "return_3m": 0.07, "return_12m": 0.21,
                "momentum_6m": 0.11, "momentum_12m": 0.19
                },
                ...
            }
            }

        The equity agent derives the `current_momentum` mapping that `score_branch`
        expects (e.g. {ticker: momentum_6m}) from this structure.

        Parameters
        ----------
        tickers
            Sector ETF tickers; defaults conceptually to SECTOR_ETFS.
        period
            Lookback window, e.g. "1y", "5y", "10y".
        metrics
            Which fields to compute; defaults to returns + momentum.

        Returns
        -------
        dict
            Structured per-sector performance (see shape above).
        """
        logger.info("yfinance tool call -> get_sector_performance: %d tickers, period=%s, as_of=%s",
                    len(tickers), period, as_of)
        logger.debug("yfinance performance tickers: %s, metrics=%s", tickers, metrics)
        t0 = time.perf_counter()
        result = await self.session().call_tool(
            "get_sector_performance",
            {"tickers": tickers, "period": period, "metrics": metrics, "as_of": as_of},
        )
        try:
            data = json.loads(result.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]
        except (json.JSONDecodeError, AttributeError, IndexError):
            logger.exception("Failed to parse get_sector_performance response")
            raise
        n = len(data.get("sectors", {}))
        logger.info("yfinance get_sector_performance returned %d sector(s) (%.0f ms)",
                    n, (time.perf_counter() - t0) * 1000.0)
        return data

    async def get_sector_valuations(
            self,
            tickers: list[str]
        ) -> dict:
        """
        TODO: Docstring
        """
        logger.info("yfinance tool call -> get_sector_valuations: %d tickers", len(tickers))
        logger.debug("yfinance valuation tickers: %s", tickers)
        t0 = time.perf_counter()
        result = await self.session().call_tool(
            "get_sector_valuations",
            {"tickers": tickers}
        )
        try:
            data = json.loads(result.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]
        except (json.JSONDecodeError, AttributeError, IndexError):
            logger.exception("Failed to parse get_sector_valuations response")
            raise
        n = len(data.get("sectors", {}))
        logger.info("yfinance get_sector_valuations returned %d sector(s) (%.0f ms)",
                    n, (time.perf_counter() - t0) * 1000.0)
        return data
