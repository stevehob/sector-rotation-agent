"""
fred_query.py

Agent tool: macroeconomic indicator retrieval from FRED (spec Section 5,
`get_macro_indicators`). Consumed by the macro agent.

IMPLEMENTATION PLAN — MCP client/server
----------------------------------------

The client inherits from a generic MCPClient base class that maintains
session state over multiple agent calls (MCP Client plumbing).

This subclass will handle the Fred Server interactions to forward
the agent's request to that server and return the structured result.
"""
from __future__ import annotations

# src/sector_rotation_agent/fred_query.py
import json
import logging
import time
from pathlib import Path

from sector_rotation_agent.base_query import MCPClientBase

logger = logging.getLogger(__name__)

class FredMCPClient(MCPClientBase):
    def __init__(
            self,
            server_path: str | Path
            ):
        # connect the base object to the correct MCP Server
        # TODO: add error checking to make sure it's fred_server.py
        super().__init__(server_path)


    # -----  Fred MCP Client Code   -------------------------------------------------
    # Now the specific implementations to call the Tools from the Fred MCP Server
    async def list_available_series(self) -> dict:
        """
        Retrieve the list of possible values for FRED macro series data that is 
        retrieveable by the MCP server.
        
        Parameters
        ----------
            None

        Returns
        -------
            dict
                Structured macro data series types.
        """
        logger.info("FRED tool call -> list_available_series")
        res = await self.session().call_tool("list_available_series")
        try:
            data = json.loads(res.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]
        except (json.JSONDecodeError, AttributeError, IndexError):
            logger.exception("Failed to parse list_available_series response")
            raise
        logger.info("FRED list_available_series returned %d series", len(data))
        return data
                


    async def get_macro_indicators(
            self,
            series_ids: list[str],
            start_date: str,
            end_date: str | None = None,
        ) -> dict:
        """
        Retrieve one or more FRED macro series via the FRED MCP server.

        Parameters
        ----------
        series_ids
            FRED series IDs, e.g. ["FEDFUNDS", "CPIAUCSL", "UNRATE"].
        start_date
            ISO start date for observations.
        end_date
            ISO end date; defaults to the latest available observation.

        Returns
        -------
        dict
            Structured macro data (see shape above), including a per-series freshness
            flag consumed by the statistical checker / audit layer.

        Expected return shape:

            {
            "retrieved_at": "<ISO timestamp>",
            "series": {
                "FEDFUNDS": {
                "last_updated": "<ISO date>",
                "stale": false,                 # freshness flag for the audit layer
                "observations": [{"date": "...", "value": 5.25}, ...]
                },
                ...
            }
            }
        """
        logger.info("FRED tool call -> get_macro_indicators: %d series, %s..%s",
                    len(series_ids), start_date, end_date or "latest")
        logger.debug("FRED series requested: %s", series_ids)
        t0 = time.perf_counter()
        res = await self.session().call_tool("get_macro_indicators",
                        {"series_ids": series_ids, "start_date": start_date, "end_date": end_date})
        try:
            data = json.loads(res.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]
        except (json.JSONDecodeError, AttributeError, IndexError):
            logger.exception("Failed to parse get_macro_indicators response")
            raise
        n_series = len(data.get("series", {}))
        logger.info("FRED get_macro_indicators returned %d series (%.0f ms)",
                    n_series, (time.perf_counter() - t0) * 1000.0)
        return data
                

