"""
fred_server.py

Agent tool: macroeconomic indicator retrieval from FRED (spec Section 5,
`get_macro_indicators`). Consumed by the mcp client (fred_query.py).

IMPLEMENTATION PLAN — MCP client/server
----------------------------------------
This module is the SERVER side of the tool.

Parameters
----------
series_ids
    FRED series IDs, e.g. ["FEDFUNDS", "CPIAUCSL", "UNRATE"].
start_date
    ISO start date for observations.
end_date
    ISO end date; defaults to the latest available observation.


Expected return shape :

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
    This server pulls data from the FredAPI every time it's called.  Not efficient, but so far the queries are quick.
        (Future TODO) Implement a sqllite data store populated from Fred
         and only refresh if the end_date is after the last pull.
"""
from mcp.server.fastmcp import FastMCP
from fredapi import Fred
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
import sector_rotation_agent.constants as const

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

server = FastMCP("fred-mcp-server", log_level=logging.getLevelName(logger.getEffectiveLevel()))        # type: ignore

"""
_series_to_payload
    input:  s (panda series)
    output: string
     
    The output string will be in the form of the expected return shape :

    "FEDFUNDS": {
            "last_updated": "<ISO date>",
            "stale": false,                 # freshness flag for the audit layer
            "observations": [{"date": "...", "value": 5.25}, ...]
    }
"""
def _release_date(fred, series_id: str) -> str | None:
    """FRED's actual publication date for a series -- the 'last_updated' field from
    get_series_info, as a YYYY-MM-DD string. This is when FRED PUBLISHED the data, which
    differs from the latest observation's PERIOD date by the release lag (e.g. May data
    published in mid-June). Best-effort: a failure here must not sink the data fetch, so
    we log and return None."""
    try:
        info = fred.get_series_info(series_id)
        raw = info.get("last_updated")   # e.g. "2026-06-11 07:31:02-05"
        return str(raw)[:10] if raw else None
    except Exception:
        logger.warning("Could not fetch release date for %s", series_id, exc_info=True)
        return None


def _series_to_payload(s, *, release_date: str | None = None):
    s = s.dropna()       # FRED series carry NaNs; NaN isn't valid JSON
    observations = [
        {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
        for idx, val in s.items()
    ]
    return {
        # latest_observation is the PERIOD date of the newest data point. FRED dates each
        # observation at its period START, so a current monthly series read mid-June shows
        # the May 1 point. release_date is when FRED actually PUBLISHED the series (from
        # get_series_info); the two differ by the release lag, which is why current data
        # can look "old". (Renamed from "last_updated", which was really the period date.)
        "latest_observation": observations[-1]["date"] if observations else None,
        "release_date": release_date,
        "stale": False,
        "observations": observations,
    }


@server.tool(
        name = "list_available_series",
        description="Lists the available economic data series available from get_macro_indictors"
)
async def list_available_series() -> dict:
    logger.info("list_available_series -> %d supported series", len(const.SUPPORTED_SERIES))
    return const.SUPPORTED_SERIES

@server.tool(
        name = "get_macro_indicators",
        description = "Queries macro economic data from the Federal Reserve"
)
async def get_macro_indicators(
    series_ids: list[str],
    start_date: str,
    end_date: str | None = None # None indicates to FredAPI the latest available data
    ) -> dict:

    output = {}
    # First, some input validation
    if len(series_ids) == 0:
        logger.warning("get_macro_indicators called with an empty series list")
        return output

    logger.info("FRED API fetch: %d series, %s..%s",
                len(series_ids), start_date, end_date or "latest")

    # With valid inputs, fetch data from FredAPI
    key = os.getenv("FRED_API_KEY")
    if not key:
        logger.error("FRED_API_KEY is not set; cannot query the FRED API")
        raise RuntimeError("Missing API key for FredAPI")

    fred = Fred(api_key=key)
    data = {}
    releases: dict[str, str | None] = {}
    t0 = time.perf_counter()
    for series in series_ids:
        if series in const.SUPPORTED_SERIES.values():
            try:
                data[series] = fred.get_series(series, observation_start=start_date, observation_end=end_date)
            except Exception:
                logger.exception("FRED API call failed for series %s", series)
                raise
            releases[series] = _release_date(fred, series)   # when FRED published it
        else:
            logger.warning("Ignoring unsupported FRED series id: %s", series)

    # Package the results into the defined output format
    output = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "series": {name: _series_to_payload(s, release_date=releases.get(name)) for name, s in data.items()},
    }
    n_obs = sum(len(v.get("observations", [])) for v in output["series"].values())
    logger.info("FRED API fetch complete: %d series, %d observations (%.0f ms)",
                len(output["series"]), n_obs, (time.perf_counter() - t0) * 1000.0)
    return output


def _configure_logging() -> None:
    """Route this server subprocess's logs to a dedicated file. A stdio MCP server speaks
    JSON-RPC over STDOUT, so log records must NEVER reach stdout; we attach a FileHandler
    to this module's logger and disable propagation so a record can't bubble up to a root
    (or FastMCP) handler that might write to stdout. Idempotent."""
    try:
        from sector_rotation_agent.config import settings
        log_dir = Path(settings.log_file).parent
        level = getattr(logging, settings.logging_level, logging.INFO)
    except Exception:
        log_dir = Path("logs")
        level = logging.INFO
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)
    logger.propagate = False   # stdio safety: keep records off any root/stdout handler
    if not any(getattr(h, "_sra_tag", None) == "fred-server-file" for h in logger.handlers):
        handler = logging.FileHandler(log_dir / "fred_server.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"))
        handler.setLevel(level)
        handler._sra_tag = "fred-server-file"  # type: ignore[attr-defined]
        logger.addHandler(handler)


if __name__ == "__main__":
    _configure_logging()
    logger.info("fred-mcp-server starting (stdio transport)")
    try:
        server.run(transport="stdio")
    finally:
        logger.info("fred-mcp-server stopped")