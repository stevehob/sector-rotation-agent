"""
yfin_server.py

Agent tool: macroeconomic indicator retrieval from Y!Finance (spec Section 5,
`get_sector_performance`). Consumed by the mcp client (yfin_query.py).

----------------------------------------
This module is the SERVER side of the tool.

Parameters
----------
tickers
    Sector ETF ticker symbols, e.g. ["XLK (Technology)", "XLE (Energy)"].
period
    String indicating the length of trend
        from the documentation (link below):
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Default: "1mo" if start & end None
            Can combine with start/end e.g. end = start + period
metrics
    metric value(s) to retrieve data for, e.g. ['returns', 'momentum_6m']

Expected return shape :

    {
        "retrieved_at": "<ISO timestamp>",
        "ticker": {
        "metric": {
            "last_updated": "<ISO date>",
            "stale": false,                 # freshness flag for the audit layer
            "observations": [{"date": "...", "value": 5.25}, ...]
        },
        ...
        }
    }
"""
# Y!Finance
# https://pypi.org/project/yfinance/
# https://ranaroussi.github.io/yfinance/
# Note: yfinance is a wrapper around Yahoo Finance's public API, which is not officially supported by Yahoo.
from mcp.server.fastmcp import FastMCP
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, cast

import yfinance as yf
import pandas as pd

import sector_rotation_agent.constants as const

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
server = FastMCP("yfin-mcp-server", log_level=logging.getLevelName(logger.getEffectiveLevel())) # ignore # pyright: ignore[reportArgumentType]

def _equity_holdings_to_dict(eh) -> dict:
    """equity_holdings (Series or 1-col DataFrame) -> {real_label: clean_value}."""
    if eh is None:
        return {}
    s = eh.iloc[:, 0] if isinstance(eh, pd.DataFrame) else eh
    return {str(k): _clean(v) for k, v in s.items()}

def _ratios_from_equity_holdings(eh) -> dict:
    """yfinance stores Price/X rows as the reciprocal (a yield), so invert to a
    conventional ratio. Fund values are the first column; Category Average is often <NA>."""
    if eh is None or getattr(eh, "empty", True):
        return {}
    fund = eh.iloc[:, 0]
    def ratio(label):
        v = _clean(fund.get(label))
        return None if (v is None or v == 0) else 1.0 / v
    return {
        "price_to_earnings": ratio("Price/Earnings"),
        "price_to_book":     ratio("Price/Book"),
        "price_to_sales":    ratio("Price/Sales"),
        "price_to_cashflow": ratio("Price/Cashflow"),
    }

def _clean(value) -> float | None:
    """Coerce a yfinance scalar to a JSON-safe value: NaN -> None, numpy float -> float.

    yfinance fields arrive as numpy floats or NaN; NaN is not valid JSON, so it
    becomes None. A non-scalar (e.g. a Series, which means the extraction picked
    up the wrong shape) also becomes None rather than emitting junk.
    """
    if value is None:
        return None
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _compound(window: pd.Series) -> float | None:
    """Compounded return over a window of periodic returns: prod(1 + r) - 1.

    The cast keeps the type-checker happy: Series.prod() is typed as a broad Scalar
    union that doesn't support arithmetic, though here the window is float returns so
    the product is a float at runtime."""
    return _clean(cast(float, (1.0 + window).prod()) - 1.0)


def _period_start(end: pd.Timestamp, period: str) -> Optional[str]:
    """Start date ("YYYY-MM-DD") for a point-in-time fetch: `end` minus `period`
    ('5y','2y','1y','6mo','3mo','5d','ytd','max'). 'max'/'' -> None (fetch from the
    earliest available). Only used when an as_of is supplied; live calls keep
    yfinance's relative `period`."""
    p = (period or "").strip().lower()
    if p in ("", "max"):
        return None
    if p == "ytd":
        return f"{end.year}-01-01"
    i = 0
    while i < len(p) and p[i].isdigit():
        i += 1
    n = int(p[:i]) if p[:i] else 0
    unit = p[i:]
    if n <= 0:
        return (end - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    offsets = {
        "d": pd.DateOffset(days=n), "wk": pd.DateOffset(weeks=n),
        "mo": pd.DateOffset(months=n), "y": pd.DateOffset(years=n),
    }
    off = offsets.get(unit) or pd.DateOffset(years=5)
    return (end - off).strftime("%Y-%m-%d")
"""
Expected Output format:

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
"""
@server.tool(
        name = "list_available_sectors",
        description="Lists the available market sectors data series available from get_sector_**"
)
async def list_available_sectors() -> dict:
    logger.info("list_available_sectors -> %d sectors", len(const.SECTOR_ETFS))
    return const.SECTOR_ETFS


@server.tool(
        name = "get_sector_valuations",
        description = "Queries sector tickers from YFinance for standard valuation metrics"
)
def get_sector_valuations(tickers: list) -> dict:
    valuations = {}
    # Note, not doing any error checking here at the moment, relying on the client to supply valid tickers
    if len(tickers)==0:
        logger.warning("get_sector_valuations called with an empty tickers list")
        return valuations

    logger.info("yfinance valuations fetch: %d tickers", len(tickers))
    t0 = time.perf_counter()
    for ticker in tickers:
        try:
            eh = yf.Ticker(ticker).funds_data.equity_holdings
            valuations[ticker] = _ratios_from_equity_holdings(eh)
        except Exception as e:
            logger.warning("yfinance valuations failed for %s: %s", ticker, e)
            valuations[ticker] = {"error": str(e)}

    n_ok = sum(1 for v in valuations.values() if "error" not in v)
    logger.info("yfinance valuations complete: %d/%d ok (%.0f ms)",
                n_ok, len(tickers), (time.perf_counter() - t0) * 1000.0)
    output = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sectors": valuations,
    }
    return output

@server.tool(
        name = "get_sector_performance",
        description = "Queries sector tickers from YFinance"
)
def get_sector_performance(tickers: list, period: str = '5y', 
                           metrics: Optional[List[str]] = None,
                           as_of: Optional[str] = None) -> dict:
    """
    Agent-callable tool. Returns returns, momentum, and valuation
    for specified sector ETFs.
    
    Returns a dict the agent can reason about directly, plus
    a freshness timestamp for the audit layer.
    """
    if metrics is None:
        metrics = ['returns', 'momentum_6m', 'momentum_12m']

    logger.info("yfinance performance fetch: %d tickers, period=%s, metrics=%s, as_of=%s",
                len(tickers), period, metrics, as_of)
    t0 = time.perf_counter()
    try:
        if as_of:
            # Point-in-time: fetch a `period`-long window ENDING at as_of (not relative to
            # today). The 2-month tail buffer is trimmed below, so the series ends exactly
            # at the as_of month regardless of yfinance bar labeling / `end` exclusivity --
            # this keeps a backtest from seeing prices after the decision date.
            end_ts = cast(pd.Timestamp, pd.Timestamp(as_of))
            start = _period_start(end_ts, period)
            end_dl = (end_ts + pd.DateOffset(months=2)).strftime("%Y-%m-%d")
            sector_data = yf.download(tickers, start=start, end=end_dl, interval='1mo',
                              auto_adjust=True, progress=False)
        else:
            sector_data = yf.download(tickers, period=period, interval='1mo',
                              auto_adjust=True, progress=False)
    except Exception:
        logger.exception("yfinance download failed for tickers=%s period=%s as_of=%s",
                         tickers, period, as_of)
        raise
    if isinstance(sector_data, pd.DataFrame) and 'Close' in sector_data.columns:
        sector_close_prices = sector_data['Close']
        if as_of:
            # drop any month after the as_of month so every return/momentum below ends
            # exactly at as_of (point-in-time guard, independent of the fetch's `end`)
            keep = pd.to_datetime(sector_close_prices.index).to_period("M") <= pd.Period(as_of[:7], freq="M")
            sector_close_prices = cast(pd.DataFrame, sector_close_prices[keep])
        # how="all" (not the default "any"): keep a month if AT LEAST ONE sector has a
        # return. A sector that didn't exist yet at this as_of (XLRE pre-2015, XLC pre-2018)
        # comes back as an all-NaN column; with the default "any", a single such column
        # drops EVERY row, leaving monthly_returns empty so the per-ticker .iloc[-1] below
        # raises IndexError. With "all", real sectors keep their series and a nonexistent
        # one stays NaN -> its momentum comes out None and the equity agent skips it.
        monthly_returns = sector_close_prices.pct_change().dropna(how="all")

        result = {
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'period': period,
            'as_of': as_of,
            'sectors': {}
        }

        for ticker in tickers:
            if ticker not in sector_close_prices.columns:
                logger.warning("yfinance: data unavailable for %s", ticker)
                result['sectors'][ticker] = {'error': 'data unavailable'}
                continue

            ticker_data = {}

            # monthly_returns is a DataFrame (one column per ticker); pin the per-ticker
            # column to a Series so the type-checker resolves .iloc / .rolling -- indexing
            # a DataFrame by a single key widens to ndarray under the pandas stubs.
            col = cast(pd.Series, monthly_returns[ticker])

            if 'returns' in metrics:
                ticker_data['return_1m']  = _clean(col.iloc[-1])
                ticker_data['return_3m'] = _compound(col.iloc[-3:])
                ticker_data['return_12m'] = _compound(col.iloc[-12:])

            if 'momentum_6m' in metrics:
                # Rolling 6-month momentum at each month-end, so the audit layer can
                # check the latest reading against the sector's own momentum history.
                # The last point equals the scalar momentum_6m the rest of the system
                # consumes (compute_sector_score, the ToT scorer).
                rets = col
                roll6 = (1 + rets).rolling(6).apply(lambda w: w.prod(), raw=True) - 1
                history = [
                    {"date": idx.strftime("%Y-%m-%d"), "value": _clean(v)} # pyright: ignore[reportAttributeAccessIssue]
                    for idx, v in roll6.dropna().items()
                ]
                ticker_data['momentum_6m'] = (
                    history[-1]["value"] if history
                    else _compound(rets.iloc[-6:])
                )
                ticker_data['momentum_6m_history'] = history

            if 'momentum_12m' in metrics:
                ticker_data['momentum_12m'] = _compound(col.iloc[-12:])

            result['sectors'][ticker] = ticker_data
        logger.info("yfinance performance complete: %d/%d sector(s) with data (%.0f ms)",
                    sum(1 for v in result['sectors'].values() if 'error' not in v),
                    len(tickers), (time.perf_counter() - t0) * 1000.0)
    else:
        logger.error("yfinance returned no 'Close' data for tickers=%s period=%s", tickers, period)
        result = {
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'error': 'Failed to retrieve data from Yahoo Finance'
        }
    return result


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
    if not any(getattr(h, "_sra_tag", None) == "yfin-server-file" for h in logger.handlers):
        handler = logging.FileHandler(log_dir / "yfin_server.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"))
        handler.setLevel(level)
        handler._sra_tag = "yfin-server-file"  # type: ignore[attr-defined]
        logger.addHandler(handler)


if __name__ == "__main__":
    _configure_logging()
    logger.info("yfin-mcp-server starting (stdio transport)")
    try:
        server.run(transport="stdio")
    finally:
        logger.info("yfin-mcp-server stopped")
