"""
equity_agent.py

The equity agent (spec Section 4) -- the second retrieval-and-analysis agent the
coordinator spawns. Its job is to retrieve sector ETF data and reduce it to the
two per-sector products the rest of the system consumes:

  * equity_data       -- {ticker: {"momentum": float, "valuation": float}}, the
                         exact shape compute_sector_score takes as `equity_data`.
  * current_momentum  -- {ticker: float}, the shape the macro agent's ToT branch
                         scorer needs for its signal-consistency check.

It calls two tools -- get_sector_performance (returns/momentum) and
get_sector_valuations (P/E etc.) -- and blends them. Like the macro agent, its data
client is injected and Protocol-typed, so the agent imports no MCP/network code and
is testable offline with a fake (see test_equity_agent).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import sector_rotation_agent.constants as const

logger = logging.getLogger(__name__)


class EquityDataClient(Protocol):
    """The slice of the yfinance tool (yfin_query.YFinMCPClient) the agent needs."""
    async def get_sector_performance(
        self, tickers: list[str], period: str = "5y", metrics: list[str] | None = None,
        as_of: str | None = None,
    ) -> dict: ...
    async def get_sector_valuations(self, tickers: list[str]) -> dict: ...


@dataclass
class EquityResult:
    """What the equity agent hands the coordinator / synthesis + macro agents."""
    equity_data: dict[str, dict]        # {ticker: {"momentum":..., "valuation":...}} -> compute_sector_score
    current_momentum: dict[str, float]  # {ticker: momentum} -> macro agent's ToT scorer
    raw: dict = field(default_factory=dict)   # raw tool payloads, for the audit log
    series_history: dict[str, list[dict]] = field(default_factory=dict)
    # ^ per-sector momentum_6m history ({ticker: [{date,value}]}), retained for the
    #   audit layer's statistical checker.


class EquityAgent:
    """
    Retrieve sector ETF performance + valuations and reduce them per sector.

    data_client is injected (yfin_query.YFinMCPClient in production, a fake in
    tests) and need only satisfy EquityDataClient.
    """

    def __init__(self, data_client: EquityDataClient) -> None:
        self._data = data_client

    async def run(
        self,
        tickers: tuple[str, ...] = const.SECTOR_ETFS_LIST,
        *,
        period: str = "5y",
        as_of: str | None = None,
    ) -> EquityResult:
        """
        Produce equity_data and current_momentum for the sector universe.

        Steps to implement:
          1. Reject an empty `tickers` (raise ValueError) -- an empty universe is a
             caller error, not a silently empty result.
          2. Call self._data.get_sector_performance(tickers, period) and
             self._data.get_sector_valuations(tickers).
          3. Per sector, derive a momentum figure from the performance payload
             (e.g. momentum_6m) and a valuation figure from the valuation payload
             (e.g. trailing_pe, or a composite of the multiples).
          4. Assemble equity_data {ticker: {"momentum":..., "valuation":...}} -- the
             shape compute_sector_score consumes -- and current_momentum
             {ticker: momentum} -- the shape the macro agent's ToT scorer needs.
             Keep the raw payloads on the result for the audit log.

        Note the momentum here and the momentum the macro agent passes into its ToT
        must be the same series; deriving both from this one retrieval is what keeps
        them consistent across the two agents.

        Parameters
        ----------
        tickers
            Sector ETF tickers; defaults to all 11 (const.SECTOR_ETFS_LIST).
        period
            Lookback window forwarded to get_sector_performance.

        Returns
        -------
        EquityResult
        """
        # Reject an empty `tickers` (raise ValueError) -- an empty universe is a
        #  caller error, not a silently empty result.
        if not tickers:
            logger.error("Equity agent called with an empty tickers list")
            raise ValueError("Caller supplied empty tickers list")

        ticker_list = list(tickers)
        
        #  2. Get data: get_sector_performance, get_sector_valuations
        logger.info("Equity agent run: %d tickers, period=%s, as_of=%s", len(ticker_list), period, as_of)
        logger.debug("Equity agent tickers: %s", ticker_list)
        sector_performance = await self._data.get_sector_performance(tickers=ticker_list, period=period, as_of=as_of)
        sector_valuations = await self._data.get_sector_valuations(tickers=ticker_list)
        # the data we want is nested below a top-level from what just came back
        sec_perf = sector_performance["sectors"]
        sec_vals = sector_valuations["sectors"]

        #  Per sector, derive a momentum and valuation figure from the performance payload
        equity_data: dict[str, dict] = {}
        current_momentum: dict[str, float] = {}
        series_history: dict[str, list[dict]] = {}
        for tkr in ticker_list:
            # Tolerate sectors with no usable data at this as_of instead of failing the
            # whole run. A backtest of an older window legitimately lacks some ETFs (XLRE
            # pre-2015, XLC pre-2018), and a single bad fetch shouldn't sink a live run
            # either. compute_sector_score treats an absent sector as neutral, and a
            # backtest's realized-return comparison drops it too. A momentum or P/E that is
            # missing OR None (yfinance NaN -> None) both count as "no data".
            sec = sec_perf.get(tkr)
            val = sec_vals.get(tkr)
            momentum_raw = sec.get("momentum_6m") if isinstance(sec, dict) else None
            pe_raw = val.get("price_to_earnings") if isinstance(val, dict) else None
            if momentum_raw is None or pe_raw is None:
                logger.warning("Equity data unavailable for %s (momentum_6m=%s, "
                               "price_to_earnings=%s); skipping sector",
                               tkr, momentum_raw, pe_raw)
                continue
            momentum = float(momentum_raw)
            valuation = float(pe_raw)
            current_momentum[tkr] = momentum
            equity_data[tkr] = {
                "momentum": momentum,
                "valuation": valuation,
            }
            # retain the sector's momentum history (if the tool provides it) for the
            # audit layer's statistical checker; absent -> the auditor skips this sector
            series_history[tkr] = sec.get("momentum_6m_history", []) if isinstance(sec, dict) else []
        # end for tkr

        # Every sector missing data is a real failure (bad fetch / wrong universe), not a
        # silently-empty result -- keep that guard, but only after tolerating partial gaps.
        if not equity_data:
            logger.error("Equity agent produced no usable sectors out of %d requested", len(ticker_list))
            raise ValueError("No sector had usable equity data")
        
        #  Assemble result
        logger.info("Equity agent computed momentum + valuation for %d sector(s)", len(equity_data))
        return EquityResult(
            equity_data=equity_data,
            current_momentum=current_momentum,
            raw={"performance": sector_performance, "valuations": sector_valuations},
            series_history=series_history,
        )

