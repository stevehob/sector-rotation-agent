"""
backtest.py -- evaluation harness (NOT part of the agent's runtime).

Runs the agent at several historical as-of dates and compares the sector ranking it
produces against what the sectors ACTUALLY did over the following horizon. For each
window it reports:

  * the regime the agent called (point-in-time: macro uses end_date=as_of, and the
    analog store is filtered to outcomes realized on or before as_of),
  * Spearman rho between the agent's per-sector score and the realized forward return
    (positive => the ranking pointed the right way),
  * the favored-minus-disfavored realized-return spread (did the call add value),

plus an aggregate across windows, and a Markdown summary for the report's evaluation
section.

POINT-IN-TIME NOTE:
    Regime, analog evidence, AND the equity momentum signal are all point-in-time: the
    macro agent fetches FRED with end_date=as_of, the analog store is filtered to outcomes
    realized by as_of, and the equity path fetches/trims prices to end at as_of. The only
    non-as-of input is sector VALUATION (P/E), which yfinance exposes only as current --
    but compute_sector_score scores momentum, not valuation, so the ranking carries no
    look-ahead.

Run:
    uv run python -m sector_rotation_agent.backtest
    uv run python -m sector_rotation_agent.backtest --windows 2018-06-30 2021-12-31
    uv run python -m sector_rotation_agent.backtest --out reports/backtest_summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yfin
from scipy.stats import spearmanr

import sector_rotation_agent.constants as const
from sector_rotation_agent.config import settings
from sector_rotation_agent.main import amain

logger = logging.getLogger(__name__)

# Default windows: each as_of is the decision date; the following `HORIZON_MONTHS` are the
# evaluation window. Chosen to span regimes -- contraction (2008), early-cycle recovery
# (2009), mid-cycle (2013), late-cycle (2018), late/inflation (2022) -- with clear sector
# dispersion. Later windows have more prior history to draw analogs from; 2008/2009 are
# thinner (only ~2001 precedes them) and lean more on the rules label + momentum.
DEFAULT_WINDOWS = ["2008-06-30", "2009-06-30", "2013-06-30", "2018-06-30", "2021-12-31"]

HORIZON_MONTHS = const.ANALOG_DEFAULT_HORIZON_MONTHS  # 6 -- matches the question below
QUESTION = "Which sectors should I overweight/underweight over the next 6 months?"


# --------------------------------------------------------------------------- #
# Realized "ground truth": what each sector actually returned over the window
# --------------------------------------------------------------------------- #
def realized_forward_returns(as_of: str, horizon_months: int) -> dict[str, float]:
    """Actual forward total return per sector ETF over `horizon_months` from `as_of`.

    Mirrors the seed convention exactly (monthly, auto-adjusted closes), so the
    benchmark is built the same way the analog returns were:
        return = close[as_of_month + horizon] / close[as_of_month] - 1
    Matching is by calendar month (PeriodIndex), which sidesteps month-end/timestamp
    alignment quirks. Sectors with no price at either endpoint (e.g. XLRE before 2015,
    XLC before 2018) are omitted, not zero-filled.
    """
    a = cast(pd.Period, pd.Period(as_of[:7], freq="M"))   # the as_of month, e.g. 2008-06
    e = a + horizon_months                                 # the exit month, e.g. 2008-12

    # pad two months on each side so both endpoints land inside the frame; derive the
    # bounds from the Period's start/end Timestamp +/- an offset (typed cleanly, unlike
    # Period-minus-int arithmetic, which the stubs widen to Period | NaT)
    start = (a.start_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    end = (e.end_time + pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    raw = yfin.download(
        list(const.SECTOR_ETFS_LIST),
        start=start, end=end,
        interval="1mo", auto_adjust=True, progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no sector data around {as_of}")

    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).to_period("M")   # match by calendar month

    out: dict[str, float] = {}
    for tkr in const.SECTOR_ETFS_LIST:
        if tkr not in close.columns:
            continue
        try:
            p0 = float(close.loc[a, tkr])   #type: ignore
            p1 = float(close.loc[e, tkr])   #type: ignore
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(p0) or math.isnan(p1) or p0 == 0.0:
            continue
        out[tkr] = p1 / p0 - 1.0
    return out


# --------------------------------------------------------------------------- #
# Per-window scoring
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    as_of: str
    regime: str
    low_confidence: bool
    n_sectors: int
    spearman: float | None
    favored_mean: float | None
    disfavored_mean: float | None
    spread: float | None
    favored: list[str] = field(default_factory=list)       # agent's top tier (tickers)
    disfavored: list[str] = field(default_factory=list)    # agent's bottom tier
    realized_leaders: list[str] = field(default_factory=list)    # actual best 3
    realized_laggards: list[str] = field(default_factory=list)   # actual worst 3


def evaluate_window(
    as_of: str,
    regime: str,
    low_confidence: bool,
    agent_ranking: list[tuple[str, float]],   # (sector, score), best-first
    realized: dict[str, float],
) -> WindowResult:
    """Compare one window's agent ranking against realized returns."""
    # restrict to sectors the agent ranked AND that have a realized return (drops
    # XLRE/XLC in windows where they didn't trade); keep the agent's order
    common = [(s, sc) for s, sc in agent_ranking if s in realized]
    sectors = [s for s, _ in common]
    scores = [sc for _, sc in common]
    rets = [realized[s] for s in sectors]
    n = len(sectors)

    rho: float | None = None
    if n >= 3 and len(set(scores)) > 1 and len(set(rets)) > 1:
        rho = float(spearmanr(scores, rets)[0])   #type: ignore  ## [0]=correlation; version-safe
        if math.isnan(rho):
            rho = None

    # tiers by the agent's rank order: top third favored, bottom third disfavored
    k = max(1, n // 3)
    fav_tkrs, dis_tkrs = sectors[:k], sectors[-k:]
    fav_mean = sum(realized[s] for s in fav_tkrs) / k if fav_tkrs else None
    dis_mean = sum(realized[s] for s in dis_tkrs) / k if dis_tkrs else None
    spread = (fav_mean - dis_mean) if (fav_mean is not None and dis_mean is not None) else None

    by_real = sorted(realized, key=lambda s: realized[s], reverse=True)
    return WindowResult(
        as_of=as_of, regime=regime, low_confidence=low_confidence, n_sectors=n,
        spearman=rho, favored_mean=fav_mean, disfavored_mean=dis_mean, spread=spread,
        favored=fav_tkrs, disfavored=dis_tkrs,
        realized_leaders=by_real[:3], realized_laggards=by_real[-3:],
    )


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #
async def run_backtest(windows: list[str], question: str) -> list[WindowResult]:
    results: list[WindowResult] = []
    for as_of in windows:
        logger.info("Backtest window: as_of=%s", as_of)
        print(f"  running agent for {as_of} ...", flush=True)
        try:
            coord = await amain(question, as_of)        # the real pipeline, point-in-time
        except Exception as exc:
            logger.exception("Agent run failed for %s; skipping", as_of)
            print(f"  !! agent run failed for {as_of}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        if not coord.rankings:
            print(f"  !! no rankings produced for {as_of}; skipping")
            continue

        agent_ranking = [(r["sector"], r["score"]) for r in coord.rankings]
        try:
            realized = realized_forward_returns(as_of, HORIZON_MONTHS)
        except Exception as exc:
            logger.exception("Realized-return fetch failed for %s; skipping", as_of)
            print(f"  !! realized-return fetch failed for {as_of}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        results.append(evaluate_window(
            as_of, coord.regime.value, coord.low_confidence, agent_ranking, realized
        ))
    return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _f(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "  n/a"
    return f"{x * 100:+5.1f}%" if pct else f"{x:+.2f}"


def _config_line() -> str:
    return (f"model_location={settings.model_location}  "
            f"cloud={settings.cloud_model_service}/{settings.cloud_model}  "
            f"local={settings.local_model_service}/{settings.local_model}")


def print_summary(results: list[WindowResult]) -> None:
    print("\n" + "=" * 78)
    print("BACKTEST SUMMARY")
    print(_config_line())
    print("NOTE: scoring inputs are point-in-time (regime, analogs, equity momentum). "
          "Sector valuation\n      is current-only (yfinance) but is not used in scoring.")
    print("=" * 78)
    hdr = (f"{'as_of':<12}{'regime':<13}{'n':>3}  {'rho':>6}  "
           f"{'favored':>8}  {'disfav':>8}  {'spread':>8}  lowconf")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.as_of:<12}{r.regime:<13}{r.n_sectors:>3}  {_f(r.spearman):>6}  "
              f"{_f(r.favored_mean, True):>8}  {_f(r.disfavored_mean, True):>8}  "
              f"{_f(r.spread, True):>8}  {r.low_confidence}")
    print("-" * len(hdr))

    rhos = [r.spearman for r in results if r.spearman is not None]
    spreads = [r.spread for r in results if r.spread is not None]
    if rhos:
        print(f"mean Spearman rho:                      {sum(rhos) / len(rhos):+.2f}  (n={len(rhos)})")
    if spreads:
        hit = sum(1 for s in spreads if s > 0)
        print(f"mean favored-minus-disfavored spread:   {sum(spreads) / len(spreads) * 100:+.1f}%")
        print(f"windows favored beat disfavored:        {hit}/{len(spreads)}")
    print()


def write_markdown(results: list[WindowResult], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Backtest results\n")
    lines.append(f"`{_config_line()}`\n")
    lines.append("> Scoring inputs are point-in-time: regime (FRED end_date=as_of), analog "
                 "outcomes (filtered to <= as_of), and equity momentum (prices trimmed to "
                 "as_of). Sector valuation is current-only (yfinance limitation) but is not "
                 "used in the score, so the ranking has no look-ahead.\n")
    lines.append("| as_of | regime | n | Spearman | favored | disfavored | spread | low-conf |")
    lines.append("|---|---|--:|--:|--:|--:|--:|:-:|")
    for r in results:
        lines.append(f"| {r.as_of} | {r.regime} | {r.n_sectors} | {_f(r.spearman)} | "
                     f"{_f(r.favored_mean, True)} | {_f(r.disfavored_mean, True)} | "
                     f"{_f(r.spread, True)} | {'yes' if r.low_confidence else 'no'} |")
    rhos = [r.spearman for r in results if r.spearman is not None]
    spreads = [r.spread for r in results if r.spread is not None]
    lines.append("")
    if rhos:
        lines.append(f"- Mean Spearman rho: **{sum(rhos) / len(rhos):+.2f}** (n={len(rhos)})")
    if spreads:
        hit = sum(1 for s in spreads if s > 0)
        lines.append(f"- Mean favored-minus-disfavored spread: **{sum(spreads) / len(spreads) * 100:+.1f}%**")
        lines.append(f"- Windows where favored beat disfavored: **{hit}/{len(spreads)}**")
    lines.append("\n## Per-window detail\n")
    for r in results:
        lines.append(f"### {r.as_of} -- agent called *{r.regime}*"
                     f"{' (low confidence)' if r.low_confidence else ''}")
        lines.append(f"- Agent favored: {', '.join(r.favored) or 'n/a'}  |  "
                     f"disfavored: {', '.join(r.disfavored) or 'n/a'}")
        lines.append(f"- Realized leaders: {', '.join(r.realized_leaders) or 'n/a'}  |  "
                     f"laggards: {', '.join(r.realized_laggards) or 'n/a'}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown summary written to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the sector-rotation agent over historical windows")
    parser.add_argument("--windows", nargs="*", default=DEFAULT_WINDOWS,
                        help="as-of decision dates (YYYY-MM-DD); the 6 months after each are evaluated")
    parser.add_argument("--question", default=QUESTION, help="the analyst question to run at each window")
    parser.add_argument("--out", default="reports/backtest_summary.md", help="path for the Markdown summary")
    args = parser.parse_args()

    # keep the console readable; the agent's own component logs still go to logs/
    logging.basicConfig(level=logging.WARNING)

    print(f"Backtesting {len(args.windows)} window(s): {', '.join(args.windows)}")
    print(_config_line())
    results = asyncio.run(run_backtest(args.windows, args.question))
    if not results:
        print("No windows produced results.")
        return
    print_summary(results)
    write_markdown(results, Path(args.out))


if __name__ == "__main__":
    main()
