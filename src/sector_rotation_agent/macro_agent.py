"""
macro_agent.py

The macro agent (spec Section 4) -- one of the two retrieval-and-analysis agents
the coordinator spawns. Its job is to turn an as-of date into a classified macro
regime by orchestrating existing pieces, owning no scoring logic of its own:

  1. pull the macro indicators from FRED (the get_macro_indicators tool),
  2. reduce them to a normalized point-in-time MacroSnapshot, and
  3. run the bounded Tree-of-Thought regime classifier (classify_regime_tot) over
     that snapshot, then retrieve the winning regime's analogs for the synthesis
     stage.

Following classify_regime_tot's pattern, every collaborator is injected and
Protocol-typed rather than imported as a concrete class, so this module pulls in
no MCP/network/LLM code and is exercised in full with fakes (see test_macro_agent).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Protocol

import sector_rotation_agent.constants as const
from sector_rotation_agent.classify_regime_tot import (
    AnalogFinder,
    BranchResult,
    HypothesisGenerator,
    MacroSnapshot,
    RegimeHypothesis,
    ToTResult,
    classify_regime_tot,
)

if TYPE_CHECKING:
    # Type-only import: keep the FedSource enum (and the chromadb it pulls in) out of
    # the runtime graph. `from __future__ import annotations` makes the NarrativeFinder
    # annotation a string, so this import exists purely for the type-checker.
    from sector_rotation_agent.fed_narrative_rag import FedSource

logger = logging.getLogger(__name__)

# cpi_inflation and pce_inflation are DERIVED as the YoY % change of their
# underlying FRED price-index levels (CPIAUCSL, PCEPI). Every other INDICATOR_KEY
# maps 1:1 to a FRED series in const.SUPPORTED_SERIES and is taken as the latest
# raw observation. The seed side (historical_analogs.build_seed_history) derives
# BOTH inflation series as YoY, so the query side must do the same -- otherwise
# those vector dimensions mean different things in the store vs. the live snapshot
# and cosine similarity collapses.
_DERIVED_CODES: dict[str, str] = {
    "cpi_inflation": const.SUPPORTED_SERIES["cpi"],            # CPIAUCSL (index level)
    "pce_inflation": const.SUPPORTED_SERIES["pce_inflation"],  # PCEPI    (index level)
}
_DIRECT_KEYS = tuple(k for k in const.INDICATOR_KEYS if k not in _DERIVED_CODES)


def _yoy_series(obs: list[dict]) -> list[dict]:
    """Year-over-year % change series from a monthly price-index observation list.

    Each output point is {"date", "value"} with
        value = 100 * (level_t - level_{t-12}) / level_{t-12}
    Mirrors the seed-side derivation (pct_change(12) * 100) exactly, so a derived
    indicator carries identical units in the vector store and the live snapshot.
    Months without a 12-months-prior point (or with a zero base) are skipped.
    """
    out: list[dict] = []
    for i in range(12, len(obs)):
        prev = float(obs[i - 12]["value"])
        if prev == 0:
            continue
        cur = float(obs[i]["value"])
        out.append({"date": obs[i]["date"], "value": (cur - prev) / prev * 100.0})
    return out


class NarrativeFinder(Protocol):
    def __call__(self, query_text: str, n: int, *,
                 source_filter: FedSource | None = None,
                 as_of: str | None = None) -> list[dict]: ...

class MacroDataClient(Protocol):
    """The slice of the FRED tool (fred_query.FredMCPClient) the agent depends on."""
    async def get_macro_indicators(
        self, series_ids: list[str], start_date: str, end_date: str | None = None
    ) -> dict: ...


class BranchScorer(Protocol):
    """score_branch's true signature: current_momentum is keyword-only.

    classify_regime_tot expects a 3-positional-arg scorer, so the agent binds
    current_momentum (functools.partial) before passing it down -- the same
    adapter the classify_regime_tot __main__ demo uses.
    """
    def __call__(
        self,
        hypothesis: RegimeHypothesis,
        analogs: list[dict],
        snapshot: MacroSnapshot,
        *,
        current_momentum: dict[str, float],
    ) -> BranchResult: ...


@dataclass
class MacroResult:
    """What the macro agent hands the coordinator / synthesis agent."""
    snapshot: MacroSnapshot     # normalized inputs, keyed by const.INDICATOR_KEYS
    regime: const.Regime        # the winning regime
    analogs: list[dict]         # winning regime's analogs (carry subsequent_sector_returns)
    low_confidence: bool        # propagated from the ToT (top branches too close)
    tot_result: ToTResult       # full ToT output incl. pruned branches, for the audit log
    series_history: dict[str, list[dict]] = field(default_factory=dict)
    # ^ per-indicator history in each indicator's OWN units ({INDICATOR_KEY: [{date,value}]}),
    #   retained for the audit layer's statistical checker (z-score / IQR / freshness).
    fed_narrative: list[dict] = field(default_factory=list)
    # ^ RAG data from published Fed documents
    series_meta: dict[str, dict] = field(default_factory=dict)
    # ^ per-indicator provenance ({INDICATOR_KEY: {"latest_observation", "release_date"}}),
    #   for the brief's macro-snapshot table -- observation PERIOD date vs FRED RELEASE date.


class MacroAgent:
    """
    Retrieve macro data and classify the current regime.

    Dependency injection (all collaborators supplied at construction)
    -----------------------------------------------------------------
    data_client
        Anything satisfying MacroDataClient -- fred_query.FredMCPClient in
        production, a fake in tests.
    generate_hypotheses, find_historical_analogs, score_branch
        The three ToT collaborators, forwarded to classify_regime_tot. score_branch
        is the raw scorer (current_momentum keyword-only); run() binds the momentum.

    Why current_momentum is a run() argument rather than fetched here: the ToT
    branch scorer compares each regime's expected sector tilts against CURRENT
    sector momentum, which is the equity agent's product. The coordinator runs the
    equity agent and passes its momentum in, so the two agents stay decoupled and
    momentum is computed once.
    """

    def __init__(
        self,
        data_client: MacroDataClient,
        *,
        generate_hypotheses: HypothesisGenerator,
        find_historical_analogs: AnalogFinder,
        score_branch: BranchScorer,
        find_fed_narrative: NarrativeFinder | None = None,
    ) -> None:
        self._data = data_client
        self._generate_hypotheses = generate_hypotheses
        self._find_historical_analogs = find_historical_analogs
        self._score_branch = score_branch
        self._find_fed_narrative = find_fed_narrative
    async def run(
        self,
        as_of: str,
        current_momentum: dict[str, float],
        *,
        lookback_start: str = const.HISTORY_SEED_START,
    ) -> MacroResult:
        """
        Produce a classified macro regime for the given as-of date.

        Parameters
        ----------
        as_of
            ISO date the snapshot is anchored to (the report's "as of").
        current_momentum
            Per-sector momentum from the equity agent, e.g. {"XLK": 0.08, ...},
            bound into score_branch for the ToT's signal-consistency check.
        lookback_start
            How far back to pull FRED history (defaults to the seed start so the
            YoY derivation has its warm-up window).

        Returns
        -------
        MacroResult
        """
        # --- Step 1: fetch raw macro series from FRED ------------------------
        logger.info("Macro agent run: as_of=%s, lookback_start=%s", as_of, lookback_start)
        codes = [const.SUPPORTED_SERIES[k] for k in _DIRECT_KEYS]
        codes.extend(_DERIVED_CODES.values())            # CPIAUCSL, PCEPI — inflation series derived below

        result = await self._data.get_macro_indicators(
            series_ids=codes,
            start_date=lookback_start,
            end_date=as_of,            # point-in-time: no observations after as_of
        )
        macro_series = result["series"]   # {fred_code: {"observations": [...], "stale": bool, ...}}

        # --- Step 2: reduce to a point-in-time MacroSnapshot -----------------
        indicators: dict[str, float] = {}
        for key in _DIRECT_KEYS:
            code = const.SUPPORTED_SERIES[key]
            obs = macro_series.get(code, {}).get("observations")
            if not obs:
                logger.error("FRED returned no observations for %s (%s)", key, code)
                raise ValueError(f"FRED returned no observations for {key} ({code})")
            indicators[key] = float(obs[-1]["value"])

        # cpi_inflation / pce_inflation: YoY % change of their underlying price-index
        # levels (CPIAUCSL, PCEPI). Derived identically to the seed side; the latest
        # YoY point is the snapshot value and the full YoY series is kept for the
        # audit layer (built once here, reused for series_history below).
        derived_history: dict[str, list[dict]] = {}
        for key, code in _DERIVED_CODES.items():
            obs = macro_series.get(code, {}).get("observations", [])
            if len(obs) < 13:
                logger.error("Insufficient history (%d obs) to derive YoY %s (%s)", len(obs), key, code)
                raise ValueError(
                    f"Insufficient history ({len(obs)} obs) to derive YoY {key} ({code})"
                )
            yoy = _yoy_series(obs)
            indicators[key] = yoy[-1]["value"]
            derived_history[key] = yoy

        # invariant: every INDICATOR_KEY is populated (don't emit a NaN-bearing vector)
        missing = set(const.INDICATOR_KEYS) - set(indicators)
        if missing:
            logger.error("Snapshot missing indicators: %s", sorted(missing))
            raise ValueError(f"snapshot missing indicators: {sorted(missing)}")

        # Retain per-indicator history (each in its OWN units) so the audit layer's
        # statistical checker can run z-score / IQR / freshness over it. Direct keys
        # carry their raw FRED observations; cpi_inflation and pce_inflation carry
        # their DERIVED YoY series (built in Step 2), so the checker compares
        # like-for-like rather than a YoY value against a raw index level.
        series_history: dict[str, list[dict]] = {}
        for key in _DIRECT_KEYS:
            code = const.SUPPORTED_SERIES[key]
            series_history[key] = macro_series.get(code, {}).get("observations", [])
        series_history.update(derived_history)   # cpi_inflation, pce_inflation as YoY

        # Per-indicator provenance for the brief: the newest observation's PERIOD date and
        # FRED's actual RELEASE date, so the macro-snapshot table can show that current
        # data isn't stale, just period-dated. Derived inflation keys borrow their
        # underlying index's release date (their YoY period date comes from series_history).
        series_meta: dict[str, dict] = {}
        for key in const.INDICATOR_KEYS:
            code = const.SUPPORTED_SERIES.get(key) or _DERIVED_CODES.get(key)
            payload = macro_series.get(code, {}) if code else {}
            hist = series_history.get(key) or []
            series_meta[key] = {
                "latest_observation": hist[-1].get("date") if hist else None,
                "release_date": payload.get("release_date"),
            }

        macro_snapshot = MacroSnapshot(as_of=as_of, indicators=indicators)
        logger.debug("Macro snapshot indicators: %s", indicators)

        # --- Step 3: classify the regime via the bounded ToT -----------------
        bound_score = partial(self._score_branch, current_momentum=current_momentum)
        tot = classify_regime_tot(
            macro_snapshot,
            generate_hypotheses=self._generate_hypotheses,
            find_historical_analogs=self._find_historical_analogs,
            score_branch=bound_score,
        )

        # --- Step 4: retrieve the winning regime's analogs for synthesis -----
        regime = tot.selected.hypothesis.regime
        logger.info("Regime classified: %s (low_confidence=%s)", regime.value, tot.low_confidence)
        analogs = self._find_historical_analogs(macro_snapshot, const.ANALOGS_PER_BRANCH, regime)
        logger.info("Retrieved %d analog(s) for regime %s", len(analogs), regime.value)


        # --- Step 5: attach narrative from Fed RAG  --------------------------
        fed_narrative: list[dict] = []
        if self._find_fed_narrative is not None:
            query = f"{regime.value.replace('_', ' ')} regime: " + ", ".join(
                f"{k}={indicators[k]:.2f}" for k in
                ("fed_funds_rate", "cpi_inflation", "unemployment", "yield_spread_10_2")
            )
            try:
                fed_narrative = self._find_fed_narrative(query, const.FED_NARRATIVE_TOP_K, as_of=as_of)
                logger.info("Fed-narrative retrieval returned %d passage(s)", len(fed_narrative))
            except Exception:
                logger.exception("Fed-narrative retrieval failed; continuing without it")

        return MacroResult(
            snapshot=macro_snapshot,
            regime=regime,
            analogs=analogs,
            low_confidence=tot.low_confidence,
            tot_result=tot,
            series_history=series_history,
            fed_narrative=fed_narrative,
            series_meta=series_meta,
        )
