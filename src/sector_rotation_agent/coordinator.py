"""
coordinator.py

The coordinator agent (spec Section 3.1 / Section 4) -- the top-level orchestrator
and the ReAct driver for the whole system. Given an analyst query it:

  1. decomposes the query into a concrete plan (horizon parsed from the query;
     as-of date and sector universe passed through),       # deterministic or LLM
  2. runs the equity agent, then the macro agent (equity FIRST, because the macro
     ToT's signal-consistency check needs the equity agent's current_momentum),
  3. routes each agent's results through the audit layer -- statistical checker
     then critic (spec Section 3.2),                        # SEAM: hook, stubbed
  4. on a raised flag, revises and re-runs, bounded by max_revision_cycles and the
     max_tool_calls cap (spec Section 9, guardrail #3),
  5. scores the sectors (compute_sector_score) once results are clean, and
  6. assembles the cited brief (generate_report).           # SEAM: tool, stubbed

BUILT today: the two retrieval agents, the sector scorer, the audit layer, the report
generator, and query decomposition (a deterministic horizon parser by default, or an
injected LLM decomposer -- see main.py).

Every collaborator is injected, so the coordinator imports no MCP/LLM/network code
and is unit-testable with fakes (see test_coordinator). The composition root that
wires the REAL clients -> agents -> coordinator is main.py.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, TYPE_CHECKING

import sector_rotation_agent.constants as const
from sector_rotation_agent.macro_agent import MacroResult
from sector_rotation_agent.equity_agent import EquityResult
from sector_rotation_agent.synthesize import build_sources
from sector_rotation_agent.trace import NullTrace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sector_rotation_agent.trace import TraceLogger


# --------------------------------------------------------------------------- #
# Query decomposition helpers (deterministic; no LLM) -- the _decompose_query seam
# --------------------------------------------------------------------------- #
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_UNIT_MONTHS = {"month": 1, "mo": 1, "quarter": 3, "qtr": 3, "year": 12, "yr": 12}

# A count (digits or one..twelve) followed by a unit -- tolerant of "6 months",
# "6-month", "six months". "a"/"an" are deliberately excluded so incidental phrases
# like "a quarter of the market" don't read as a horizon.
_HORIZON_NUM_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"[\s-]+(months?|mo|quarters?|qtr|years?|yrs?)\b"
)
# A bare "next/coming/following <unit>" with no number -> one of that unit.
_HORIZON_CUE_RE = re.compile(r"\b(?:next|coming|following)[\s-]+(month|quarter|year)\b")


def _parse_horizon(query: str) -> str | None:
    """Best-effort, deterministic extraction of an investment horizon from the query,
    normalized to '<N> months' (e.g. 'over the next 6 months' -> '6 months', 'next
    year' -> '12 months', 'next quarter' -> '3 months'). Returns None when no clear
    horizon phrase is present -- intentionally conservative: a miss leaves the plan's
    horizon unset rather than guessing. A future LLM decomposer can supersede this."""
    if not query:
        return None
    q = query.lower()

    m = _HORIZON_NUM_RE.search(q)
    if m:
        count_token, unit_token = m.group(1), m.group(2)
        count = int(count_token) if count_token.isdigit() else _WORD_NUMBERS[count_token]
        months = count * _UNIT_MONTHS[unit_token.rstrip("s")]
        return f"{months} months"

    m = _HORIZON_CUE_RE.search(q)
    if m:
        return f"{_UNIT_MONTHS[m.group(1)]} months"

    return None


# --------------------------------------------------------------------------- #
# Sector-focus parsing (deterministic) -- the query "focus" sub-universe seam
# --------------------------------------------------------------------------- #
# A query can ask to rank only a SUBSET of the 11 sectors ("which DEFENSIVE sectors
# ..."). We resolve that to a tuple of tickers (the focus), which the scorer ranks over;
# the full universe is still fetched and fed to the macro ToT, so the focus never moves
# the regime call -- only which sectors appear in the ranking.
_FOCUS_KEYWORDS = {
    "defensive": "defensive",
    "defensives": "defensive",
    "cyclical": "cyclical",
    "cyclicals": "cyclical",
    "rate-sensitive": "rate_sensitive",
    "rate sensitive": "rate_sensitive",
    "interest-rate": "rate_sensitive",
    "interest rate": "rate_sensitive",
    "growth": "growth",
}
_TICKER_RE = re.compile(r"\b(XL[A-Z])\b")


def _parse_focus(query: str) -> tuple[str, ...] | None:
    """Best-effort, deterministic extraction of a sector sub-universe to RANK, as a tuple
    of tickers, or None when the question names no subset (rank all 11). Explicit sector
    tickers in the query (e.g. 'XLU and XLP') win; otherwise a recognized group keyword
    ('defensive', 'cyclicals', 'rate-sensitive', 'growth') maps through
    const.SECTOR_GROUPS. Conservative: an unrecognized or absent focus leaves the universe
    full rather than guessing. The LLM decomposer can supersede this."""
    if not query:
        return None
    found = {m.group(1) for m in _TICKER_RE.finditer(query.upper())}
    explicit = tuple(t for t in const.SECTOR_ETFS_LIST if t in found)
    if explicit:
        return explicit
    q = query.lower()
    for kw, group in _FOCUS_KEYWORDS.items():
        if kw in q:
            return const.SECTOR_GROUPS[group]
    return None


def _resolve_focus(value: object) -> tuple[str, ...] | None:
    """Coerce a decomposer-supplied focus to a validated tuple of known tickers in
    canonical (SECTOR_ETFS_LIST) order, or None. Accepts either a group NAME (mapped via
    const.SECTOR_GROUPS) or a LIST of tickers; unknown groups/tickers are dropped and an
    empty result becomes None (rank all). This validates whatever the LLM decomposer
    returns before it can reach the scorer."""
    if value is None:
        return None
    if isinstance(value, str):
        return const.SECTOR_GROUPS.get(value.strip().lower())
    if isinstance(value, (list, tuple)):
        wanted = {str(t).strip().upper() for t in value}
        picked = tuple(t for t in const.SECTOR_ETFS_LIST if t in wanted)
        return picked or None
    return None


# --------------------------------------------------------------------------- #
# Query decomposition -- LLM seam (the deterministic parser above is the fallback)
# --------------------------------------------------------------------------- #
# When a model is injected (llm_decompose_query, wired in main.py) it extracts the
# horizon more robustly than the regex can -- "through the back half of next year", "a
# couple of quarters" -- and we fall back to the regex, then to None, on any failure.
# as_of is deliberately NOT taken from the query: it stays an explicit parameter so the
# run's point-in-time / no-lookahead discipline can't be moved by free text.
DECOMPOSE_SYSTEM_PROMPT = (
    "You extract structured parameters from an equity analyst's question. "
    "Return ONLY a JSON object, no prose and no markdown fences, with exactly these keys:\n"
    '  "horizon_months": the investment horizon the question asks about, as an integer '
    "number of months, or null if the question does not state one. "
    "Convert units: quarters x3, years x12 (e.g. 'next year' -> 12, 'a couple of "
    "quarters' -> 6, 'through the next 18 months' -> 18).\n"
    '  "focus": the subset of sectors the question asks to rank, if any. Either one of the '
    'group names "defensive", "cyclical", "rate_sensitive", "growth", OR a JSON list of '
    "GICS sector ETF tickers (from XLK, XLF, XLE, XLV, XLI, XLU, XLP, XLY, XLB, XLRE, XLC). "
    "Use null when the question asks about sectors generally rather than a subset.\n"
    'Example: {"horizon_months": 6, "focus": "defensive"}'
)


def _strip_json_fence(raw: str) -> str:
    """Drop a ```json ... ``` fence if the model wrapped its JSON in one."""
    return raw.replace("```json", "").replace("```", "").strip()


def _normalize_horizon_months(value: object) -> str | None:
    """Coerce a model-supplied month count to the canonical '<N> months', or None.
    Guards the range (1..120) so a nonsense value degrades to 'unset' rather than
    flowing downstream."""
    try:
        months = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f"{months} months" if 1 <= months <= 120 else None


def deterministic_decompose(
    query: str, *, as_of: str, tickers: tuple[str, ...], period: str
) -> QueryPlan:
    """The no-LLM decomposer (the Coordinator's default): packages the explicit
    parameters and parses the horizon from the query with the regex above."""
    return QueryPlan(
        as_of=as_of,
        tickers=tuple(tickers),
        period=period,
        horizon=_parse_horizon(query),
        focus=_parse_focus(query),
    )


def llm_decompose_query(
    query: str,
    *,
    as_of: str,
    tickers: tuple[str, ...],
    period: str,
    call_model: Callable[[str, str], str] | None,
) -> QueryPlan:
    """LLM decomposer (wired in main.py via functools.partial binding call_model).

    Starts from the deterministic plan, then asks the model for a structured horizon and
    overrides only when the model returns a usable value. Any failure -- model error,
    non-JSON, out-of-range -- silently falls back to the deterministic plan, so the loop
    never breaks on decomposition. as_of / tickers / period pass through unchanged; only
    the horizon is model-derived today (the JSON shape is structured so universe /
    scenario fields can be added later)."""
    base = deterministic_decompose(query, as_of=as_of, tickers=tickers, period=period)
    if not query or not call_model:
        return base
    try:
        raw = call_model(DECOMPOSE_SYSTEM_PROMPT, query)
        data = json.loads(_strip_json_fence(raw))
        horizon = _normalize_horizon_months(data.get("horizon_months"))
        focus = _resolve_focus(data.get("focus"))
    except Exception:
        logger.warning("LLM query decomposition failed; using deterministic parse.", exc_info=True)
        return base
    return replace(
        base,
        horizon=horizon if horizon is not None else base.horizon,
        focus=focus if focus is not None else base.focus,
    )


# --------------------------------------------------------------------------- #
# Injected collaborators (Protocols -> any object with the right shape fits)
# --------------------------------------------------------------------------- #
class MacroRunner(Protocol):
    async def run(self, as_of: str, current_momentum: dict[str, float]) -> MacroResult: ...


class EquityRunner(Protocol):
    async def run(self, tickers: tuple[str, ...], *, period: str = "5y", as_of: str | None = None) -> EquityResult: ...


class SectorScorer(Protocol):
    """compute_sector_score's call shape."""
    def __call__(
        self, macro_regime: const.Regime, analog_data: list[dict], equity_data: dict[str, dict],
        *, horizon: str | None = None, universe: tuple[str, ...] | None = None,
    ) -> list[dict]: ...


class AuditLayer(Protocol):
    """Reviews one labeled tool result against the working plan; returns any flags.

    The real implementation will compose check_statistical_anomaly (pure Python,
    unconditional) then run_critic_check (narrow LLM) per spec Section 3.2. The
    default (_NullAudit) raises nothing -- the happy path -- so the coordinator is
    constructible before the audit layer exists.
    """
    def review(self, label: str, result: object, plan: QueryPlan) -> list[AuditFlag]: ...


class ReportGenerator(Protocol):
    """generate_report's call shape (spec Section 5)."""
    def __call__(
        self,
        *,
        query: str,
        regime: const.Regime,
        as_of: str,
        rankings: list[dict],
        confidence: float,
        flags: list[AuditFlag],
        audit_log: AuditLog,
        sources: list[dict],
        analogs: list[dict] | None = None,
        regime_analysis: dict | None = None,
        low_confidence: bool = False,
        series_history: dict | None = None,
        series_meta: dict | None = None,
        horizon: str | None = None,
        universe: tuple[str, ...] | None = None,
    ) -> str | None: ...


class QueryDecomposer(Protocol):
    """The query-decomposition seam (spec Section 3.1). Maps the analyst query plus the
    explicit run parameters to a QueryPlan. Default is deterministic_decompose (regex
    horizon parse); main.py injects llm_decompose_query (call_model bound) for the LLM
    version."""
    def __call__(
        self, query: str, *, as_of: str, tickers: tuple[str, ...], period: str
    ) -> QueryPlan: ...


# --------------------------------------------------------------------------- #
# Data carried through a run
# --------------------------------------------------------------------------- #
@dataclass
class QueryPlan:
    """The decomposed query: what the deterministic pipeline actually needs."""
    as_of: str
    tickers: tuple[str, ...]
    period: str
    horizon: str | None = None      # SEAM: filled by the LLM decomposer later
    focus: tuple[str, ...] | None = None   # sub-universe to RANK (None = all 11); the
    #                                        FETCH stays full, so the macro ToT is unaffected


@dataclass
class AuditFlag:
    """One concern raised by the audit layer."""
    source: str         # "statistical" | "critic"
    label: str          # what was under review (e.g. "macro", "equity", a series id)
    message: str


@dataclass
class AuditLog:
    """Append-only record of the run (spec Section 6.1 / guardrail #7).

    Tool calls are logged via record_tool_call (which also advances the tool_calls
    counter); decision events go through record. tool_calls is kept as an independent
    counter -- not merely len(entries) -- so reconcile() can verify at session end
    that the logged tool-call entries and the counter agree, which is guardrail #7's
    "audit-log entries == tool calls".
    """
    entries: list[dict] = field(default_factory=list)
    tool_calls: int = 0

    def record(self, entry: dict) -> None:
        self.entries.append(entry)

    def record_tool_call(self, tool: str, *, cycle: int) -> None:
        """Log one tool invocation AND advance the tool-call counter in lockstep.

        Funneling both through one method is what keeps the counter and the logged
        entries consistent; reconcile() then guards against any future code path that
        bumps one without the other."""
        self.tool_calls += 1
        self.entries.append(
            {"event": "tool_call", "tool": tool, "cycle": cycle, "n": self.tool_calls}
        )

    def reconcile(self) -> dict:
        """Guardrail #7: at session end, verify the log accounts for every tool call.

        Counts the logged tool-call entries and compares them with the independently
        maintained tool_calls counter, then records and returns the outcome. A
        mismatch means a tool ran without being logged (or vice versa) -- a provenance
        gap -- so it is recorded and logged as a warning rather than passing silently."""
        logged = sum(1 for e in self.entries if e.get("event") == "tool_call")
        reconciled = logged == self.tool_calls
        result = {
            "event": "reconciliation",
            "tool_calls": self.tool_calls,
            "logged_tool_calls": logged,
            "reconciled": reconciled,
        }
        self.record(result)
        if not reconciled:
            logger.warning(
                "Audit-log reconciliation FAILED (guardrail #7): %d tool calls but "
                "%d logged tool-call entries.",
                self.tool_calls, logged,
            )
        return result


@dataclass
class CoordinatorResult:
    """The coordinator's final output."""
    query: str
    regime: const.Regime
    rankings: list[dict]            # compute_sector_score output, ranked best-first
    low_confidence: bool            # macro ToT low-confidence OR unresolved audit flags
    flags: list[AuditFlag]          # any flags still standing at the end of the run
    macro: MacroResult
    equity: EquityResult
    audit_log: AuditLog
    report: str | None = None       # generate_report output once that tool exists


# --------------------------------------------------------------------------- #
# No-op defaults for the not-yet-built seams
# --------------------------------------------------------------------------- #
class _NullAudit:
    """Default audit layer: passes everything (no flags). Replace once the real
    statistical checker + critic exist."""
    def review(self, label: str, result: object, plan: QueryPlan) -> list[AuditFlag]:
        return []


def _null_report(
    *,
    query: str,
    regime: const.Regime,
    as_of: str,
    rankings: list[dict],
    confidence: float,
    flags: list[AuditFlag],
    audit_log: AuditLog,
    sources: list[dict],
    analogs: list[dict] | None = None,
    regime_analysis: dict | None = None,
    low_confidence: bool = False,
    series_history: dict | None = None,
    series_meta: dict | None = None,
    horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
) -> str | None:
    """Default report generator: produces nothing until generate_report is built."""
    return None


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class Coordinator:
    """
    Orchestrates one research run end-to-end and drives the ReAct revision loop.

    Dependency injection
    --------------------
    macro_agent, equity_agent
        The two retrieval agents (MacroAgent / EquityAgent in production, fakes in
        tests). Equity runs first; its current_momentum feeds the macro ToT.
    score_sectors
        compute_sector_score (injected so the coordinator needn't import the scipy /
        score_branch chain, and so tests can pass a fake).
    audit
        The audit layer (spec Section 3.2). Defaults to a no-op pass-through.
    generate_report
        The report tool (spec Section 5). Defaults to a no-op.
    decompose
        The query-decomposition seam (spec Section 3.1). Defaults to
        deterministic_decompose (regex horizon parse); main.py injects
        llm_decompose_query for the LLM version.
    max_tool_calls, max_revision_cycles
        Safety caps (spec Section 9, guardrail #3): 20 tool calls / 3 revisions.
        Defaults live here for now; promote to constants.py if reused elsewhere.
    """

    def __init__(
        self,
        macro_agent: MacroRunner,
        equity_agent: EquityRunner,
        score_sectors: SectorScorer,
        *,
        audit: AuditLayer | None = None,
        generate_report: ReportGenerator | None = None,
        decompose: QueryDecomposer | None = None,
        max_tool_calls: int = 20,
        max_revision_cycles: int = 3,
        trace: "TraceLogger | None" = None,
    ) -> None:
        self._macro = macro_agent
        self._equity = equity_agent
        self._score = score_sectors
        self._audit: AuditLayer = audit if audit is not None else _NullAudit()
        self._generate_report: ReportGenerator = (
            generate_report if generate_report is not None else _null_report
        )
        self._decompose: QueryDecomposer = (
            decompose if decompose is not None else deterministic_decompose
        )
        self._max_tool_calls = max_tool_calls
        self._max_revision_cycles = max_revision_cycles
        # Telemetry seam (trace.TraceLogger). Null-object default so every call site can
        # use self._trace.span(...) / .event(...) unconditionally; main.py injects the
        # real one. Distinct from AuditLog: developer telemetry, not the audited record.
        self._trace = trace if trace is not None else NullTrace()

    def _decompose_query(
        self, query: str, *, as_of: str, tickers: tuple[str, ...], period: str
    ) -> QueryPlan:
        """Turn the analyst query into a concrete plan via the injected decomposer
        (deterministic by default; an LLM decomposer in production -- see main.py).

        The horizon is the field the decomposer derives from the question text; as_of,
        tickers, and period pass through unchanged. as_of in particular is never read
        from free text, preserving the run's point-in-time / no-lookahead discipline.
        The focus sub-universe (which sectors to RANK) is also derived from the question
        when present; reading scenario assumptions remains deferred (section 10.1).
        """
        return self._decompose(query, as_of=as_of, tickers=tuple(tickers), period=period)

    async def run(
        self,
        query: str,
        *,
        as_of: str,
        tickers: tuple[str, ...] = const.SECTOR_ETFS_LIST,
        period: str = "5y",
    ) -> CoordinatorResult:
        """
        Drive one research run: decompose -> (equity, macro) -> audit -> revise? ->
        score -> report.

        Parameters
        ----------
        query
            The analyst's natural-language request (carried through for provenance;
            full LLM decomposition is a later seam).
        as_of
            The point-in-time date the run is anchored to.
        tickers, period
            Sector universe and lookback window, forwarded to the agents.

        Returns
        -------
        CoordinatorResult
        """
        # make sure we got a query to process
        if not query:
            raise ValueError("Received an empty query string, can't do any work")
        
        # Decompose the incoming query and create a plan
        self._trace.event("coordinator", "run_start", query=query, as_of=as_of,
                          tickers=list(tickers), period=period)
        with self._trace.span("coordinator", "decompose"):
            query_plan = self._decompose_query(
                query=query,
                as_of=as_of,
                tickers=tickers,
                period=period,
            )
        self._trace.event("coordinator", "query_plan", horizon=query_plan.horizon,
                          focus=list(query_plan.focus) if query_plan.focus else None,
                          as_of=query_plan.as_of)

        # initialize the audit log
        audit_log: AuditLog = AuditLog()
        audit_flags: list[AuditFlag] = []
        macro_agent: MacroResult | None = None
        equity_agent: EquityResult | None = None

        # ------------   ReAct loop (decompose -> retrieve -> audit -> revise?) ------
        plan = query_plan  # working plan; a revision narrows the sector universe
        for cycle in range(0, self._max_revision_cycles + 1):

            # run the equity agent first (must be before macro agent)
            audit_log.record_tool_call("equity agent", cycle=cycle)
            with self._trace.span("coordinator", "equity_agent.run", cycle=cycle):
                equity_agent = await self._equity.run(
                    tickers=plan.tickers,
                    period=plan.period,
                    as_of=plan.as_of,
                )

            # with equity agent done, can now run the macro agent
            audit_log.record_tool_call("macro agent", cycle=cycle)
            with self._trace.span("coordinator", "macro_agent.run", cycle=cycle):
                macro_agent = await self._macro.run(
                    as_of=plan.as_of,
                    current_momentum=equity_agent.current_momentum,
                )

            # audit both results (statistical checker then critic, per spec 3.2)
            with self._trace.span("coordinator", "audit.review", cycle=cycle):
                audit_flags = (
                    self._audit.review(label="equity agent", result=equity_agent, plan=plan)
                    + self._audit.review(label="macro agent", result=macro_agent, plan=plan)
                )

            if not audit_flags:
                audit_log.record({"event": "audit_clean", "cycle": cycle})
                break  # evidence is clean, proceed to synthesis

            # Decide whether a revision can actually change anything. The only flags a
            # re-run can act on are STATISTICAL flags that name a sector in the current
            # universe: that's a number we distrust, so we quarantine the sector and
            # re-score without it. A macro statistical flag names an indicator, which
            # can't be dropped (the regime needs the whole vector), and every critic
            # flag is a judgment a re-run would only reproduce -- those are carried
            # forward as caveats (they lower confidence and surface in the report).
            suspect = {
                f.label for f in audit_flags
                if f.source == "statistical" and f.label in plan.tickers
            }
            remaining = tuple(t for t in plan.tickers if t not in suspect)
            budget_left = (
                cycle < self._max_revision_cycles
                and audit_log.tool_calls < self._max_tool_calls
            )

            if suspect and remaining and budget_left:
                audit_log.record({
                    "event": "revision",
                    "cycle": cycle,
                    "dropped": sorted(suspect),
                    "flags": [f.label for f in audit_flags],
                })
                logger.info(
                    "Revision %d: quarantining %s and re-running.",
                    cycle + 1, sorted(suspect),
                )
                plan = replace(plan, tickers=remaining)
                self._trace.event("coordinator", "revision", cycle=cycle,
                                  dropped=sorted(suspect), remaining=list(remaining))
                
                continue  # re-run with the narrowed universe

            # Nothing a re-run can fix (only macro / critic flags), dropping would empty
            # the universe, or we're out of budget -> stop and carry the flags forward.
            reason = (
                "budget" if not budget_left
                else "no_actionable_flags" if not suspect
                else "would_empty_universe"
            )
            audit_log.record({
                "event": "revision_halt",
                "cycle": cycle,
                "reason": reason,
                "flags": [f.label for f in audit_flags],
            })
            logger.info(
                "Audit raised %d flag(s); halting revision (%s) and carrying them forward.",
                len(audit_flags), reason,
            )
            break
        # ------------ end of ReAct loop --------------------------------------------

        # Guardrail #7: reconcile the audit log against the tool-call counter at
        # session end -- before anything consumes the log (the report surfaces the
        # reconciliation outcome in its audit trail).
        audit_log.reconcile()

        # ------------ Construct the output --------------------------------------------------
        # first, make sure we at least ran once
        assert macro_agent is not None and equity_agent is not None

        # The ranking universe is the requested focus sub-universe (plan.focus) when the
        # query named one, else all 11 -- intersected with plan.tickers so a sector
        # quarantined out of the fetch is dropped from the ranking too (focus and
        # quarantine compose). Falls back to the bare focus if that intersection is empty
        # (every focus sector was quarantined), so a focused query still ranks what it
        # asked for, with the quarantine flags carried forward as caveats.
        ranking_universe = tuple(
            t for t in (plan.focus or const.SECTOR_ETFS_LIST) if t in plan.tickers
        ) or plan.focus or const.SECTOR_ETFS_LIST

        # score the result to get rank-ordered data scores (horizon selects the analog
        # forward-return window; falls back to the default when unset or not seeded;
        # universe restricts the ranking to the requested focus sub-universe)
        with self._trace.span("coordinator", "score_sectors", universe=list(ranking_universe)):
            rankings = self._score(
                macro_agent.regime, macro_agent.analogs, equity_agent.equity_data,
                horizon=plan.horizon, universe=ranking_universe,
            )

        # compute a confidence (mean)
        confidence = sum(r["confidence"] for r in rankings) / len(rankings) if rankings else 0.0
        
        with self._trace.span("coordinator", "generate_report"):
            report_out = self._generate_report(
                query=query,
                regime=macro_agent.regime,
                as_of=macro_agent.snapshot.as_of,
                rankings=rankings,
                confidence=confidence,
                flags=audit_flags,
                audit_log=audit_log,
                sources=build_sources(
                        as_of=macro_agent.snapshot.as_of,
                        indicators=macro_agent.snapshot.indicators,
                        analogs=macro_agent.analogs,
                        equity_data=equity_agent.equity_data,
                        fed_narrative=macro_agent.fed_narrative),
                analogs=macro_agent.analogs,
                regime_analysis=macro_agent.tot_result.audit_entry,
                low_confidence=macro_agent.low_confidence,
                series_history=macro_agent.series_history,
                series_meta=macro_agent.series_meta,
                horizon=plan.horizon,
                universe=plan.focus,
            )

        self._trace.event("coordinator", "run_complete", regime=macro_agent.regime.value,
                          confidence=round(confidence, 4), n_flags=len(audit_flags),
                          low_confidence=bool(macro_agent.low_confidence or audit_flags))
        return CoordinatorResult(
            query=query,
            regime=macro_agent.regime,
            rankings=rankings,
            low_confidence=macro_agent.low_confidence or bool(audit_flags) ,
            flags=audit_flags,
            macro=macro_agent,
            equity=equity_agent,
            audit_log=audit_log,
            report=report_out,
        )
