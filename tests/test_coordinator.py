"""
test_coordinator.py

Tests for the coordinator (src/sector_rotation_agent/coordinator.py).

Status when written:
  * Coordinator.run is a STUB (raises NotImplementedError). Written against the
    documented ReAct contract and marked xfail(strict=False) below, so the suite
    stays green now and flips to XPASS once run() is implemented.

run() is async, so the tests drive it with asyncio.run(...). Every collaborator is
a fake injected at construction: fake macro/equity runners (returning prebuilt
MacroResult/EquityResult), a fake sector scorer that records its arguments, and --
for the revision test -- an audit layer that always flags. This isolates the
coordinator's orchestration logic from the agents and the scorer.
"""
from __future__ import annotations

import asyncio

import pytest

import sector_rotation_agent.coordinator as co
import sector_rotation_agent.constants as const
from sector_rotation_agent.classify_regime_tot import (
    BranchResult,
    MacroSnapshot,
    RegimeHypothesis,
    ToTResult,
)
from sector_rotation_agent.macro_agent import MacroResult
from sector_rotation_agent.equity_agent import EquityResult

# --------------------------------------------------------------------------- #
# Prebuilt agent outputs
# --------------------------------------------------------------------------- #
def _macro_result(regime: const.Regime = const.Regime.MID_CYCLE, low_conf: bool = False) -> MacroResult:
    hyp = RegimeHypothesis(regime, "rationale", 0.6)
    branch = BranchResult(hypothesis=hyp, analog_similarity=0.8, signal_consistency=0.7, support_score=0.75)
    tot = ToTResult(selected=branch, branches=[branch], low_confidence=low_conf, audit_entry={})
    analogs = [{"similarity": 0.85, "subsequent_sector_returns": {t: 0.02 for t in const.SECTOR_ETFS_LIST}}]
    return MacroResult(
        snapshot=MacroSnapshot(as_of="2026-06-01", indicators={k: 0.0 for k in const.INDICATOR_KEYS}),
        regime=regime,
        analogs=analogs,
        low_confidence=low_conf,
        tot_result=tot,
    )


def _equity_result() -> EquityResult:
    return EquityResult(
        equity_data={t: {"momentum": 0.0, "valuation": 20.0} for t in const.SECTOR_ETFS_LIST},
        current_momentum={t: 0.0 for t in const.SECTOR_ETFS_LIST},
        raw={},
    )


# --------------------------------------------------------------------------- #
# Fake collaborators
# --------------------------------------------------------------------------- #
class FakeMacroRunner:
    def __init__(self, result: MacroResult) -> None:
        self.result = result
        self.calls = 0
        self.last_momentum: dict[str, float] | None = None

    async def run(self, as_of, current_momentum, *, lookback_start=const.HISTORY_SEED_START) -> MacroResult:
        self.calls += 1
        self.last_momentum = current_momentum
        return self.result


class FakeEquityRunner:
    def __init__(self, result: EquityResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self, tickers=const.SECTOR_ETFS_LIST, *, period="5y", as_of=None) -> EquityResult:
        self.calls += 1
        if not tickers:                      # mirrors the real EquityAgent guard
            raise ValueError("EquityAgent.run requires at least one ticker")
        return self.result


class FakeScorer:
    """Records the args it was called with; returns a canned 11-row ranking."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, macro_regime, analog_data, equity_data, **kw) -> list[dict]:
        self.calls.append((macro_regime, analog_data, equity_data))
        return [
            {"sector": t, "score": 0.5, "rank": i, "confidence": 0.5, "detail": {}}
            for i, t in enumerate(const.SECTOR_ETFS_LIST, start=1)
        ]


class FlaggingAudit:
    """Always raises a flag -- exercises the bounded revision loop."""

    def __init__(self) -> None:
        self.calls = 0

    def review(self, label, result, plan) -> list:
        self.calls += 1
        return [co.AuditFlag(source="critic", label=label, message="contradiction")]


class QuarantineThenCleanAudit:
    """Flags one sector (statistical) while it's still in the universe, then passes.

    Drives a *real* revision: the coordinator should drop the flagged sector and
    re-run, after which this returns clean."""

    def __init__(self, bad: str = "XLK") -> None:
        self.bad = bad

    def review(self, label, result, plan) -> list:
        if label.startswith("equity") and self.bad in plan.tickers:
            return [co.AuditFlag(source="statistical", label=self.bad, message="momentum outlier")]
        return []


class AlwaysFlagsFirstSectorAudit:
    """Each pass flags whichever sector leads the (shrinking) universe, so a revision
    always has something to drop -- exercises the bound under genuine revisions."""

    def review(self, label, result, plan) -> list:
        if label.startswith("equity") and plan.tickers:
            return [co.AuditFlag(source="statistical", label=plan.tickers[0], message="outlier")]
        return []


# --------------------------------------------------------------------------- #
# Valid: the happy path wires equity -> macro -> scorer and assembles a result
# --------------------------------------------------------------------------- #
def test_coordinator_run_orchestrates_pipeline():
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    scorer = FakeScorer()
    coord = co.Coordinator(macro, equity, scorer)

    result = asyncio.run(coord.run("which sectors should I overweight?", as_of="2026-06-01"))

    assert isinstance(result, co.CoordinatorResult)
    assert result.regime is const.Regime.MID_CYCLE
    assert len(result.rankings) == len(const.SECTOR_ETFS_LIST)
    # equity ran first and its momentum was handed to the macro agent
    assert macro.last_momentum == equity.result.current_momentum
    # the scorer was called with the macro regime + the equity agent's equity_data
    assert scorer.calls and scorer.calls[0][0] is const.Regime.MID_CYCLE
    assert scorer.calls[0][2] == equity.result.equity_data
    # clean audit -> not low confidence, no flags
    assert result.low_confidence is False
    assert result.flags == []


# --------------------------------------------------------------------------- #
# Invalid: a sub-agent error (empty universe) propagates, not swallowed
# --------------------------------------------------------------------------- #
def test_coordinator_empty_tickers_raises():
    coord = co.Coordinator(FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), FakeScorer())

    with pytest.raises(ValueError):
        asyncio.run(coord.run("q", as_of="2026-06-01", tickers=()))


# --------------------------------------------------------------------------- #
# ReAct: a persistently flagging audit must bound revisions and degrade gracefully
# --------------------------------------------------------------------------- #
def test_coordinator_revision_is_bounded():
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(macro, equity, FakeScorer(), audit=FlaggingAudit(), max_revision_cycles=2)

    result = asyncio.run(coord.run("q", as_of="2026-06-01"))

    # bounded by max_revision_cycles + 1 -- it does not loop forever
    assert equity.calls <= 3
    assert macro.calls <= 3
    # unresolved flags are surfaced and the run degrades to low confidence
    assert result.flags
    assert result.low_confidence is True


# --------------------------------------------------------------------------- #
# ReAct: a statistical flag naming a sector drives a REAL revision (drop + re-run)
# --------------------------------------------------------------------------- #
def test_coordinator_revises_by_quarantining_flagged_sector():
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(
        macro, equity, FakeScorer(),
        audit=QuarantineThenCleanAudit(bad="XLK"), max_revision_cycles=3,
    )

    result = asyncio.run(coord.run("q", as_of="2026-06-01"))

    # cycle 0 flags XLK; cycle 1 (XLK quarantined) comes back clean -> exactly 2 runs
    assert equity.calls == 2 and macro.calls == 2
    # the revision is recorded with what it dropped, and the run ends clean
    revisions = [e for e in result.audit_log.entries if e.get("event") == "revision"]
    assert revisions and revisions[0]["dropped"] == ["XLK"]
    assert result.flags == []
    assert result.low_confidence is False


# --------------------------------------------------------------------------- #
# ReAct: genuine revisions still terminate -- the universe shrinks, bounded by cap
# --------------------------------------------------------------------------- #
def test_coordinator_real_revision_is_bounded():
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(
        macro, equity, FakeScorer(),
        audit=AlwaysFlagsFirstSectorAudit(), max_revision_cycles=2,
    )

    result = asyncio.run(coord.run("q", as_of="2026-06-01"))

    # cycle 0 drop, cycle 1 drop, cycle 2 out of budget -> at most 3 runs
    assert equity.calls == 3 and macro.calls == 3
    # two revisions happened before the budget halt, dropping distinct sectors
    dropped = [e["dropped"] for e in result.audit_log.entries if e.get("event") == "revision"]
    assert len(dropped) == 2 and dropped[0] != dropped[1]
    # still flagged at the halt -> degrades to low confidence
    assert result.flags
    assert result.low_confidence is True


# --------------------------------------------------------------------------- #
# Guardrail #7: every tool call is logged, and the session-end reconciliation
# entry confirms the tool_calls counter and the logged entries agree
# --------------------------------------------------------------------------- #
def test_coordinator_audit_log_reconciles_clean_run():
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(macro, equity, FakeScorer())

    result = asyncio.run(coord.run("q", as_of="2026-06-01"))
    log = result.audit_log

    # one clean cycle = two tool calls (equity then macro), each logged in order
    assert log.tool_calls == 2
    tool_calls = [e for e in log.entries if e.get("event") == "tool_call"]
    assert [e["tool"] for e in tool_calls] == ["equity agent", "macro agent"]
    # session-end reconciliation entry: counter and logged entries agree
    recon = [e for e in log.entries if e.get("event") == "reconciliation"]
    assert len(recon) == 1
    assert recon[0]["reconciled"] is True
    assert recon[0]["tool_calls"] == recon[0]["logged_tool_calls"] == 2


def test_coordinator_audit_log_reconciles_across_revisions():
    """Tool calls accumulate across revision cycles and the log still reconciles --
    the counter and the logged tool-call entries stay in lockstep through re-runs."""
    macro = FakeMacroRunner(_macro_result())
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(
        macro, equity, FakeScorer(),
        audit=AlwaysFlagsFirstSectorAudit(), max_revision_cycles=2,
    )

    result = asyncio.run(coord.run("q", as_of="2026-06-01"))
    log = result.audit_log

    # cycles 0,1,2 x (equity + macro) = 6 tool calls, all logged, and reconciled
    assert log.tool_calls == 6
    assert sum(1 for e in log.entries if e.get("event") == "tool_call") == 6
    recon = next(e for e in log.entries if e.get("event") == "reconciliation")
    assert recon["reconciled"] is True
    assert recon["logged_tool_calls"] == recon["tool_calls"] == 6


# --------------------------------------------------------------------------- #
# Query decomposition: the horizon is parsed from the question (deterministic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query, expected", [
    ("Which sectors should I overweight over the next 6 months?", "6 months"),
    ("best sectors for the next 12 months", "12 months"),
    ("outlook over a 2 year horizon", "24 months"),
    ("what looks good next quarter?", "3 months"),
    ("which sectors for next year", "12 months"),
    ("give me a six-month sector view", "6 months"),
    ("which sectors should I overweight or underweight?", None),
    ("year-over-year inflation and sectors", None),
    ("", None),
])
def test_parse_horizon(query, expected):
    assert co._parse_horizon(query) == expected


def test_decompose_query_carries_horizon_into_plan():
    """_decompose_query now populates QueryPlan.horizon from the question text while
    still taking as_of / tickers / period as explicit parameters."""
    coord = co.Coordinator(FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), FakeScorer())
    plan = coord._decompose_query(
        query="which sectors over the next 18 months?",
        as_of="2026-06-01",
        tickers=const.SECTOR_ETFS_LIST,
        period="5y",
    )
    assert plan.horizon == "18 months"
    assert plan.as_of == "2026-06-01"
    assert plan.tickers == tuple(const.SECTOR_ETFS_LIST)


# --------------------------------------------------------------------------- #
# LLM decomposition: model-extracted horizon, with deterministic fallback
# --------------------------------------------------------------------------- #
def _decomp(query, model):
    """Run llm_decompose_query with a fake call_model and fixed run parameters."""
    return co.llm_decompose_query(
        query, as_of="2026-06-01", tickers=const.SECTOR_ETFS_LIST,
        period="5y", call_model=model,
    )


def test_llm_decompose_extracts_horizon_from_model():
    """A clean JSON horizon from the model becomes the plan's horizon."""
    assert _decomp("what's the outlook?", lambda s, u: '{"horizon_months": 9}').horizon == "9 months"


def test_llm_decompose_strips_json_fence():
    """A ```json-fenced object is still parsed."""
    assert _decomp("outlook?", lambda s, u: '```json\n{"horizon_months": 18}\n```').horizon == "18 months"


def test_llm_decompose_overrides_regex_when_model_succeeds():
    """The model's horizon wins over the regex parse of the same query (6 -> 12)."""
    assert _decomp("over the next 6 months", lambda s, u: '{"horizon_months": 12}').horizon == "12 months"


def test_llm_decompose_falls_back_to_regex_on_bad_json():
    """Non-JSON output falls back to the deterministic regex parse."""
    assert _decomp("what about the next quarter?", lambda s, u: "sorry, no").horizon == "3 months"


def test_llm_decompose_falls_back_to_regex_on_model_error():
    """A model exception falls back to the deterministic regex parse."""
    def boom(system, user):
        raise RuntimeError("model down")
    assert _decomp("over the next 6 months", boom).horizon == "6 months"


def test_llm_decompose_null_horizon_keeps_regex_result():
    """horizon_months=null means the model found none -> keep the regex result (None here)."""
    assert _decomp("which sectors should I overweight?", lambda s, u: '{"horizon_months": null}').horizon is None


def test_llm_decompose_out_of_range_is_rejected():
    """An absurd month count is rejected by the range guard -> regex fallback (None here)."""
    assert _decomp("which sectors look good?", lambda s, u: '{"horizon_months": 9999}').horizon is None


def test_coordinator_uses_injected_decomposer():
    """The Coordinator routes decomposition through the injected decomposer, and the
    resulting horizon flows into generate_report."""
    captured: dict = {}
    def fake_report(**kwargs):
        captured.update(kwargs)
        return "report"
    def fake_decompose(query, *, as_of, tickers, period):
        return co.QueryPlan(as_of=as_of, tickers=tuple(tickers), period=period, horizon="99 months")

    coord = co.Coordinator(
        FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), FakeScorer(),
        generate_report=fake_report, decompose=fake_decompose,
    )
    asyncio.run(coord.run("anything", as_of="2026-06-01"))
    assert captured["horizon"] == "99 months"


def test_coordinator_passes_horizon_to_scorer():
    """plan.horizon reaches compute_sector_score (the scorer) via the run() call, so the
    ranking can honor the requested window."""
    captured: dict = {}
    def fake_score(macro_regime, analog_data, equity_data, *, horizon=None, universe=None):
        captured["horizon"] = horizon
        captured["universe"] = universe
        return [{"sector": "XLK", "score": 1.0, "rank": 1, "confidence": 0.5, "detail": {}}]
    def fake_decompose(query, *, as_of, tickers, period):
        return co.QueryPlan(as_of=as_of, tickers=tuple(tickers), period=period, horizon="3 months")

    coord = co.Coordinator(
        FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), fake_score,
        decompose=fake_decompose,
    )
    asyncio.run(coord.run("anything", as_of="2026-06-01"))
    assert captured["horizon"] == "3 months"


# --------------------------------------------------------------------------- #
# Query decomposition: the focus sub-universe is parsed from the question
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query, expected", [
    ("which defensive sectors should I overweight?", const.SECTOR_GROUPS["defensive"]),
    ("rank the cyclicals for me", const.SECTOR_GROUPS["cyclical"]),
    ("how do rate-sensitive sectors look?", const.SECTOR_GROUPS["rate_sensitive"]),
    ("which growth sectors lead?", const.SECTOR_GROUPS["growth"]),
    ("should I favor XLU and XLP?", ("XLU", "XLP")),       # explicit tickers, canonical order
    ("is XLK defensive right now?", ("XLK",)),             # explicit ticker wins over keyword
    ("which sectors should I overweight?", None),          # no subset named
    ("", None),
])
def test_parse_focus(query, expected):
    assert co._parse_focus(query) == expected


@pytest.mark.parametrize("value, expected", [
    ("defensive", const.SECTOR_GROUPS["defensive"]),       # group name from the LLM
    ("DEFENSIVE", const.SECTOR_GROUPS["defensive"]),       # case-insensitive
    (["xlp", "xlu"], ("XLU", "XLP")),                      # ticker list -> canonical order
    (["XLU", "BOGUS"], ("XLU",)),                          # unknown tickers dropped
    (["NOPE"], None),                                      # nothing valid -> None (rank all)
    ("not_a_group", None),
    (None, None),
])
def test_resolve_focus(value, expected):
    assert co._resolve_focus(value) == expected


def test_llm_decompose_extracts_focus_group():
    """A group name in the model's JSON resolves to that group's tickers."""
    plan = _decomp("what should I buy?", lambda s, u: '{"horizon_months": 6, "focus": "defensive"}')
    assert plan.focus == const.SECTOR_GROUPS["defensive"]
    assert plan.horizon == "6 months"


def test_llm_decompose_extracts_focus_ticker_list():
    """An explicit ticker list in the model's JSON is validated to canonical order."""
    plan = _decomp("rank these", lambda s, u: '{"horizon_months": null, "focus": ["XLP", "XLU"]}')
    assert plan.focus == ("XLU", "XLP")


def test_llm_decompose_null_focus_keeps_regex_result():
    """focus=null falls back to the deterministic parse of the query (defensive here)."""
    plan = _decomp("which defensive sectors look good?", lambda s, u: '{"horizon_months": null, "focus": null}')
    assert plan.focus == const.SECTOR_GROUPS["defensive"]


def test_llm_decompose_falls_back_to_regex_focus_on_bad_json():
    """Non-JSON output falls back to the deterministic focus parse."""
    plan = _decomp("rank the cyclicals", lambda s, u: "nope")
    assert plan.focus == const.SECTOR_GROUPS["cyclical"]


# --------------------------------------------------------------------------- #
# The coordinator narrows the RANKING to the focus sub-universe (fetch stays full)
# --------------------------------------------------------------------------- #
class RecordingScorer:
    """Records the universe (and horizon) it was called with; returns rows for exactly
    the sectors in that universe, so callers can assert what got ranked."""

    def __init__(self) -> None:
        self.universe: object = "unset"   # sentinel: distinguish 'not called' from None
        self.horizon = None

    def __call__(self, macro_regime, analog_data, equity_data, *, horizon=None, universe=None):
        self.universe = universe
        self.horizon = horizon
        tickers = universe if universe is not None else const.SECTOR_ETFS_LIST
        return [
            {"sector": t, "score": 0.5, "rank": i, "confidence": 0.5, "detail": {}}
            for i, t in enumerate(tickers, start=1)
        ]


def _focus_decompose(focus):
    def fake_decompose(query, *, as_of, tickers, period):
        return co.QueryPlan(as_of=as_of, tickers=tuple(tickers), period=period, focus=focus)
    return fake_decompose


def test_coordinator_passes_focus_universe_to_scorer():
    """A plan with a focus sub-universe restricts the scorer's ranking universe to that
    subset; the equity fetch (plan.tickers) stays the full 11, so the macro ToT input is
    unaffected."""
    scorer = RecordingScorer()
    equity = FakeEquityRunner(_equity_result())
    coord = co.Coordinator(
        FakeMacroRunner(_macro_result()), equity, scorer,
        decompose=_focus_decompose(("XLU", "XLP", "XLV")),
    )
    result = asyncio.run(coord.run("which defensive sectors?", as_of="2026-06-01"))

    assert scorer.universe == ("XLU", "XLP", "XLV")          # ranking narrowed to the focus
    assert {r["sector"] for r in result.rankings} == {"XLU", "XLP", "XLV"}
    assert equity.calls == 1                                 # fetch ran once, not narrowed


def test_coordinator_no_focus_ranks_full_universe():
    """No focus -> the scorer's universe is the full 11 (backward compatible)."""
    scorer = RecordingScorer()
    coord = co.Coordinator(
        FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), scorer,
        decompose=_focus_decompose(None),
    )
    asyncio.run(coord.run("which sectors should I overweight?", as_of="2026-06-01"))
    assert scorer.universe == tuple(const.SECTOR_ETFS_LIST)


def test_coordinator_focus_composes_with_quarantine():
    """Focus and quarantine compose: a focus sector quarantined out of the fetch is also
    dropped from the ranking universe (the fix for 'quarantine never dropped a sector
    from the ranking')."""
    scorer = RecordingScorer()
    coord = co.Coordinator(
        FakeMacroRunner(_macro_result()), FakeEquityRunner(_equity_result()), scorer,
        audit=QuarantineThenCleanAudit(bad="XLU"),
        decompose=_focus_decompose(("XLU", "XLP", "XLV")),
        max_revision_cycles=3,
    )
    asyncio.run(coord.run("which defensive sectors?", as_of="2026-06-01"))
    assert scorer.universe == ("XLP", "XLV")     # XLU quarantined out of the focus ranking
