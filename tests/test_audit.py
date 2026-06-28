"""
test_audit.py

Tests for the audit layer's statistical checker
(src/sector_rotation_agent/audit.py :: check_statistical_anomaly).

Status when written:
  * check_statistical_anomaly is a STUB (raises NotImplementedError). These tests
    encode the documented contract (spec Section 3.2 / 5.2 / guardrails #1, #2) and
    the whole module is marked xfail(strict=False) below, so the suite stays green
    now and every test flips to XPASS the moment you implement the body -- that's
    your signal to delete the module-level marker. (Same pattern as
    test_synthesize.py / test_historical_analogs.py.)

check_statistical_anomaly is pure, deterministic Python -- no FRED/Yahoo/LLM/Chroma
-- so there is nothing to Integration-gate; every test here always runs offline.
Freshness is made deterministic by passing an explicit `as_of` rather than letting
the check read the system clock.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import sector_rotation_agent.audit as audit
import sector_rotation_agent.constants as const

if TYPE_CHECKING:
    # Type-only import: review() is typed to take coordinator.QueryPlan, but these
    # tests pass a duck-typed SimpleNamespace (review reads only .as_of). Importing
    # QueryPlan for real would pull the coordinator graph into this light test module,
    # so it stays TYPE_CHECKING-only and _PLAN is cast to satisfy the checker.
    from sector_rotation_agent.coordinator import QueryPlan

# A fixed point-in-time anchor for every freshness assertion.
AS_OF = "2026-06-01"


def make_history(values: list[float], *, end: str = AS_OF, step_days: int = 30) -> list[dict]:
    """Ascending {"date","value"} observations, one per value, `step_days` apart,
    with the most recent dated `end`.

    Mirrors a get_macro_indicators per-series payload, so the checker is exercised
    on exactly the shape it will see in production. `end` controls the freshness
    check independently of the values (push it into the past to model a stale series).
    """
    end_d = date.fromisoformat(end)
    n = len(values)
    return [
        {
            "date": (end_d - timedelta(days=step_days * (n - 1 - i))).isoformat(),
            "value": float(v),
        }
        for i, v in enumerate(values)
    ]


# A tight, fresh, 10-point window: enough points to clear min_history, low variance
# so a moderate deviation reads as anomalous.
STEADY = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.95, 5.05]


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_returns_documented_flag_dict():
    """The result is a self-describing flag dict: the series under test, a boolean
    overall verdict, human-readable reasons, and a per-check breakdown -- the shape
    the audit log and generate_report consume."""
    out = audit.check_statistical_anomaly("FEDFUNDS", 5.0, make_history(STEADY), as_of=AS_OF)

    assert isinstance(out, dict)
    assert out["series_id"] == "FEDFUNDS"
    assert isinstance(out["flagged"], bool)
    assert isinstance(out["reasons"], list)
    assert {"z_score", "iqr", "freshness"} <= set(out["checks"])
    for check in out["checks"].values():
        assert isinstance(check["flagged"], bool)


# --------------------------------------------------------------------------- #
# The three checks
# --------------------------------------------------------------------------- #
def test_clean_value_is_not_flagged():
    """A value sitting in the middle of a fresh, well-behaved series trips nothing
    -- the unconditional gate must not cry wolf on normal data."""
    out = audit.check_statistical_anomaly("FEDFUNDS", 5.0, make_history(STEADY), as_of=AS_OF)
    assert out["flagged"] is False
    assert out["reasons"] == []


def test_gross_outlier_trips_distributional_checks():
    """A wildly out-of-range value (the guardrail #1 'fabricated figure' case) is
    caught by BOTH distributional checks, and the overall verdict is flagged."""
    out = audit.check_statistical_anomaly("FEDFUNDS", 1000.0, make_history(STEADY), as_of=AS_OF)
    assert out["flagged"] is True
    assert out["checks"]["z_score"]["flagged"] is True
    assert out["checks"]["iqr"]["flagged"] is True


def test_stale_series_flagged_on_freshness_alone():
    """Guardrail #2: a series whose most recent observation predates `as_of` by more
    than the freshness ceiling is flagged even when the value itself is perfectly
    normal -- so freshness is isolated here from the distributional checks."""
    stale = make_history(STEADY, end="2026-04-02")  # ~60 days before AS_OF
    out = audit.check_statistical_anomaly("FEDFUNDS", 5.0, stale, as_of=AS_OF)
    assert out["checks"]["freshness"]["flagged"] is True
    assert out["checks"]["z_score"]["flagged"] is False
    assert out["checks"]["iqr"]["flagged"] is False
    assert out["flagged"] is True


def test_fresh_series_not_flagged_on_freshness():
    """The mirror image: a recent observation is within the freshness ceiling and
    does not trip the freshness check."""
    fresh = make_history(STEADY, end="2026-05-20")  # well within 45 days of AS_OF
    out = audit.check_statistical_anomaly("FEDFUNDS", 5.0, fresh, as_of=AS_OF)
    assert out["checks"]["freshness"]["flagged"] is False


# --------------------------------------------------------------------------- #
# Edge case
# --------------------------------------------------------------------------- #
def test_short_history_skips_distributional_checks_without_raising():
    """Too few points to define a distribution: the z-score and IQR checks are
    SKIPPED (flagged False, no stats) rather than raising -- a brand-new series with
    little history must not crash the unconditional gate. Freshness still applies."""
    out = audit.check_statistical_anomaly("FEDFUNDS", 999.0, make_history([5.0, 5.1, 4.9]), as_of=AS_OF)
    assert out["checks"]["z_score"]["flagged"] is False
    assert out["checks"]["z_score"]["z"] is None
    assert out["checks"]["iqr"]["flagged"] is False


def test_short_history_still_runs_freshness():
    """Freshness is UNCONDITIONAL (spec 3.2 / guardrail #2): a brand-new series with
    too few points for the distributional checks can still be stale, and must be
    flagged anyway. Pins freshness running OUTSIDE the min_history gate -- the case
    that slips through if the freshness block is nested inside it."""
    stale_short = make_history([5.0, 5.1, 4.9], end="2026-04-02")  # 3 points (< min_history), ~60d old
    out = audit.check_statistical_anomaly("FEDFUNDS", 5.0, stale_short, as_of=AS_OF)
    # distributional checks skipped (too little history) ...
    assert out["checks"]["z_score"]["flagged"] is False
    assert out["checks"]["z_score"]["z"] is None
    assert out["checks"]["iqr"]["flagged"] is False
    # ... but freshness still fires
    assert out["checks"]["freshness"]["flagged"] is True
    assert out["flagged"] is True


def test_constant_history_flags_departure_without_zero_division():
    """A series held perfectly flat has zero variance, so statistics.stdev is 0 and a
    naive z-score divides by zero. The check must not raise: a value that departs
    from the constant is still anomalous and is flagged, while a value equal to the
    constant is clean."""
    flat = make_history([5.0] * 10)  # 10 identical points, fresh (end defaults to AS_OF)

    departed = audit.check_statistical_anomaly("FEDFUNDS", 9.0, flat, as_of=AS_OF)
    assert departed["checks"]["z_score"]["flagged"] is True
    assert departed["flagged"] is True

    same = audit.check_statistical_anomaly("FEDFUNDS", 5.0, flat, as_of=AS_OF)
    assert same["checks"]["z_score"]["flagged"] is False


# --------------------------------------------------------------------------- #
# run_critic_check  (LLM half of the audit layer -- STUB)
# --------------------------------------------------------------------------- #
# These pin run_critic_check's contract while it's a NotImplementedError stub, so the
# group is marked xfail(strict=False): green now, flipping to XPASS the day the body
# lands (drop the class marker then). The LLM call is injected via `call_model`, so
# the verdict/parse/clamp logic is exercised with canned responses and no network --
# the same dependency injection the ToT classifier uses.
HYPOTHESIS = "The macro regime is late_cycle."
EVIDENCE = "The 10y-2y Treasury spread rose to +0.40 this month."


def make_fake_model(response: str):
    """A (system, user) -> raw_text stand-in for the LLM call that records the
    prompts it received, so a test can assert what the critic actually sent."""
    calls: list[tuple[str, str]] = []

    def call(system: str, user: str) -> str:
        calls.append((system, user))
        return response

    return call, calls


class TestRunCriticCheck:
    def test_returns_verdict_contract(self):
        """The result carries a verdict from the allowed set, a reason string, and a
        confidence in [0, 1]."""
        call, _ = make_fake_model('{"verdict": "contradicts", "reason": "spread rose", "confidence": 0.7}')
        out = audit.run_critic_check(HYPOTHESIS, EVIDENCE, call_model=call)
        assert out["verdict"] in {"supports", "weakens", "contradicts"}
        assert isinstance(out["reason"], str)
        assert isinstance(out["confidence"], float)
        assert 0.0 <= out["confidence"] <= 1.0

    def test_echoes_each_verdict(self):
        """Each of the three verdicts round-trips from the model response."""
        for verdict in ("supports", "weakens", "contradicts"):
            call, _ = make_fake_model(f'{{"verdict": "{verdict}", "reason": "r", "confidence": 0.5}}')
            out = audit.run_critic_check(HYPOTHESIS, EVIDENCE, call_model=call)
            assert out["verdict"] == verdict

    def test_isolates_context_to_hypothesis_and_evidence(self):
        """Guardrail #6: the critic is called once and its prompt carries the
        hypothesis and the single evidence item -- nothing else is threaded in,
        because the signature gives it nothing else to thread."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        audit.run_critic_check(HYPOTHESIS, EVIDENCE, call_model=call)
        assert len(calls) == 1                     # single-shot, stateless
        _system, user = calls[0]
        assert HYPOTHESIS in user
        assert EVIDENCE in user

    def test_rejects_unknown_verdict(self):
        """A verdict outside the allowed set is a parse failure, not a fourth option
        -- the critic must not silently coerce 'maybe' into a pass."""
        call, _ = make_fake_model('{"verdict": "maybe", "reason": "r", "confidence": 0.5}')
        with pytest.raises(ValueError):
            audit.run_critic_check(HYPOTHESIS, EVIDENCE, call_model=call)

    def test_clamps_confidence(self):
        """Out-of-range confidence is clamped to [0, 1] -- the same discipline the
        hypothesis prior gets in generate_hypotheses."""
        call, _ = make_fake_model('{"verdict": "weakens", "reason": "r", "confidence": 1.7}')
        out = audit.run_critic_check(HYPOTHESIS, EVIDENCE, call_model=call)
        assert out["confidence"] == 1.0


# --------------------------------------------------------------------------- #
# ResultAuditor  (the composed AuditLayer the coordinator injects)
# --------------------------------------------------------------------------- #
# Phase 1: the statistical checker is wired into review() but yields nothing yet
# (neither agent result carries per-series history); the LLM critic is LIVE and
# cross-checks the macro regime call against the strongest historical analog.
# review() reads `plan` only structurally (.as_of, unused today), so these pass a
# SimpleNamespace and never import coordinator.QueryPlan -- keeping this module's
# import light. The flags returned ARE real coordinator.AuditFlags (built lazily
# inside audit._flag), so .source / .label / .message are asserted directly.
_PLAN = cast("QueryPlan", SimpleNamespace(as_of=AS_OF, tickers=("XLK",), period="5y"))

_ANALOGS = [
    {"date": "2001-03", "similarity": 0.71, "regime": "contraction"},
    {"date": "2007-11", "similarity": 0.88, "regime": "early_cycle"},  # strongest match
]


def _macro_result(regime, analogs):
    """Minimal MacroResult stand-in: the auditor reads only .regime and .analogs."""
    return SimpleNamespace(regime=regime, analogs=analogs)


def _macro_result_with_history(regime, indicators, series_history, analogs=None):
    """MacroResult stand-in carrying the snapshot + per-indicator history the
    statistical checker reads, alongside .regime/.analogs for the critic path."""
    return SimpleNamespace(
        regime=regime,
        analogs=analogs if analogs is not None else _ANALOGS,
        snapshot=SimpleNamespace(indicators=indicators),
        series_history=series_history,
    )


MOM_STEADY = [0.05, 0.04, 0.06, 0.05, 0.03, 0.05, 0.04, 0.06, 0.05, 0.04]  # ~5% momenta


def _equity_result_with_history(equity_data, series_history):
    """EquityResult stand-in carrying equity_data + per-sector momentum history."""
    return SimpleNamespace(equity_data=equity_data, current_momentum={}, series_history=series_history)

def make_fake_freshness(report: dict):
    """A check_corpus_freshness stand-in: records the `as_of` it was called with and
    returns a canned flag dict, so the auditor's corpus-freshness path is exercised
    offline (no Chroma store). Mirrors make_fake_model's record-and-return shape."""
    calls: list[str | None] = []

    def check(*, as_of=None) -> dict:
        calls.append(as_of)
        return report

    return check, calls


# Canned check_corpus_freshness outputs (the shape ResultAuditor consumes).
_STALE_CORPUS = {
    "source_id": "fed_narrative",
    "flagged": True,
    "reasons": ["newest fed_narrative doc is 84 days old (> 60)"],
    "checks": {"freshness": {"flagged": True, "newest_date": "2026-03-09",
                             "age_days": 84, "max_age_days": 60}},
}
_FRESH_CORPUS = {
    "source_id": "fed_narrative",
    "flagged": False,
    "reasons": [],
    "checks": {"freshness": {"flagged": False, "newest_date": "2026-05-20",
                             "age_days": 12, "max_age_days": 60}},
}


class TestResultAuditor:
    def test_contradicting_critic_raises_one_flag(self):
        """A 'contradicts' verdict on the regime-vs-analog check becomes a single
        critic AuditFlag naming the verdict and the regime."""
        call, _ = make_fake_model(
            '{"verdict": "contradicts", "reason": "analog was early_cycle", "confidence": 0.8}')
        auditor = audit.ResultAuditor(call_model=call)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert len(flags) == 1
        assert flags[0].source == "critic"
        assert flags[0].label == "macro"
        assert "contradicts" in flags[0].message
        assert "late_cycle" in flags[0].message

    def test_supporting_critic_raises_no_flag(self):
        """A 'supports' verdict is the clean path -- no flag."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "consistent", "confidence": 0.6}')
        auditor = audit.ResultAuditor(call_model=call)
        assert auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN) == []

    def test_critic_isolated_to_strongest_analog_single_call(self):
        """Guardrail #6: the critic is called once, and its prompt carries the regime
        hypothesis and ONLY the single strongest analog (0.88, not the 0.71 one)."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert len(calls) == 1
        _system, user = calls[0]
        assert "late_cycle" in user
        assert "2007-11" in user
        assert "2001-03" not in user

    def test_no_analogs_skips_the_critic(self):
        """No analogs => no evidence to test the call against => no LLM call, no flag."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        assert auditor.review("macro agent", _macro_result(const.Regime.MID_CYCLE, []), _PLAN) == []
        assert calls == []

    def test_critic_failure_degrades_to_soft_flag(self):
        """A flaky model (run_critic_check raises) must not crash the run: the auditor
        emits a soft 'could not be completed' flag instead."""
        def boom(_system, _user):
            raise ValueError("model returned garbage")
        auditor = audit.ResultAuditor(call_model=boom)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert len(flags) == 1
        assert flags[0].source == "critic"
        assert "could not be completed" in flags[0].message

    def test_equity_review_is_a_noop_in_phase1(self):
        """Equity gets no critic yet (Phase 2 hook): no flags, no LLM call."""
        call, calls = make_fake_model('{"verdict": "contradicts", "reason": "r", "confidence": 0.9}')
        auditor = audit.ResultAuditor(call_model=call)
        equity = SimpleNamespace(equity_data={"XLK": {"momentum": 0.1}}, current_momentum={"XLK": 0.1})
        assert auditor.review("equity agent", equity, _PLAN) == []
        assert calls == []

    # --- statistical checker (now live for macro): runs BEFORE the critic --------
    def test_stale_indicator_flags_statistical_and_skips_critic(self):
        """A stale indicator trips the statistical checker, which short-circuits the
        critic (spec 3.2: don't spend an LLM call on already-suspect data)."""
        call, calls = make_fake_model('{"verdict": "contradicts", "reason": "r", "confidence": 0.9}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.LATE_CYCLE,
            indicators={"fed_funds_rate": 5.0},
            series_history={"fed_funds_rate": make_history(STEADY, end="2026-04-02")},  # ~60d stale
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert any(f.source == "statistical" and f.label == "fed_funds_rate" for f in flags)
        assert calls == []   # gate failed -> critic not called

    def test_outlier_indicator_flags_statistical(self):
        """A gross-outlier indicator value is caught by the statistical checker."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.LATE_CYCLE,
            indicators={"fed_funds_rate": 1000.0},
            series_history={"fed_funds_rate": make_history(STEADY)},   # fresh, steady history
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert any(f.source == "statistical" for f in flags)
        assert calls == []

    def test_clean_indicators_pass_gate_then_run_critic(self):
        """Clean, fresh indicators raise no statistical flag, so the gate passes and
        the critic runs (spec 3.2 ordering: statistical first, then critic)."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.LATE_CYCLE,
            indicators={"fed_funds_rate": 5.0},
            series_history={"fed_funds_rate": make_history(STEADY)},   # clean + fresh
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert flags == []          # supports verdict, nothing flagged
        assert len(calls) == 1      # gate passed -> critic ran

    def test_indicator_without_history_is_skipped(self):
        """An indicator with no retained history is skipped (not guessed at, not an
        error); a clean run still proceeds to the critic."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.LATE_CYCLE,
            indicators={"fed_funds_rate": 5.0},
            series_history={},      # nothing retained for this indicator
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert flags == []
        assert len(calls) == 1

    # --- statistical checker (equity momentum) -------------------------------
    def test_equity_momentum_outlier_flags_statistical(self):
        """A sector whose current momentum is a gross outlier vs its own momentum
        history is flagged; equity has no critic, so nothing else runs."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={"XLK": {"momentum": 2.0, "valuation": 20.0}},   # 200% vs ~5% history
            series_history={"XLK": make_history(MOM_STEADY)},
        )
        flags = auditor.review("equity agent", result, _PLAN)
        assert any(f.source == "statistical" and f.label == "XLK" for f in flags)
        assert calls == []   # equity has no critic

    def test_equity_clean_momentum_no_flags(self):
        """In-range, fresh momentum trips nothing (and equity has no critic)."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={"XLK": {"momentum": 0.05, "valuation": 20.0}},
            series_history={"XLK": make_history(MOM_STEADY)},
        )
        assert auditor.review("equity agent", result, _PLAN) == []
        assert calls == []

    def test_equity_without_history_is_skipped(self):
        """A sector with no retained momentum history is skipped, not errored."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={"XLK": {"momentum": 999.0, "valuation": 20.0}},  # would be an outlier...
            series_history={},                                            # ...but no history to check
        )
        assert auditor.review("equity agent", result, _PLAN) == []

    # --- equity critic (momentum-vs-valuation), runs only if the gate passes --
    def test_equity_critic_flags_expensive_momentum_leader(self):
        """With no statistical flag, the critic checks the strongest 'rising but
        expensive' sector; a 'weakens' verdict flags it. One call, isolated to that
        single leader (guardrail #6)."""
        call, calls = make_fake_model('{"verdict": "weakens", "reason": "stretched multiple", "confidence": 0.7}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={
                "XLK": {"momentum": 0.20, "valuation": 35.0},   # expensive + rising (leader)
                "XLF": {"momentum": 0.12, "valuation": 32.0},   # expensive + rising
                "XLU": {"momentum": 0.03, "valuation": 12.0},   # cheap
                "XLP": {"momentum": -0.05, "valuation": 14.0},  # cheap + falling
            },
            series_history={},   # no statistical flags -> critic runs
        )
        flags = auditor.review("equity agent", result, _PLAN)
        assert len(calls) == 1
        assert len(flags) == 1
        assert flags[0].source == "critic" and flags[0].label == "XLK"
        _system, user = calls[0]
        assert "XLK" in user and "XLF" not in user   # the single strongest leader, isolated

    def test_equity_critic_supports_no_flag(self):
        """A 'supports' verdict on the expensive leader raises nothing."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "earnings justify it", "confidence": 0.6}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={
                "XLK": {"momentum": 0.20, "valuation": 35.0},
                "XLU": {"momentum": 0.03, "valuation": 12.0},
            },
            series_history={},
        )
        assert auditor.review("equity agent", result, _PLAN) == []

    def test_equity_critic_skips_when_no_expensive_riser(self):
        """When the momentum leaders are all cheap (no rising-but-expensive tension),
        the critic isn't called at all."""
        call, calls = make_fake_model('{"verdict": "contradicts", "reason": "r", "confidence": 0.9}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _equity_result_with_history(
            equity_data={
                "XLK": {"momentum": 0.20, "valuation": 12.0},   # cheap + rising (agree)
                "XLF": {"momentum": 0.15, "valuation": 14.0},   # cheap + rising
                "XLU": {"momentum": -0.05, "valuation": 35.0},  # expensive + falling (agree)
                "XLP": {"momentum": -0.03, "valuation": 32.0},
            },
            series_history={},
        )
        assert auditor.review("equity agent", result, _PLAN) == []
        assert calls == []

    # --- per-series freshness ceilings (guardrail #2): quarterly GDP ----------
    def test_quarterly_gdp_within_its_window_not_flagged(self):
        """gdp_growth is quarterly, so its per-series freshness ceiling is far wider
        than the 45-day default: a ~150-day-old GDP print is current, not stale, and
        must not trip the freshness check (it would under the flat 45-day rule)."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.MID_CYCLE,
            indicators={"gdp_growth": 5.0},
            series_history={"gdp_growth": make_history(STEADY, end="2026-01-02")},  # ~150d old
            analogs=[],   # skip the critic; isolate the statistical gate
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert not any(f.source == "statistical" for f in flags)

    def test_quarterly_gdp_beyond_its_window_flagged(self):
        """The wider ceiling still catches genuinely stale quarterly data: a GDP print
        older than a quarter + reporting lag (~210 days here) is flagged."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.MID_CYCLE,
            indicators={"gdp_growth": 5.0},
            series_history={"gdp_growth": make_history(STEADY, end="2025-11-03")},  # ~210d old
            analogs=[],
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert any(f.source == "statistical" and f.label == "gdp_growth" for f in flags)

    def test_freshness_ceiling_is_per_series_not_global(self):
        """The wide GDP window is an override, not a global loosening: a MONTHLY
        indicator at the same ~150-day age still trips the 45-day default."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        auditor = audit.ResultAuditor(call_model=call)
        result = _macro_result_with_history(
            const.Regime.MID_CYCLE,
            indicators={"fed_funds_rate": 5.0},
            series_history={"fed_funds_rate": make_history(STEADY, end="2026-01-02")},  # ~150d old
            analogs=[],
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert any(f.source == "statistical" and f.label == "fed_funds_rate" for f in flags)

# --- corpus freshness (guardrail #2, corpus level): injected, macro-only -----
    def test_stale_corpus_raises_fed_narrative_flag(self):
        """A flagged corpus-freshness report becomes one statistical AuditFlag labeled
        'fed_narrative', carrying the report's reason. The checker is called with the
        run's as_of (point-in-time)."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        fresh, fresh_calls = make_fake_freshness(_STALE_CORPUS)
        auditor = audit.ResultAuditor(call_model=call, check_freshness=fresh)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        corpus = [f for f in flags if f.label == "fed_narrative"]
        assert len(corpus) == 1
        assert corpus[0].source == "statistical"
        assert "84 days old" in corpus[0].message
        assert fresh_calls == [AS_OF]            # called point-in-time

    def test_stale_corpus_does_not_suppress_critic(self):
        """The corpus check sits OUTSIDE the statistical-gate/critic short-circuit: a
        stale corpus and a contradicting critic both surface. With no per-series history
        the statistical gate passes, so the critic runs and its flag stands beside the
        fed_narrative one."""
        call, calls = make_fake_model(
            '{"verdict": "contradicts", "reason": "analog was early_cycle", "confidence": 0.8}')
        fresh, _ = make_fake_freshness(_STALE_CORPUS)
        auditor = audit.ResultAuditor(call_model=call, check_freshness=fresh)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert len(calls) == 1                                   # critic still ran
        assert any(f.source == "critic" and f.label == "macro" for f in flags)
        assert any(f.source == "statistical" and f.label == "fed_narrative" for f in flags)

    def test_clean_corpus_raises_no_flag(self):
        """A fresh corpus adds no flag (and the supports verdict adds none either), but
        the check still ran point-in-time."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        fresh, fresh_calls = make_fake_freshness(_FRESH_CORPUS)
        auditor = audit.ResultAuditor(call_model=call, check_freshness=fresh)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert flags == []
        assert fresh_calls == [AS_OF]

    def test_no_freshness_checker_means_no_corpus_flag(self):
        """With no checker injected (the default), the corpus path is skipped entirely --
        no fed_narrative flag, regardless of corpus state."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        auditor = audit.ResultAuditor(call_model=call)            # check_freshness defaults to None
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert not any(f.label == "fed_narrative" for f in flags)

    def test_corpus_freshness_not_run_on_equity(self):
        """The corpus belongs to the macro result: the checker is not called on an equity
        review, and no fed_narrative flag appears there."""
        call, _ = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.5}')
        fresh, fresh_calls = make_fake_freshness(_STALE_CORPUS)
        auditor = audit.ResultAuditor(call_model=call, check_freshness=fresh)
        result = _equity_result_with_history(
            equity_data={"XLK": {"momentum": 0.03, "valuation": 12.0}},   # cheap -> no equity critic
            series_history={},
        )
        flags = auditor.review("equity agent", result, _PLAN)
        assert fresh_calls == []                                  # never called on equity
        assert not any(f.label == "fed_narrative" for f in flags)

    def test_corpus_freshness_failure_degrades_to_no_flag(self):
        """A checker that raises (e.g. the Chroma store is unavailable) must not crash the
        run: the corpus check degrades to no flag and the rest of the review proceeds."""
        call, calls = make_fake_model('{"verdict": "supports", "reason": "r", "confidence": 0.6}')
        def boom(*, as_of=None):
            raise RuntimeError("chroma unavailable")
        auditor = audit.ResultAuditor(call_model=call, check_freshness=boom)
        flags = auditor.review("macro agent", _macro_result(const.Regime.LATE_CYCLE, _ANALOGS), _PLAN)
        assert not any(f.label == "fed_narrative" for f in flags)
        assert len(calls) == 1                                    # critic still ran

    def test_stale_corpus_appended_alongside_statistical_flag(self):
        """When a statistical flag fires (stale indicator -> critic short-circuited), the
        corpus flag is STILL appended -- it lives outside that gate. Both flags surface
        and the critic is not called."""
        call, calls = make_fake_model('{"verdict": "contradicts", "reason": "r", "confidence": 0.9}')
        fresh, _ = make_fake_freshness(_STALE_CORPUS)
        auditor = audit.ResultAuditor(call_model=call, check_freshness=fresh)
        result = _macro_result_with_history(
            const.Regime.LATE_CYCLE,
            indicators={"fed_funds_rate": 5.0},
            series_history={"fed_funds_rate": make_history(STEADY, end="2026-04-02")},  # stale
        )
        flags = auditor.review("macro agent", result, _PLAN)
        assert any(f.source == "statistical" and f.label == "fed_funds_rate" for f in flags)
        assert any(f.source == "statistical" and f.label == "fed_narrative" for f in flags)
        assert calls == []  # gate failed -> critic skipped


# =========================================================================== #
# ADDITIONAL TEST CASES TO IMPLEMENT LATER
# =========================================================================== #
# ResultAuditor (Phase 2, as the audit layer deepens):
#   - _statistical_flags: once MacroResult/EquityResult carry per-series history,
#     a stale/outlier series yields a "statistical" flag AND short-circuits the
#     critic (statistical gate fails -> no LLM call)
#   - equity critic: momentum-vs-valuation disagreement raises a flag
#   - the macro critic runs only after the statistical gate passes (ordering)
#
# check_statistical_anomaly:
#   - IQR catches an outlier that the z-score misses (value just past the Tukey
#     fence but within 3 sigma) -- the reason the spec runs both
#   - reasons list carries one entry per tripped check, each naming the series_id
#   - exactly-at-threshold boundaries (z == z_threshold, age == max_age_days) are
#     NOT flagged (strictly greater trips)
#   - None values inside history are ignored, not treated as 0.0
#   - custom z_threshold / iqr_multiplier / max_age_days change the verdict
