"""
test_synthesize.py

Tests for the synthesis layer (src/sector_rotation_agent/synthesize.py).

Status when written:
  * compute_sector_score is a STUB (raises NotImplementedError). These tests are
    written against the documented contract and the whole module is marked
    xfail(strict=False) below, so the suite stays green now and every test flips
    to XPASS the moment you implement the body -- that's your signal to delete the
    module-level marker. (Same pattern test_historical_analogs.py used.)

compute_sector_score is a pure, deterministic function -- no FRED/Yahoo/LLM/Chroma
calls -- so unlike the data-layer suites there is nothing to Integration-gate and
no paths to redirect. Every test here always runs offline.

The behavioural tests are deliberately built so all three signals (regime tilt,
analog returns, current momentum) point the SAME way. That keeps the asserted
ordering robust to how you end up normalizing/weighting internally: as long as the
combination is monotonic in each signal, the result holds.
"""
from __future__ import annotations

from statistics import mean
from types import SimpleNamespace

import pytest

import sector_rotation_agent.synthesize as syn
import sector_rotation_agent.constants as const

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def equity_data_all(**momentum_overrides: float) -> dict[str, dict]:
    """Per-sector equity state for all 11 ETFs; override individual momenta by ticker.

    Shape mirrors get_sector_performance / get_sector_valuations output:
    {ticker: {"momentum": float, "valuation": float}}.
    """
    data: dict[str, dict] = {
        t: {"momentum": 0.0, "valuation": 18.0} for t in const.SECTOR_ETFS_LIST
    }
    for ticker, m in momentum_overrides.items():
        data[ticker]["momentum"] = m
    return data


def make_analog(
    date: str,
    similarity: float,
    regime: const.Regime,
    returns: dict[str, float | None] | None = None,
) -> dict:
    """One analog row shaped like a find_historical_analogs result.

    Unspecified tickers default to a mild positive return so the row is realistic.
    Pass an explicit None to model a sector that did not exist yet (XLRE pre-2015,
    XLC pre-2018) -- the contract says those must be skipped, not read as 0.0.
    """
    base: dict[str, float | None] = {t: 0.02 for t in const.SECTOR_ETFS_LIST}
    if returns:
        base.update(returns)
    return {
        "date": date,
        "similarity": similarity,
        "regime": regime.value,  # metadata is stored/returned as the enum's .value
        "subsequent_sector_returns": base,
    }


@pytest.fixture
def strong_mid_cycle_analogs() -> list[dict]:
    """Four STRONG (similarity >= 0.75) MID_CYCLE analogs where cyclicals led.

    MID_CYCLE favors XLK/XLC/XLI (+1) and disfavors XLU/XLP/XLE (-1); these returns
    agree with that tilt, so analog evidence and the regime prior reinforce.
    """
    cyclicals_led = {"XLK": 0.12, "XLC": 0.10, "XLI": 0.08, "XLU": -0.06, "XLP": -0.05}
    return [
        make_analog("2017-05", 0.88, const.Regime.MID_CYCLE, cyclicals_led),        #type: ignore
        make_analog("2014-09", 0.81, const.Regime.MID_CYCLE, cyclicals_led),        #type: ignore
        make_analog("2006-02", 0.79, const.Regime.MID_CYCLE, cyclicals_led),        #type: ignore
        make_analog("1997-11", 0.76, const.Regime.MID_CYCLE, cyclicals_led),        #type: ignore
    ]


# --------------------------------------------------------------------------- #
# End-to-end: full output contract
# --------------------------------------------------------------------------- #
def test_compute_sector_score_end_to_end(strong_mid_cycle_analogs):
    """A realistic call returns a complete, well-formed ranking of all 11 sectors."""
    equity = equity_data_all(XLK=0.10, XLC=0.08, XLI=0.06, XLU=-0.05, XLP=-0.04)

    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity
    )

    # one entry per sector -- nothing dropped, nothing invented
    assert isinstance(result, list)
    assert {r["sector"] for r in result} == set(const.SECTOR_ETFS_LIST)
    assert len(result) == len(const.SECTOR_ETFS_LIST)

    # every row carries the documented fields, with confidence in range
    for r in result:
        assert {"sector", "score", "rank", "confidence", "detail"} <= set(r)
        assert 0.0 <= r["confidence"] <= 1.0

    # ranks are a clean 1..N permutation, ordered by descending score
    ranks = sorted(r["rank"] for r in result)
    assert ranks == list(range(1, len(const.SECTOR_ETFS_LIST) + 1))
    by_rank = sorted(result, key=lambda r: r["rank"])
    scores = [r["score"] for r in by_rank]
    assert scores == sorted(scores, reverse=True)


def test_compute_sector_score_aligned_signals_outrank(strong_mid_cycle_analogs):
    """When tilt, analogs, and momentum all agree, the favored sector wins.

    XLK is favored in MID_CYCLE (+1), led in the analogs, and we give it strong
    current momentum. XLU is the mirror image (-1, lagged, weak). XLK must outrank
    XLU regardless of the exact internal weighting.
    """
    equity = equity_data_all(XLK=0.12, XLU=-0.08)

    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity
    )
    rank = {r["sector"]: r["rank"] for r in result}

    assert rank["XLK"] < rank["XLU"]


# --------------------------------------------------------------------------- #
# Failure / edge cases
# --------------------------------------------------------------------------- #
def test_compute_sector_score_weights_must_sum_to_one(strong_mid_cycle_analogs):
    """Weights are a convex blend; a non-1.0 sum is a programming error, not a
    silent rescale. Mirrors score_branch's w_analog + w_signal == 1.0 guard."""
    with pytest.raises(ValueError):
        syn.compute_sector_score(
            const.Regime.MID_CYCLE,
            strong_mid_cycle_analogs,
            equity_data_all(),
            w_regime=0.9, w_analog=0.9, w_equity=0.9,
        )


def test_compute_sector_score_empty_analogs_still_ranks_all():
    """No historical support is a legitimate state (a regime with no close analogs).
    The model must fall back to regime + equity signals and still rank all 11 --
    not crash and not drop sectors."""
    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, [], equity_data_all(XLK=0.10)
    )
    assert {r["sector"] for r in result} == set(const.SECTOR_ETFS_LIST)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in result)


def test_compute_sector_score_skips_none_subsequent_returns():
    """None forward returns (sector not yet in existence) must be SKIPPED when
    averaging, never folded in as 0.0 -- doing so would dilute the mean.

    XLK and XLC share the MID_CYCLE tilt (+1) and we give them identical momentum,
    so the analog return is the ONLY difference:
        XLC: [None, 0.20]  -> skip-None mean = 0.20
        XLK: [0.10, 0.10]  -> mean = 0.10
    Correct (skip) => XLC outranks XLK. The None-as-0.0 bug halves XLC to ~0.10 and
    the two tie / flip.
    """
    analogs = [
        make_analog("2016-04", 0.85, const.Regime.MID_CYCLE, {"XLC": None, "XLK": 0.10}),
        make_analog("2019-07", 0.82, const.Regime.MID_CYCLE, {"XLC": 0.20, "XLK": 0.10}),
    ]
    equity = equity_data_all(XLC=0.05, XLK=0.05)  # identical momentum

    result = syn.compute_sector_score(const.Regime.MID_CYCLE, analogs, equity)
    rank = {r["sector"]: r["rank"] for r in result}

    assert rank["XLC"] < rank["XLK"]


def test_compute_sector_score_confidence_tracks_analog_strength():
    """The spec's guardrail: many strong analogs -> higher confidence than few weak
    ones. Asserted as a relative comparison so no absolute scale is baked in."""
    strong = [
        make_analog(f"20{10 + i:02d}-05", 0.85, const.Regime.MID_CYCLE) for i in range(4)
    ]
    weak = [make_analog("2011-05", 0.40, const.Regime.MID_CYCLE)]
    equity = equity_data_all()

    strong_conf = mean(
        r["confidence"]
        for r in syn.compute_sector_score(const.Regime.MID_CYCLE, strong, equity)
    )
    weak_conf = mean(
        r["confidence"]
        for r in syn.compute_sector_score(const.Regime.MID_CYCLE, weak, equity)
    )

    assert strong_conf > weak_conf


def test_compute_sector_score_tolerates_partial_equity_data(strong_mid_cycle_analogs):
    """equity_data need not cover every sector; missing sectors get a neutral equity
    signal rather than raising KeyError. Still ranks all 11."""
    partial = {"XLK": {"momentum": 0.10, "valuation": 20.0}}  # only one sector present

    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, partial
    )
    assert {r["sector"] for r in result} == set(const.SECTOR_ETFS_LIST)


# --------------------------------------------------------------------------- #
# generate_report  (sibling tool: a deterministic placeholder brief)
# --------------------------------------------------------------------------- #
# generate_report is a WORKING placeholder, not a NotImplementedError stub: the
# coordinator calls it unconditionally every run, so it has to return a real string
# today. These tests pin the contract it already honors -- header facts, every
# sector present, flags surfaced inline, determinism -- and the structural
# duck-typing it leans on (flags and audit_log are read by attribute, not via
# imported coordinator types), so the eventual full spec-Section-5 rewrite can't
# silently regress them.
QUERY = "Which sectors should I overweight/underweight over the next 6 months?"

# Minimal provenance list shaped like build_sources output (id/label/tool/value/as_of).
# generate_report renders these in a Sources section (see
# test_generate_report_cites_sources_and_audit_trail), so every call passes one.
SOURCES = [
    {"id": "fed_funds_rate", "label": "Federal Funds Rate",
     "tool": "FRED:FEDFUNDS", "value": 4.5, "as_of": "2026-06-01"},
    {"id": "XLK:momentum", "label": "XLK 6m momentum",
     "tool": "yfinance", "value": 0.08, "as_of": "2026-06-01"},
]

AS_OF_DATE = "2026-06-01"

def make_flag(source: str, label: str, message: str) -> SimpleNamespace:
    """A minimal stand-in for coordinator.AuditFlag.

    generate_report consumes flags structurally (.source / .label / .message), so
    the test mirrors that duck-typing rather than importing the coordinator's
    dataclass -- the same no-dependency stance the function's own docstring takes.
    """
    return SimpleNamespace(source=source, label=label, message=message)


def make_audit_log(tool_calls: int = 0) -> SimpleNamespace:
    """Minimal stand-in for coordinator.AuditLog (only .tool_calls is read)."""
    return SimpleNamespace(tool_calls=tool_calls)


@pytest.fixture
def simple_rankings() -> list[dict]:
    """Three pre-ranked rows shaped like compute_sector_score output.

    Hand-built rather than run through compute_sector_score so these tests stay
    isolated to generate_report -- a change in the scorer shouldn't ripple here.
    """
    return [
        {"sector": "XLK", "score": 0.82, "rank": 1, "confidence": 0.74, "detail": {}},
        {"sector": "XLF", "score": 0.55, "rank": 2, "confidence": 0.40, "detail": {}},
        {"sector": "XLU", "score": 0.10, "rank": 3, "confidence": 0.00, "detail": {}},
    ]


def test_generate_report_header_and_disclaimer(simple_rankings):
    """The brief is a single string carrying its header facts -- the question it
    answers, the regime (by .value, not the enum repr), and the confidence -- plus
    the not-advice disclaimer the synthesis layer must always attach."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.LATE_CYCLE,
        rankings=simple_rankings,
        confidence=0.39,
        flags=[],
        audit_log=make_audit_log(tool_calls=2),
        sources=SOURCES,
    )
    assert isinstance(out, str)
    assert QUERY in out
    assert const.Regime.LATE_CYCLE.value in out      # "late_cycle", not "Regime.LATE_CYCLE"
    assert "0.39" in out
    assert "does not constitute investment advice" in out.lower()


def test_generate_report_includes_every_ranked_sector(strong_mid_cycle_analogs):
    """Every sector in the ranking must appear -- nothing silently dropped. Sourced
    from a real compute_sector_score call so the two synthesis tools stay in sync
    on row shape."""
    rankings = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity_data_all(XLK=0.10)
    )
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
    )
    for tkr in const.SECTOR_ETFS_LIST:
        assert tkr in out


def test_generate_report_surfaces_flags_inline(simple_rankings):
    """Raised flags must show up in the brief -- source, label, and message -- with
    the count in the section header, so the reader sees every caveat."""
    flags = [
        make_flag("statistical", "macro", "fed_funds_rate z-score 3.4 exceeds 3.0"),
        make_flag("critic", "equity", "momentum and valuation disagree on XLK"),
    ]
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.CONTRACTION,
        rankings=simple_rankings,
        confidence=0.2,
        flags=flags,
        audit_log=make_audit_log(tool_calls=4),
        sources=SOURCES,
    )
    assert "Audit flags (2)" in out
    for f in flags:
        assert f.source in out
        assert f.label in out
        assert f.message in out


def test_generate_report_no_flags_reads_none(simple_rankings):
    """A clean run says so explicitly rather than emitting an empty flags block."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.EARLY_CYCLE,
        rankings=simple_rankings,
        confidence=0.8,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
    )
    assert "Audit flags: none" in out


def test_generate_report_is_deterministic(simple_rankings):
    """No LLM, no clock, no RNG: identical inputs give identical output. That
    reproducibility is the whole point of keeping the brief deterministic (the same
    reason compute_sector_score is)."""
    kwargs = dict(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.42,
        flags=[make_flag("statistical", "macro", "stale series: USSLIND")],
        audit_log=make_audit_log(tool_calls=2),
        sources=SOURCES,
    )
    assert syn.generate_report(**kwargs) == syn.generate_report(**kwargs) #type: ignore


def test_generate_report_tolerates_audit_log_without_tool_calls(simple_rankings):
    """audit_log is read via getattr(..., "tool_calls", None); an object lacking that
    attribute must not crash the brief -- the tool-calls line is simply omitted.
    Guards the structural-typing contract against a stricter or different AuditLog."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=object(),          # no .tool_calls attribute
        sources=SOURCES,
    )
    assert isinstance(out, str)
    assert "Tool calls audited" not in out


def test_generate_report_cites_sources_and_audit_trail(simple_rankings):
    """The brief renders its provenance (each source id + formatted value) and
    reconciles the run against the audit log -- the tool-call count and the Phase-3
    revision history (which sectors were quarantined, and why the loop stopped)."""
    audit_log = SimpleNamespace(
        tool_calls=6,
        entries=[
            {"event": "revision", "cycle": 0, "dropped": ["XLE"], "flags": ["XLE"]},
            {"event": "revision_halt", "cycle": 1, "reason": "budget", "flags": ["XLB"]},
        ],
    )
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.4,
        flags=[],
        audit_log=audit_log,
        sources=SOURCES,
    )
    # provenance: each source id and the formatted fed-funds value appear
    for s in SOURCES:
        assert s["id"] in out
    assert "4.50" in out
    # audit trail: tool-call count + revision history
    assert "Tool calls audited: 6" in out
    assert "quarantined XLE" in out
    assert "Revision halted (budget)" in out


def test_generate_report_surfaces_reconciliation(simple_rankings):
    """A reconciliation entry (guardrail #7) is surfaced in the audit trail: when the
    logged tool-call entries match the counter, the brief says so explicitly."""
    audit_log = SimpleNamespace(
        tool_calls=2,
        entries=[
            {"event": "tool_call", "tool": "equity agent", "cycle": 0, "n": 1},
            {"event": "tool_call", "tool": "macro agent", "cycle": 0, "n": 2},
            {"event": "audit_clean", "cycle": 0},
            {"event": "reconciliation", "tool_calls": 2, "logged_tool_calls": 2,
             "reconciled": True},
        ],
    )
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.4,
        flags=[],
        audit_log=audit_log,
        sources=SOURCES,
    )
    assert "Audit-log reconciled (guardrail #7)" in out
    assert "2 logged tool-call entries match 2 tool calls" in out


# --------------------------------------------------------------------------- #
# Executive summary  (LLM-written opening; the call_model seam)
# --------------------------------------------------------------------------- #
# generate_report gains an optional call_model seam: given one, it asks the LLM for a
# two-paragraph plain-English opening and renders it under the title; omitted, the brief
# is unchanged. call_model is a (system, user) -> str function (the model_client.complete
# shape), faked here so the tests stay offline and pin the WIRING -- placement, the
# facts-only digest, and graceful degradation -- not the model's prose. The call lives
# in synthesize (no audit_log in reach) and runs during assembly, so it cannot touch the
# tool-call counter; that guarantee is structural and needs no test here.
SUMMARY_TEXT = "First paragraph about the outcome.\n\nSecond paragraph about the caveats."


def _fake_summary_model(system: str, user: str) -> str:
    """Stand-in for model_client.complete: ignores the prompt, returns fixed prose."""
    return SUMMARY_TEXT


def test_generate_report_includes_executive_summary_when_call_model_given(simple_rankings):
    """With a call_model, the brief opens with an 'Executive summary' section carrying
    the model's prose, placed ABOVE the regime narrative and the ranking table."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(tool_calls=2),
        sources=SOURCES,
        call_model=_fake_summary_model,
    )
    assert "Executive summary" in out
    assert SUMMARY_TEXT in out
    # it is the OPENING: the summary precedes the regime narrative and the ranking
    assert out.index("Executive summary") < out.index("**Question:**")
    assert out.index("Executive summary") < out.index("Sector ranking")


def test_generate_report_omits_summary_without_call_model(simple_rankings):
    """No call_model (the default) -> no summary section; the brief is exactly the
    deterministic placeholder it was before the feature (backward compatible)."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
    )
    assert "Executive summary" not in out


def test_generate_report_degrades_when_summary_model_raises(simple_rankings):
    """A failing model must not break report assembly: the summary is omitted and the
    rest of the brief still renders (graceful degradation)."""
    def boom(system: str, user: str) -> str:
        raise RuntimeError("model unavailable")

    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.LATE_CYCLE,
        rankings=simple_rankings,
        confidence=0.3,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
        call_model=boom,
    )
    assert isinstance(out, str)
    assert "Executive summary" not in out
    assert const.Regime.LATE_CYCLE.value in out          # the structured brief survives
    assert "does not constitute investment advice" in out.lower()


def test_generate_executive_summary_feeds_only_final_facts(simple_rankings):
    """generate_executive_summary hands the model a digest of the FINAL outcome -- the
    question, regime, ranked sectors, and any flags -- so the prose is grounded and the
    model isn't invited to invent. Capture the prompts and assert the facts (and the
    two-paragraph, no-advice contract) are actually in them."""
    captured: dict[str, str] = {}

    def recording_model(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return SUMMARY_TEXT

    summary = syn.generate_executive_summary(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[make_flag("statistical", "macro", "fed_funds_rate z-score 3.4")],
        call_model=recording_model,
    )
    assert summary == SUMMARY_TEXT
    # the digest carries the question, the regime, the top sector, and the flag message
    assert QUERY in captured["user"]
    assert const.Regime.MID_CYCLE.value in captured["user"]
    assert "XLK" in captured["user"]
    assert "fed_funds_rate z-score 3.4" in captured["user"]
    # the system prompt pins the format + not-advice guardrail
    assert "two paragraph" in captured["system"].lower()


def test_generate_executive_summary_empty_on_model_failure(simple_rankings):
    """The function swallows a model error and returns "" so report assembly can omit
    the section rather than crash -- the contract generate_report relies on."""
    def boom(system: str, user: str) -> str:
        raise RuntimeError("model unavailable")

    assert syn.generate_executive_summary(
        query=QUERY, 
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags= [],
        call_model=boom
    ) == ""


# --------------------------------------------------------------------------- #
# Methodology appendix  (derived from the scoring parameters, not hardcoded)
# --------------------------------------------------------------------------- #
def test_generate_report_methodology_reflects_scoring_weights(simple_rankings):
    """The methodology appendix is rendered from the scoring parameters passed in, not
    fixed text: custom weights/thresholds appear and the defaults do not. (The default
    figures are exercised implicitly by the other generate_report tests.)"""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
        w_regime=0.50, w_analog=0.30, w_equity=0.20,
        strong_similarity=0.80, min_strong_analogs=5,
    )
    assert "weight 0.50" in out
    assert "weight 0.30" in out
    assert "weight 0.20" in out
    assert "similarity >= 0.80, saturating at 5" in out
    # the defaults are not silently shown when other weights were used
    assert "weight 0.40" not in out
    assert "saturating at 3" not in out


# --------------------------------------------------------------------------- #
# Horizon  (decomposed upstream; surfaced here, honored in scoring when seeded)
# --------------------------------------------------------------------------- #
def test_generate_report_flags_horizon_not_seeded(simple_rankings):
    """A horizon the store wasn't seeded at is shown in the header AND flagged: the brief
    says it falls back to the default window rather than silently answering with it."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
        horizon="9 months",   # not in const.ANALOG_HORIZONS_MONTHS
    )
    assert "**Requested horizon:** 9 months" in out
    assert "not one of the seeded analog windows" in out


def test_generate_report_seeded_horizon_raises_no_caveat(simple_rankings):
    """A seeded, non-default horizon (12 months) is displayed and honored -- no caveat."""
    assert 12 in const.ANALOG_HORIZONS_MONTHS  # guard: this test assumes 12 is seeded
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
        horizon="12 months",
    )
    assert "**Requested horizon:** 12 months" in out
    assert "not one of the seeded analog windows" not in out


def test_generate_report_without_horizon_is_unchanged(simple_rankings):
    """No horizon -> no horizon line and no mismatch caveat (backward compatible)."""
    out = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(),
        sources=SOURCES,
    )
    assert "Requested horizon" not in out
    assert "not one of the seeded analog windows" not in out


# --------------------------------------------------------------------------- #
# Horizon selection in scoring (compute_sector_score reads the matching slice)
# --------------------------------------------------------------------------- #
def _analog_multi(similarity: float, by_horizon: dict[int, dict[str, float | None]]) -> dict:
    """An analog carrying per-horizon returns (what the multi-horizon store returns). The
    default (6m) slice is mirrored into subsequent_sector_returns, as the reader does."""
    return {
        "date": "2014-09",
        "similarity": similarity,
        "regime": const.Regime.MID_CYCLE.value,
        "subsequent_sector_returns": by_horizon.get(const.ANALOG_DEFAULT_HORIZON_MONTHS, {}),
        "subsequent_returns_by_horizon": by_horizon,
    }


def test_compute_sector_score_selects_horizon_slice():
    """The analog signal reads the slice for the requested horizon: XLK leads at 3m and
    XLU leads at 12m, so the two horizons rank those sectors oppositely."""
    flat = {t: 0.0 for t in const.SECTOR_ETFS_LIST}
    by_h = {
        3:  {**flat, "XLK": 0.20, "XLU": -0.20},
        6:  flat,
        12: {**flat, "XLK": -0.20, "XLU": 0.20},
    }
    analogs = [_analog_multi(0.9, by_h)] # pyright: ignore[reportArgumentType]
    equity = equity_data_all()

    at_3 = {r["sector"]: r["detail"]["analog_return"] for r in
            syn.compute_sector_score(const.Regime.MID_CYCLE, analogs, equity, horizon="3 months")}
    at_12 = {r["sector"]: r["detail"]["analog_return"] for r in
             syn.compute_sector_score(const.Regime.MID_CYCLE, analogs, equity, horizon="12 months")}

    assert at_3["XLK"] > 0 and at_3["XLU"] < 0      # 3m slice
    assert at_12["XLK"] < 0 and at_12["XLU"] > 0    # 12m slice -- opposite


def test_compute_sector_score_unseeded_horizon_falls_back_to_default():
    """A horizon the analog wasn't seeded at uses the default (6m) slice, not nothing."""
    flat = {t: 0.0 for t in const.SECTOR_ETFS_LIST}
    by_h = {6: {**flat, "XLK": 0.15}}            # only the default horizon present
    analogs = [_analog_multi(0.9, by_h)]  # pyright: ignore[reportArgumentType]
    equity = equity_data_all()

    rows = {r["sector"]: r["detail"]["analog_return"] for r in
            syn.compute_sector_score(const.Regime.MID_CYCLE, analogs, equity, horizon="9 months")}
    assert rows["XLK"] > 0      # fell back to the 6m slice rather than vanishing


def test_report_to_pdf_writes_a_file(simple_rankings, tmp_path):
    """report_to_pdf renders the Markdown brief to a non-empty PDF. Skipped unless the
    optional markdown-pdf dependency is installed -- it isn't needed for the rest of
    the suite, only for an actual export."""
    pytest.importorskip("markdown_pdf")
    md = syn.generate_report(
        query=QUERY,
        as_of=AS_OF_DATE,
        regime=const.Regime.MID_CYCLE,
        rankings=simple_rankings,
        confidence=0.5,
        flags=[],
        audit_log=make_audit_log(tool_calls=2),
        sources=SOURCES,
    )
    out = syn.report_to_pdf(md, tmp_path / "brief.pdf")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Focus sub-universe  (rank only a requested subset; normalization basis + guard)
# --------------------------------------------------------------------------- #
def test_compute_sector_score_ranks_only_focus_subset(strong_mid_cycle_analogs):
    """A focus universe restricts the ranking to that subset -- only those sectors are
    returned, ranked 1..k -- while the rest of the 11 are absent."""
    focus = ("XLU", "XLP", "XLV")
    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity_data_all(), universe=focus
    )
    assert {r["sector"] for r in result} == set(focus)
    assert sorted(r["rank"] for r in result) == [1, 2, 3]


def test_compute_sector_score_normalizes_within_subset(strong_mid_cycle_analogs):
    """With a focus of >=3 sectors, each signal is min-max normalized WITHIN the subset
    (Option A): the best sector on a signal pins to 1.0 and the worst to 0.0, measured
    among the focus sectors only. Isolated to the equity signal via w_equity=1.0."""
    focus = ("XLU", "XLP", "XLV")
    equity = equity_data_all(XLU=0.10, XLP=0.05, XLV=0.00)
    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity, universe=focus,
        w_regime=0.0, w_analog=0.0, w_equity=1.0,
    )
    by_sector = {r["sector"]: r for r in result}
    # within the 3-sector subset, XLU has the max momentum (-> 1.0), XLV the min (-> 0.0)
    assert by_sector["XLU"]["detail"]["normalized_equity_signal"] == 1.0
    assert by_sector["XLV"]["detail"]["normalized_equity_signal"] == 0.0
    assert by_sector["XLU"]["rank"] == 1 and by_sector["XLV"]["rank"] == 3


def test_compute_sector_score_small_subset_falls_back_to_market_normalization(strong_mid_cycle_analogs):
    """Guard: a focus of fewer than 3 sectors normalizes across all 11 (market-wide), not
    within the tiny subset -- so the sectors take their market-relative values rather than
    a degenerate 1.0/0.0. Here XLF (-0.20) and XLK (0.20) set the market span (0.40), so
    XLU=0.05 -> 0.625 and XLP=0.00 -> 0.5. Isolated to the equity signal."""
    equity = equity_data_all(XLK=0.20, XLF=-0.20, XLU=0.05, XLP=0.00)
    result = syn.compute_sector_score(
        const.Regime.MID_CYCLE, strong_mid_cycle_analogs, equity, universe=("XLU", "XLP"),
        w_regime=0.0, w_analog=0.0, w_equity=1.0,
    )
    by_sector = {r["sector"]: r for r in result}
    assert set(by_sector) == {"XLU", "XLP"}                   # only the 2 requested returned
    assert by_sector["XLU"]["detail"]["normalized_equity_signal"] == pytest.approx(0.625)
    assert by_sector["XLP"]["detail"]["normalized_equity_signal"] == pytest.approx(0.5)


def test_generate_report_shows_sector_universe_for_subset():
    """A focus sub-universe is surfaced in the brief header ('Sector universe: N of 11')."""
    focus_rankings = [
        {"sector": "XLU", "score": 0.7, "rank": 1, "confidence": 0.5, "detail": {}},
        {"sector": "XLP", "score": 0.5, "rank": 2, "confidence": 0.4, "detail": {}},
        {"sector": "XLV", "score": 0.3, "rank": 3, "confidence": 0.3, "detail": {}},
    ]
    out = syn.generate_report(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.MID_CYCLE, rankings=focus_rankings,
        confidence=0.5, flags=[], audit_log=make_audit_log(), sources=SOURCES,
        universe=("XLU", "XLP", "XLV"),
    )
    assert "Sector universe:" in out
    assert "3 of 11" in out


def test_generate_report_no_universe_line_for_full_run(simple_rankings):
    """No focus (universe=None) -> no 'Sector universe' line (backward compatible)."""
    out = syn.generate_report(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.MID_CYCLE, rankings=simple_rankings,
        confidence=0.5, flags=[], audit_log=make_audit_log(), sources=SOURCES,
    )
    assert "Sector universe:" not in out


def test_generate_report_small_subset_normalization_caveat():
    """A focus of <3 sectors triggers the market-wide-normalization caveat in the brief."""
    rankings = [
        {"sector": "XLU", "score": 0.6, "rank": 1, "confidence": 0.5, "detail": {}},
        {"sector": "XLP", "score": 0.5, "rank": 2, "confidence": 0.4, "detail": {}},
    ]
    out = syn.generate_report(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.MID_CYCLE, rankings=rankings,
        confidence=0.5, flags=[], audit_log=make_audit_log(), sources=SOURCES,
        universe=("XLU", "XLP"),
    )
    assert "normalized across all 11 sectors" in out


# --------------------------------------------------------------------------- #
# Fed-narrative citations  (build_sources tags retrieved passages "FedNarrative")
# --------------------------------------------------------------------------- #
def test_build_sources_cites_fed_narrative():
    """build_sources emits one citation row per Fed-narrative passage, tagged
    'FedNarrative', carrying the source/date, the similarity as the value, and a quoted
    excerpt of the matched text in the label."""
    fed = [
        {"source": "fomc_minutes", "date": "2026-04-30", "title": "Minutes", "url": "",
         "text": "Participants judged that inflation remained elevated.", "similarity": 0.83},
        {"source": "beige_book", "date": "2026-03-05", "title": "Beige Book", "url": "",
         "text": "Economic activity expanded modestly.", "similarity": 0.71},
    ]
    sources = syn.build_sources(
        as_of="2026-06-01", indicators={}, analogs=[], equity_data={}, fed_narrative=fed
    )
    fed_rows = [s for s in sources if s["tool"] == "FedNarrative"]
    assert len(fed_rows) == 2
    assert "fomc minutes" in fed_rows[0]["label"] and "2026-04-30" in fed_rows[0]["label"]
    assert "inflation remained elevated" in fed_rows[0]["label"]
    assert fed_rows[0]["value"] == 0.83
    assert fed_rows[0]["as_of"] == "2026-06-01"


def test_build_sources_without_fed_narrative_is_unchanged():
    """No fed_narrative (the default) -> no FedNarrative rows (backward compatible)."""
    sources = syn.build_sources(as_of="2026-06-01", indicators={}, analogs=[], equity_data={})
    assert all(s["tool"] != "FedNarrative" for s in sources)


def test_generate_report_renders_fed_citations():
    """A Fed-narrative source row is rendered in the brief's Sources block."""
    sources = syn.build_sources(
        as_of="2026-06-01", indicators={}, analogs=[], equity_data={},
        fed_narrative=[{"source": "fomc_minutes", "date": "2026-04-30", "title": "Minutes",
                        "url": "", "text": "Inflation remained elevated.", "similarity": 0.83}],
    )
    out = syn.generate_report(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.LATE_CYCLE,
        rankings=[{"sector": "XLU", "score": 0.5, "rank": 1, "confidence": 0.4, "detail": {}}],
        confidence=0.4, flags=[], audit_log=make_audit_log(), sources=sources,
    )
    assert "FedNarrative" in out
    assert "fomc minutes" in out


# --------------------------------------------------------------------------- #
# Fed narrative in the executive summary (qualitative context, NOT a scoring input)
# --------------------------------------------------------------------------- #
def test_summary_facts_includes_fed_narrative():
    """_summary_facts surfaces the top Fed passages as qualitative context, explicitly
    labelled NOT a scoring input; absent fed_sources, the block is omitted."""
    fed_sources = [
        {"id": "fed#1: fomc minutes 2026-04-30", "tool": "FedNarrative", "value": 0.83,
         "as_of": "2026-06-01",
         "label": "Fed fomc minutes (2026-04-30): 'inflation remained elevated'"},
    ]
    rankings = [{"sector": "XLU", "score": 0.5, "rank": 1, "confidence": 0.4, "detail": {}}]
    with_fed = syn._summary_facts(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.LATE_CYCLE, rankings=rankings,
        confidence=0.4, flags=[], fed_sources=fed_sources
    )
    assert "Fed narrative" in with_fed
    assert "NOT a scoring input" in with_fed
    assert "inflation remained elevated" in with_fed

    without = syn._summary_facts(query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.LATE_CYCLE,
                                 rankings=rankings, confidence=0.4, flags=[])
    assert "Fed narrative" not in without


def test_generate_executive_summary_passes_fed_context_to_model():
    """generate_executive_summary threads the Fed passages into the model's prompt, and
    the system prompt tells the editor to treat them as non-scoring corroboration."""
    captured: dict[str, str] = {}

    def recording_model(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return SUMMARY_TEXT

    fed_sources = [
        {"id": "fed#1: fomc minutes 2026-04-30", "tool": "FedNarrative", "value": 0.83,
         "as_of": "2026-06-01",
         "label": "Fed fomc minutes (2026-04-30): 'inflation remained elevated'"},
    ]
    rankings = [{"sector": "XLU", "score": 0.5, "rank": 1, "confidence": 0.4, "detail": {}}]
    syn.generate_executive_summary(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.LATE_CYCLE, rankings=rankings,
        confidence=0.4, flags=[], fed_sources=fed_sources, call_model=recording_model,
    )
    assert "inflation remained elevated" in captured["user"]
    assert "Fed-narrative" in captured["system"]
    assert "NOT inputs to the sector scores" in captured["system"]


def test_generate_report_summary_draws_fed_from_sources():
    """generate_report pulls the FedNarrative rows out of `sources` and feeds them to the
    summary, so the passages reach the model with no extra plumbing."""
    captured: dict[str, str] = {}

    def recording_model(system: str, user: str) -> str:
        captured["user"] = user
        return SUMMARY_TEXT

    sources = syn.build_sources(
        as_of="2026-06-01", indicators={}, analogs=[], equity_data={},
        fed_narrative=[{"source": "fomc_minutes", "date": "2026-04-30", "title": "Minutes",
                        "url": "", "text": "Participants judged inflation remained elevated.",
                        "similarity": 0.83}],
    )
    syn.generate_report(
        query=QUERY, as_of=AS_OF_DATE, regime=const.Regime.LATE_CYCLE,
        rankings=[{"sector": "XLU", "score": 0.5, "rank": 1, "confidence": 0.4, "detail": {}}],
        confidence=0.4, flags=[], audit_log=make_audit_log(), sources=sources,
        call_model=recording_model,
    )
    assert "Fed narrative" in captured["user"]
    assert "inflation remained elevated" in captured["user"]


# =========================================================================== #
# ADDITIONAL TEST CASES TO IMPLEMENT LATER
# =========================================================================== #
# compute_sector_score:
#   - parametrize across all four regimes (the tilt prior changes; ranking shifts)
#   - signal isolation: with w_regime=1.0, ranking is exactly the tilt order;
#     with w_equity=1.0, exactly the momentum order (one knob at a time)
#   - normalization: a single huge momentum value must not swamp the tilt/analog
#     signals (the magnitude lesson from _vectorize)
#   - similarity weighting: a high-similarity analog should move a sector more than
#     a low-similarity one with the same returns
#   - low-confidence FLAG surfaced when < min_strong_analogs strong analogs
#   - detail sub-scores are present and internally consistent with score
#   - empty equity_data dict (not just partial) still ranks all 11
#   - ties in score -> stable, deterministic rank assignment
#
# generate_report (placeholder brief covered above; pending the full Section 5 version):
#   - cites its sources inline: build_sources now produces the `sources` list and
#     generate_report accepts it, but _render_sources is not yet wired into the body
#     (its call is commented out); once it is, assert the source ids/values appear
#   - per-sector rationale drawn from each row's `detail` sub-scores
#   - a regime narrative wrapped around the ranking (not just the bare numbers)
#
# build_sources / _render_sources (new in synthesize.py, not yet covered here):
#   - build_sources: ordering (indicators -> momentum -> analogs strongest-first),
#     cpi_inflation tagged FRED:CPIAUCSL (YoY derived), None/missing similarity sorts
#     last, partial equity + empty analogs degrade cleanly
#   - _render_sources: id-prefixed lines, _fmt_value (None -> n/a, 2dp numbers),
#     empty sources -> explicit 'none recorded'
