"""
Tests for the sector_rotation_agent package.

Seeded from the validation of `classify_regime_tot` and `score_branch`. These
exercise the public API plus the ToT prune logic and a full fan-out/converge run
with stubbed retrieval (dependency injection makes that possible without the
vector store or any network access).

Run with:
    uv run pytest
"""

from functools import partial

import pytest

import sector_rotation_agent.constants as const
import sector_rotation_agent as sra
from sector_rotation_agent import (
    BranchResult,
    MacroSnapshot,
    RegimeHypothesis,
    classify_regime_tot,
    score_branch,
)
from sector_rotation_agent.classify_regime_tot import _select_and_prune


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def momentum() -> dict[str, float]:
    """An expansionary tape: tech/comm/industrials strong, defensives weak."""
    return {
        "XLK": 0.11, "XLC": 0.09, "XLI": 0.06,
        "XLF": 0.02, "XLY": 0.01, "XLB": -0.01,
        "XLRE": -0.02, "XLE": -0.03,
        "XLV": -0.04, "XLP": -0.05, "XLU": -0.06,
    }


@pytest.fixture
def snapshot() -> MacroSnapshot:
    return MacroSnapshot(as_of="2026-06-01", indicators={})


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #

def test_package_exports():
    for name in [
        "classify_regime_tot", "score_branch", "MacroSnapshot",
        "RegimeHypothesis", "BranchResult", "ToTResult", "SECTOR_TILTS",
    ]:
        assert hasattr(sra, name), f"missing export: {name}"


# --------------------------------------------------------------------------- #
# score_branch
# --------------------------------------------------------------------------- #

def test_consistent_regime_outscores_contradicted_one(momentum, snapshot):
    scorer = partial(score_branch, current_momentum=momentum)
    strong = [
        {"date": "2017-05", "similarity": 0.88},
        {"date": "2014-09", "similarity": 0.81},
        {"date": "2006-02", "similarity": 0.79},
    ]
    weak = [{"date": "2008-10", "similarity": 0.64}]

    mid = scorer(RegimeHypothesis(const.Regime.MID_CYCLE, "expansion", 0.6), strong, snapshot)
    con = scorer(RegimeHypothesis(const.Regime.CONTRACTION, "downturn", 0.2), weak, snapshot)

    assert mid.support_score > con.support_score
    # Markets agree with mid-cycle's tilts and oppose contraction's.
    assert mid.signal_consistency > 0.5
    assert con.signal_consistency < 0.5


def test_thin_analogs_are_discounted(momentum, snapshot):
    # One non-strong analog -> zero strong -> strength_factor 0 -> analog 0.
    res = score_branch(
        RegimeHypothesis(const.Regime.MID_CYCLE, "x", 0.5),
        [{"date": "x", "similarity": 0.50}],
        snapshot,
        current_momentum=momentum,
    )
    assert res.analog_similarity == 0.0
    assert res.detail["analog"]["n_strong"] == 0


def test_no_analogs_scores_zero_analog(momentum, snapshot):
    res = score_branch(
        RegimeHypothesis(const.Regime.MID_CYCLE, "x", 0.5), [], snapshot,
        current_momentum=momentum,
    )
    assert res.analog_similarity == 0.0


def test_signal_consistency_neutral_on_insufficient_overlap(snapshot):
    # Only two shared sectors -> cannot correlate -> neutral 0.5.
    res = score_branch(
        RegimeHypothesis(const.Regime.MID_CYCLE, "x", 0.5),
        [{"date": "x", "similarity": 0.90}],
        snapshot,
        current_momentum={"XLK": 0.1, "XLC": 0.1},
    )
    assert res.signal_consistency == 0.5


def test_weights_must_sum_to_one(momentum, snapshot):
    with pytest.raises(ValueError):
        score_branch(
            RegimeHypothesis(const.Regime.MID_CYCLE, "x", 0.5), [], snapshot,
            current_momentum=momentum, w_analog=0.7, w_signal=0.7,
        )


# --------------------------------------------------------------------------- #
# classify_regime_tot — prune logic
# --------------------------------------------------------------------------- #

def _branch(regime: const.Regime, support: float) -> BranchResult:
    return BranchResult(
        hypothesis=RegimeHypothesis(regime, "x", 0.5),
        analog_similarity=support, signal_consistency=support,
        support_score=support,
    )


def test_select_and_prune_picks_highest_support():
    b1 = _branch(const.Regime.MID_CYCLE, 0.86)
    b2 = _branch(const.Regime.LATE_CYCLE, 0.55)
    winner, low = _select_and_prune([b1, b2], tie_margin=0.05)
    assert winner.hypothesis.regime is const.Regime.MID_CYCLE
    assert b2.pruned and not b1.pruned
    assert low is False


def test_select_and_prune_flags_ties():
    b1 = _branch(const.Regime.MID_CYCLE, 0.86)
    b2 = _branch(const.Regime.LATE_CYCLE, 0.84)
    _, low = _select_and_prune([b1, b2], tie_margin=0.05)
    assert low is True


# --------------------------------------------------------------------------- #
# classify_regime_tot — full fan-out/converge with stubbed retrieval
# --------------------------------------------------------------------------- #

def test_tot_end_to_end_selects_market_consistent_regime(momentum, snapshot):
    def fake_generate(snap, n):
        return [
            RegimeHypothesis(const.Regime.MID_CYCLE, "expansion", 0.5),
            RegimeHypothesis(const.Regime.LATE_CYCLE, "peak", 0.3),
            RegimeHypothesis(const.Regime.CONTRACTION, "downturn", 0.2),
        ][:n]

    def fake_analogs(snap, n, regime_filter=None):
        # Mid-cycle gets strong analogs; the others only a weak one.
        if regime_filter is const.Regime.MID_CYCLE:
            return [
                {"date": "2017-05", "similarity": 0.88},
                {"date": "2014-09", "similarity": 0.82},
                {"date": "2006-02", "similarity": 0.78},
            ]
        return [{"date": "2008-10", "similarity": 0.55}]

    result = classify_regime_tot(
        snapshot,
        generate_hypotheses=fake_generate,          # type: ignore
        find_historical_analogs=fake_analogs,       # type: ignore
        score_branch=partial(score_branch, current_momentum=momentum),
        max_branches=3,
    )

    assert result.selected.hypothesis.regime is const.Regime.MID_CYCLE
    assert len(result.branches) == 3
    assert result.audit_entry["selected_regime"] == "mid_cycle"
    assert sum(1 for b in result.branches if b.pruned) == 2
