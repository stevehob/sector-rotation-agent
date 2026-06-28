"""
Tests for sector_rotation_agent.classify_regime_tot.

Regression coverage:
  - 

NOTE: 
"""
import pytest

from sector_rotation_agent.generate_hypotheses import generate_hypotheses, RegimeHypothesis
from sector_rotation_agent.historical_analogs import find_historical_analogs
from sector_rotation_agent.score_branch import score_branch
from sector_rotation_agent.classify_regime_tot import MacroSnapshot, BranchResult, classify_regime_tot
import sector_rotation_agent.constants as const

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def macro_snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        as_of="2026-06-01",
        indicators={
            "fed_funds_rate": 4.50,
            "cpi_inflation": 3.1,
            "pce_inflation": 2.5,
            "unemployment": 4.3,
            "yield_spread_10_2": 0.15,
            "gdp_growth": 2.0,
            "ism_pmi": 12950.0,
            "leading_index": -0.2,
        },
    )

@pytest.fixture
def momentum() -> dict[str, float]:
    return {
        "XLK": 0.08, "XLC": 0.05, "XLI": 0.04, "XLY": 0.03, "XLF": 0.01,
        "XLB": 0.00, "XLRE": -0.01, "XLE": -0.02, "XLV": -0.03,
        "XLP": -0.04, "XLU": -0.05,
    }


# --------------------------------------------------------------------------- #
# Functional tests
# --------------------------------------------------------------------------- #
def test_classify_regime_tot_end_to_end(macro_snapshot, momentum):

    # This block shows the intended call shape. The three injected callables
    # below are placeholders — wire them to your real implementations.
    def _stub_generate(snapshot: MacroSnapshot, n: int) -> list[RegimeHypothesis]:
        return generate_hypotheses(snapshot, n)

    def _stub_analogs(snapshot: MacroSnapshot, n:int , regime_filter: const.Regime | None=None) -> list[dict]:
        return find_historical_analogs(snapshot, n, regime_filter)

    def _stub_score(hypothesis, analogs, snapshot) -> BranchResult:
        return score_branch(hypothesis=hypothesis, analogs=analogs, snapshot=snapshot, current_momentum=momentum)

    result = classify_regime_tot(
        macro_snapshot,
        generate_hypotheses=_stub_generate,                             # type: ignore
        find_historical_analogs=_stub_analogs,                          # type: ignore
        score_branch=_stub_score,
    )

    assert result.selected.hypothesis.regime in const.Regime
    assert result.selected.hypothesis.regime == const.Regime.MID_CYCLE

