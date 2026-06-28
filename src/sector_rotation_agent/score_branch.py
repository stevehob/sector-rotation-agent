"""
score_branch.py

Concrete branch-scoring implementation for the Tree-of-Thought regime
classifier (plugs into `classify_regime_tot`, see project spec Section 3.3).

A branch's support score answers: "how well does this regime hypothesis hold up
under independent evidence?" It combines two deliberately different signals:

  1. analog_similarity  — BACKWARD looking. Do historical periods that resemble
     today actually fall under this regime, and are there enough strong matches
     to trust? Sourced from the vector store via find_historical_analogs.

  2. signal_consistency — PRESENT looking. Does the sector leadership this regime
     *predicts* agree with what sector momentum is *actually* doing right now?
     This is the cross-check: a regime can have great historical analogs yet be
     contradicted by current market behavior, and that disagreement should cost it.

support_score = w_analog * analog_similarity + w_signal * signal_consistency

Keeping the two signals separate (rather than one blended number) is what lets
the synthesis agent and the audit log explain *why* a regime won or lost.
"""
from __future__ import annotations
import logging
from typing import cast
from statistics import mean

from scipy.stats import spearmanr   # already in the project stack

import sector_rotation_agent.constants as const

from sector_rotation_agent.classify_regime_tot import (
    BranchResult,
    MacroSnapshot,
    RegimeHypothesis,
)

# iniitalize logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Domain assumption: expected sector leadership per business-cycle regime.
#
# +1 = regime theory expects this sector to LEAD (outperform)
# -1 = regime theory expects this sector to LAG (underperform)
#  0 = neutral / no strong directional expectation
#
# This is the classic business-cycle sector-rotation heuristic. It is a
# SIMPLIFICATION and should be validated against your own backtest — the exact
# tilts are a domain assumption worth defending explicitly in the final report.
# Tickers are the 11 GICS sector ETFs used throughout the project.
# --------------------------------------------------------------------------- #

SECTOR_TILTS: dict[const.Regime, dict[str, int]] = {
    const.Regime.EARLY_CYCLE: {   # recovery — cyclicals lead, defensives lag
        "XLY": +1, "XLF": +1, "XLI": +1, "XLB": +1, "XLRE": +1,
        "XLU": -1, "XLP": -1, "XLV": -1,
        "XLK": 0, "XLE": 0, "XLC": 0,
    },
    const.Regime.MID_CYCLE: {     # steady expansion — tech / comm / industrials lead
        "XLK": +1, "XLC": +1, "XLI": +1,
        "XLU": -1, "XLP": -1, "XLE": -1,
        "XLY": 0, "XLF": 0, "XLB": 0, "XLV": 0, "XLRE": 0,
    },
    const.Regime.LATE_CYCLE: {    # peak / slowing — inflation hedges + early defensives
        "XLE": +1, "XLB": +1, "XLV": +1, "XLP": +1,
        "XLY": -1, "XLK": -1, "XLRE": -1,
        "XLF": 0, "XLI": 0, "XLU": 0, "XLC": 0,
    },
    const.Regime.CONTRACTION: {   # recession — defensives lead, cyclicals lag
        "XLP": +1, "XLU": +1, "XLV": +1,
        "XLY": -1, "XLF": -1, "XLI": -1, "XLK": -1, "XLB": -1, "XLRE": -1,
        "XLE": 0, "XLC": 0,
    },
}


def score_branch(
    hypothesis: RegimeHypothesis,
    analogs: list[dict],
    snapshot: MacroSnapshot,
    *,
    current_momentum: dict[str, float],
    w_analog: float = 0.5,
    w_signal: float = 0.5,
    strong_similarity: float = 0.75,
    min_strong_analogs: int = 3,
) -> BranchResult:
    """
    Score one regime hypothesis into a fully populated BranchResult.

    Parameters
    ----------
    hypothesis
        The candidate regime from fan-out.
    analogs
        Historical analogs retrieved under this regime's filter. Each analog is
        expected to be a dict shaped like:
            {
              "date": "2001-03",
              "similarity": 0.87,                  # cosine match to today, 0..1
              "regime": "contraction",
              "subsequent_sector_returns": {...},  # optional, for reporting
            }
    snapshot
        Current macro snapshot (unused in the default scoring but passed through
        so custom implementations can reference raw indicators).
    current_momentum
        Live sector momentum keyed by ETF ticker, e.g. {"XLK": 0.08, "XLF": -0.02}.
        Bind this with functools.partial when wiring into classify_regime_tot
        (see the __main__ example).
    w_analog, w_signal
        Weights for the two components. Must sum to 1.0; tune against a backtest.
    strong_similarity
        Similarity at or above which an analog counts as "strong". Matches the
        0.75 threshold used by the downstream confidence guardrail.
    min_strong_analogs
        How many strong analogs are needed before historical support is trusted
        at full weight. Fewer than this discounts analog_similarity proportionally.

    Returns
    -------
    BranchResult
        With analog_similarity, signal_consistency, support_score, and a `detail`
        dict capturing the sub-scores for the audit log.
    """
    if abs((w_analog + w_signal) - 1.0) > 1e-9:
        logger.error("Invalid branch-score weights: w_analog=%s + w_signal=%s != 1.0",
                     w_analog, w_signal)
        raise ValueError("w_analog + w_signal must equal 1.0")

    analog_similarity, analog_detail = _aggregate_analog_similarity(
        analogs, strong_similarity, min_strong_analogs
    )
    signal_consistency, signal_detail = _signal_consistency(
        hypothesis.regime, current_momentum
    )

    support_score = w_analog * analog_similarity + w_signal * signal_consistency
    logger.debug("Branch %s: support=%.4f (analog=%.4f, signal=%.4f)",
                 hypothesis.regime.value, support_score, analog_similarity, signal_consistency)

    return BranchResult(
        hypothesis=hypothesis,
        analog_similarity=analog_similarity,
        signal_consistency=signal_consistency,
        support_score=support_score,
        detail={
            "weights": {"analog": w_analog, "signal": w_signal},
            "analog": analog_detail,
            "signal": signal_detail,
        },
    )


# --------------------------------------------------------------------------- #
# Component 1 — historical analog support (backward looking)
# --------------------------------------------------------------------------- #

def _aggregate_analog_similarity(
    analogs: list[dict],
    strong_similarity: float,
    min_strong_analogs: int,
) -> tuple[float, dict]:
    """
    Collapse a list of analogs into a single 0..1 support figure.

    Two ideas combine here:
      - raw match quality: the mean cosine similarity of the retrieved analogs.
      - evidence sufficiency: thin support is untrustworthy, so the raw figure is
        scaled by how close we are to `min_strong_analogs` strong matches. With
        enough strong analogs the factor is 1.0 (no discount); with none it is 0.

    This folds the "<3 strong analogs => low confidence" guardrail directly into
    the score rather than bolting it on afterward.
    """
    if not analogs:
        return 0.0, {"n_analogs": 0, "n_strong": 0, "mean_similarity": 0.0,
                     "strength_factor": 0.0, "note": "no analogs retrieved"}

    sims = [a.get("similarity", 0.0) for a in analogs]
    n_strong = sum(1 for s in sims if s >= strong_similarity)
    mean_sim = mean(sims)
    strength_factor = min(1.0, n_strong / min_strong_analogs)
    analog_similarity = mean_sim * strength_factor

    return analog_similarity, {
        "n_analogs": len(analogs),
        "n_strong": n_strong,
        "mean_similarity": round(mean_sim, 4),
        "strength_factor": round(strength_factor, 4),
    }


# --------------------------------------------------------------------------- #
# Component 2 — agreement with live sector momentum (present looking)
# --------------------------------------------------------------------------- #

def _signal_consistency(
    regime: const.Regime,
    current_momentum: dict[str, float],
) -> tuple[float, dict]:
    """
    Measure how well the regime's expected sector leadership agrees with what
    sectors are actually doing right now.

    Method: rank-correlate the regime's expected tilt vector (+1/0/-1 per sector)
    against current momentum across the sectors they share, using Spearman's rho:
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    rho in [-1, 1] is mapped to [0, 1]:
        +1  -> 1.0   markets are rotating exactly as the regime predicts
         0  -> 0.5   no relationship
        -1  -> 0.0   markets are doing the OPPOSITE of the regime's prediction

    Rank correlation (not Pearson) is used because we care about the ordering of
    sector leadership, not the magnitude of returns.
    """
    tilts = SECTOR_TILTS[regime]
    shared = [t for t in tilts if t in current_momentum]

    # Need at least 3 shared sectors with some tilt variation to correlate.
    expected = [tilts[t] for t in shared]
    actual = [current_momentum[t] for t in shared]
    if len(shared) < 3 or len(set(expected)) < 2:
        return 0.5, {"n_sectors": len(shared), "rho": None,
                     "note": "insufficient overlap/variation; neutral 0.5"}

    rho, _ = spearmanr(expected, actual)
    if rho != rho:   # NaN guard (e.g. zero variance slipped through)
        return 0.5, {"n_sectors": len(shared), "rho": None,
                     "note": "undefined correlation; neutral 0.5"}
    # rho is a numpy float, so this is fine, but pylint is complaining so strongly type with cast to be explicit
    consistency = (cast(float, rho) + 1.0) / 2.0
    return consistency, {
        "n_sectors": len(shared),
        "rho": round(cast(float, rho), 4),
        "expected_leaders": [t for t in shared if tilts[t] > 0],
        "expected_laggards": [t for t in shared if tilts[t] < 0],
    }


# --------------------------------------------------------------------------- #
# Example wiring + self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from functools import partial

    # Current sector momentum (6-month) from the equity agent.
    momentum = {
        "XLK": 0.11, "XLC": 0.09, "XLI": 0.06,      # tech/comm/industrials strong
        "XLF": 0.02, "XLY": 0.01, "XLB": -0.01,
        "XLRE": -0.02, "XLE": -0.03,
        "XLV": -0.04, "XLP": -0.05, "XLU": -0.06,   # defensives weak
    }
    snap = MacroSnapshot(as_of="2026-06-01", indicators={})

    # Strong analogs for the candidate regime.
    strong_analogs = [
        {"date": "2017-05", "similarity": 0.88, "regime": "mid_cycle"},
        {"date": "2014-09", "similarity": 0.81, "regime": "mid_cycle"},
        {"date": "2006-02", "similarity": 0.79, "regime": "mid_cycle"},
        {"date": "1997-11", "similarity": 0.72, "regime": "mid_cycle"},
    ]

    scorer = partial(score_branch, current_momentum=momentum)

    # MID_CYCLE should score well: tech/comm/industrials leading matches its tilts,
    # and it has 3 strong analogs.
    mid = scorer(RegimeHypothesis(const.Regime.MID_CYCLE, "expansion", 0.6), strong_analogs, snap)

    # CONTRACTION should score poorly on consistency: it predicts defensives lead,
    # but defensives are the weakest sectors right now (opposite rotation).
    weak_analogs = [{"date": "2008-10", "similarity": 0.64, "regime": "contraction"}]
    con = scorer(RegimeHypothesis(const.Regime.CONTRACTION, "downturn", 0.2), weak_analogs, snap)

    for label, br in [("MID_CYCLE", mid), ("CONTRACTION", con)]:
        print(f"{label:12} support={br.support_score:.3f} "
              f"analog={br.analog_similarity:.3f} signal={br.signal_consistency:.3f}")
        print(f"             detail={br.detail}")
