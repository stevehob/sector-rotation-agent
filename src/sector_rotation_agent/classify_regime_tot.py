"""
classify_regime_tot.py

Tree-of-Thought (ToT) macro regime classification for the Sector
Rotation Research Agent (see project spec, Section 3.3).

This module implements the ONE bounded Tree-of-Thought step in the system. The
rest of the agent runs a ReAct loop.

ToT is scoped here because macro regime classification is the single decision point
where the same data can support several valid hypotheses and committing early causes
revision-loop thrashing.

Control flow:

    macro snapshot
         │
         ▼
    _generate_hypotheses()      ← fan out: LLM proposes N plausible regimes (divergent)
         │
         ▼
    _evaluate_branch() × N       ← expand: each branch pulls its own analogs + scores
         │                          (run in parallel; independent of one another)
         ▼
    _select_and_prune()          ← converge: pick best-supported branch, prune rest
         │
         ▼
    ToTResult (winner + all branch reasoning for the audit log)

Integration points that depend on your existing code are marked `# INTEGRATE:`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol
import logging

import sector_rotation_agent.constants as const

logger = logging.getLogger(__name__)

@dataclass
class MacroSnapshot:
    """
    Normalized point-in-time view of the macro environment.
    """
    as_of: str                          # ISO date of the snapshot
    indicators: dict[str, float]        # raw values, e.g. {"FEDFUNDS": 5.25, ...}


@dataclass
class RegimeHypothesis:
    """A single candidate branch produced during fan-out."""
    regime: const.Regime
    rationale: str                      # why the LLM thinks this regime is plausible
    prior: float                        # initial plausibility 0..1 from fan-out


@dataclass
class BranchResult:
    """Fully evaluated branch — the unit recorded in the audit log."""
    hypothesis: RegimeHypothesis
    analog_similarity: float            # strength of historical analog match 0..1
    signal_consistency: float           # agreement of implied tilts vs live momentum 0..1
    support_score: float                # combined score used for selection
    pruned: bool = False
    prune_reason: str | None = None
    detail: dict = field(default_factory=dict)   # raw analogs, sub-scores, etc.


@dataclass
class ToTResult:
    """Return value of classify_regime_tot()."""
    selected: BranchResult
    branches: list[BranchResult]        # ALL branches, including pruned ones
    low_confidence: bool                # True if top branches were too close to separate
    audit_entry: dict                   # ready to append to the session audit log


# --------------------------------------------------------------------------- #
# Injected dependencies (kept as parameters so the function stays testable)
# --------------------------------------------------------------------------- #

class AnalogFinder(Protocol):
    """Signature of existing find_historical_analogs tool."""
    def __call__(self, snapshot: MacroSnapshot, n: int,
                 regime_filter: const.Regime | None = None) -> list[dict]: ...


class HypothesisGenerator(Protocol):
    """Wraps a single LLM call that proposes candidate regimes (fan-out)."""
    def __call__(self, snapshot: MacroSnapshot, max_hypotheses: int
                 ) -> list[RegimeHypothesis]: ...



# --------------------------------------------------------------------------- #
# Agent entry point
# --------------------------------------------------------------------------- #

def classify_regime_tot(
    snapshot: MacroSnapshot,
    *,
    generate_hypotheses: HypothesisGenerator,
    find_historical_analogs: AnalogFinder,
    score_branch: Callable[[RegimeHypothesis, list[dict], MacroSnapshot], BranchResult],
    max_branches: int = const.MAX_BRANCHES,
    analogs_per_branch: int = const.ANALOGS_PER_BRANCH,
    tie_margin: float = const.TIE_MARGIN,
) -> ToTResult:
    """
    Classify the current macro regime via a bounded Tree-of-Thought.

    Parameters
    ----------
    snapshot
        The current normalized macro snapshot.
    generate_hypotheses
        Fan-out step. An LLM-backed callable that proposes the most plausible
        candidate regimes. Divergent on purpose — favor recall over precision;
        the evaluation step enforces rigor.
    find_historical_analogs
        Your existing retrieval tool. Called once per branch, filtered to that
        branch's regime so each hypothesis is tested against its own analogs.
    score_branch
        Convergent scoring. Combines analog-match strength with how well the
        regime's implied sector tilts agree with current momentum. Returns a
        fully populated BranchResult (minus prune flags).
    max_branches
        Hard cap on fan-out width. Three is plenty; more burns tokens for little gain.
    analogs_per_branch
        How many historical analogs to retrieve when evaluating each branch.
    tie_margin
        If the top two support scores differ by less than this, the result is
        flagged low_confidence (feeds the confidence guardrail downstream).

    Returns
    -------
    ToTResult
        The selected branch, every branch's reasoning (for the audit log), a
        low-confidence flag, and a ready-to-store audit entry.
    """
    # 1. FAN OUT ------------------------------------------------------------- #
    logger.info("ToT classify: as_of=%s, max_branches=%d", snapshot.as_of, max_branches)
    hypotheses = generate_hypotheses(snapshot, max_branches)
    if not hypotheses:
        logger.error("ToT fan-out produced no candidate regimes for as_of=%s", snapshot.as_of)
        raise ValueError("Fan-out produced no candidate regimes")
    hypotheses = hypotheses[:max_branches]
    logger.info("ToT fan-out: %d candidate regime(s): %s",
                len(hypotheses), [h.regime.value for h in hypotheses])

    # 2. EXPAND each branch independently (safe to parallelize: no shared state) #
    def _expand(h: RegimeHypothesis) -> BranchResult:
        return _evaluate_branch(
            h, snapshot, find_historical_analogs, score_branch, analogs_per_branch
        )

    with ThreadPoolExecutor(max_workers=max_branches) as pool:
        branches = list(pool.map(_expand, hypotheses))

    # 3. CONVERGE: select the strongest branch, prune the rest ---------------- #
    selected, low_confidence = _select_and_prune(branches, tie_margin)
    logger.info("ToT selected regime: %s (support=%.4f, low_confidence=%s)",
                selected.hypothesis.regime.value, selected.support_score, low_confidence)
    if low_confidence:
        logger.warning("ToT low confidence: top branches within tie margin %.3f for as_of=%s",
                       tie_margin, snapshot.as_of)

    # 4. Package the audit entry --------------------------------------------- #
    audit_entry = _build_audit_entry(snapshot, branches, selected, low_confidence)

    return ToTResult(
        selected=selected,
        branches=branches,
        low_confidence=low_confidence,
        audit_entry=audit_entry,
    )


# --------------------------------------------------------------------------- #
# Internal steps
# --------------------------------------------------------------------------- #

def _evaluate_branch(
    hypothesis: RegimeHypothesis,
    snapshot: MacroSnapshot,
    find_historical_analogs: AnalogFinder,
    score_branch: Callable[[RegimeHypothesis, list[dict], MacroSnapshot], BranchResult],
    analogs_per_branch: int,
) -> BranchResult:
    """
    Expand a single hypothesis into a scored branch.

    Each branch is tested against analogs retrieved *under its own regime label*,
    so a 'contraction' branch is judged only against historical contractions. This
    is what makes the comparison fair across branches.
    """
    # INTEGRATE: your retrieval tool, filtered to this branch's regime.
    analogs = find_historical_analogs(
        snapshot, n=analogs_per_branch, regime_filter=hypothesis.regime
    )
    logger.debug("ToT branch %s: retrieved %d analog(s)",
                 hypothesis.regime.value, len(analogs))

    # Scoring model:
    #   support_score = w1 * analog_similarity + w2 * signal_consistency
    # where analog_similarity is the mean cosine match of retrieved analogs and
    # signal_consistency measures whether the regime's expected sector leadership
    # agrees with current momentum data. Tune weights against a backtest.
    return score_branch(hypothesis, analogs, snapshot)


def _select_and_prune(
    branches: list[BranchResult],
    tie_margin: float,
) -> tuple[BranchResult, bool]:
    """
    Pick the highest-support branch; mark the others pruned with a reason.

    Returns (selected_branch, low_confidence_flag).
    """
    ranked = sorted(branches, key=lambda b: b.support_score, reverse=True)
    logger.debug("ToT branch ranking: %s",
                 [(b.hypothesis.regime.value, round(b.support_score, 4)) for b in ranked])
    winner = ranked[0]

    low_confidence = (
        len(ranked) > 1
        and (winner.support_score - ranked[1].support_score) < tie_margin
    )

    for b in ranked[1:]:
        b.pruned = True
        b.prune_reason = (
            f"support {b.support_score:.3f} below winning "
            f"{winner.hypothesis.regime.value} ({winner.support_score:.3f})"
        )

    return winner, low_confidence


def _build_audit_entry(
    snapshot: MacroSnapshot,
    branches: list[BranchResult],
    selected: BranchResult,
    low_confidence: bool,
) -> dict:
    """
    Produce the append-only audit-log record for this ToT step.

    Records EVERY branch (including pruned ones) so the synthesis agent and the
    final report can show the reasoning that was considered and rejected — not
    just the surviving conclusion.
    """
    return {
        "step": "classify_regime_tot",
        "as_of": snapshot.as_of,
        "selected_regime": selected.hypothesis.regime.value,
        "selected_support": round(selected.support_score, 4),
        "low_confidence": low_confidence,
        "branches": [
            {
                "regime": b.hypothesis.regime.value,
                "support": round(b.support_score, 4),
                "analog_similarity": round(b.analog_similarity, 4),
                "signal_consistency": round(b.signal_consistency, 4),
                "pruned": b.pruned,
                "prune_reason": b.prune_reason,
                "rationale": b.hypothesis.rationale,
            }
            for b in branches
        ],
    }


# --------------------------------------------------------------------------- #
# Example wiring (remove once integrated into the macro agent)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from sector_rotation_agent.generate_hypotheses import generate_hypotheses
    from sector_rotation_agent.historical_analogs import find_historical_analogs
    from sector_rotation_agent.score_branch import score_branch

    # Chroma DB problem debugging  ---------------------------------
    #import sys, chromadb; print(sys.executable, chromadb.__version__)

    #import sys, chromadb
    #print(sys.executable)
    #c = chromadb.PersistentClient(path=r"C:\Users\steve\GitHub\cmu-agentic-ai-capstone-project\data\chroma")
    #print([x.name for x in c.list_collections()])
    # -----------------------------------------------------------------


    # This block shows the intended call shape. The three injected callables
    # below are placeholders — wire them to your real implementations.

    def _stub_generate(snapshot: MacroSnapshot, n: int) -> list[RegimeHypothesis]:
        return generate_hypotheses(snapshot, n)

    def _stub_analogs(snapshot: MacroSnapshot, n:int , regime_filter: const.Regime | None=None) -> list[dict]:
        return find_historical_analogs(snapshot, n, regime_filter)

    # current sector momentum normally comes from the equity agent
    # (get_sector_performance over the momentum lookback). Hardcoded here only so
    # the example shows the call shape.
    _demo_momentum: dict[str, float] = {
        "XLK": 0.08, "XLC": 0.05, "XLI": 0.04, "XLY": 0.03, "XLF": 0.01,
        "XLB": 0.00, "XLRE": -0.01, "XLE": -0.02, "XLV": -0.03,
        "XLP": -0.04, "XLU": -0.05,
    }

    _demo_macro_snapshot = MacroSnapshot(
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

    def _stub_score(hypothesis, analogs, snapshot) -> BranchResult:
        return score_branch(hypothesis=hypothesis, analogs=analogs, snapshot=snapshot, current_momentum=_demo_momentum)

    result = classify_regime_tot(
        _demo_macro_snapshot,
        generate_hypotheses=_stub_generate,                             # type: ignore
        find_historical_analogs=_stub_analogs,                          # type: ignore
        score_branch=_stub_score,
    )
    print(f"Regime analysis concludes that we are in: {result.selected.hypothesis.regime}")
