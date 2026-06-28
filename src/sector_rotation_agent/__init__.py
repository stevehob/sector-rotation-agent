"""
sector_rotation_agent

Macro-Driven Sector Rotation Research Agent (CMU Agentic AI capstone).

Public API is re-exported here so callers can do, e.g.:

    from sector_rotation_agent import classify_regime_tot, Regime, score_branch
"""

from sector_rotation_agent.classify_regime_tot import (
    BranchResult,
    MacroSnapshot,
    RegimeHypothesis,
    ToTResult,
    classify_regime_tot,
)
from sector_rotation_agent.score_branch import SECTOR_TILTS, score_branch

__all__ = [
    "classify_regime_tot",
    "score_branch",
    "MacroSnapshot",
    "RegimeHypothesis",
    "BranchResult",
    "ToTResult",
    "SECTOR_TILTS",
]
