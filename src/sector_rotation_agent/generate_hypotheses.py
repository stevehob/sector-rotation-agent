"""
generate_hypotheses.py

Fan-out step of the Tree-of-Thought regime classifier (spec Section 3.3).

WHAT THIS DOES
--------------
Given the current macro snapshot, propose the handful of regimes the data could
*plausibly* support — each with a short rationale and an initial plausibility
(`prior`). This is the DIVERGENT half of the ToT: favor recall over precision.
A regime never proposed here can never be selected later, so it is better to
surface a borderline candidate and let the rigorous evaluation step
(find_historical_analogs + score_branch) prune it than to omit it now.

This is the right place for an LLM: turning a vector of indicators into a small
set of *named, reasoned* interpretations is judgment, not arithmetic. Contrast
with score_branch, which is deliberately deterministic.

HOW IT FITS
-----------
classify_regime_tot() calls this as its `generate_hypotheses` dependency:

    hypotheses = generate_hypotheses(snapshot, max_branches)

It must return a list[RegimeHypothesis]. classify_regime_tot raises if the list
is empty, so surfacing parse failures (rather than silently returning []) is the
safer choice.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import sector_rotation_agent.constants as const
from sector_rotation_agent.classify_regime_tot import (
    MacroSnapshot,
    RegimeHypothesis,
)

from sector_rotation_agent.model_client import make_model_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The SYSTEM_PROMPT defines the contract with the model. It MUST be clear and specific
# It MUST make the model:
#   (1) choose ONLY from the known regimes (the Regime enum values);
#   (2) return STRICT JSON and nothing else — no prose, no markdown fences;
#   (3) return a JSON array of objects, each with keys: regime, rationale, prior;
#   (4) think DIVERGENTLY — include every regime the data could plausibly fit,
#       up to the requested maximum, not just the single best guess.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
    You are evaluating the current macroeconomic environment based on a set of indicators.

    Your task is to propose which economic regimes could plausibly fit this environment.

    Be diverse in your proposals — include every regime that could reasonably match the indicators, up to the maximum of %MAX_HYPOTHESES%.
    This is a brainstorming step, not a final decision.

    You should only choose from the following known regimes: ["early_cycle", "mid_cycle", "late_cycle", "contraction"].
      - early_cycle indicates a period of recovery or early expansion;
      - mid_cycle indicates a period of steady expansion;
      - late_cycle indicates a period of peak or slowing expansion;
      - contraction indicates a period of recession or early downturn.

    For each regime you propose, provide a short rationale explaining why this regime could fit the current indicators
      and assign a prior probability (between 0.0 and 1.0) reflecting how plausible you think this regime is given the data.

    Return your answer STRICTLY as a JSON array of objects, where each object has the following keys:
      - "regime": one of the allowed regime strings (e.g. "early_cycle")
      - "rationale": a brief explanation of why this regime could fit the current macroeconomic indicators
      - "prior": a number between 0.0 and 1.0 representing the plausibility of this regime given the data

    Here is an example of the expected output format:
    [
      {
        "regime": "early_cycle",
        "rationale": "The leading index is negative, which often signals a downturn, but the fed funds rate is still relatively low, suggesting we could be in an early cycle.",
        "prior": 0.4
      },
      {
        "regime": "mid_cycle",
        "rationale": "The unemployment rate is moderate and the yield spread is positive, which can be consistent with a mid-cycle expansion, though the negative leading index tempers this view.",
        "prior": 0.3
      },
      {
        "regime": "late_cycle",
        "rationale": "The fed funds rate is relatively high and the yield spread is flattening, which can be signs of a late cycle, but the negative leading index and moderate unemployment suggest caution in assigning a high probability to this regime.",
        "prior": 0.2
      }
    ]

    Do not include any prose, explanations, or markdown fences in your response.
"""

def generate_hypotheses(
    snapshot: MacroSnapshot,
    max_hypotheses: int,
    *,
    call_model: Callable[[str, str], str] | None = None,
) -> list[RegimeHypothesis]:
    """
    Propose up to `max_hypotheses` candidate regimes for the given snapshot.

    Returns
    -------
    list[RegimeHypothesis]
        Non-empty on success. classify_regime_tot treats an empty list as an
        error, so prefer raising inside _parse_hypotheses over returning [].
    """
    if call_model is None:
        call_model = make_model_client().complete

    user_prompt = _build_user_prompt(snapshot)
    system_prompt = SYSTEM_PROMPT.replace("%MAX_HYPOTHESES%", str(max_hypotheses))
    raw = call_model(system_prompt, user_prompt)
    hypotheses = _parse_hypotheses(raw, max_hypotheses)
    logger.info("Hypothesis fan-out parsed %d regime(s): %s",
                len(hypotheses), [h.regime.value for h in hypotheses])
    return hypotheses


def _build_user_prompt(snapshot: MacroSnapshot) -> str:
    """
    Turn the macro snapshot into the user message.

    Consideration: the more legible the indicator block, the better the
    rationales. Bare floats with no labels produce weak reasoning.
    """
    question = "Which economic regimes could plausibly fit this environment?"

    return (
        f"Given the following macroeconomic snapshot as of {snapshot.as_of}:\n"
        + "\n".join(f"{k}: {v}" for k, v in snapshot.indicators.items())
        + "\n\n"
        + question
    )


def _parse_hypotheses(raw: str, max_hypotheses: int) -> list[RegimeHypothesis]:
    """
    Parse the model's JSON text into validated RegimeHypothesis objects.

    Returns
    -------
    list[RegimeHypothesis]

    Expected Format of `raw`
    ------------------
    [
      {
        "regime": "early_cycle",
        "rationale": "The leading index is negative, which often signals a downturn, but the fed funds rate is still relatively low, suggesting we could be in an early cycle.",
        "prior": 0.4
      },
      {
        "regime": "mid_cycle",
        "rationale": "The unemployment rate is moderate and the yield spread is positive, which can be consistent with a mid-cycle expansion, though the negative leading index tempers this view.",
        "prior": 0.3
      },
      {
        "regime": "late_cycle",
        "rationale": "The fed funds rate is relatively high and the yield spread is flattening, which can be signs of a late cycle, but the negative leading index and moderate unemployment suggest caution in assigning a high probability to this regime.",
        "prior": 0.2
      }
    ]
    """
    # Clean the raw output before parsing. Models often wrap JSON in Markdown
    # fences, and strip("```json") would remove arbitrary matching characters.

    cleaned = raw.strip()
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    cleaned = "\n".join(lines).strip()

    if not cleaned.startswith(("[", "{")):
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        if array_start != -1 and array_end != -1:
            cleaned = cleaned[array_start:array_end + 1]
        elif object_start != -1 and object_end != -1:
            cleaned = cleaned[object_start:object_end + 1]


    # attempt loading the JSON, with error handling for malformed input
    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Hypothesis JSON parse failed: %s", e)
        raise ValueError(
            f"Failed to parse JSON from model output: {e}\nRaw output was:\n{raw}"
        ) from e
    
    
    if not isinstance(parsed_json, list):
        logger.error("Model output was not a JSON array (got %s)", type(parsed_json).__name__)
        raise ValueError("Model output must be a JSON array of hypotheses")

    results: list[RegimeHypothesis] = []
    for item in parsed_json:
        if len(results) >= max_hypotheses:
            break
        if not isinstance(item, dict):
            continue
        # Check for required fields
        if "regime" not in item:
            logger.warning("Skipping non-compliant hypothesis item: %r", item)
            # for now we skip such items
            continue
        # Check if the regime is one of the known regimes
        if item["regime"] not in {r.value for r in const.Regime}:
            logger.debug("Skipping hallucinated regime not in the enum: %r", item["regime"])
            continue
        # Check if the regime has already been proposed (deduplication)
        if any(h.regime.value == item["regime"] for h in results):
            logger.debug("Skipping duplicate regime already proposed: %r", item["regime"])
            continue

        # Should be clean for adding this regime now — validate and clamp the prior
        try:
            prior = float(item.get("prior", 0.5))  # default to 0.5 if missing
        except (ValueError, TypeError):
            logger.debug(
                "Invalid prior %r for regime %r; defaulting to 0.5",
                item.get("prior"), item.get("regime"),
            )
            prior = 0.5
        prior = max(0.0, min(1.0, prior))  # clamp to [0.0, 1.0]

        results.append(
            RegimeHypothesis(
                regime=const.Regime(item["regime"]),
                rationale=str(item.get("rationale", "")),
                prior=prior,
            )
        )

    if len(results) == 0:
        logger.error("No valid hypotheses parsed from model output (after cleaning/validation)")
        raise ValueError(f"No valid hypotheses parsed from model output. Raw output was:\n{raw}")

    return results
