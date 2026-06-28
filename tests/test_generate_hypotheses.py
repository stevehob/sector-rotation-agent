"""
Tests for sector_rotation_agent.generate_hypotheses.

Covers the pure parser (_parse_hypotheses) against real-world model output, plus
generate_hypotheses end-to-end through its injected `call_model` seam (a fake
model -- no SDK, no network, no API key).

Regression coverage:
  - _parse_hypotheses must ACCEPT a JSON array and REJECT a bare object.
  - generate_hypotheses must fill %MAX_HYPOTHESES% into the system prompt, send the
    snapshot in the user prompt, and surface a parse failure as an error.

Provider routing now lives in model_client.ModelClient; the old _call_model tests
(Anthropic text-block joining, missing-key ValueError) move with it, into a future
test_model_client.py covering AnthropicClient.
"""

import json
import os
import dotenv
import pytest

import sector_rotation_agent.constants as const
from sector_rotation_agent.generate_hypotheses import _parse_hypotheses, generate_hypotheses
from sector_rotation_agent.classify_regime_tot import MacroSnapshot

dotenv.load_dotenv()

# =========================================================================== #
# _parse_hypotheses
# =========================================================================== #

def test_parses_valid_array():
    """Regression: the model returns a JSON ARRAY, which must parse."""
    raw = json.dumps([
        {"regime": "late_cycle", "rationale": "flat curve", "prior": 0.5},
        {"regime": "mid_cycle", "rationale": "low unemployment", "prior": 0.3},
    ])
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert [h.regime for h in out] == [const.Regime.LATE_CYCLE, const.Regime.MID_CYCLE]
    assert out[0].rationale == "flat curve"
    assert out[0].prior == 0.5


def test_rejects_top_level_object():
    """Regression: a bare object is not the contract; an array is required."""
    raw = json.dumps({"regime": "late_cycle", "rationale": "x", "prior": 0.5})
    with pytest.raises(ValueError):
        _parse_hypotheses(raw, max_hypotheses=3)


def test_strips_markdown_fences():
    raw = '```json\n[{"regime": "mid_cycle", "rationale": "x", "prior": 0.6}]\n```'
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert [h.regime for h in out] == [const.Regime.MID_CYCLE]


def test_extracts_array_from_surrounding_prose():
    raw = ('Here are the regimes: '
           '[{"regime": "contraction", "rationale": "x", "prior": 0.4}] '
           'Hope this helps.')
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert [h.regime for h in out] == [const.Regime.CONTRACTION]


def test_skips_hallucinated_regime():
    raw = json.dumps([
        {"regime": "recovery", "rationale": "not a real regime", "prior": 0.7},
        {"regime": "mid_cycle", "rationale": "valid", "prior": 0.5},
    ])
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert [h.regime for h in out] == [const.Regime.MID_CYCLE]


def test_dedupes_repeated_regime():
    raw = json.dumps([
        {"regime": "late_cycle", "rationale": "first", "prior": 0.6},
        {"regime": "late_cycle", "rationale": "second", "prior": 0.4},
    ])
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert len(out) == 1
    assert out[0].rationale == "first"   # first occurrence wins


def test_clamps_and_defaults_priors():
    raw = json.dumps([
        {"regime": "contraction", "rationale": "too high", "prior": 1.4},
        {"regime": "mid_cycle", "rationale": "non-numeric", "prior": "high"},
    ])
    out = _parse_hypotheses(raw, max_hypotheses=3)
    by_regime = {h.regime: h for h in out}
    assert by_regime[const.Regime.CONTRACTION].prior == 1.0      # clamped down
    assert by_regime[const.Regime.MID_CYCLE].prior == 0.5        # defaulted (non-numeric)


def test_defaults_missing_fields():
    raw = json.dumps([{"regime": "early_cycle"}])           # no prior, no rationale
    out = _parse_hypotheses(raw, max_hypotheses=3)
    assert out[0].prior == 0.5
    assert out[0].rationale == ""


def test_truncates_to_max_hypotheses():
    raw = json.dumps([
        {"regime": "early_cycle", "rationale": "a", "prior": 0.4},
        {"regime": "mid_cycle", "rationale": "b", "prior": 0.3},
        {"regime": "late_cycle", "rationale": "c", "prior": 0.2},
        {"regime": "contraction", "rationale": "d", "prior": 0.1},
    ])
    out = _parse_hypotheses(raw, max_hypotheses=2)
    assert len(out) == 2


def test_raises_when_all_invalid():
    raw = json.dumps([
        {"regime": "recovery", "rationale": "bad", "prior": 0.5},
        {"regime": "stagflation", "rationale": "also bad", "prior": 0.5},
    ])
    with pytest.raises(ValueError):
        _parse_hypotheses(raw, max_hypotheses=3)


def test_raises_on_malformed_json():
    with pytest.raises(ValueError):
        _parse_hypotheses("this is not json at all", max_hypotheses=3)


# =========================================================================== #
# generate_hypotheses  (orchestration, via the injected call_model seam)
# =========================================================================== #
# _call_model is gone -- provider routing now lives in model_client.ModelClient,
# and generate_hypotheses takes the model call as an injected `call_model` seam
# (default: make_model_client().complete). The orchestration is exercised here with
# a fake call_model: no SDK, no network, no API key.

SNAPSHOT = MacroSnapshot(
    as_of="2026-06-01",
    indicators={"fed_funds_rate": 4.5, "unemployment": 4.3, "yield_spread_10_2": 0.15},
)


def make_fake_model(response: str):
    """A (system, user) -> raw_text stand-in for the model call that records the
    prompts it received, so a test can assert what generate_hypotheses sent."""
    calls: list[tuple[str, str]] = []

    def call(system: str, user: str) -> str:
        calls.append((system, user))
        return response

    return call, calls


def test_generate_hypotheses_parses_injected_model_output():
    """End-to-end with no network: a canned model response flows through
    _build_user_prompt -> call_model -> _parse_hypotheses into RegimeHypothesis objects."""
    call, _ = make_fake_model(json.dumps([
        {"regime": "late_cycle", "rationale": "flat curve", "prior": 0.5},
        {"regime": "mid_cycle", "rationale": "low unemployment", "prior": 0.3},
    ]))
    out = generate_hypotheses(SNAPSHOT, max_hypotheses=3, call_model=call)
    assert [h.regime for h in out] == [const.Regime.LATE_CYCLE, const.Regime.MID_CYCLE]


def test_generate_hypotheses_substitutes_max_and_sends_snapshot():
    """The system prompt has %MAX_HYPOTHESES% filled in, and the user prompt carries
    the snapshot the model is asked to classify."""
    call, calls = make_fake_model(
        json.dumps([{"regime": "mid_cycle", "rationale": "x", "prior": 0.5}]))
    generate_hypotheses(SNAPSHOT, max_hypotheses=2, call_model=call)
    assert len(calls) == 1
    system, user = calls[0]
    assert "%MAX_HYPOTHESES%" not in system     # placeholder was substituted
    assert "maximum of 2" in system             # ... with the requested max
    assert "fed_funds_rate" in user             # snapshot rendered into the prompt
    assert SNAPSHOT.as_of in user


def test_generate_hypotheses_propagates_parse_failure():
    """No valid hypotheses in the response must raise, not return [] -- classify_regime_tot
    treats an empty list as an error."""
    call, _ = make_fake_model(
        json.dumps([{"regime": "recovery", "rationale": "not a regime", "prior": 0.5}]))
    with pytest.raises(ValueError):
        generate_hypotheses(SNAPSHOT, max_hypotheses=3, call_model=call)


@pytest.mark.skipif(os.getenv("TEST_MODE") != "Integration",
                    reason="Hits the live model; Integration only")
def test_live_model_generate_hypotheses(capsys):
    demo = MacroSnapshot(
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

    for h in generate_hypotheses(demo, max_hypotheses=3):
        #TODO: use logging for durable test results, for debugging, just run pytest -s
        print(f"{h.regime.value:14} prior={h.prior:.2f}  {h.rationale}")
