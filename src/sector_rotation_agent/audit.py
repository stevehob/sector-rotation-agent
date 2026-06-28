"""
audit.py

The audit layer's statistical checker (spec Section 3.2, Section 5.2 "audit
functions", Section 9 guardrails #1 and #2).

Section 3.2 puts two checks in front of every tool result: a pure-Python
statistical checker that runs FIRST and UNCONDITIONALLY (it costs nothing and
validates data quality), then -- only when the statistical check passes -- a
narrow LLM critic. This module is the home of the statistical half:

  - check_statistical_anomaly  -- pure Python; z-score + IQR + freshness.   (here)
  - run_critic_check           -- narrow LLM call; supports/weakens/contradicts.  (TODO)

The composed AuditLayer the coordinator injects (coordinator.AuditLayer Protocol,
currently satisfied by the no-op _NullAudit) will eventually live here too: it will
run check_statistical_anomaly over each result's numeric claims, then run_critic_check,
and translate any returned flag dicts into coordinator.AuditFlag entries. Keeping the
raw checks here -- returning plain dicts, importing no coordinator/LLM/network code --
keeps them trivially unit-testable and reusable by either caller.
"""
from __future__ import annotations
import json
import logging
import statistics
from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

import sector_rotation_agent.constants as const
from sector_rotation_agent.model_client import make_model_client

if TYPE_CHECKING:
    # Type-only import. The AuditLayer adapter (ResultAuditor, below) returns
    # coordinator.AuditFlag, but we keep that import OUT of the runtime graph so
    # `import audit` stays free of the coordinator/scipy chain -- construction goes
    # through _flag(), which imports lazily. (audit -> coordinator is one-directional;
    # coordinator must never import audit, or this becomes a cycle.)
    from sector_rotation_agent.coordinator import AuditFlag, QueryPlan


logger = logging.getLogger(__name__)


def check_statistical_anomaly(
    series_id: str,
    value: float,
    history: list[dict],
    *,
    as_of: str | None = None,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    max_age_days: int = 45,
    min_history: int = 8,
) -> dict:
    """
    Validate one freshly observed data point against its own history (spec 3.2 / 5.2).

    Parameters
    ----------
    series_id
        The indicator/series under test (FRED code or sector metric); echoed into
        the result and the human-readable reasons so a flag is self-describing.
    value
        The latest observed value being validated against the historical window.
    history
        The reference observations, ASCENDING, each shaped like a get_macro_indicators
        per-series row: {"date": "YYYY-MM-DD", "value": <float>}. The z-score and IQR
        checks use the values; the freshness check uses the most recent date
        (history[-1]["date"]). Pass the full series with `value` == the last value to
        check "is the newest point an anomaly, and is the series fresh?".
    as_of
        Reference "today" (ISO date) for the freshness check; None -> the system
        clock. Keyword-only and optional so the spec's positional signature
        (series_id, value, history) is preserved while freshness stays deterministic
        in tests and honors a point-in-time `as_of` run (no lookahead), the same
        discipline the macro agent uses.
    z_threshold, iqr_multiplier
        Sensitivity knobs for the two distributional checks (domain assumptions to
        tune by backtest, like the scorers' weights). Defaults: 3.0 sigma, 1.5*IQR.
    max_age_days
        Freshness ceiling in days for THIS series (spec guardrail #2; default 45).
        Per-series ceilings live in const.FRESHNESS_MAX_AGE_DAYS and are resolved by
        the ResultAuditor caller (quarterly gdp_growth gets a wider window, say); this
        pure checker just applies whatever ceiling it is handed.
    min_history
        Minimum number of historical values required before the DISTRIBUTIONAL
        checks (z-score, IQR) run at all. Below it those checks are SKIPPED -- not
        errored -- (reported flagged=False with None stats), because a handful of
        points can't define a distribution; the freshness check still runs.

    Returns
    -------
    dict
        A flag dict (spec 5.2) -- the per-check breakdown, not a coordinator.AuditFlag
        (the AuditLayer adapter does that translation):

            {
              "series_id": str,
              "value": float,
              "flagged": bool,                 # True iff ANY sub-check flagged
              "reasons": list[str],            # one human-readable line per tripped check
              "checks": {
                  "z_score":   {"flagged": bool, "z": float | None, "threshold": float},
                  "iqr":       {"flagged": bool, "lower": float | None, "upper": float | None},
                  "freshness": {"flagged": bool, "age_days": int | None, "max_age_days": int},
              },
            }

        A check whose inputs are insufficient (too little history, or no dates)
        reports flagged=False with None stats -- skipped, never raising.

    """
    reasons: list = []

    # Run three independent checks and reports each separately so the
    #  audit log (and generate_report) can say exactly what tripped:

    #1. z-score   -- |value - mean(history)| / std(history) > `z_threshold`
    #                (guardrail #1: catch out-of-range / fabricated figures).
    
    # values to compute and include in output
    z_flag: bool = False
    iqr_flag = False
    date_flag = False
    z_score: float | None = None
    lower: float | None = None
    upper: float | None = None
    age_days: int | None = None

    # get the data from the dict
    values = [item["value"] for item in history if item.get("value") is not None]

    # check that we have enough history data to check
    n = len(values)
    if n < min_history:
        logger.debug(
            "%s: only %d/%d history points; skipping z-score/IQR (freshness still runs)",
            series_id, n, min_history,
        )
    if n >= min_history:  # if yes, then do checking
        # compute z-score
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values)
        if std_value > 0:
            # z-score
            z_score = float((value - mean_value) / std_value)
            if abs(z_score) > z_threshold:
                reasons.append(f"z-score: {z_score:.2f} exceeded threshold: {z_threshold}")
                z_flag = True
        elif value != mean_value:
            z_score = float("inf")
            z_flag = True
            reasons.append(f"{series_id}: value {value} departs from constant history ({mean_value})")
        # compute IQR       -- value outside the Tukey fences
        #                [Q1 - k*IQR, Q3 + k*IQR], k = `iqr_multiplier`. A
        #                distribution-shape check that catches outliers a z-score can
        #                miss on skewed series.
        q1, _, q3 = statistics.quantiles(values, n=4)
        iqr = q3 - q1
        if iqr > 0:
            lower = float(q1 - iqr_multiplier * iqr)
            upper = float(q3 + iqr_multiplier * iqr)
            if value < lower or value > upper:
                iqr_flag = True
                reasons.append(f"{series_id}: value {value} outside the IQR fences [{lower:.2f}, {upper:.2f}]")
    # End If
    #         
    # data freshness -- age of the most recent observation in `history` exceeds
    #                `max_age_days` (guardrail #2: stale data presented as current;
    #                the spec's threshold is 45 days).
    # Convert strings to date objects
    as_of_date = date.fromisoformat(as_of) if as_of else date.today()
    max_obs_date = date.fromisoformat(history[-1]["date"]) if history else None
    if max_obs_date:
        age_days = (as_of_date - max_obs_date).days
        if age_days > max_age_days:
            reasons.append(
                f"stale data: most recent observation ({max_obs_date.isoformat()}) is "
                f"{age_days} days old as of {as_of_date.isoformat()}, exceeding the "
                f"{max_age_days}-day freshness ceiling"
            )
            date_flag = True


    flag_exist = z_flag or iqr_flag or date_flag # we had at least one thing flagged=True

    output = {
        "series_id": series_id,
        "value": value,
        "flagged": flag_exist,
        "reasons": reasons,
        "checks": {
            "z_score":  {"flagged": z_flag, "z": z_score, "threshold": z_threshold},
            "iqr":      {"flagged": iqr_flag, "lower": lower, "upper": upper},
            "freshness":{"flagged": date_flag, "age_days": age_days, "max_age_days": max_age_days},
        }

    }
    
    return output


# The three verdicts the critic may return (spec 5.2). Anything else the model
# emits is a parse failure, not a fourth option.
_CRITIC_VERDICTS = frozenset({"supports", "weakens", "contradicts"})

CRITIC_SYSTEM_PROMPT = """
        You are an independent critic. You are given ONE working hypothesis and ONE new
        piece of evidence, and nothing else. Judge only whether that single piece of
        evidence supports, weakens, or contradicts the hypothesis. Do not assume facts
        beyond the evidence shown, and do not try to reconstruct the wider analysis --
        your isolation from the surrounding reasoning is deliberate.

        Return STRICTLY a JSON object, no prose and no markdown fences, with keys:
        - "verdict":    one of "supports", "weakens", "contradicts"
        - "reason":     a brief explanation grounded only in the evidence shown
        - "confidence": a number between 0.0 and 1.0 -- how sure you are of the verdict

        Example:
        {"verdict": "contradicts", "reason": "A rising 10y-2y spread is inconsistent with the late-cycle call.", "confidence": 0.7}
"""

def run_critic_check(
    hypothesis: str,
    new_evidence: str,
    *,
    call_model: Callable[[str, str], str] | None = None,
) -> dict:
    """
    Independently judge ONE evidence item against ONE working hypothesis
    (spec 5.2; Section 3.2 critic; Section 4; guardrail #6).

    The second, LLM half of the audit layer, and the opposite kind of check from
    check_statistical_anomaly. The statistical gate asks "is this number plausible
    on its own?"; the critic asks "does this (statistically fine) value actually
    agree with what we currently believe?" -- the contradiction a range check cannot
    see. It fires only after the statistical check passes (Section 3.2), since
    spending an LLM call on already-flagged data is wasteful.

    Context isolation is the whole point (guardrail #6 / Section 4): the critic is a
    stateless, single-shot call that sees ONLY `hypothesis` and ONE `new_evidence`
    item -- never the broader reasoning chain -- so it cannot rubber-stamp the
    agent's own narrative. Passing minimal strings here, rather than rich objects or
    the run-so-far, is how that isolation is enforced in code; keep the caller's
    rendering down to the single claim and the single datum.

    Parameters
    ----------
    hypothesis
        The current working hypothesis as an isolated one-line statement
        (e.g. "The macro regime is late_cycle.").
    new_evidence
        A single new evidence item rendered to text (e.g. one analog, one indicator
        reading). Exactly one -- batching evidence would defeat the isolation.
    call_model
        Seam for the LLM call: a callable (system_prompt, user_prompt) -> raw_text,
        matching generate_hypotheses._call_model. Keyword-only and optional so the
        spec's positional signature (hypothesis, new_evidence) is preserved; None
        routes through the project's real model path, while tests inject a fake that
        returns canned JSON -- the same dependency-injection discipline the ToT
        classifier uses to stay unit-testable with no network (Section 3.3).

    Returns
    -------
    dict
        The critic verdict:

            {
              "verdict": "supports" | "weakens" | "contradicts",
              "reason": str,
              "confidence": float,   # clamped to [0.0, 1.0]
            }

        Implementation note (for when this body lands): build the user prompt from
        ONLY hypothesis + new_evidence, call `call_model` with CRITIC_SYSTEM_PROMPT,
        then parse/validate the JSON the way generate_hypotheses does -- strip any
        markdown fences, json.loads, require "verdict" in _CRITIC_VERDICTS (raise
        rather than inventing a fourth verdict), coerce and clamp "confidence" to
        [0, 1]. A verdict the model can't produce cleanly should surface as an
        error, not a silent "supports".
    """
    if call_model is None:
        call_model = make_model_client().complete

    user_prompt =  f"Given the current hypothesis: \n {hypothesis} \n"
    user_prompt += f"Evaluate against this new evidence: \n {new_evidence}"
    system_prompt = CRITIC_SYSTEM_PROMPT
    raw = call_model(system_prompt, user_prompt)

    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> dict:
    """
    Parse the critic's JSON into a validated verdict dict.

    The contract here is a single JSON OBJECT --
      {"verdict", "reason", "confidence"}
    """
    # Strip markdown fences the model may have wrapped the JSON in. (identical)
    cleaned = raw.strip()
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    cleaned = "\n".join(lines).strip()

    # If the model wrapped the object in prose, carve out the JSON object.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from critic output: {e}\nRaw output was:\n{raw}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Critic output must be a JSON object, got {type(parsed).__name__}")

    verdict = parsed.get("verdict")
    if verdict not in _CRITIC_VERDICTS:
        raise ValueError(f"Critic returned an unknown verdict: {verdict!r}")

    # Same coerce-and-clamp discipline as the hypothesis prior.
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "verdict": verdict,
        "reason": str(parsed.get("reason", "")),
        "confidence": confidence,
    }


# --------------------------------------------------------------------------- #
# AuditLayer adapter (spec Section 3.2): statistical checker THEN narrow critic
# --------------------------------------------------------------------------- #
def _flag(source: str, label: str, message: str) -> AuditFlag:
    """Build a coordinator.AuditFlag, imported lazily so audit.py's module import
    stays free of the coordinator/scipy chain. (audit -> coordinator is
    one-directional; coordinator must never import audit, or this becomes a cycle.)"""
    from sector_rotation_agent.coordinator import AuditFlag
    return AuditFlag(source=source, label=label, message=message)


class ResultAuditor:
    """
    The composed audit layer the coordinator injects (satisfies coordinator.AuditLayer),
    replacing the no-op _NullAudit. Per spec Section 3.2 it runs the pure-Python
    statistical checker FIRST and, only when that raises nothing, the narrow LLM
    critic -- then translates either into coordinator.AuditFlag entries.

    Scope
    -----
    Macro is fully wired: the statistical checker runs check_statistical_anomaly over
    every indicator (z-score / IQR / freshness) against the per-indicator history the
    macro agent now retains, and -- only if that gate passes -- the critic cross-checks
    the regime call against the single strongest historical analog, in isolation
    (guardrail #6: it sees only the regime statement and that one analog, never the
    agent's reasoning chain).

    Equity: the statistical checker covers each sector's 6-month momentum (the yfin
    server emits a rolling momentum history that the equity agent retains), and -- if
    that gate passes -- the critic adjudicates the strongest momentum-vs-valuation
    tension (the highest-momentum sector trading above the universe's median P/E).
    Valuation has no time series of its own, so it isn't run through the statistical
    checker.

    The critic call is injected (`call_model`) and wrapped: a flaky model degrades to
    a soft flag rather than crashing the whole research run.

    The Fed-narrative corpus freshness check (guardrail #2, corpus level) is injected too
    (`check_freshness`) and run on the macro result; a stale corpus surfaces as a
    carried-forward caveat, independent of the statistical-gate / critic ordering.
    """

    def __init__(
        self,
        *,
        call_model: Callable[[str, str], str] | None = None,
        check_freshness: Callable[..., dict] | None = None,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        max_age_days: int = const.DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        # call_model=None lets run_critic_check fall back to the default (local) model;
        # main.py can inject a cloud client's .complete for the spec's hybrid setup.
        self._call_model = call_model
        # Corpus-freshness seam (guardrail #2, corpus level). main.py injects
        # fed_narrative_rag.check_corpus_freshness; None skips the check (e.g. unit
        # tests). Called as check_freshness(as_of=...) -> flag dict.
        self._check_freshness = check_freshness
        # Statistical-checker knobs, forwarded to check_statistical_anomaly.
        self._z_threshold = z_threshold
        self._iqr_multiplier = iqr_multiplier
        # Default freshness ceiling (guardrail #2); per-series overrides in
        # const.FRESHNESS_MAX_AGE_DAYS take precedence (see _max_age_for).
        self._max_age_days = max_age_days
        self._logger = logging.getLogger(__name__)

    def _max_age_for(self, series_id: str) -> int:
        """Per-series freshness ceiling (guardrail #2): a cadence-aware override from
        const.FRESHNESS_MAX_AGE_DAYS when one exists (e.g. quarterly gdp_growth), else
        this auditor's default. Macro series are keyed by INDICATOR_KEY; equity tickers
        aren't in the map and fall through to the default."""
        return const.FRESHNESS_MAX_AGE_DAYS.get(series_id, self._max_age_days)

    def review(self, label: str, result: object, plan: QueryPlan) -> list[AuditFlag]:
        """Statistical checker first; the critic fires only if it passed (spec 3.2) --
        spending an LLM call on already-suspect data would be wasteful. The Fed-corpus
        freshness check (guardrail #2, corpus level) sits OUTSIDE that gate: a stale
        corpus is a carried-forward caveat, not a reason to skip contradiction-checking,
        so it is appended to whichever branch runs."""
        corpus_flags = self._corpus_freshness_flags(label, result, plan)
        stat_flags = self._statistical_flags(label, result, plan)
        if stat_flags:
            # Statistical anomalies short-circuit the critic (spec 3.2) and trigger a
            # revision upstream, so surface them prominently.
            self._logger.warning(
                "Audit[%s]: %d statistical flag(s) -> skipping critic this pass: %s",
                label, len(stat_flags), "; ".join(f.label for f in stat_flags),
            )
            return stat_flags + corpus_flags
        critic_flags = self._critic_flags(label, result)
        self._logger.debug(
            "Audit[%s]: statistical gate clean; %d critic flag(s), %d corpus flag(s)",
            label, len(critic_flags), len(corpus_flags),
        )
        return critic_flags + corpus_flags

    def _statistical_flags(self, label: str, result: object, plan: QueryPlan) -> list[AuditFlag]:
        """Run check_statistical_anomaly over each numeric claim that carries history.

        Macro: every indicator in the snapshot is checked against its retained
        per-indicator history. Equity: each sector's current 6-month momentum is
        checked against its retained momentum history. Valuation isn't checked --
        get_sector_valuations returns a single current P/E with no time series.
        """
        if "macro" in label.lower():
            return self._statistical_macro(result, plan)
        if "equity" in label.lower():
            return self._statistical_equity(result, plan)
        return []

    def _statistical_macro(self, result: object, plan: QueryPlan) -> list[AuditFlag]:
        """Check each macro indicator's latest value against its own history.

        Reads the values from result.snapshot.indicators and the per-indicator history
        from result.series_history (same INDICATOR_KEY keys, same units -- the macro
        agent derives cpi_inflation's YoY history so this stays like-for-like). An
        indicator with no retained history is skipped rather than guessed at.
        """
        snapshot = getattr(result, "snapshot", None)
        indicators = getattr(snapshot, "indicators", None) or {}
        history = getattr(result, "series_history", None) or {}
        as_of = getattr(plan, "as_of", None)

        flags: list[AuditFlag] = []
        for series_id, value in indicators.items():
            series_hist = history.get(series_id)
            if not series_hist:
                continue  # nothing retained for this indicator -> can't check it
            report = check_statistical_anomaly(
                series_id,
                value,
                series_hist,
                as_of=as_of,
                z_threshold=self._z_threshold,
                iqr_multiplier=self._iqr_multiplier,
                max_age_days=self._max_age_for(series_id),
            )
            if report["flagged"]:
                self._logger.warning("Statistical anomaly: %s -- %s",
                                     series_id, "; ".join(report["reasons"]))
                flags.append(_flag("statistical", series_id, "; ".join(report["reasons"])))
        return flags

    def _statistical_equity(self, result: object, plan: QueryPlan) -> list[AuditFlag]:
        """Check each sector's current 6-month momentum against its own momentum
        history (from result.series_history). Valuation has no time series from the
        tools, so it isn't checked; a sector with no retained history is skipped.
        """
        equity_data = getattr(result, "equity_data", None) or {}
        history = getattr(result, "series_history", None) or {}
        as_of = getattr(plan, "as_of", None)

        flags: list[AuditFlag] = []
        for ticker, row in equity_data.items():
            value = row.get("momentum")
            series_hist = history.get(ticker)
            if value is None or not series_hist:
                continue  # nothing to check this sector against
            report = check_statistical_anomaly(
                f"{ticker} momentum",
                value,
                series_hist,
                as_of=as_of,
                z_threshold=self._z_threshold,
                iqr_multiplier=self._iqr_multiplier,
                max_age_days=self._max_age_for(ticker),
            )
            if report["flagged"]:
                self._logger.warning("Statistical anomaly: %s momentum -- %s",
                                     ticker, "; ".join(report["reasons"]))
                flags.append(_flag("statistical", ticker, "; ".join(report["reasons"])))
        return flags

    def _corpus_freshness_flags(self, label: str, result: object, plan: QueryPlan) -> list[AuditFlag]:
        """Guardrail #2 at the corpus level: is the Fed-narrative corpus fresh as of the
        run date? Runs only on the macro result (which owns the Fed corpus) and only when
        a freshness checker was injected -- main.py wires
        fed_narrative_rag.check_corpus_freshness; tests inject a fake or leave it None to
        skip. A flagged corpus becomes a statistical flag labeled 'fed_narrative' (not a
        ticker), so the coordinator carries it forward as a low-confidence caveat rather
        than triggering a revision. Wrapped so a DB/embedding hiccup degrades to no flag
        rather than crashing the run."""
        if self._check_freshness is None or "macro" not in label.lower():
            return []
        as_of = getattr(plan, "as_of", None)
        try:
            report = self._check_freshness(as_of=as_of)
        except Exception as err:  # a DB/embedding hiccup must not crash the run
            self._logger.warning("Corpus freshness check failed: %s", err)
            return []
        if not report or not report.get("flagged"):
            return []
        source_id = report.get("source_id", "fed_narrative")
        reasons = "; ".join(report.get("reasons", [])) or f"{source_id} corpus is stale"
        self._logger.warning("Corpus freshness flag: %s -- %s", source_id, reasons)
        return [_flag("statistical", source_id, reasons)]

    def _critic_flags(self, label: str, result: object) -> list[AuditFlag]:
        if "macro" in label.lower():
            return self._critic_macro(result)
        if "equity" in label.lower():
            return self._critic_equity(result)
        return []

        
    def _critic_equity(self, result: object) -> list[AuditFlag]:
        """Adjudicate the strongest momentum-vs-valuation tension in one critic call.

        Candidate = the highest-momentum sector trading ABOVE the universe's median
        P/E (a "rising but expensive" leader) -- the classic momentum/value conflict.
        Using the median keeps this relative (no magic P/E threshold), and if no such
        sector exists there's no tension to question, so the critic isn't called. Like
        the macro critic, the call is isolated to one hypothesis + one evidence string
        (guardrail #6) and wrapped so a flaky model degrades to a soft flag.
        """
        equity_data = getattr(result, "equity_data", None) or {}
        rows = [
            (t, r.get("momentum"), r.get("valuation"))
            for t, r in equity_data.items()
            if isinstance(r.get("momentum"), (int, float))
            and isinstance(r.get("valuation"), (int, float))
        ]
        if len(rows) < 2:
            return []  # need a universe to judge "expensive" against

        median_pe = statistics.median(sorted(v for _, _, v in rows))
        expensive_risers = [(t, m, v) for t, m, v in rows if v > median_pe and m > 0]
        if not expensive_risers:
            return []  # no rising-but-expensive sector -> no tension to check

        ticker, momentum, pe = max(expensive_risers, key=lambda r: r[1])
        hypothesis = f"{ticker} is a momentum leader, with 6-month momentum of {momentum:.1%}."
        evidence = (
            f"{ticker}'s price-to-earnings is {pe:.1f}, above the sector-universe "
            f"median of {median_pe:.1f}."
        )
        try:
            verdict = run_critic_check(hypothesis, evidence, call_model=self._call_model)
        except Exception as err:  # a flaky model must not crash the whole run
            self._logger.warning("Critic check failed for equity %s: %s", ticker, err)
            return [_flag("critic", ticker,
                          f"critic check could not be completed ({type(err).__name__})")]

        if verdict["verdict"] in ("weakens", "contradicts"):
            self._logger.warning("Critic %s the %s momentum case: %s",
                                 verdict["verdict"], ticker, verdict["reason"])
            return [_flag("critic", ticker,
                          f"{verdict['verdict']} the {ticker} momentum case: {verdict['reason']}")]
        return []

    def _regime_critic(
        self,
        hypothesis: str,
        evidence: str,
        regime: str,
        *,
        evidence_label: str | None = None,
    ) -> list[AuditFlag]:
        """One isolated regime critic call -- one hypothesis + ONE evidence item
        (guardrail #6) -- translated into at most one AuditFlag. A flaky model degrades
        to a soft flag rather than crashing the run. `evidence_label`, when given, names
        the evidence source in the flag message (e.g. 'Fed narrative'); omitted for the
        analog so its message stays the bare '... the <regime> call: <reason>'.
        """
        try:
            verdict = run_critic_check(hypothesis, evidence, call_model=self._call_model)
        except Exception as err:  # a flaky model must not crash the whole run
            self._logger.warning(
                "Critic check failed for macro regime (%s): %s", evidence_label or "analog", err
            )
            return [_flag("critic", "macro",
                          f"critic check could not be completed ({type(err).__name__})")]

        if verdict["verdict"] in ("weakens", "contradicts"):
            suffix = f" vs {evidence_label}" if evidence_label else ""
            self._logger.warning("Critic %s the %s call%s: %s",
                                 verdict["verdict"], regime, suffix, verdict["reason"])
            return [_flag("critic", "macro",
                          f"{verdict['verdict']} the {regime} call{suffix}: {verdict['reason']}")]
        return []

    def _critic_macro(self, result: object) -> list[AuditFlag]:
        regime_obj = getattr(result, "regime", None)
        if regime_obj is None:
            return []  # no regime to test
        regime = getattr(regime_obj, "value", str(regime_obj))
        hypothesis = f"The current macro regime is {regime}."

        flags: list[AuditFlag] = []

        # Evidence 1: the single strongest historical analog (numeric neighbor).
        analogs = getattr(result, "analogs", None) or []
        if analogs:
            top = max(analogs, key=lambda a: a.get("similarity") or float("-inf"))
            evidence = (
                f"The closest historical analog is {top.get('date')} "
                f"(cosine similarity {top.get('similarity')}), a period classified "
                f"'{top.get('regime')}'."
            )
            flags += self._regime_critic(hypothesis, evidence, regime)

        # Evidence 2: the most relevant Fed-narrative passage (qualitative). A SEPARATE,
        # isolated critic call (guardrail #6) -- never merged into the analog evidence
        # above. The macro agent now attaches `fed_narrative` (retrieval is wired), so
        # this runs whenever passages were retrieved; the getattr/empty guard just skips
        # it when the corpus returned nothing or no finder was injected.
        fed_narrative = getattr(result, "fed_narrative", None) or []
        if fed_narrative:
            passage = fed_narrative[0]   # find_fed_narrative returns nearest-first
            evidence = (
                f"From the {passage.get('source')} dated {passage.get('date')} "
                f"(\"{passage.get('title')}\"): {passage.get('text')}"
            )
            flags += self._regime_critic(hypothesis, evidence, regime, evidence_label="Fed narrative")

        return flags