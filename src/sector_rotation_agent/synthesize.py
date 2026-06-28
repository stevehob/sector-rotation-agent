"""
synthesize.py

Synthesis layer (spec Section 4 "Synthesis agent", Section 5 tool table).

Where classify_regime_tot decides WHICH regime we're in, the synthesis stage
decides WHAT TO DO about it: it turns the chosen regime -- plus the historical
analogs and current equity data -- into a ranked list of sectors with
confidence, which generate_report then writes up.

This is deliberately a DETERMINISTIC, statistical scoring model (no LLM). The
Synthesis agent (the LLM) calls it as a tool so the ranking is reproducible and
auditable; the agent's job is to reconcile and narrate, not to invent the
numbers.

Two scorers, two different questions, run in sequence -- don't conflate them:
  - score_branch          scores a REGIME hypothesis (4 regimes) DURING classification.
  - compute_sector_score  scores the 11 SECTORS AFTER the regime is settled.

generate_report (spec Section 5) is the sibling tool that belongs in this module
too, once compute_sector_score is producing rankings.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import sector_rotation_agent.constants as const

from sector_rotation_agent.score_branch import SECTOR_TILTS

logger = logging.getLogger(__name__)

# Default scoring parameters -- the single source of truth for both the scorer and the
# report's methodology appendix. compute_sector_score defaults to these, and the
# methodology appendix is rendered from whatever generate_report is handed (defaulting to
# the same values), so the appendix always reflects the blend actually used (spec §3.4).
DEFAULT_W_REGIME = 0.40
DEFAULT_W_ANALOG = 0.35
DEFAULT_W_EQUITY = 0.25
DEFAULT_STRONG_SIMILARITY = 0.75
DEFAULT_MIN_STRONG_ANALOGS = 3

# The analog store seeds forward sector returns at several horizons
# (const.ANALOG_HORIZONS_MONTHS); compute_sector_score selects the slice matching the
# query's horizon, falling back to const.ANALOG_DEFAULT_HORIZON_MONTHS when the query
# states none or asks for a horizon the store wasn't seeded at -- in which case
# generate_report flags the gap (spec section 3.3 / 3.4 / 10.1). ANALOG_HORIZON_MONTHS is
# kept as an alias for that default window.
ANALOG_HORIZON_MONTHS = const.ANALOG_DEFAULT_HORIZON_MONTHS


def _horizon_months(horizon: str | None) -> int | None:
    """Months in a normalized '<N> months' horizon string (what _decompose_query
    produces), or None if absent/unparseable. Forgiving so a hand-passed horizon in
    another format degrades to 'unknown' rather than raising."""
    if not horizon:
        return None
    head = horizon.strip().split()[0]
    return int(head) if head.isdigit() else None


def _effective_horizon_months(horizon: str | None) -> int:
    """The analog window the ranking ACTUALLY uses: the requested horizon when the store
    is seeded at it (it's in const.ANALOG_HORIZONS_MONTHS), otherwise the default."""
    months = _horizon_months(horizon)
    if months is not None and months in const.ANALOG_HORIZONS_MONTHS:
        return months
    return const.ANALOG_DEFAULT_HORIZON_MONTHS


def _analog_returns(analog: dict, horizon_months: int | None) -> dict:
    """The sector-returns slice to score one analog against: the requested horizon when
    the analog carries it, otherwise the analog's default (6m) slice. So a horizon the
    store wasn't seeded at, or a legacy analog with only the default field, both degrade
    to the default returns rather than vanishing."""
    if horizon_months is not None:
        by_h = analog.get("subsequent_returns_by_horizon")
        if isinstance(by_h, dict):
            chosen = by_h.get(horizon_months)
            if chosen:
                return chosen
    return analog.get("subsequent_sector_returns") or {}


def _build_sector_lists(analog_data: list[dict], equity_data: dict[str, dict], regime: const.Regime, horizon_months: int | None = None, universe: tuple[str, ...] = const.SECTOR_ETFS_LIST) -> tuple[dict[str, float], dict[str,float], dict[str,float]]:
    raw_regimes: dict[str, float] = {}
    raw_analogs: dict[str, float] = {}
    raw_equity: dict[str, float] = {}
    
    for tkr in universe:
        # for the regime provided, get tilt
        # the +1 / 0 / -1 expectation
        raw_regimes[tkr] = float(SECTOR_TILTS[regime][tkr]) 
        #  in the 6 months after macro periods that looked like today, how did sector s actually do?
        num = 0.0   # running Σ similarity * return
        den = 0.0   # running Σ similarity  (only over analogs that HAVE a return for s)
        for a in analog_data:
           r = _analog_returns(a, horizon_months).get(tkr)  # horizon-appropriate slice; None if absent
           if r is None:    # sector didn't exist yet -> no vote
              continue
           w = a.get("similarity", 0.0)
           num += w * r
           den += w
        raw_analogs[tkr] = num / den if den else 0.0
         # in the case it's missing, fill in 0.0 (neutral) instead of error
        raw_equity[tkr] = equity_data.get(tkr, {}).get("momentum", 0.0)
    return raw_regimes, raw_analogs, raw_equity


def _normalize(signal: dict[str, float]) -> dict[str, float]:
    """Min-max a per-sector signal onto [0, 1] across the sectors in this call.

    Returns 0.5 for every sector when the signal is flat (all values equal): that
    both avoids a divide-by-zero and correctly says "this signal can't separate
    the sectors", so it nudges everyone equally and changes no rankings.
    """
    lo = min(signal.values())
    hi = max(signal.values())
    span = hi - lo
    if span < 1e-12:                       # flat signal -> no information to scale
        return {s: 0.5 for s in signal}
    return {s: (v - lo) / span for s, v in signal.items()}

def _signal_agreement(nr: float, na: float, ne: float) -> float:
    """1.0 when the three normalized signals coincide, 0.0 when maximally split."""
    return 1.0 - (max(nr, na, ne) - min(nr, na, ne))

def compute_sector_score(
    macro_regime: const.Regime,
    analog_data: list[dict],
    equity_data: dict[str, dict],
    *,
    horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
    w_regime: float = DEFAULT_W_REGIME,
    w_analog: float = DEFAULT_W_ANALOG,
    w_equity: float = DEFAULT_W_EQUITY,
    strong_similarity: float = DEFAULT_STRONG_SIMILARITY,
    min_strong_analogs: int = DEFAULT_MIN_STRONG_ANALOGS,
) -> list[dict]:
    """
    Rank the 11 GICS sector ETFs for the given macro regime.

    The statistical scoring model the Synthesis agent calls (spec Section 5).
    Computes one blended score per sector from three independent signals, then
    ranks. Keeping the signals separate (rather than one opaque number) is what
    lets generate_report and the audit log explain why a sector ranked where it
    did -- the same principle score_branch follows.

      1. regime prior   -- SECTOR_TILTS[macro_regime] (+1 favored / -1 disfavored
         / 0 neutral): the textbook business-cycle expectation for this regime.
         Reuse SECTOR_TILTS from score_branch.py so there is one source of truth.
      2. analog evidence -- what sectors ACTUALLY did after similar past periods:
         per sector, the mean of `subsequent_sector_returns` across analog_data,
         ideally weighted by each analog's `similarity`. Realized, not theoretical.
      3. equity signal  -- current per-sector state from equity_data (momentum,
         valuation). The present-looking cross-check.

    Parameters
    ----------
    macro_regime
        The regime classify_regime_tot settled on (the winning ToTResult.regime).
    analog_data
        Analogs from find_historical_analogs, each shaped like:
            {
              "date": "2001-03",
              "similarity": 0.87,                  # 0..1 cosine match to today
              "regime": "contraction",
              "subsequent_sector_returns": {       # forward 6m return per sector
                  "XLK": -0.22, "XLF": -0.18, ..., "XLRE": None,   # None = no data
              },
            }
        NOTE: `subsequent_sector_returns` values can be None (the sector didn't
        exist yet, e.g. XLRE pre-2015, XLC pre-2018). Skip Nones when averaging --
        do not treat a missing return as 0.0.
    equity_data
        Current per-sector equity state keyed by ETF ticker, e.g.:
            {
              "XLK": {"momentum": 0.08, "valuation": 22.1},
              "XLF": {"momentum": -0.02, "valuation": 14.3},
              ...
            }
        Sourced from get_sector_performance / get_sector_valuations.
    w_regime, w_analog, w_equity
        Weights for the three signals. Should sum to 1.0; they are domain
        assumptions to tune against a backtest (same spirit as score_branch's
        w_analog / w_signal).
    strong_similarity, min_strong_analogs
        Confidence guardrail, mirroring score_branch and the spec: with fewer
        than `min_strong_analogs` analogs at or above `strong_similarity`, the
        analog evidence (and the affected sectors' confidence) should be
        discounted rather than trusted at full weight.
    horizon
        The requested investment horizon (normalized '<N> months'), or None. Selects the
        matching slice of each analog's subsequent_returns_by_horizon; when None, or a
        horizon the store wasn't seeded at, the default (6m) slice is used. Affects only
        which forward-return window the analog signal reads -- the regime/equity signals
        and the blend math are unchanged.
    universe
        The sector sub-universe to RANK (a tuple of tickers), or None for all 11. When the
        analyst's question names a subset ("which defensive sectors ..."), the coordinator
        passes it here; only those sectors are scored and returned. Min-max normalization
        spans this subset when it has >=3 sectors (so scores are standing WITHIN the
        requested group, spec 3.4), else the full 11 -- a guard against degenerate 1-2
        sector scaling. The fetch and the macro regime classification are unaffected;
        narrowing changes only which sectors appear in the ranking.

    Returns
    -------
    list[dict]
        The sectors ranked best-first -- one dict per sector, e.g.:
            [
              {
                "sector": "XLK",
                "score": 0.82,             # blended score used for the ranking
                "rank": 1,
                "confidence": 0.74,        # from analog strength + signal agreement
                "detail": {                # sub-scores, for the report / audit log
                    "regime_tilt": 1,
                    "analog_return": 0.061,
                    "equity_signal": 0.08,
                    "n_strong_analogs": 4,
                },
              },
              ...
            ]
    """
    # First, make sure inputs are valid
    if abs((w_regime + w_analog + w_equity) - 1.0) > 1e-9:
      logger.error("Invalid sector-score weights: %s + %s + %s != 1.0", w_regime, w_analog, w_equity)
      raise ValueError("w_regime + w_analog + w_equity must equal 1.0")

    # Resolve the requested horizon to the slice the analogs carry (default when unset or
    # not seeded), then build the three raw per-sector dicts against it.
    horizon_months = _horizon_months(horizon)
    # Resolve the ranking universe (None -> all 11) and the normalization basis: the focus
    # subset when it has >=3 sectors (so scores express standing WITHIN the subset, spec
    # 3.4), else the full 11 -- a guard so a 1-2 sector focus doesn't get a degenerate 0/1
    # min-max. Raw signals are built over the basis; rows are emitted only for `universe`.
    universe = const.SECTOR_ETFS_LIST if universe is None else tuple(universe)
    norm_basis = universe if len(universe) >= 3 else const.SECTOR_ETFS_LIST
    raw_regimes, raw_analogs, raw_equity = _build_sector_lists(analog_data, equity_data, macro_regime, horizon_months, norm_basis)

    # normalize onto common scale
    # tilt is +/-1, momentum is small decimal, tilt would overwhelm momentum
    # min-max each signal to [0, 1] across the 11 sectors
    n_regimes = _normalize(raw_regimes)
    n_analogs = _normalize(raw_analogs)
    n_equity = _normalize(raw_equity)

    # Blend the 3 normalized together
    scores: dict[str, float] = {}
    for tkr in universe:
      scores[tkr] = w_regime * n_regimes[tkr] + w_analog * n_analogs[tkr] + w_equity * n_equity[tkr]

    # Assemble the rows. One dict per sector
    scored: list[dict] = []
    for tkr in universe:
       # Compute confidence from analog strength - per ticker
       n_strong_tkr = sum(
          1 for a in analog_data
          if a.get("similarity", 0.0) >= strong_similarity
          and _analog_returns(a, horizon_months).get(tkr) is not None
       )
       strength_tkr = min(1.0, n_strong_tkr / min_strong_analogs)
       confidence_tkr = strength_tkr * _signal_agreement(n_regimes[tkr], n_analogs[tkr], n_equity[tkr])
       score: dict = {}
       score["sector"] = tkr
       score["score"] = scores[tkr]
       score["rank"] = 0
       score["confidence"] = confidence_tkr
       score["detail"] = {
           "regime_tilt": int(raw_regimes[tkr]),
           "normalized_regime_tilt": float(n_regimes[tkr]),
           "analog_return": raw_analogs[tkr],
           "normalized_analog_return": n_analogs[tkr],
           "equity_signal": raw_equity[tkr],
           "normalized_equity_signal": n_equity[tkr],
           "n_strong_analogs": n_strong_tkr
       }
       scored.append(score)

    # Rank and return - in descending order
    scored.sort(key=lambda x: x["score"], reverse=True)
    # fill in the rank for the sorted list
    i=1
    for r in scored:
        r["rank"] = i
        i += 1

    if scored:
        top = scored[0]
        logger.debug("Sector scoring: regime=%s, %d sector(s) ranked; top=%s (score=%.3f, confidence=%.2f)",
                     macro_regime.value, len(scored), top["sector"], top["score"], top["confidence"])
    return scored

def _analog_sim(a: dict) -> float:
    """Sort key for analogs: numeric similarity, or -inf so a None/missing
    similarity sorts last instead of raising in the float-vs-None comparison."""
    s = a.get("similarity")
    return s if isinstance(s, (int, float)) else float("-inf")

def _fred_tool_tag(indicator_key: str) -> str:
    """FRED provenance tag for a snapshot indicator. cpi_inflation is derived from
    CPIAUCSL rather than pulled directly, so it's tagged explicitly; every other
    indicator's code comes straight from SUPPORTED_SERIES."""
    if indicator_key == "cpi_inflation":
        return f"FRED:{const.SUPPORTED_SERIES['cpi']} (YoY derived)"
    return f"FRED:{const.SUPPORTED_SERIES[indicator_key]}"

def build_sources(
    as_of: str,
    indicators: dict[str, float],
    analogs: list[dict],
    equity_data: dict[str, dict],
    fed_narrative: list[dict] | None = None,
    ) -> list[dict]:
        """
    Assemble the provenance list generate_report cites (spec Section 5; guardrail #1).

    Every numeric input the brief can claim is tagged here to the tool that produced
    it and the value observed, so the report never states a figure the audit trail
    can't source. Deterministic and side-effect free -- it only reshapes data the
    agents already retrieved, which keeps it trivially testable and keeps the
    "no LLM in the synthesis tooling" guarantee.

    Takes flat primitives rather than the MacroResult/EquityResult objects (the same
    style generate_report follows), so synthesize.py needs no import of the agent
    result types. The coordinator unpacks them from the run's results:
        as_of        -- macro.snapshot.as_of (the point-in-time anchor)
        indicators   -- macro.snapshot.indicators ({INDICATOR_KEY: value})
        analogs      -- macro.analogs (each {date, similarity, regime, ...})
        equity_data  -- equity.equity_data ({ticker: {"momentum":..., "valuation":...}})
        fed_narrative -- macro.fed_narrative (each {source, date, title, text, similarity})

    Returns
    -------
    list[dict]
        One citation per claimable value, each shaped
            {"id":..., "label":..., "tool":..., "value":..., "as_of":...}
        ordered: macro indicators (INDICATOR_KEYS order), then per-sector momentum
        (SECTOR_ETFS_LIST order), then the historical analogs (strongest first), then
        any Fed-narrative passages (most relevant first; tagged "FedNarrative").
        `id` is a stable key so generate_report can reference a claim's source inline.
    """
        sources: list[dict] = []
        
        # 1) MACRO INDICATORS
        for key in const.INDICATOR_KEYS:
            if key not in indicators:
                continue
            
            sources.append(
                {
                    "id": key,
                    "label": const._INDICATOR_LABELS.get(key, key),
                    "tool": _fred_tool_tag(key),
                    "value": indicators[key],
                    "as_of": as_of,
                }
            )

        # 2) EQUITY - per sector momentum
        for tkr in const.SECTOR_ETFS_LIST:
            sec = equity_data.get(tkr)
            if not sec or "momentum" not in sec:
                continue
            sources.append(
                {
                    "id": f"{tkr}: momentum",
                    "label": f"{tkr}: 6m momentum",
                    "tool": "yfinance",
                    "value": sec["momentum"],
                    "as_of": as_of,
                }
            )

        # 3) Historical analogs
        for a in sorted(analogs, key=_analog_sim, reverse=True):  # descending sort
            date = a.get("date")
            sources.append(
                {
                    "id": f"analog: {date}",
                    "label": f"Historical analog {date} ({a.get("regime", "n/a")})",
                    "tool": "ChromaDB",
                    "value": a.get("similarity"),
                    "as_of": as_of,
                }
            )

        # 4) FED NARRATIVE -- retrieved policy passages (qualitative evidence, spec 3.5).
        # One citation per passage, most-relevant-first (the macro agent returns them
        # ordered), tagged "FedNarrative" so the brief's Sources block distinguishes them
        # from the numeric analogs; the matched text is quoted (truncated) in the label so
        # the citation actually conveys what the Fed said, not just a similarity score.
        for i, passage in enumerate(fed_narrative or [], start=1):
            src = str(passage.get("source", "fed")).replace("_", " ")
            pub = passage.get("date", "n/a")
            text = " ".join(str(passage.get("text", "")).split())
            excerpt = text[:160] + ("\u2026" if len(text) > 160 else "")
            label = f"Fed {src} ({pub})"
            if excerpt:
                label += f": '{excerpt}'"
            sources.append(
                {
                    "id": f"fed#{i}: {src} {pub}",
                    "label": label,
                    "tool": "FedNarrative",
                    "value": passage.get("similarity"),
                    "as_of": as_of,
                }
            )
        return sources


# -------------  Composable pieces for output report  --------------------
def _fmt_value(value) -> str:
    """Render a source value for the brief: fixed precision for numbers, an explicit
    marker for missing data, str() otherwise -- so the citations never print a bare
    'None' and the value column stays readable."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):          # bool is an int subclass; keep it as True/False
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)

def _heading_1(text: str) -> str:
    return "# " + text

def _heading_2(text: str) -> str:
    return "## " + text

def _render_header() -> list[str]:
    return [_heading_1("Sector Rotation Research Brief")]

def _render_executive_summary(summary: str) -> list[str]:
    """The LLM-written two-paragraph opening (spec Section 4 narration). Omitted when
    the summary is empty -- model unavailable or the call failed -- so the brief still
    renders cleanly from its structured sections."""
    if not summary:
        return []
    return [_heading_2("Executive summary"), "", summary]

def _render_regime_narrative(
    query: str, regime: const.Regime, as_of: str, confidence: float, horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
) -> list[str]:
    lines = [
        f"- **Question:** {query}",
        f"- **Regime:** {regime.value}",
        f"- **As of:** {as_of}",
        f"- **Overall confidence:** {confidence:.2f}",
    ]
    if horizon:
        lines.append(f"- **Requested horizon:** {horizon}")
    if universe is not None and len(universe) < len(const.SECTOR_ETFS_LIST):
        lines.append(
            f"- **Sector universe:** {len(universe)} of {len(const.SECTOR_ETFS_LIST)} "
            f"sectors ({', '.join(universe)}) -- ranked among these"
        )
    return lines

def _render_rankings(rankings: list[dict]) -> list[str]:
    """The ranking as a Markdown table (rank / sector / score / confidence), best
    first. A table survives MD -> PDF cleanly, unlike space-aligned columns."""
    rank_out: list[str] = [
        _heading_2("Sector ranking"),
        "",
        "| Rank | Sector | Score | Confidence |",
        "| ---: | :--- | ---: | ---: |",
    ]
    for r in rankings:
        rank_out.append(
            f"| {r['rank']} | {r['sector']} | {r['score']:.3f} | {r['confidence']:.2f} |"
        )
    return rank_out

def _render_flags(flags: list) -> list[str]:
    if not flags:
        return [_heading_2("Audit flags: none"), "", "No audit flags were raised."]
    flags_out: list[str] = [_heading_2(f"Audit flags ({len(flags)})"), ""]
    for f in flags:
        flags_out.append(f"- **{f.source}** ({f.label}): {f.message}")
    return flags_out

def _render_sources(sources: list[dict]) -> list[str]:
    """
    Render the provenance block (spec Section 5; guardrail #1) -- one cited line per
    claimable value, in the order build_sources emitted them (macro indicators, then
    per-sector momentum, then analogs strongest-first), so no sorting is needed here.

    Each line leads with the source `id` so other sections can cite a figure by that
    key and the reader resolves it here. Returns a self-contained block (heading +
    lines); empty `sources` yields an explicit 'none recorded' rather than a silently
    missing section, since a cited brief with no provenance is itself a red flag.
    """
    sources_out: list[str] = [_heading_2("Sources"), ""]
    if not sources:
        sources_out.append("- (none recorded)")
        return sources_out

    for src in sources:
        ref = src.get("id", "?")
        label = src.get("label", ref)
        value = _fmt_value(src.get("value"))
        tool = src.get("tool", "unknown")
        as_of = src.get("as_of", "n/a")
        sources_out.append(f"- **[{ref}]** {label} = {value} ({tool}, as of {as_of})")
    return sources_out

def _render_methodology(
    *,
    w_regime: float = DEFAULT_W_REGIME,
    w_analog: float = DEFAULT_W_ANALOG,
    w_equity: float = DEFAULT_W_EQUITY,
    strong_similarity: float = DEFAULT_STRONG_SIMILARITY,
    min_strong_analogs: int = DEFAULT_MIN_STRONG_ANALOGS,
) -> list[str]:
    """Describe the deterministic scoring model (spec Section 10 methodology appendix),
    rendered from the SAME parameters compute_sector_score was run with -- so the appendix
    reflects the actual blend rather than asserting fixed numbers. Defaults mirror
    compute_sector_score's (the shared module constants), so a default run reads exactly
    as before; a run scored with custom weights renders those instead."""
    return [
        _heading_2("Methodology"),
        "",
        (
            "Each sector's score blends three normalized signals: the regime tilt "
            f"(business-cycle prior, weight {w_regime:.2f}), the analog evidence (similarity-"
            f"weighted forward returns after comparable past regimes, weight {w_analog:.2f}), "
            f"and current 6-month equity momentum (weight {w_equity:.2f}). Each signal is min-max "
            "scaled across the sectors before blending so no single raw scale dominates."
        ),
        "",
        (
            "Confidence = strength x agreement: strength rises with the count of strong "
            f"analogs (similarity >= {strong_similarity:.2f}, saturating at {min_strong_analogs}) and agreement is high when "
            "the three signals point the same way. Scoring is fully deterministic -- no "
            "model call -- so an identical run reproduces an identical ranking."
        ),
    ]

def _render_sector_rationale(rankings: list[dict], flags: list | None = None) -> list[str]:
    """Per-sector 'why it ranked here' lines from each row's `detail` sub-scores, with
    any audit flag that NAMES a sector annotated inline beneath it (guardrail #6 -- the
    reader sees the caveat right next to the affected sector)."""
    flagged: dict[str, list[str]] = {}
    for f in (flags or []):
        flagged.setdefault(getattr(f, "label", ""), []).append(
            f"{getattr(f, 'source', '?')}: {getattr(f, 'message', '')}"
        )

    rationale: list[str] = [_heading_2("Sector rationale"), ""]
    for r in rankings:
        detail = r.get("detail") or {}
        line = f"- **{r['sector']}** (rank {r['rank']}): "
        if not detail:
            line += "no sub-scores available."
        else:
            tilt = detail.get("regime_tilt", 0)
            word = "favored" if tilt > 0 else "disfavored" if tilt < 0 else "neutral"
            line += (
                f"regime tilt {tilt:+d} ({word}); "
                f"analogs {detail.get('analog_return', 0.0):+.3f} over "
                f"{detail.get('n_strong_analogs', 0)} strong match(es); "
                f"momentum {detail.get('equity_signal', 0.0):+.2f}"
            )
            nr = detail.get("normalized_regime_tilt")
            na = detail.get("normalized_analog_return")
            ne = detail.get("normalized_equity_signal")
            if nr is not None and na is not None and ne is not None:
                agree = _signal_agreement(nr, na, ne)
                conviction = (
                    "signals aligned" if agree >= 0.66
                    else "signals conflicting" if agree <= 0.34
                    else "signals mixed"
                )
                line += f"; {conviction} (agreement {agree:.2f})"
        rationale.append(line)
        for msg in flagged.get(r["sector"], []):
            rationale.append(f"  - flagged ({msg})")
    return rationale

def _fmt_pct(x: float) -> str:
    """Render an analog forward return (a decimal, e.g. -0.22) as a signed percent."""
    return f"{x * 100:+.0f}%"

def _analog_moves_text(returns: dict, k: int = 3) -> str:
    """Format one analog's forward sector returns as 'best ...; worst ...' (top/bottom k),
    or a single best-first list when there are too few sectors to split cleanly. Skips
    sectors with no data (None); returns '' when none are populated."""
    pairs = [(s, r) for s, r in returns.items() if isinstance(r, (int, float))]
    if not pairs:
        return ""
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    if len(pairs) <= 2 * k:                       # too few to split without overlap
        return ", ".join(f"{s} {_fmt_pct(r)}" for s, r in pairs)
    top = ", ".join(f"{s} {_fmt_pct(r)}" for s, r in pairs[:k])
    bottom = ", ".join(f"{s} {_fmt_pct(r)}" for s, r in pairs[-k:][::-1])
    return f"best {top}; worst {bottom}"

def _render_macro_snapshot(series_history: dict | None, series_meta: dict | None = None) -> list[str]:
    """The macro picture behind the regime call: each indicator's latest reading (the
    snapshot the ToT classified), as a compact table. Reads the newest point of each
    series in series_history -- which equals the snapshot value, since the macro agent
    anchors the snapshot to the last observation. When series_meta is supplied, a
    'Released' column shows FRED's actual publication date, distinguishing the data's
    PERIOD (which FRED dates at period start, so it reads a bit back) from when it was
    PUBLISHED -- so current inputs don't look stale. Returns [] when no history is
    available, so the section opts out."""
    if not series_history:
        return []
    meta = series_meta or {}
    has_release = any((meta.get(k) or {}).get("release_date") for k in series_history)
    rows: list[str] = []
    for key in const.INDICATOR_KEYS:
        hist = series_history.get(key)
        if not hist:
            continue
        latest = hist[-1]
        label = const._INDICATOR_LABELS.get(key, key)
        observed = latest.get("date", "n/a")
        if has_release:
            released = (meta.get(key) or {}).get("release_date") or "n/a"
            rows.append(f"| {label} | {_fmt_value(latest.get('value'))} | {observed} | {released} |")
        else:
            rows.append(f"| {label} | {_fmt_value(latest.get('value'))} | {observed} |")
    if not rows:
        return []
    if has_release:
        intro = (
            "The indicator readings the regime classification was computed from. FRED dates "
            "each observation at the START of its period, so a current monthly series reads "
            "about a month back and quarterly GDP about a quarter back; the **Released** "
            "column is when FRED actually published that figure -- confirming the inputs are "
            "the latest available, not stale."
        )
        header = ["| Indicator | Latest | Observed (period) | Released |",
                  "| :--- | ---: | :--- | :--- |"]
    else:
        intro = "The indicator readings the regime classification was computed from:"
        header = ["| Indicator | Latest | Observed |", "| :--- | ---: | :--- |"]
    return [
        _heading_2("Macro snapshot"),
        "",
        intro,
        "",
        *header,
        *rows,
    ]

def _render_sector_tiers(rankings: list[dict]) -> list[str]:
    """Headline overweight/underweight read: group the ranked sectors into ranking
    tertiles -- favored (top third), neutral (middle), disfavored (bottom third) -- so the
    reader gets the actionable grouping before the detailed table. DESCRIBES where each
    sector landed in the model's ranking; it is not investment advice (guardrail #5).
    Returns [] for a universe too small to tier meaningfully (the table already says it)."""
    n = len(rankings)
    if n < 3:
        return []
    cut = max(1, n // 3)
    favored = [r["sector"] for r in rankings[:cut]]
    disfavored = [r["sector"] for r in rankings[-cut:]]
    neutral = [r["sector"] for r in rankings[cut:n - cut]]
    return [
        _heading_2("Favored / neutral / disfavored"),
        "",
        f"- **Favored** (top {cut} of {n}): {', '.join(favored)}",
        f"- **Neutral** (middle {len(neutral)}): {', '.join(neutral) or 'none'}",
        f"- **Disfavored** (bottom {cut}): {', '.join(disfavored)}",
        "",
        "Grouping describes where each sector landed in the model's ranking; it is not "
        "investment advice.",
    ]

def _render_run_metadata(
    *,
    as_of: str,
    model_label: str | None = None,
    run_id: str | None = None,
    w_regime: float = DEFAULT_W_REGIME,
    w_analog: float = DEFAULT_W_ANALOG,
    w_equity: float = DEFAULT_W_EQUITY,
) -> list[str]:
    """A compact reproducibility footer: when the brief was generated, the as-of anchor,
    which model wrote the summary, the run id (pairs the brief with its
    trace-<run_id>.jsonl), and the scoring blend used -- so two briefs are comparable
    across models/runs at a glance. model_label / run_id are omitted when not supplied."""
    lines = [
        _heading_2("Run metadata"),
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Evaluation as of: {as_of}",
    ]
    if model_label:
        lines.append(f"- Summary model: {model_label}")
    if run_id:
        lines.append(f"- Run id: {run_id}")
    lines.append(
        f"- Scoring weights: regime {w_regime:.2f}, analog {w_analog:.2f}, equity {w_equity:.2f}"
    )
    return lines

def _render_alternatives(regime_analysis: dict | None) -> list[str]:
    """Show the regimes the Tree-of-Thought weighed and REJECTED (spec 3.3 / 10.1), not
    just the survivor: a table of each candidate's support and sub-scores with its outcome,
    then the one-line rationale per branch. Reads the ToT audit_entry structurally (dict
    access) so synthesize needs no classify_regime_tot import; returns [] when there is no
    branch record or only the winner, so the section opts out."""
    if not regime_analysis:
        return []
    branches = regime_analysis.get("branches") or []
    if len(branches) < 2:
        return []
    ordered = sorted(branches, key=lambda b: b.get("support", 0.0), reverse=True)
    out: list[str] = [
        _heading_2("Alternatives considered"),
        "",
        (
            "The regime was chosen by a bounded Tree-of-Thought: it fans out several "
            "candidate regimes, scores each against its own historical analogs and current "
            "sector momentum, then keeps the best-supported. The candidates it weighed:"
        ),
        "",
        "| Regime | Support | Analog match | Signal fit | Outcome |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for b in ordered:
        outcome = "**selected**" if not b.get("pruned") else "rejected"
        out.append(
            f"| {b.get('regime', '?')} | {b.get('support', 0.0):.3f} | "
            f"{b.get('analog_similarity', 0.0):.2f} | {b.get('signal_consistency', 0.0):.2f} | "
            f"{outcome} |"
        )
    out.append("")
    for b in ordered:
        rationale = (b.get("rationale") or "").strip()
        if not rationale:
            continue
        tag = "selected" if not b.get("pruned") else "rejected"
        out.append(f"- **{b.get('regime', '?')}** ({tag}): {rationale}")
    return out

def _render_analog_outcomes(
    analogs: list[dict] | None, horizon: str | None = None, *, max_analogs: int = 5
) -> list[str]:
    """Show what sectors ACTUALLY did after the closest historical analogs -- the realized
    payoff behind the analog signal, which the Sources block (dates + similarity only)
    doesn't convey. Uses the SAME forward-return window the ranking scored against
    (_analog_returns / the requested horizon, default 6m); top and bottom movers per
    analog. Returns [] when no analog returns are available, so the section opts out."""
    if not analogs:
        return []
    horizon_months = _horizon_months(horizon)
    window = _effective_horizon_months(horizon)
    ranked = sorted(analogs, key=_analog_sim, reverse=True)[:max_analogs]
    rows: list[str] = []
    for a in ranked:
        moves = _analog_moves_text(_analog_returns(a, horizon_months))
        if not moves:
            continue
        sim = a.get("similarity")
        sim_txt = f"{sim:.2f}" if isinstance(sim, (int, float)) else "n/a"
        rows.append(
            f"- **{a.get('date', '?')}** ({a.get('regime', 'n/a')}, similarity {sim_txt}): {moves}"
        )
    if not rows:
        return []
    return [
        _heading_2("Historical analogs: what happened next"),
        "",
        (
            f"The closest past periods to today's macro snapshot, and how sectors performed "
            f"in the {window} months after each (the same forward window the ranking scores "
            f"against) -- the realized evidence behind the analog signal:"
        ),
        "",
        *rows,
    ]

def _render_caveats(
    confidence: float, flags: list, rankings: list[dict], horizon: str | None = None,
    universe: tuple[str, ...] | None = None, *, low_confidence: bool = False,
) -> list[str]:
    """Surface the run's reliability caveats (guardrail #4) when there are any: a
    near-tie regime call (the ToT's low_confidence flag), audit flags raised, sectors
    whose ranking rests on thin analog support (fewer than 3 strong matches), and --
    when the query asked for a horizon different from the analog evidence window
    (ANALOG_HORIZON_MONTHS) -- an explicit horizon-mismatch note, so the brief never
    silently answers a long-horizon question with short-horizon evidence. Returns []
    when the run is clean, so the section is omitted."""
    thin = [
        r["sector"] for r in rankings
        if (r.get("detail") or {}) and r["detail"].get("n_strong_analogs", 0) < 3
    ]
    notes: list[str] = []
    if low_confidence:
        notes.append(
            "The regime call was a near-tie -- the top Tree-of-Thought branches fell "
            "within the confidence margin, so the regime (and the ranking it drives) is "
            "lower-confidence; treat it as provisional and weigh the alternatives "
            "considered above."
        )
    if flags:
        notes.append(
            f"{len(flags)} audit flag(s) were raised (see Audit flags); read the "
            "ranking with that in mind."
        )
    if thin:
        notes.append("Thin analog support (<3 strong matches) for: " + ", ".join(thin) + ".")
    months = _horizon_months(horizon)
    if months is not None and months not in const.ANALOG_HORIZONS_MONTHS:
        seeded = ", ".join(str(h) for h in const.ANALOG_HORIZONS_MONTHS)
        notes.append(
            f"Requested horizon ({horizon}) is not one of the seeded analog windows "
            f"({seeded} months): the ranking falls back to the "
            f"{const.ANALOG_DEFAULT_HORIZON_MONTHS}-month window, so read it as directional "
            "for the requested horizon rather than calibrated to it."
        )
    if universe is not None and len(universe) < 3:
        notes.append(
            f"The requested sub-universe has only {len(universe)} sector(s), so scores are "
            "normalized across all 11 sectors (market-wide) rather than within the subset, "
            "to avoid a degenerate two-point scaling."
        )
    if not notes:
        return []
    return [_heading_2("Confidence & caveats"), "", *(f"- {n}" for n in notes)]


def _render_data_freshness(series_history: dict | None, as_of: str | None) -> list[str]:
    """Always-on input-freshness summary (guardrail #2, reader-facing): how current the
    macro inputs are relative to the as-of date. The audit only surfaces freshness as a
    FLAG when a series breaches its ceiling; this states the data vintage on EVERY run --
    especially useful for back-dated runs, where it confirms point-in-time inputs (no
    observation dated after as_of). Display-only: it reports the dates, it does not
    re-apply the audit's per-series staleness thresholds. Returns [] when no history or
    no as_of is available, so the section opts out."""
    if not series_history or not as_of:
        return []
    try:
        as_of_date = date.fromisoformat(as_of)
    except (ValueError, TypeError):
        return []
    ages: list[tuple[str, str, int]] = []          # (series_id, newest_date, age_days)
    for series_id, hist in series_history.items():
        newest = hist[-1].get("date") if hist else None
        if not newest:
            continue
        try:
            age = (as_of_date - date.fromisoformat(newest)).days
        except (ValueError, TypeError):
            continue
        ages.append((series_id, newest, age))
    if not ages:
        return []
    def _line(label: str, item: tuple[str, str, int]) -> str:
        sid, d, age = item
        unit = "day" if age == 1 else "days"
        return f"- {label}: {sid} -- {d} ({age} {unit} before the as-of date)"
    return [
        _heading_2("Data freshness"),
        "",
        (
            f"Macro inputs are point-in-time as of {as_of} (no observation dated after it). "
            f"Vintage of the {len(ages)} indicator series feeding the snapshot:"
        ),
        "",
        _line("Freshest", min(ages, key=lambda a: a[2])),
        _line("Most lagging", max(ages, key=lambda a: a[2])),
    ]


def _render_audit_trail(audit_log) -> list[str]:
    """Reconcile the run against its audit log (guardrail #7): the audited tool-call
    count, the session-end reconciliation result (do the logged tool-call entries
    match the counter?), and the revision history -- which sectors were quarantined
    and re-run, and why the loop stopped. Read structurally (getattr / event tags) so
    an audit_log lacking these simply yields an omitted section rather than an error."""
    audit_trail: list[str] = []
    entries = getattr(audit_log, "entries", None) or []
    revisions = [e for e in entries if e.get("event") == "revision"]
    halts = [e for e in entries if e.get("event") == "revision_halt"]
    reconciliation = next((e for e in entries if e.get("event") == "reconciliation"), None)
    tool_calls = getattr(audit_log, "tool_calls", None)

    if not (revisions or halts or reconciliation is not None or tool_calls is not None):
        return audit_trail

    audit_trail.append(_heading_2("Audit trail"))
    audit_trail.append("")
    if tool_calls is not None:
        audit_trail.append(f"- Tool calls audited: {tool_calls}")
    if reconciliation is not None:
        logged = reconciliation.get("logged_tool_calls")
        counted = reconciliation.get("tool_calls")
        if reconciliation.get("reconciled"):
            audit_trail.append(
                f"- Audit-log reconciled (guardrail #7): {logged} logged tool-call "
                f"entries match {counted} tool calls."
            )
        else:
            audit_trail.append(
                f"- Audit-log reconciliation FAILED (guardrail #7): {logged} logged "
                f"tool-call entries vs {counted} tool calls."
            )
    if revisions:
        audit_trail.append(f"- Revisions: {len(revisions)}")
        for e in revisions:
            dropped = ", ".join(e.get("dropped", [])) or "(none)"
            audit_trail.append(f"  - cycle {e.get('cycle', '?')}: quarantined {dropped}")
    for e in halts:
        audit_trail.append(
            f"- Revision halted ({e.get('reason', '?')}) after cycle {e.get('cycle', '?')}."
        )
    return audit_trail

def _render_disclaimer() -> list[str]:
    """Guardrail #5: the fixed not-advice disclaimer, verbatim from constants, set off
    by a horizontal rule and italicized as a footer."""
    return ["---", "", f"*{const.DISCLAIMER}*"]


# -------------  LLM-written opening summary (the synthesis agent's narration)  ------
EXEC_SUMMARY_SYSTEM_PROMPT = (
    "You are a buy-side research editor. Write a clear, plain-English opening summary "
    "of a sector-rotation analysis for a portfolio manager who will then read the "
    "detailed tables below.\n\n"
    "Rules:\n"
    "- EXACTLY two paragraphs of prose. No headings, no bullet points, no markdown.\n"
    "- Use ONLY the facts provided below. Do not invent figures, sectors, or reasons.\n"
    "- Paragraph 1: what the analysis concluded -- the macro regime, the sectors it "
    "favors and disfavors, and the overall confidence.\n"
    "- Paragraph 2: the main reasons behind the ranking and the key caveats (thin "
    "evidence, audit flags, conflicting signals) the reader should keep in mind.\n"
    "- The analysis reflects the macro regime as of the run date and the ranked sectors "
    "given to you; it is NOT conditioned on any hypothetical scenario in the question. "
    "If the question poses a scenario or assumption, summarize what this current-data "
    "analysis found -- do not imply the scenario was modeled.\n"
    "- If the facts say the ranking was restricted to a sector sub-universe, describe the "
    "favored and disfavored sectors as being WITHIN that subset, not the whole market.\n"
    "- The ranking is based on the analog evidence window stated in the facts. If a "
    "requested horizon differs from it, say so rather than implying the ranking is "
    "calibrated to the requested horizon.\n"
    "- If Fed-narrative passages are provided, you may note in the second paragraph what "
    "policymakers were signaling, as QUALITATIVE context, attributed to the source and date "
    "(e.g. 'the April FOMC minutes noted ...'). Treat them as corroboration or tension only; "
    "they are NOT inputs to the sector scores, so never imply the ranking was computed from "
    "them, and do not fabricate quotes beyond the excerpts given.\n"
    "- If historical analog outcomes are provided, you may cite them in the second "
    "paragraph as the realized evidence behind the analog signal, attributing each to its "
    "period (e.g. 'in the six months after the 2001-03 analog, technology fell sharply'). "
    "They are REALIZED PAST returns, not a forecast -- never present them as what WILL "
    "happen, and invent no figures beyond those given.\n"
    "- This is research, not advice: do not tell the reader to buy, sell, or allocate.\n"
    "- Be concise and professional -- roughly four to six sentences per paragraph."
)


def _summary_facts(
    query: str,
    regime: const.Regime,
    as_of: str,
    rankings: list[dict],
    confidence: float,
    flags: list,
    horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
    fed_sources: list[dict] | None = None,
    analogs: list[dict] | None = None,
) -> str:
    """Compact, factual digest of the FINAL outcome -- the only material the summary
    LLM is allowed to draw on (it is instructed to invent nothing beyond this)."""
    def _row(r: dict) -> str:
        d = r.get("detail") or {}
        return (
            f"{r['sector']} (rank {r['rank']}, score {r['score']:.3f}, "
            f"confidence {r['confidence']:.2f}, regime tilt {d.get('regime_tilt', 0):+d}, "
            f"{d.get('n_strong_analogs', 0)} strong analog(s))"
        )

    lines = [
        f"Analyst question: {query}",
        f"Macro regime: {regime.value}",
        f"Overall confidence: {confidence:.2f}",
        f"Horizon requested in the question: {horizon or 'unspecified'}",
        f"With an evaluation point as of: {as_of}",
        f"Analog evidence window used for the ranking: {_effective_horizon_months(horizon)} months",
        "Top-ranked sectors: " + ("; ".join(_row(r) for r in rankings[:3]) or "n/a"),
        "Bottom-ranked sectors: " + ("; ".join(_row(r) for r in rankings[-3:]) or "n/a"),
    ]
    if universe is not None and len(universe) < len(const.SECTOR_ETFS_LIST):
        lines.append(
            "Sector universe: ranking restricted to "
            + ", ".join(universe)
            + " (ranked among these, not the full 11)"
        )
    # Realized analog outcomes -- the forward sector returns AFTER the closest past
    # periods, i.e. the evidence behind the analog signal. Uses the same horizon window
    # and helpers as the report's "what happened next" section, so the prose and that
    # table agree. Offered as historical evidence to explain the ranking; the system
    # prompt reinforces that these are realized outcomes, not a forecast.
    if analogs:
        horizon_months = _horizon_months(horizon)
        window = _effective_horizon_months(horizon)
        outcome_lines: list[str] = []
        for a in sorted(analogs, key=_analog_sim, reverse=True)[:3]:
            moves = _analog_moves_text(_analog_returns(a, horizon_months))
            if not moves:
                continue
            sim = a.get("similarity")
            sim_txt = f"{sim:.2f}" if isinstance(sim, (int, float)) else "n/a"
            outcome_lines.append(
                f"- {a.get('date', '?')} ({a.get('regime', 'n/a')}, similarity {sim_txt}): {moves}"
            )
        if outcome_lines:
            lines.append(
                f"Historical analog outcomes (realized sector returns in the {window} months "
                "after the closest comparable periods -- the evidence behind the analog "
                "signal, NOT a forecast):"
            )
            lines.extend(outcome_lines)
    # Qualitative Fed context (top passages), drawn from the already-formatted provenance
    # rows so the summary cites exactly what the Sources block does. Flagged as NOT a
    # scoring input so the editor (and the reader) never mistakes it for a ranking driver.
    if fed_sources:
        lines.append("Fed narrative (qualitative context only -- NOT a scoring input):")
        for s in fed_sources[:3]:
            lines.append(f"- {s.get('label', s.get('id', 'Fed passage'))}")
    if flags:
        lines.append(
            f"Audit flags raised ({len(flags)}): "
            + "; ".join(f"{getattr(f, 'label', '?')} -- {getattr(f, 'message', '')}" for f in flags)
        )
    else:
        lines.append("Audit flags raised: none")
    return "\n".join(lines)


def generate_executive_summary(
    query: str,
    as_of: str,
    regime: const.Regime,
    rankings: list[dict],
    confidence: float,
    flags: list,
    *,
    horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
    fed_sources: list[dict] | None = None,
    analogs: list[dict] | None = None,
    call_model: Callable[[str, str], str],
) -> str:
    """Write the brief's two-paragraph, plain-English opening from the FINAL outcome
    (regime, ranked sectors, confidence, flags, the realized outcomes of the closest
    historical analogs, and any retrieved Fed-narrative passages as qualitative context)
    via the injected LLM seam.

    This is the synthesis agent's narration step (spec Section 4) and the one place an
    LLM touches the report. It runs during report assembly, AFTER the ReAct loop and
    audit reconciliation, and is NOT a retrieval tool call -- so, like the critic's
    calls, it never goes through the coordinator's tool-call counter and does not count
    against the guardrail #3 cap.

    Returns "" on any model failure (logged) so report assembly degrades gracefully --
    the brief simply opens with its structured sections instead of crashing.
    """
    user_prompt = (
        "Write the opening summary from these analysis results:\n\n"
        + _summary_facts(query=query,
                         regime=regime,
                        as_of=as_of,
                        rankings=rankings,
                        confidence=confidence,
                        flags=flags,
                        horizon=horizon,
                        universe=universe,
                        fed_sources=fed_sources,
                        analogs=analogs
                    )
    )
    try:
        return call_model(EXEC_SUMMARY_SYSTEM_PROMPT, user_prompt).strip()
    except Exception:
        logger.exception("Executive-summary generation failed; omitting the section")
        return ""


def generate_report(
    *,
    query: str,
    regime: const.Regime,
    as_of: str,
    rankings: list[dict],
    confidence: float,
    flags: list,
    audit_log, # 
    sources: list[dict],
    analogs: list[dict] | None = None,
    regime_analysis: dict | None = None,
    low_confidence: bool = False,
    series_history: dict | None = None,
    series_meta: dict | None = None,
    call_model: Callable[[str, str], str] | None = None,
    model_label: str | None = None,
    run_id: str | None = None,
    pdf_path: str | Path | None = None,
    horizon: str | None = None,
    universe: tuple[str, ...] | None = None,
    w_regime: float = DEFAULT_W_REGIME,
    w_analog: float = DEFAULT_W_ANALOG,
    w_equity: float = DEFAULT_W_EQUITY,
    strong_similarity: float = DEFAULT_STRONG_SIMILARITY,
    min_strong_analogs: int = DEFAULT_MIN_STRONG_ANALOGS,
) -> str:
    """
    Assemble the analyst-facing brief from the synthesis output (spec Section 5).

    Matches coordinator.ReportGenerator: the coordinator calls this with keyword
    args query / regime / rankings / confidence / flags / audit_log. `flags` and
    `audit_log` are left loosely typed on purpose so this module need not import
    the coordinator's AuditFlag / AuditLog types (no synthesize -> coordinator
    dependency); they are consumed structurally (flag.source/.label/.message,
    audit_log.tool_calls).

    Parameters
    ----------
    query
        The analyst's original question, echoed in the brief header.
    regime
        The macro regime classify_regime_tot settled on (the run's headline finding).
    as_of
        The as-of date provided as the anchor point for analysis (could be in the past)
    rankings
        The ranked sector rows from compute_sector_score (best first).
    confidence
        Overall confidence the coordinator derived from the per-sector rows.
    flags
        Audit flags raised during the run (empty when nothing tripped).
    audit_log
        The run's audit log; its tool-call count and revision history are surfaced.
    sources
        Provenance rows from build_sources, rendered as the cited Sources block.
    analogs
        The winning regime's historical analogs (macro.analogs); rendered as the
        "what happened next" section showing realized forward sector returns over the
        same window the ranking scored. None/empty omits the section.
    regime_analysis
        The ToT audit_entry (macro.tot_result.audit_entry): every candidate regime the
        Tree-of-Thought weighed, with support/sub-scores and rationale, rendered as
        "Alternatives considered". None or a lone branch omits the section.
    low_confidence
        The ToT's low_confidence flag (macro.low_confidence): True when the top regime
        branches fell within the tie margin. Surfaced as a leading caveat so a near-tie
        regime call is never presented as settled. Defaults False.
    series_history
        Per-indicator history (macro.series_history): {INDICATOR_KEY: [{date, value}]}.
        Used only for the always-on "Data freshness" summary (input vintage vs the as-of
        date). None/empty omits that section.
    call_model
        Optional (system_prompt, user_prompt) -> text seam for the LLM that writes the
        two-paragraph opening summary. When None, the summary is skipped and the brief
        opens with the structured sections. This call runs during report assembly --
        after the ReAct loop and audit reconciliation -- so it is NOT a retrieval tool
        call and never goes through the coordinator's tool-call counter; it does not
        count against the guardrail #3 cap (like the audit critic's calls).
    model_label
        Optional "service/model" label for the model that wrote the summary (e.g.
        "anthropic/claude-sonnet-4-5-20250929"), rendered in the run-metadata footer so a
        brief is self-identifying across model-comparison runs. Bound by main alongside
        call_model; None omits the line.
    run_id
        Optional run identifier, rendered in the footer so the brief pairs with its
        trace-<run_id>.jsonl. Bound by main; None omits the line.
    pdf_path
        If given, the rendered brief is also written to this path as a PDF (via the
        markdown-pdf library) as a side effect; the Markdown string is still returned.
    horizon
        The investment horizon parsed from the query (normalized '<N> months'), or
        None. Surfaced in the brief header and the executive summary, and -- when it
        differs from the analog evidence window (ANALOG_HORIZON_MONTHS) -- raised as a
        caveat, so a long-horizon question isn't silently answered with short-horizon
        evidence.
    w_regime, w_analog, w_equity, strong_similarity, min_strong_analogs
        The scoring parameters compute_sector_score was run with; passed straight to the
        methodology appendix so it states the actual blend rather than fixed text. They
        default to the same module constants compute_sector_score defaults to, so a
        default run is unchanged; a caller scoring with custom weights should pass the
        SAME values here and to compute_sector_score.

    Returns
    -------
    str
        A strict-Markdown brief suitable for printing or conversion to PDF.
    """
    # The Fed-narrative passages are already formatted as provenance rows in `sources`
    # (build_sources tags them "FedNarrative"); reuse those so the summary describes the
    # same passages the Sources block cites instead of re-formatting them. They feed the
    # narration as qualitative context only -- the deterministic ranking never reads them.
    fed_sources = [s for s in sources if s.get("tool") == "FedNarrative"]
    summary = (
        generate_executive_summary(
            query=query, regime=regime, as_of=as_of, rankings=rankings,
            confidence=confidence, flags=flags, horizon=horizon, universe=universe,
            fed_sources=fed_sources, analogs=analogs, call_model=call_model
        )
        if call_model is not None
        else ""
    )

    sections: list[list[str]] = [
        _render_header(),
        _render_executive_summary(summary),
        _render_regime_narrative(query, regime, as_of, confidence, horizon, universe),
        _render_macro_snapshot(series_history, series_meta),
        _render_alternatives(regime_analysis),
        _render_sector_tiers(rankings),
        _render_rankings(rankings),
        _render_sector_rationale(rankings, flags),
        _render_analog_outcomes(analogs, horizon),
        _render_caveats(confidence, flags, rankings, horizon, universe, low_confidence=low_confidence),
        _render_flags(flags),
        _render_audit_trail(audit_log),
        _render_data_freshness(series_history, as_of),
        _render_sources(sources),
        _render_methodology(
            w_regime=w_regime, w_analog=w_analog, w_equity=w_equity,
            strong_similarity=strong_similarity, min_strong_analogs=min_strong_analogs,
        ),
        _render_run_metadata(
            as_of=as_of, model_label=model_label, run_id=run_id,
            w_regime=w_regime, w_analog=w_analog, w_equity=w_equity,
        ),
    ]
    lines: list[str] = []
    for block in sections:
        if block:                       # optional sections (caveats, audit trail) opt out by returning []
            lines.extend(block)
            lines.append("")            # blank line between Markdown blocks
    lines.extend(_render_disclaimer())  # footer; no trailing blank

    report = "\n".join(lines)
    logger.info("Report assembled: regime=%s, %d sector(s), %d flag(s), exec_summary=%s, %d chars",
                regime.value, len(rankings), len(flags), bool(summary), len(report))
    if pdf_path is not None:
        report_to_pdf(report, pdf_path)
    return report


def report_to_pdf(
    markdown_text: str,
    out_path: str | Path,
    *,
    title: str = "Sector Rotation Research Brief",
) -> Path:
    """Render a generate_report() Markdown brief to a PDF file (spec Section 11.1).

    Uses the `markdown-pdf` library (markdown-it -> PyMuPDF), imported lazily so the
    rest of the synthesis module -- and the test suite -- carry no PDF dependency
    unless an export is actually requested. Returns the written path.
    """
    try:
        from markdown_pdf import MarkdownPdf, Section
    except ImportError as err:  # keep the failure actionable
        logger.error("PDF export requested but 'markdown-pdf' is not installed")
        raise RuntimeError(
            "PDF export requires the 'markdown-pdf' package; install it with "
            "`uv add markdown-pdf`."
        ) from err

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)   # reports/ is gitignored -> absent on a fresh clone
    pdf = MarkdownPdf(toc_level=0)          # single-section brief, no table of contents
    pdf.add_section(Section(markdown_text))
    pdf.meta["title"] = title
    pdf.save(str(out))
    logger.info("Report PDF written to %s", out)
    return out
