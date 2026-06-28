Capstone project for Carnegie Mellon's **Agentic AI** course.
**Steve Hoberecht**

# Macro-Driven Sector Rotation Research Agent

A multi-agent research assistant that answers one recurring question for an equity
analyst: **given current macroeconomic conditions, which equity sectors are likely to
outperform over a defined horizon?** It pulls live macro and equity data, classifies
the current business-cycle regime, matches it against ~20 years of historical analogs
held in a vector store, scores the eleven GICS sectors, and produces a sourced,
confidence-scored research brief — with a dedicated audit layer checking every data
point and reasoning step along the way.


This README is the practical guide to running it.

> **Disclaimer:** This system is an AI research agent for educational and
> analytical purposes only. It's output does not constitute investment advice.

---

## What it demonstrates

The project is organized around the six capstone rubric concepts:

| Concept | Where it lives |
|---|---|
| **Tool calling** | Six wrapped tools over FRED, yfinance (performance + valuation), the vector store, scoring, and reporting |
| **Reasoning (ReAct)** | The coordinator's think → act → observe → revise loop |
| **Knowledge & memory (RAG)** | ChromaDB store of labeled historical regime snapshots with their subsequent sector returns, plus a second text-RAG corpus of Fed narrative (Beige Book / FOMC minutes / policy statements) |
| **Further reasoning (Tree-of-Thought)** | Scoped to macro regime classification — fan out competing regimes, score each, keep the best |
| **Multi-agent coordination** | Coordinator + macro + equity + synthesis + critic agents |
| **Safety** | Statistical checker, isolated LLM critic, per-series freshness checks, tool-call/revision caps, audit-log reconciliation, fixed disclaimer |

---

## How it works

The coordinator decomposes the analyst's query (today an LLM extracts the investment
horizon and an optional sector **focus** sub-universe — "which defensive sectors …" — with
a deterministic fallback), then runs the two retrieval agents **in sequence — equity first**,
because the macro agent's Tree-of-Thought needs current sector momentum for one of its 
scoring signals:

1. **Equity agent** pulls the 11 sector ETFs and computes 6-month momentum (point-in-time,
   as of the decision date) and valuation.
2. **Macro agent** pulls the 8 FRED indicators, reduces them to a normalized point-in-time
   snapshot, and runs the **Tree-of-Thought** regime classifier: an LLM proposes a few
   candidate regimes (`early_cycle` / `mid_cycle` / `late_cycle` / `contraction`), each is
   scored deterministically against historical analogs *and* current market behavior, and
   the best-supported regime wins.
3. **Audit layer** screens every result: a pure-Python **statistical checker** (z-score,
   IQR, per-series freshness) runs first and unconditionally; only if it passes does the
   **critic** — an isolated, single-shot LLM call — judge whether the evidence actually
   supports the regime assessment.
4. **Revision loop:** a statistical flag naming a sector, quarantines that sector and
   re-runs, bounded by hard caps (20 tool calls / 3 cycles). Other flags are carried
   forward as low-confidence caveats.
5. **Synthesis** scores and ranks the sectors, and `generate_report` writes the cited
   brief.

### Fed-narrative corpus (second RAG)

Alongside the numeric analog store, the project includes a second retrieval corpus over
**Fed narrative text** — the Beige Book, FOMC minutes, and monetary-policy statements —
in `fed_narrative_rag.py`. It parses and chunks the source PDFs (Docling), embeds them
with a sentence-transformers model into a separate `fed_narrative` ChromaDB collection,
and retrieves passages by semantic similarity with a point-in-time (`as_of`) guard, plus
a corpus freshness check. The intent is qualitative evidence — *what policymakers were
saying* — to complement the numeric analogs' *what happened to sectors*.

Fed documents are provided in /data/fed_source_data from 1/1/2025 - 6/15/2026.  Before
first run, seeding is required (instructions below).

The macro agent attaches retrieved passages to its result, so the macro critic now weighs
the top Fed passage as a second isolated evidence item against the regime call. The 
auditor now runs the corpus freshness check on the macro result, and the retrieved 
passages are cited in the brief's Sources block (tagged `FedNarrative`).

---

## Project layout

```
src/sector_rotation_agent/
  constants.py          # fixed config: indicator keys, FRED series, regimes, guardrail params
  config.py             # AppConfig (model provider, logging); loads .env
  model_client.py       # provider-agnostic LLM client (Anthropic / OpenRouter / Ollama / HF)

  fred_query.py         # get_macro_indicators  (FRED)            ─┐ data tools
  yfin_query.py         # get_sector_performance / _valuations    ─┘ (MCP clients)
  mcp_servers/          # FastMCP stdio servers wrapping FRED + yfinance

  generate_hypotheses.py  # ToT fan-out: snapshot -> candidate regimes (LLM)
  score_branch.py         # ToT branch scoring (deterministic)
  classify_regime_tot.py  # ToT orchestrator (fan out -> score -> select)
  historical_analogs.py   # seed + query the ChromaDB analog store (numeric RAG)
  fed_narrative_rag.py    # seed + query the Fed-narrative text corpus (Beige Book / FOMC / policy)

  macro_agent.py        # snapshot -> regime + analogs
  equity_agent.py       # sector data -> momentum / valuation
  coordinator.py        # ReAct orchestration + revision loop + audit log
  audit.py              # check_statistical_anomaly + run_critic_check + ResultAuditor
  synthesize.py         # compute_sector_score + build_sources + generate_report (+ PDF)
  main.py               # composition root / CLI entry point

tests/                  # offline unit tests (LLM + network seams are injected/faked)
data/                   # ChromaDB store + norm_stats.json (created by the seed step)
data/fed_source_data    # PDF files downloaded from https://www.federalreserve.gov
```

---

## Prerequisites

- **Python 3.14+** and **[uv](https://docs.astral.sh/uv/)** for environment + dependency management.
- A **FRED API key** (free: <https://fredaccount.stlouisfed.org/apikeys>).
- An LLM provider. A single `model_location` switch (see [Configuration](#configuration))
  selects **cloud-only**, **local-only**, or a **mixed** split across Anthropic, OpenRouter,
  Ollama (local), and Hugging Face; the shipped default is `cloud_only` via OpenRouter.
  Cloud providers need their own API key; local runs need Ollama.
   **Always ensure .env is in .gitignore**
---

## Setup

```powershell
# 1. install the package (editable) and all dependencies, into a managed venv
uv sync

# 2. create a .env file in the repo root with your keys, e.g.:
#    FRED_API_KEY=your_fred_key
#    ANTHROPIC_API_KEY=your_key        # if using the Anthropic provider
#    OPENROUTER_API_KEY=your_key       # if using OpenRouter
```

Keys are read from `.env` (via `python-dotenv`); never commit them. **Always ensure .env is in .gitignore**

PDF export works out of the box — `markdown-pdf` is a bundled dependency.

---

## Seed the memory store (one-time)

Build the ChromaDB analog store and the normalization statistics. This pulls ~20 years
of monthly history from FRED and Yahoo, so it makes live network calls and takes a
minute or two:

```powershell
uv run python -m sector_rotation_agent.historical_analogs
```

This writes `data/chroma/` (the vector store) and `data/norm_stats.json`. The main app
refuses to run until both exist.

The agent also uses a second retrieval corpus over **Fed narrative text** (Beige Book,
FOMC minutes, monetary-policy statements). Seed it from the source PDFs in
`data/fed_source_data/` (listed in `local_fed_files.json`):

```powershell
uv run python -m sector_rotation_agent.fed_narrative_rag
```

This builds the separate `fed_narrative` ChromaDB collection that the macro critic weighs
and that the brief cites in its Sources block.

> **Note:** the leading-index indicator uses the OECD composite leading indicator for the
> US (`USALOLITOAASTSAM`), an index centered on 100. It replaced FRED's `USSLIND`, which
> was discontinued in 2020 and had been silently truncating the analog store at ~2020.
> Re-running this seed step picks up the current series and restores analog coverage
> through the present — re-seed any time the indicator set or one of its series changes.

---

## Run the agent

Pass the analyst question and the point-in-time "as of" date as flags. Both are optional —
omit `--ao` to default to today, and omit `--q` to be prompted for the question:

```powershell
uv run python -m sector_rotation_agent.main --q "Which sectors should I overweight/underweight over the next 6 months?" --ao 2026-06-01
```

The brief prints to the console, and you're then prompted whether to also save a PDF
(written to `reports/sector_rotation_brief.pdf`).

The brief opens with an LLM-written two-paragraph executive summary, then contains: a
regime narrative, the ranked sector table with per-sector confidence, a per-sector
rationale, confidence caveats, any audit flags, the audit trail (tool-call count,
reconciliation result, and any revisions), the cited sources, a methodology appendix, and
the fixed disclaimer.

Note: the investment horizon is extracted from your question by an LLM (e.g. "next 6
months", "next year", "a couple of quarters"), with a deterministic fallback, and shown in
the brief. The analog store is seeded at several forward windows (3 / 6 / 12 months), so a
seeded horizon **changes the sector ranking** — the scorer reads that window's realized
returns; a horizon outside that set, or an unstated one, falls back to the 6-month default
and the brief flags it.

Note: if your question names a **sector subset** — a group ("which **defensive** sectors
should I overweight?", also cyclical / rate-sensitive / growth) or explicit tickers ("how
do XLU and XLP look?") — the brief ranks **only those sectors**, scored relative to one
another, and says so in the header. Groups are defined in `SECTOR_GROUPS` (`constants.py`)
and overlap by design. A subset of one or two sectors is normalized market-wide instead (a
guard against degenerate two-point scaling), noted in the caveats. The full eleven are
always still fetched and used to classify the regime, so the focus changes only *which
sectors are presented*, never the regime call.


The as-of date is **not** read from the question, so apart from the horizon and the focus
sub-universe the analysis is a function of the as-of date, not the wording of the question
(spec §3.4 / §10.1).

Note: the methodology appendix is rendered from the scoring parameters the run used —
the weight blend (default 0.40 / 0.35 / 0.25) and confidence thresholds (strong-analog
similarity 0.75, saturating at 3) — so it always matches what was actually computed
rather than restating fixed text.

---

## Configuration

Model selection and runtime knobs live in `src/sector_rotation_agent/config.py`
(`AppConfig`). The `model_client` layer is provider-agnostic and supports four backends —
**Anthropic** (cloud), **OpenRouter**, **Ollama** (local), and **Hugging Face**. A single
`model_location` switch decides how the run is wired:

- `cloud_only` — every LLM seam uses the cloud model (`cloud_model_service` / `cloud_model`).
- `local_only` — every seam uses the local model (`local_model_service` / `local_model`).
- `mixed` — the spec's (§11.2) **hybrid split**: the high-value seams (the Tree-of-Thought
  hypothesis fan-out and the executive summary) run on the cloud model, while the cheaper,
  high-frequency seams (the audit critic and query decomposition) run on the local model —
  which also gives the critic a perspective independent of the hypothesis generator.

Set `model_location` and the two `(service, model)` pairs in `AppConfig`. The composition
root (`main.py`) builds only the client(s) the chosen mode needs, so `cloud_only` doesn't
require a local Ollama and `local_only` doesn't require a cloud API key.

Logging is configured at import: set `LOGGING_LEVEL` (`DEBUG` / `INFO` / …) in the
environment; the app's own output goes to a `sector_rotation_agent.log` file rather than
the console, so console output stays limited to the brief itself. Each run also writes
per-component logs under `logs/` (`main`, `coordinator`, `model_client`, and the
`fred_server` / `yfin_server` MCP subprocesses) plus a structured per-run trace
(`trace-<run_id>.jsonl`) capturing phase latencies and full LLM-call detail.

```powershell
$env:LOGGING_LEVEL = "DEBUG"   # verbose run, written to the log file
```

---

## Testing

The suite is **offline by default** — the LLM and network seams are dependency-injected,
so orchestration, scoring, and audit logic are exercised with fakes (no API key, no live
data):

```powershell
uv run pytest                         # full suite
uv run pytest tests/test_audit.py     # one file
uv run pytest -k vectorize            # tests matching a name
uv run pytest -s                      # show print() output
```

Tests that hit live data or a real model are gated behind an environment flag, so they're
skipped unless you opt in:

```powershell
$env:TEST_MODE = "Integration"; uv run pytest
```

Lint / type-check:

```powershell
uv run ruff check          # lint
uv run pyright             # static type-check
```

---

## Backtesting (point-in-time evaluation)

Beyond the unit suite, a backtest harness runs the **full agent** at several historical
"as of" dates and compares the ranking it produced against what the sectors *actually*
did over the following six months:

```powershell
uv run python -m sector_rotation_agent.backtest
uv run python -m sector_rotation_agent.backtest --windows 2018-06-30 2021-12-31
```

For each window it reports the regime the agent called, the Spearman correlation between
the per-sector score and realized forward return, and the favored-minus-disfavored return
spread, then writes a Markdown summary to `reports/backtest_summary.md`.

The evaluation is **point-in-time**: the macro agent fetches FRED with `end_date=as_of`,
the analog store is filtered to outcomes realized on or before `as_of`, and the equity
path fetches and trims prices to end at `as_of` — so a back-dated run cannot see prices or
outcomes from after the decision date. Sector *valuation* is the one current-only input,
but it isn't used in scoring, so the ranking carries no look-ahead. (FRED returns the
latest *revisions* rather than true vintage data — a disclosed limitation, not a
future-period leak.)

---

## Safety guardrails

The audit layer implements the spec's seven guardrails (§9). In brief: every numeric
claim is tagged to its source tool; data freshness is checked with **per-series** age
ceilings (45 days by default, ~200 for quarterly GDP so a current-but-quarterly print
isn't falsely flagged, ~75 for the monthly OECD leading index given its publication lag); the run is capped at 20 tool calls and 3 revision cycles; thin or
conflicting evidence is labeled low-confidence; the critic sees only one hypothesis and
one evidence item to prevent confirmation bias; the audit log is reconciled against the
tool-call count at session end; and every brief carries the fixed not-advice disclaimer.

---

## License

MIT — see [`LICENSE`](./LICENSE).
