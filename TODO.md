# Basic Implementation

Goal: one real query runs end to end and produces a cited, disclaimered report,
demonstrating each rubric concept once (tool calling, ReAct, RAG memory, ToT,
≥1 specialized agent, ≥1 safety guardrail). Elaborate pieces are deferred — see
the bottom.

---

## Setup (do first)

- [x] `uv sync` to materialize the env (installs the package editable + scipy + pytest)
- [x] `uv add anthropic fredapi yfinance chromadb sentence-transformers pandas numpy`
- [x] `uv run pytest` — confirm the 9 existing tests pass
- [x] Put `ANTHROPIC_API_KEY` and `FRED_API_KEY` in your environment (not in code)

## Baseline already in place

- [x] Package structure, `pyproject.toml`, build system
- [x] `classify_regime_tot.py` (ToT orchestrator) — done + tested
- [x] `score_branch.py` (branch scoring) — done + tested
- [x] `tests/` seeded, 9 tests green
- [x] Stubs: `generate_hypotheses.py`, `fred_query.py`, `yfin_query.py`

---

## Phase 1 — Data foundation & memory

> Everything downstream needs data, so build the tools and the vector store first.

- [x] **MCP client/server vs direct calls for the two data tools.** MCP client/server implementations for Fed, YFin data fetching.
- [x] **`get_macro_indicators`** (`fred_query.py`)
  - [x] Pull a series (e.g. `FEDFUNDS`); return the documented shape incl. the per-series freshness flag
  - [x] Smoke test: `FEDFUNDS`, `CPIAUCSL`, `UNRATE`, `T10Y2Y`
- [x] **`get_sector_performance`** (`yfin_query.py`)
  - [x] Pull the 11 sector ETFs; compute `momentum_6m`/`momentum_12m`, `trailing_pe`
  - [x] Smoke test one ticker, then all 11
- [x] Full MCP client/server
- [ ] **Memory layer (RAG)**
  - [x] Pull ~20y monthly history for the 8 indicators
  - [x] Rules-based regime label per month (early / mid / late / contraction)
  - [x] Normalize → snapshot vectors; store in ChromaDB with metadata (date, regime, subsequent 6-month sector returns)
  - [x] Implement **`find_historical_analogs(snapshot, n, regime_filter)`** returning the analog dict shape `score_branch` expects (`similarity`, `regime`, …)
  - [x] Test: a query returns analogs with similarity scores and the `regime_filter` actually filters

**End of Phase 1:** you can pull live macro + sector data and retrieve historical analogs.

---

## Phase 2 — Reasoning core

- [x] **Implement `generate_hypotheses`** (your stub) — `SYSTEM_PROMPT`, `_build_user_prompt`, `_call_model`, `_parse_hypotheses`
  - [x] Unit-test `_parse_hypotheses` with canned JSON: valid, ```-fenced, hallucinated regime, duplicate, out-of-range prior (no API key needed)
- [x] **Wire `classify_regime_tot` end to end** with the real `generate_hypotheses` + `find_historical_analogs` + `score_branch` (bind `current_momentum` via `functools.partial`)
  - [x] Integration test on a real snapshot
- [x] **`compute_sector_score`** — given the selected regime + analog data + equity momentum, produce a ranked sector list with confidence
  - [x] Test
- [x] **Macro agent**: snapshot → `generate_hypotheses` → `classify_regime_tot` → selected regime
- [x] **Equity agent**: `get_sector_performance` → `current_momentum` dict

**End of Phase 2:** a snapshot produces a classified regime and a ranked sector list.

---

## Phase 3: Loop, safety, output

- [x] **Orchestration loop (coordinator)**: user query → run macro + equity agents → classify regime → score sectors → report. Sequential is fine for "basic"; parallel is stretch.
- [x] **Safety — statistical checker**: implement `check_statistical_anomaly` (z-score + IQR + freshness, pure Python); call it after each data-tool result; append flags to a simple JSON-lines audit log
- [x] **Safety — critic**: implement `run_critic_check` (narrow Anthropic call → supports / weakens / contradicts). Basic version is fine.
- [x] **`generate_report`**: ranked sectors, cited sources, audit flags inline, fixed disclaimer footer
- [x] **End-to-end run** on a real query (e.g. "which sectors to overweight/underweight over the next 6 months")
- [x] `uv run pytest` (all green) + `uv run ruff check` + update README / spec methodology notes

**End of Phase 3: MVP** one query runs end to end and produces a cited, disclaimered report.

---

## Full V1 implementation

- [x] Full revision loop (max 3 cycles) with coordinator re-dispatch on flags
- [x] Audit-log reconciliation check at session end (entries == tool calls)
- [x] PDF report output (markdown is enough for basic)
- [x] Add a RAG over Beige Book, FOMC minutes, Monetary policy narrative
- [x] Add a 'freshness' check for above RAG pdf..
- [x] Get HF token, add to .env, apply it for the vector creation
- [ ] Move the `__main__` self-test blocks into proper `tests/`
- [x] Test with different models and compare result:
  - [x] Local Ollama gemma4:12b
  - [x] Claude claude-sonnet-4-5-20250929

## Definition of "done"

- [x] One real user query runs end to end → cited report with disclaimer
- [x] Each rubric concept demonstrated once: tool calling · ReAct · RAG memory · ToT · ≥1 specialized agent · ≥1 safety guardrail
- [x] Test suite green

---

## Validation

- [x] Backtesting 6mo windows (as-of date); compare weightings against historical market behavior
- [x] Ask non-macro questions to see if hallucination checks work
- [x] Test with OpenRouter cloud & local models
- [x] Test Anthropic cloud model
- [x] Test Ollama local models
- [x] Test HuggingFace cloud model

## Hygiene

- [x] Logging - general cleanup (model_client now logs every call uniformly via the
      template-method base — service/model/tokens/latency, full prompts at DEBUG; the
      per-component log files landed with the TraceLogger. Remaining: audit/other modules)
- [x] TraceLogger for model calls, latency, and token usage tracking — `trace.py`,
      SEPARATE from AuditLog (telemetry vs. provenance). Per-run JSONL
      (`logs/trace-<runid>.jsonl`) with full LLM-call detail, span latencies, and a run
      summary; injected into the model client + coordinator; per-component log files
      (main/coordinator/model_client/trace) via `configure_component_logging`.
- [x] Make the LLM query a Class

---

## Future Ideas

- [ ] Clean up control parameters - currently mixed between config, constants, and .env 
- [ ] Run MCP servers in docker container(s)
- [ ] Explore CrewAI for agent coordination
- [ ] Parallel sub-agent execution (asyncio / `concurrent.futures`) — sequential is acceptable for basic
- [ ] Smarter way to load new pdfs to Fed data corpus (today, it's manipulate the json file)
- [ ] Expand the Fed data corpus by dynamically pulling from Federal Reserve web site to get updated .pdfs