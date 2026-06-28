"""
trace.py

Run-level observability for the agent -- the TraceLogger the project TODO calls for --
kept deliberately SEPARATE from coordinator.AuditLog. They answer different questions:

  - AuditLog (coordinator.py)  -- PROVENANCE / guardrails: which tool calls ran, which
    guardrail flags were raised, and the session-end reconciliation (#7). It is part of
    the audited reasoning record and is surfaced in the analyst's brief.
  - TraceLogger (here)         -- TELEMETRY / debugging: a timestamped timeline of one
    run, tagged by component (main / coordinator / model_client), the latency of each
    phase, and FULL detail of every LLM call (model, service, prompts, response, token
    usage, latency, errors). It is for the developer and never shown to the analyst.

Two outputs:
  1. A structured, per-run JSONL file (one event per line) under the log directory --
     the full-fidelity record, with full prompts and responses, easy to grep / jq.
  2. Readable lines via the `sector_rotation_agent.trace` logger, which
     configure_component_logging() routes to its own trace.log (and the combined log).

Everything takes the logger by injection (main builds it; the coordinator and the model
clients accept it). A NullTrace no-op stands in when none is supplied -- the same
null-object pattern the coordinator uses for its other optional seams -- so the trace is
genuinely optional and a run that doesn't want it pays nothing and writes nothing.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sector_rotation_agent.config import settings

_TRACE_LOGGER_NAME = "sector_rotation_agent.trace"
# How much prompt/response text the READABLE log line shows; the JSONL keeps everything.
_PREVIEW_CHARS = 600


def _utc_iso() -> str:
    """Wall-clock timestamp (UTC, millisecond precision) for ordering events across runs."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _short(value: Any, n: int) -> str:
    """Collapse whitespace and truncate for a tidy readable log line (full text lives in
    the JSONL)."""
    s = " ".join(str(value).split())
    return s if len(s) <= n else s[:n] + "\u2026"


class TraceLogger:
    """A per-run telemetry recorder. Construct one at the top of a run (main does), inject
    it into the model client(s) and the coordinator, then call summary() at the end.

    Events carry a wall-clock timestamp, a monotonic elapsed-ms offset from run start, the
    run id, a `component` tag, an `event` name, and arbitrary fields. They are appended to
    `self.events` (programmatic access), written to a per-run JSONL file (full fidelity),
    and echoed as a readable line through the trace logger.
    """

    def __init__(
        self,
        run_id: str | None = None,
        *,
        log_dir: str | Path | None = None,
        to_file: bool = True,
        preview_chars: int = _PREVIEW_CHARS,
    ) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
        self._t0 = time.perf_counter()
        self.events: list[dict] = []
        self._logger = logging.getLogger(_TRACE_LOGGER_NAME)
        self._preview = preview_chars
        # Aggregates for summary().
        self._llm_calls = 0
        self._llm_errors = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._llm_latency_ms = 0.0
        # Per-run JSONL file (best-effort: a logging failure must never take down a run).
        self._path: Path | None = None
        if to_file:
            try:
                base = Path(log_dir) if log_dir is not None else Path(settings.log_file).parent
                base.mkdir(parents=True, exist_ok=True)
                self._path = base / f"trace-{self.run_id}.jsonl"
            except Exception:
                self._logger.warning(
                    "TraceLogger could not open a trace file; using the trace logger only",
                    exc_info=True,
                )
                self._path = None

    # ------------------------------------------------------------------ core
    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _emit(self, record: dict) -> None:
        """Append to the in-memory timeline and write one JSONL line (best-effort)."""
        self.events.append(record)
        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
            except Exception:
                self._logger.warning("TraceLogger failed to write a trace event", exc_info=True)

    def event(self, component: str, event: str, **fields: Any) -> dict:
        """Record one timeline event (component-tagged) and echo a readable line."""
        record = {
            "ts": _utc_iso(),
            "t_ms": round(self._elapsed_ms(), 1),
            "run_id": self.run_id,
            "component": component,
            "event": event,
        }
        record.update(fields)
        self._emit(record)
        extra = " ".join(f"{k}={_short(v, self._preview)}" for k, v in fields.items())
        self._logger.info("[%s] %s %s", component, event, extra)
        return record

    @contextmanager
    def span(self, component: str, event: str, **fields: Any) -> Iterator[None]:
        """Time a block and emit ONE event on exit carrying its `latency_ms` (and `error`
        if it raised, which is re-raised). Wrapping an `await` is fine -- the latency is
        wall time including the await."""
        start = time.perf_counter()
        try:
            yield
        except Exception as err:
            self.event(
                component, event,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 1),
                error=repr(err), **fields,
            )
            raise
        else:
            self.event(
                component, event,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 1),
                **fields,
            )

    def llm_call(
        self,
        *,
        model: str,
        service: str,
        system: str,
        user: str,
        response: str,
        latency_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        finish_reason: str | None = None,
        error: str | None = None,
    ) -> dict:
        """Record one LLM call with FULL visibility -- model, service, latency, token
        usage, finish reason, and the complete prompts + response (in the JSONL; the
        readable log shows a one-line summary). Called from ModelClient.complete for
        every provider, so coverage is uniform. Token fields a provider doesn't expose
        are None and simply aren't aggregated."""
        self._llm_calls += 1
        if error is not None:
            self._llm_errors += 1
        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        self._prompt_tokens += prompt_tokens or 0
        self._completion_tokens += completion_tokens or 0
        self._total_tokens += total_tokens or 0
        self._llm_latency_ms += latency_ms or 0.0

        record = {
            "ts": _utc_iso(),
            "t_ms": round(self._elapsed_ms(), 1),
            "run_id": self.run_id,
            "component": "model_client",
            "event": "llm_call",
            "model": model,
            "service": service,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "error": error,
            "system": system,        # full text, by design -- this is the trace artifact
            "user": user,
            "response": response,
        }
        self._emit(record)
        if error is not None:
            self._logger.warning(
                "[model_client] llm_call ERROR %s/%s after %.0f ms: %s",
                service, model, latency_ms, error,
            )
        else:
            self._logger.info(
                "[model_client] llm_call %s/%s %s+%s=%s tok finish=%s %.0f ms",
                service, model, prompt_tokens, completion_tokens, total_tokens,
                finish_reason, latency_ms,
            )
        return record

    def summary(self) -> dict:
        """Emit and return a one-shot run summary: event counts (overall and per
        component), LLM call/error counts, token totals, time spent in LLM calls, and
        wall time. Logged as a single readable line and written as a final JSONL event."""
        by_component: dict[str, int] = {}
        for e in self.events:
            by_component[e["component"]] = by_component.get(e["component"], 0) + 1
        data = {
            "ts": _utc_iso(),
            "t_ms": round(self._elapsed_ms(), 1),
            "run_id": self.run_id,
            "component": "trace",
            "event": "summary",
            "events": len(self.events),
            "events_by_component": by_component,
            "llm_calls": self._llm_calls,
            "llm_errors": self._llm_errors,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "llm_latency_ms": round(self._llm_latency_ms, 1),
            "wall_ms": round(self._elapsed_ms(), 1),
            "trace_file": str(self._path) if self._path else None,
        }
        self._emit(data)
        self._logger.info(
            "[trace] run %s: %d events, %d LLM call(s) (%d error), %s tokens "
            "(%s prompt + %s completion), %.0f ms in LLM / %.0f ms wall%s",
            self.run_id, data["events"], self._llm_calls, self._llm_errors,
            self._total_tokens, self._prompt_tokens, self._completion_tokens,
            self._llm_latency_ms, data["wall_ms"],
            f", trace -> {self._path}" if self._path else "",
        )
        return data


class NullTrace:
    """No-op TraceLogger (null-object pattern). Lets the coordinator and the model clients
    hold a trace unconditionally; a run with no trace injected pays nothing and writes
    nothing. Mirrors coordinator._NullAudit / _null_report."""

    run_id = "null"
    events: list[dict] = []   # always empty; never written

    def event(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    @contextmanager
    def span(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        yield

    def llm_call(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    def summary(self) -> dict:
        return {}


def configure_component_logging(
    log_dir: str | Path | None = None, *, level: int | None = None
) -> dict[str, Path]:
    """Give the agent-run components their own log files -- the 'separate logging of the
    agent runs' ask -- on top of the combined app log config.basicConfig already set up.

    Attaches one FileHandler per component logger (main, coordinator, model_client, and
    the trace logger), writing logs/<component>.log. Propagation stays ON, so each line
    ALSO lands in the combined sector_rotation_agent.log -- you get both the per-component
    view and the unified stream. Idempotent: handlers are tagged, so re-calling won't
    double them up. Returns the {component: path} mapping it wired.

    Set LOGGING_LEVEL=DEBUG (env) to capture the full LLM prompts + response in
    model_client.log; at INFO the per-call summary line (tokens / latency / finish) is
    logged and the full text lives in the per-run trace JSONL.
    """
    base = Path(log_dir) if log_dir is not None else Path(settings.log_file).parent
    base.mkdir(parents=True, exist_ok=True)
    lvl = level if level is not None else getattr(logging, settings.logging_level, logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    components = {
        "main": "sector_rotation_agent.main",
        "coordinator": "sector_rotation_agent.coordinator",
        "model_client": "sector_rotation_agent.model_client",
        "trace": _TRACE_LOGGER_NAME,
    }
    paths: dict[str, Path] = {}
    for short_name, logger_name in components.items():
        path = base / f"{short_name}.log"
        paths[short_name] = path
        lg = logging.getLogger(logger_name)
        lg.setLevel(lvl)
        tag = f"component-file:{short_name}"
        if any(getattr(h, "_sra_tag", None) == tag for h in lg.handlers):
            continue  # already wired (idempotent)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(fmt)
        handler.setLevel(lvl)
        handler._sra_tag = tag  # type: ignore[attr-defined]
        lg.addHandler(handler)
    return paths
