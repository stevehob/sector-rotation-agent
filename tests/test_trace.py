"""
Tests for sector_rotation_agent.trace and the model_client tracing seam.

Covers:
  - TraceLogger.event / span / llm_call / summary: in-memory timeline, the per-run JSONL
    file, span latency capture (including the error path), and the aggregate summary.
  - NullTrace: the no-op null object the coordinator/model clients default to.
  - configure_component_logging: the per-component log files mapping, and idempotency.
  - ModelClient.complete (the template method): every provider call is timed, returns the
    LLMResult text, and -- when a trace is injected -- records a full llm_call (prompts,
    response, token usage); a provider error is traced AND re-raised.

A tiny _FakeClient subclass exercises the base complete() without any SDK or network: it
returns a canned LLMResult (or raises), so this stays offline-safe like test_model_client.
"""
from __future__ import annotations

import json
import logging

import pytest

#import sector_rotation_agent.trace as trace_mod
import sector_rotation_agent.model_client as mc
from sector_rotation_agent.trace import TraceLogger, NullTrace, configure_component_logging


# --------------------------------------------------------------------------- #
# TraceLogger.event
# --------------------------------------------------------------------------- #
def test_event_records_in_memory_and_writes_jsonl(tmp_path):
    """event() appends a component-tagged record to the timeline AND writes one JSON line
    to the per-run trace file, with the fields preserved."""
    tr = TraceLogger(run_id="t1", log_dir=tmp_path)
    tr.event("coordinator", "query_plan", horizon="6 months", focus=None)

    assert len(tr.events) == 1
    rec = tr.events[0]
    assert rec["component"] == "coordinator"
    assert rec["event"] == "query_plan"
    assert rec["horizon"] == "6 months"
    assert "ts" in rec and "t_ms" in rec and rec["run_id"] == "t1"

    path = tmp_path / "trace-t1.jsonl"
    assert path.is_file()
    line = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["event"] == "query_plan"
    assert parsed["horizon"] == "6 months"


def test_to_file_false_keeps_everything_in_memory_only(tmp_path):
    """to_file=False records events in memory but writes no file -- handy for tests and
    for callers that only want the in-process timeline."""
    tr = TraceLogger(run_id="mem", log_dir=tmp_path, to_file=False)
    tr.event("main", "run_start", query="q")
    assert len(tr.events) == 1
    assert not (tmp_path / "trace-mem.jsonl").exists()


# --------------------------------------------------------------------------- #
# TraceLogger.span
# --------------------------------------------------------------------------- #
def test_span_emits_single_event_with_latency(tmp_path):
    """A span emits exactly one event on exit, carrying a numeric latency_ms and the
    fields passed in."""
    tr = TraceLogger(run_id="s1", log_dir=tmp_path)
    with tr.span("coordinator", "equity_agent.run", cycle=0):
        pass
    assert len(tr.events) == 1
    rec = tr.events[0]
    assert rec["event"] == "equity_agent.run"
    assert rec["cycle"] == 0
    assert isinstance(rec["latency_ms"], (int, float)) and rec["latency_ms"] >= 0


def test_span_records_error_and_reraises(tmp_path):
    """If the wrapped block raises, the span records the error (with latency) and lets the
    exception propagate -- a failed phase is visible in the trace, not swallowed."""
    tr = TraceLogger(run_id="s2", log_dir=tmp_path)
    with pytest.raises(ValueError):
        with tr.span("coordinator", "score_sectors"):
            raise ValueError("boom")
    assert len(tr.events) == 1
    rec = tr.events[0]
    assert rec["event"] == "score_sectors"
    assert "boom" in rec["error"]
    assert "latency_ms" in rec


# --------------------------------------------------------------------------- #
# TraceLogger.llm_call + summary
# --------------------------------------------------------------------------- #
def test_llm_call_records_full_prompts_and_usage(tmp_path):
    """llm_call keeps the FULL prompts and response (the trace artifact) plus the token
    usage and latency."""
    tr = TraceLogger(run_id="l1", log_dir=tmp_path)
    tr.llm_call(
        model="gemma", service="open_router", system="SYS", user="USER",
        response="ANSWER", latency_ms=12.3,
        prompt_tokens=10, completion_tokens=4, total_tokens=14, finish_reason="stop",
    )
    rec = tr.events[0]
    assert rec["event"] == "llm_call"
    assert rec["system"] == "SYS" and rec["user"] == "USER" and rec["response"] == "ANSWER"
    assert rec["prompt_tokens"] == 10 and rec["completion_tokens"] == 4 and rec["total_tokens"] == 14
    assert rec["finish_reason"] == "stop"


def test_llm_call_infers_total_when_missing(tmp_path):
    """When total_tokens isn't supplied but the parts are, it's inferred (prompt +
    completion) so the summary still totals correctly."""
    tr = TraceLogger(run_id="l2", log_dir=tmp_path)
    tr.llm_call(model="m", service="s", system="", user="", response="", latency_ms=1.0,
                prompt_tokens=7, completion_tokens=3)
    assert tr.events[0]["total_tokens"] == 10


def test_summary_aggregates_calls_tokens_and_latency(tmp_path):
    """summary() totals LLM calls, errors, tokens, and LLM latency across the run, and
    counts events per component."""
    tr = TraceLogger(run_id="sum", log_dir=tmp_path)
    tr.event("coordinator", "run_start")
    tr.llm_call(model="m", service="s", system="", user="", response="", latency_ms=10.0,
                prompt_tokens=5, completion_tokens=5, total_tokens=10, finish_reason="stop")
    tr.llm_call(model="m", service="s", system="", user="", response="", latency_ms=20.0,
                prompt_tokens=1, completion_tokens=2, total_tokens=3, error="kaboom")

    data = tr.summary()
    assert data["llm_calls"] == 2
    assert data["llm_errors"] == 1
    assert data["total_tokens"] == 13
    assert data["prompt_tokens"] == 6 and data["completion_tokens"] == 7
    assert data["llm_latency_ms"] == pytest.approx(30.0)
    # run_start + 2 llm_calls + the summary event itself
    assert data["events_by_component"]["model_client"] == 2
    assert data["events_by_component"]["coordinator"] == 1


# --------------------------------------------------------------------------- #
# NullTrace
# --------------------------------------------------------------------------- #
def test_nulltrace_is_a_silent_noop():
    """NullTrace satisfies the same surface (event/span/llm_call/summary) and does
    nothing -- the default a run with no trace injected pays for."""
    nt = NullTrace()
    assert nt.event("c", "e", x=1) == {}
    with nt.span("c", "e"):
        pass
    assert nt.llm_call(model="m", service="s", system="", user="", response="", latency_ms=0.0) == {}
    assert nt.summary() == {}
    assert nt.events == []


# --------------------------------------------------------------------------- #
# configure_component_logging
# --------------------------------------------------------------------------- #
def test_configure_component_logging_maps_and_is_idempotent(tmp_path):
    """Wires one file per component logger, returns the {component: path} mapping, and a
    second call doesn't double up handlers (tagged + skipped)."""
    component_loggers = [
        "sector_rotation_agent.main",
        "sector_rotation_agent.coordinator",
        "sector_rotation_agent.model_client",
        "sector_rotation_agent.trace",
    ]
    # snapshot existing handlers so we only clean up what THIS test added
    before = {name: list(logging.getLogger(name).handlers) for name in component_loggers}
    try:
        paths = configure_component_logging(tmp_path)
        assert set(paths) == {"main", "coordinator", "model_client", "trace"}

        def added(name):
            return [h for h in logging.getLogger(name).handlers
                    if getattr(h, "_sra_tag", None) and h not in before[name]]

        first = {name: added(name) for name in component_loggers}
        assert all(len(v) == 1 for v in first.values())

        configure_component_logging(tmp_path)  # idempotent
        second = {name: added(name) for name in component_loggers}
        assert all(len(v) == 1 for v in second.values())
    finally:
        # remove + close only the handlers this test added, so the suite's logging
        # state is left as we found it
        for name in component_loggers:
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                if getattr(h, "_sra_tag", None) and h not in before[name]:
                    lg.removeHandler(h)
                    h.close()


# --------------------------------------------------------------------------- #
# ModelClient.complete -- the tracing template method (offline via a fake subclass)
# --------------------------------------------------------------------------- #
class _FakeClient(mc.ModelClient):
    """Minimal concrete client: no SDK, no network. _build_client returns a dummy and
    _invoke returns a canned LLMResult so the base complete() (timing/logging/tracing)
    can be exercised in isolation."""
    _SERVICE = "fake"

    def _build_client(self):
        return object()

    def _invoke(self, system: str, user: str) -> mc.LLMResult:
        return mc.LLMResult(text="hi", prompt_tokens=3, completion_tokens=2,
                            total_tokens=5, finish_reason="stop")


class _BoomClient(_FakeClient):
    """A client whose provider call fails, to exercise the error path."""
    def _invoke(self, system: str, user: str) -> mc.LLMResult:
        raise RuntimeError("provider exploded")


def test_complete_returns_text_and_traces_the_call():
    """complete() returns the LLMResult text and, with a trace injected, records ONE
    llm_call carrying the prompts, response, and token usage."""
    tr = TraceLogger(run_id="mc1", to_file=False)
    client = _FakeClient(model="fake-model", trace=tr)

    out = client.complete("SYS", "USER")

    assert out == "hi"
    calls = [e for e in tr.events if e["event"] == "llm_call"]
    assert len(calls) == 1
    rec = calls[0]
    assert rec["service"] == "fake" and rec["model"] == "fake-model"
    assert rec["system"] == "SYS" and rec["user"] == "USER" and rec["response"] == "hi"
    assert rec["prompt_tokens"] == 3 and rec["completion_tokens"] == 2 and rec["total_tokens"] == 5
    assert rec["error"] is None


def test_complete_without_trace_still_returns_text():
    """No trace injected -> complete() still works (the trace is purely additive)."""
    client = _FakeClient(model="fake-model")
    assert client.complete("SYS", "USER") == "hi"


def test_complete_traces_error_and_reraises():
    """A provider failure is recorded as an llm_call with an error (and empty response)
    and then re-raised, so a failed call is never silently lost."""
    tr = TraceLogger(run_id="mc2", to_file=False)
    client = _BoomClient(model="fake-model", trace=tr)

    with pytest.raises(RuntimeError):
        client.complete("SYS", "USER")

    calls = [e for e in tr.events if e["event"] == "llm_call"]
    assert len(calls) == 1
    assert "provider exploded" in calls[0]["error"]
    assert calls[0]["response"] == ""
