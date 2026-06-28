"""
model_client.py

A class implementation of an LLM caller. Since LLMs are called from several places
(query decomposition, the critic, the executive summary, hypothesis generation),
that is extracted to one provider-agnostic class.

Different models can be used -- local and cloud -- to save cost while testing, to
experiment, and to route hypotheses vs. checks to different models to limit bias.

Observability (the TraceLogger seam)
------------------------------------
complete() is a TEMPLATE METHOD: the base class owns timing, uniform logging, and
tracing, and each provider subclass implements _invoke(), which makes the provider call
and returns a normalized LLMResult (text + token usage + finish reason). So EVERY
provider is observed identically -- one INFO line per call (service / model / tokens /
latency / finish), the full prompts+response at DEBUG, and, when a TraceLogger is
injected, a full structured record of the call. A failure is logged and traced with its
latency before being re-raised, so a call is never silently lost.
"""
from __future__ import annotations
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic
from openai import OpenAI
from ollama import Client
from huggingface_hub import InferenceClient

from sector_rotation_agent.config import settings
import sector_rotation_agent.constants as const

if TYPE_CHECKING:                       # type-only; avoids any import cycle at runtime
    from sector_rotation_agent.trace import TraceLogger


@dataclass
class LLMResult:
    """One provider call's outcome, normalized across providers so the base class can log
    and trace every call identically. `text` is what complete() returns; the rest is
    telemetry extracted best-effort -- any field a provider doesn't expose stays None and
    simply isn't counted."""
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


def _get(obj: Any, *names: str) -> Any:
    """Best-effort read of a field that may be a dict key OR an attribute, returning the
    first present non-None value or None. Lets token-usage extraction work across the
    providers' differing response shapes (dicts, pydantic models, SimpleNamespace) without
    ever raising on a missing field."""
    for n in names:
        try:
            if isinstance(obj, dict):
                if obj.get(n) is not None:
                    return obj[n]
            else:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
        except Exception:
            continue
    return None


# finish_reason values that mean the model did NOT stop on its own -- the output is
# likely truncated or filtered, so the base complete() warns on them. The vocabulary is
# PROVIDER-SPECIFIC, which is why this is a union rather than a single equality check:
#   OpenAI / OpenRouter : clean = "stop";              bad = "length", "content_filter"
#   Anthropic           : clean = "end_turn";          bad = "max_tokens", "refusal"
#   Ollama              : clean = "stop";              bad = "length"
#   HuggingFace / TGI   : clean = "stop" / "eos_token"; bad = "length"
# "tool_use" / "tool_calls" / "stop_sequence" are normal completions, intentionally absent.
_INCOMPLETE_FINISH_REASONS = frozenset({"length", "max_tokens", "content_filter", "refusal"})


class ModelClient[ClientT](ABC):
    """Provider-agnostic LLM caller. One concrete subclass per provider;
        ClientT is that provider's SDK type;
        One instance per (model, temperature, max_tokens) config.

    Inject a TraceLogger (`trace`) to capture every call into the run's structured trace;
    omit it and the client still logs each call (the trace is purely additive)."""

    _SERVICE: str = "unknown"           # provider tag for logs/trace; set per subclass

    def __init__(self, *, model=None, temperature=None, max_tokens=None,
                 num_ctx=None, trace: "TraceLogger | None" = None):
        if model is not None:
            self._model = model  # model explicitly provided
        elif settings.model_location == const.ModelLocations.LOCAL_ONLY:
            self._model = settings.local_model  # some providers support both
        else:
            self._model = settings.cloud_model             # default to cloud model
        self._temperature = temperature if temperature is not None else settings.default_temperature
        self._max_tokens = max_tokens if max_tokens is not None else settings.default_max_tokens
        # Ollama context window (local only); cloud providers manage their own context and ignore it.
        self._num_ctx = num_ctx if num_ctx is not None else getattr(settings, "model_num_ctx", None)
        self._trace = trace
        self._logger = logging.getLogger(__name__)
        self._client: ClientT = self._build_client()                  # Each derived class returns its own client object (SDK)

    @abstractmethod
    def _build_client(self) -> ClientT:
        ...

    @abstractmethod
    def _invoke(self, system: str, user: str) -> LLMResult:
        """Make the provider call and return a normalized LLMResult. Subclasses build the
        request from self._model / self._temperature / self._max_tokens and parse their
        provider's response; timing, logging, tracing and error capture are handled by
        complete() so every provider gets identical observability."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Provider-agnostic entry point. Times the call, records FULL visibility of it
        (model, service, token usage, latency and -- via the injected TraceLogger -- the
        prompts and response), then returns the raw text. A failure is logged and traced
        with its latency before being re-raised, so no call is silently lost."""
        t0 = time.perf_counter()
        try:
            result = self._invoke(system, user)
        except Exception as err:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.exception(
                "LLM call FAILED (%s/%s) after %.0f ms", self._SERVICE, self._model, latency_ms
            )
            if self._trace is not None:
                self._trace.llm_call(
                    model=self._model, service=self._SERVICE, system=system, user=user,
                    response="", latency_ms=latency_ms, error=repr(err),
                )
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._logger.info(
            "LLM call %s/%s: %s prompt + %s completion = %s tokens, finish=%s, %.0f ms",
            self._SERVICE, self._model, result.prompt_tokens, result.completion_tokens,
            result.total_tokens, result.finish_reason, latency_ms,
        )
        self._logger.debug(
            "LLM call %s/%s prompts+response:\n--- system ---\n%s\n--- user ---\n%s\n--- response ---\n%s",
            self._SERVICE, self._model, system, user, result.text,
        )
        # Surface a degraded response so a weak/truncated model is LOUD, not silent (the
        # executive summary is the first section to vanish on a marginal local model). The
        # text and finish_reason are already normalized by _invoke; only the SET of "bad"
        # finish reasons is provider-specific (see _INCOMPLETE_FINISH_REASONS).
        if not result.text.strip():
            self._logger.warning(
                "LLM call %s/%s returned an EMPTY response (finish=%s) -- downstream "
                "sections (e.g. the executive summary) may be dropped.",
                self._SERVICE, self._model, result.finish_reason,
            )
        elif result.finish_reason and str(result.finish_reason).lower() in _INCOMPLETE_FINISH_REASONS:
            self._logger.warning(
                "LLM call %s/%s did not finish cleanly (finish=%s) -- the response may be "
                "truncated or filtered.",
                self._SERVICE, self._model, result.finish_reason,
            )
        if self._trace is not None:
            self._trace.llm_call(
                model=self._model, service=self._SERVICE, system=system, user=user,
                response=result.text, latency_ms=latency_ms,
                prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens, finish_reason=result.finish_reason,
            )
        return result.text


class AnthropicClient(ModelClient[Anthropic]):
    _SERVICE = "anthropic"

    def _build_client(self) -> Anthropic:

        self._logger.info(f"Initializing Anthropic Client, {self._model} with temp: {self._temperature} and max tokens: {self._max_tokens}")

        key = os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set"
                        " — required to call the Anthropic API.")
        else:
            return Anthropic(api_key=key)

    def _invoke(self, system: str, user: str) -> LLMResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user
                }
            ],
        )
        # Anthropic Messages API can return content in multiple blocks, so we concatenate them
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        # Token usage lives on response.usage (input_tokens / output_tokens); absent on
        # some stubs/responses, so read defensively.
        usage = getattr(response, "usage", None)
        prompt = _get(usage, "input_tokens") if usage is not None else None
        completion = _get(usage, "output_tokens") if usage is not None else None
        total = (prompt + completion) if (prompt is not None and completion is not None) else None
        return LLMResult(
            text="".join(parts),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            finish_reason=getattr(response, "stop_reason", None),
        )


class OpenRouterClient(ModelClient[OpenAI]):
    _SERVICE = "open_router"

    def _build_client(self) -> OpenAI:
        if settings.model_location == const.ModelLocations.LOCAL_ONLY or settings.model_location == const.ModelLocations.MIXED:
            url = os.getenv("LOCAL_OLLAMA_URL")
            if url is None:
                raise ValueError("LOCAL_OLLAMA_URL environment variable not set"
                                 " - required to call local OpenRouter host.")
            api_key = "not-needed"
        else:
            url = settings.openrouter_url
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY environment variable not set"
                        " — required to call the OpenRouter API.")
        return OpenAI(
             base_url=url,
             api_key=api_key
        )

    def _invoke(self, system: str, user: str) -> LLMResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,    # bound completion on EVERY path; for the local
                                               # Ollama OpenAI-compat endpoint this maps to
                                               # num_predict. Omitting it let local runs generate
                                               # to the context wall and return empty.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False
        }
        # Local OpenRouter is going to call Ollama: set the context window (num_ctx). It is not a standard OpenAI
        # field, so it rides along in extra_body (Ollama's OpenAI-compat endpoint reads
        # options from there); cloud OpenRouter neither needs nor receives it.
        if (settings.model_location == const.ModelLocations.LOCAL_ONLY or settings.model_location == const.ModelLocations.MIXED) and self._num_ctx:
            payload["extra_body"] = {"options": {"num_ctx": self._num_ctx}}

        response = self._client.chat.completions.create(**payload)

        # OpenAI's response is a single content block; usage carries the token counts.
        choice = response.choices[0]
        
        usage = getattr(response, "usage", None)
        # OpenAI/OpenRouter return content=None (not "") when there's nothing to say; coerce
        # to "" so the base complete() empty-response check can see it (str(None) would be
        # the literal "None"). NOTE: finish_reason lives on the CHOICE, not the response.
        content = choice.message.content
        return LLMResult(
            text="" if content is None else str(content),
            prompt_tokens=_get(usage, "prompt_tokens") if usage is not None else None,
            completion_tokens=_get(usage, "completion_tokens") if usage is not None else None,
            total_tokens=_get(usage, "total_tokens") if usage is not None else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )


class OllamaClient(ModelClient[Client]):
    # TODO: make this both local and Ollama cloud
    _SERVICE = "ollama-local"

    def _build_client(self) -> Client:
        url = os.getenv("LOCAL_OLLAMA_URL")
        if url is None:
            raise ValueError("LOCAL_OLLAMA_URL environment variable not set"
                             " - required to call the local Ollama host.")
        # LOCAL_OLLAMA_URL is shared with the OpenAI-compat path (OpenRouterClient), which
        # needs the OpenAI '/v1' base; the NATIVE ollama client wants the bare host and
        # appends its own '/api/...' routes, so a '/v1' suffix would produce '/v1/api/chat'
        # -> '404 page not found'. Strip a trailing '/v1' (and slashes) so one env var
        # serves both clients.
        host = url
        if host.endswith("/"):
            host = host.rstrip("/")

        if host.endswith("/v1"):
            host = host[: -len("/v1")]
        return Client(host=host)

    def _invoke(self, system: str, user: str) -> LLMResult:
        # Native Ollama: generation knobs go in `options`. num_predict caps the completion
        # length; num_ctx sizes the context window (its small default truncates long
        # generations like the executive summary). num_ctx is sent only when configured.
        options: dict[str, float] = {}
        options["num_predict"] = self._max_tokens
        options["temperature"] = self._temperature
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx
        response = self._client.chat(
            model = self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
                ],
            options=options,
        )
        # ollama returns a ChatResponse (mapping- AND attribute-accessible); read both ways.
        message = _get(response, "message") or {}
        text = _get(message, "content") or ""
        return LLMResult(
            text=str(text),
            prompt_tokens=_get(response, "prompt_eval_count"),
            completion_tokens=_get(response, "eval_count"),
            finish_reason=_get(response, "done_reason"),
        )


class HuggingFaceClient(ModelClient[InferenceClient]):
    _SERVICE = "huggingface"

    def _build_client(self) -> InferenceClient:
        key = os.getenv("HUGGINGFACE_HUB_KEY")
        if key is None:
            raise ValueError("HUGGINGFACE_HUB_KEY environment variable not set"
                        " — required to call the HuggingFace API.")
        return InferenceClient(token=key)

    def _invoke(self, system: str, user: str) -> LLMResult:
        response = self._client.chat_completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self._max_tokens,
            )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        # HuggingFace/TGI is OpenAI-shaped; coerce a None content to "" (see OpenRouter).
        content = choice.message.content
        return LLMResult(
            text="" if content is None else str(content),
            prompt_tokens=_get(usage, "prompt_tokens") if usage is not None else None,
            completion_tokens=_get(usage, "completion_tokens") if usage is not None else None,
            total_tokens=_get(usage, "total_tokens") if usage is not None else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )


# -------------------  Class Factory  -------------------------------------
_PROVIDERS: dict[str, type[ModelClient]] = {
    "anthropic":    AnthropicClient,
    "open_router":  OpenRouterClient,
    "ollama-local": OllamaClient,
    "huggingface":  HuggingFaceClient,
}

def make_model_client(service: str | None = None, **kwargs) -> ModelClient:
    """Build the configured provider client. Extra kwargs (model / temperature /
    max_tokens / trace) flow straight through to the subclass constructor -- e.g.
    make_model_client(trace=run_trace) to capture its calls, or
    make_model_client("anthropic", model="claude-...") for the hybrid split."""
    service = service or settings.cloud_model_service   # default to cloud model
    try:
        return _PROVIDERS[service](**kwargs)
    except KeyError:
        raise ValueError(f"Unsupported model service: {service!r}")
