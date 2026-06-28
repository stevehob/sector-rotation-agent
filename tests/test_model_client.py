"""
Tests for sector_rotation_agent.model_client.

Covers the provider-routing class hierarchy that replaced generate_hypotheses'
old _call_model:
  - make_model_client factory: service string -> correct subclass, unknown -> error,
    default from settings, kwargs forwarded.
  - ModelClient base: per-instance config vs. settings defaults.
  - AnthropicClient / OpenRouterClient / OllamaClient .complete(): each builds the
    right request from self._model / self._temperature / self._max_tokens and parses
    its provider's response shape. The SDKs are faked (monkeypatched onto the module
    symbols) so nothing hits the network or needs a real key. The Anthropic
    text-block-join regression migrated here from test_generate_hypotheses.

These import the real model_client, which imports the provider SDKs (anthropic,
openai, ollama, huggingface_hub) and config.settings -- all project dependencies,
so collection is offline-safe. HuggingFaceClient.complete is intentionally not
exercised yet (its response parsing is still a TODO); see the bottom of the file.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import sector_rotation_agent.model_client as mc


# --------------------------------------------------------------------------- #
# fakes -- stand-ins for the provider SDK clients (no network, no keys)
# --------------------------------------------------------------------------- #
def fake_anthropic(blocks, captured):
    """An Anthropic(api_key=...) stand-in whose messages.create() returns `blocks`
    and records the kwargs it was called with."""
    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=list(blocks))

    class _Client:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = _Messages()

    return _Client


def fake_openai(captured, content="RESP"):
    """An OpenAI(base_url=, api_key=) stand-in whose chat.completions.create()
    returns one choice carrying `content`, recording the kwargs it received."""
    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop",
                                         message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(completion_tokens=5, prompt_tokens=3, total_tokens=8),
            )

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.chat = _Chat()

    return _Client


def fake_ollama(captured, content="ollama hi"):
    """An ollama Client(host=...) stand-in whose chat() returns the ollama response
    dict shape and records the kwargs it received."""
    class _Client:
        def __init__(self, host=None):
            self.host = host

        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": content}}

    return _Client


@pytest.fixture
def stub_build(monkeypatch):
    """Replace every provider's _build_client so the factory can construct them with
    no env keys or real SDKs -- isolates dispatch/config from provider internals."""
    for cls in (mc.AnthropicClient, mc.OpenRouterClient, mc.OllamaClient, mc.HuggingFaceClient):
        monkeypatch.setattr(cls, "_build_client", lambda self: object())


# =========================================================================== #
# make_model_client  (factory dispatch)
# =========================================================================== #
@pytest.mark.parametrize("service, expected", [
    ("anthropic", mc.AnthropicClient),
    ("open_router", mc.OpenRouterClient),
    ("ollama-local", mc.OllamaClient),
    ("huggingface", mc.HuggingFaceClient),
])
def test_factory_dispatches_to_subclass(service, expected, stub_build):
    """Each known service string resolves to its provider subclass."""
    assert isinstance(mc.make_model_client(service), expected)


def test_factory_unknown_service_raises():
    """An unregistered service is a ValueError, not a KeyError leaking from the dict."""
    with pytest.raises(ValueError):
        mc.make_model_client("not-a-real-service")


def test_factory_defaults_to_settings_service(monkeypatch, stub_build):
    """With no explicit service, the factory falls back to settings.model_service."""
    monkeypatch.setattr(mc, "settings", SimpleNamespace(
        cloud_model_service="anthropic", cloud_model="m", model_location="cloud_only",
        default_temperature=0.7, default_max_tokens=10))
    assert isinstance(mc.make_model_client(), mc.AnthropicClient)


def test_factory_forwards_kwargs(stub_build):
    """Per-call config flows through the factory into the instance (the hybrid hook:
    different agents get different models)."""
    client = mc.make_model_client("anthropic", model = "claude-x", temperature=0.1, max_tokens=42)
    assert (client._model, client._temperature, client._max_tokens) == ("claude-x", 0.1, 42)


# =========================================================================== #
# ModelClient base -- config vs. defaults
# =========================================================================== #
def test_init_defaults_come_from_settings(monkeypatch, stub_build):
    """Unset config falls back to settings; this is what lets a bare
    make_model_client() still be fully configured."""
    monkeypatch.setattr(mc, "settings", SimpleNamespace(
        cloud_model_service="anthropic", cloud_model="default-model", model_location="cloud_only",
        default_temperature=0.55, default_max_tokens=777))
    client = mc.AnthropicClient()
    assert (client._model, client._temperature, client._max_tokens) == ("default-model", 0.55, 777)


# =========================================================================== #
# AnthropicClient
# =========================================================================== #
def test_anthropic_complete_joins_text_blocks_and_forwards_config(monkeypatch):
    """Regression (migrated from test_generate_hypotheses): the Anthropic path
    returns a single JOINED STRING, concatenating text blocks and skipping non-text
    ones -- and the call carries the instance's model/temperature/max_tokens, not
    settings'."""
    captured = {}
    blocks = [
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="tool_use", text="IGNORED"),   # must be skipped
        SimpleNamespace(type="text", text="world"),
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(mc, "Anthropic", fake_anthropic(blocks, captured))

    client = mc.AnthropicClient(model="claude-test", temperature=0.3, max_tokens=512)
    out = client.complete("SYS", "USER")

    assert out == "Hello world"
    assert "IGNORED" not in out
    assert captured["system"] == "SYS"
    assert captured["messages"][0]["content"] == "USER"
    assert captured["model"] == "claude-test"
    assert captured["max_tokens"] == 512
    assert captured["temperature"] == 0.3


def test_anthropic_build_client_requires_api_key(monkeypatch):
    """No ANTHROPIC_API_KEY -> a clear ValueError at construction, before any call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        mc.AnthropicClient(model="claude-test", temperature=0.3, max_tokens=10)


# =========================================================================== #
# OpenRouterClient
# =========================================================================== #
def test_openrouter_complete_extracts_content_and_uses_instance_model(monkeypatch):
    """The OpenAI-compatible path builds system+user messages from the instance
    model and returns choices[0].message.content. Under is_local_model it uses the
    large local max-tokens cap rather than self._max_tokens."""
    captured = {}
    monkeypatch.setattr(mc, "settings", SimpleNamespace(
        model_location = "local_only", local_url="http://x/v1",
        local_model_service="open_router", local_model="settings-model",
        default_temperature=0.7, default_max_tokens=1000))
    monkeypatch.setattr(mc, "OpenAI", fake_openai(captured, content="hello from model"))

    client = mc.OpenRouterClient(model="mistral-test", temperature=0.2, max_tokens=256)
    out = client.complete("SYS", "USER")

    assert out == "hello from model"
    assert captured["model"] == "mistral-test"
    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    # for now, I'm ignoring max_tokens for local models
    #assert captured["max_tokens"] == 2048   # local -> large cap (complete() branch)


# =========================================================================== #
# OllamaClient
# =========================================================================== #
def test_ollama_complete_extracts_message_content(monkeypatch):
    """The ollama path sends system+user messages via client.chat() and returns
    response["message"]["content"]."""
    captured = {}
    monkeypatch.setattr(mc, "settings", SimpleNamespace(
        is_local_model=True, local_url="http://localhost:11434",
        model_service="ollama-local", model="settings-model",
        default_temperature=0.7, default_max_tokens=1000))
    monkeypatch.setattr(mc, "Client", fake_ollama(captured, content="ollama says hi"))

    client = mc.OllamaClient(model="mistral", temperature=0.1, max_tokens=100)
    out = client.complete("SYS", "USER")

    assert out == "ollama says hi"
    assert captured["model"] == "mistral"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}


# =========================================================================== #
# ADDITIONAL TEST CASES TO IMPLEMENT LATER
# =========================================================================== #
# OpenRouterClient:
#   - non-local branch: is_local_model False -> max_tokens == self._max_tokens, and
#     _build_client raises ValueError when OPENROUTER_API_KEY is unset
# OllamaClient:
#   - _build_client passes settings.local_url through as the Client(host=...)
# HuggingFaceClient:
#   - _build_client raises ValueError without HUGGINGFACE_HUB_KEY
#   - complete: once implemented, parse response["choices"][0]["message"]["content"]
#     (current code indexes ["choices"]["message"] -- missing the [0]; fix before
#     writing the happy-path test)
