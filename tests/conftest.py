"""
conftest.py — shared pytest setup for the whole test suite.

Two import-time hazards this resolves once, centrally:

  * The MCP server modules read ``LOGGING_LEVEL`` at *import* time and raise a
    ``RuntimeError`` if it is unset. The offline unit tests import those modules
    directly, so the variable has to exist before collection.
  * The integration tests need the API keys (and ``TEST_MODE``) that live in
    ``.env``.

pytest imports ``conftest.py`` before it collects any test module, so loading
the environment here means a plain ``uv run pytest`` behaves the same whether or
not the shell already exported these — and the offline unit tests still run even
with no ``.env`` present (e.g. in CI), because of the ``setdefault`` fallback.
"""
import os

import dotenv

# Pull in .env (FRED_API_KEY, ANTHROPIC_API_KEY, TEST_MODE, LOGGING_LEVEL) if present.
dotenv.load_dotenv()

# Guarantee the server modules can import even when no .env is available, so the
# offline unit tests are runnable in a bare checkout. A real .env value (loaded
# above) takes precedence over this fallback.
os.environ.setdefault("LOGGING_LEVEL", "WARNING")
