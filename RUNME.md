
## API keys
Keys are read with os.getenv() calls, so add them to .env, local environment,
 or pass as runtime $env variables

An example of how to check local .env keys
```
uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print({k: bool(os.getenv(k)) for k in ('HF_TOKEN','HUGGINGFACE_HUB_KEY','HUGGINGFACEHUB_API_TOKEN')})"
```
 **Always ensure .env is in .gitignore**

## Seeding the historical data cache
```
*analogs from FredAPI*
uv run python -m sector_rotation_agent.historical_analogs
- or -
uv run --env-file .env -m sector_rotation_agent.historical_analogs

*Rag data from Federal Reservce documents*
uv run python -m sector_rotation_agent.fed_narrative_rag

```

## Clearing and reseeding the cache

- drop ONLY the stale numeric collection (Fed corpus is preserved)
```
uv run python -c "import chromadb as cdb, sector_rotation_agent.constants as const; cdb.PersistentClient(path=str(const.STORE_PATH)).delete_collection(const.COLLECTION_NAME); print('dropped', const.COLLECTION_NAME)"
```
- re-seed the analog store (needs FRED_API_KEY in your env); type 'yes' at the prompt
```
uv run python -m sector_rotation_agent.historical_analogs
```
- VERIFY the stats actually rebuilt against OECD
```
uv run python -c "import json; print(json.load(open('data/norm_stats.json'))['leading_index'])"
```
---

## Launch the analyst
```
uv run python -m sector_rotation_agent.main

uv run python -m sector_rotation_agent.main --q "Which sectors should I overweight/underweight over the next 6 months?" --ao "2026-06-15"

```

## Run the backtest
```
uv run python -m sector_rotation_agent.backtest

# or a quick one or two-window test:
uv run python -m sector_rotation_agent.backtest --windows 2008-06-30

uv run python -m sector_rotation_agent.backtest --windows 2018-06-30 2021-12-31

```
## Logs & traces

Every run writes to your configured log folder (the directory of `sector_rotation_agent.log`):

- `trace-<runid>.jsonl` — the per-run structured trace: one JSON line per event, with
  the FULL detail of every LLM call (model, service, prompts, response, token usage,
  latency, errors), per-phase latencies, and a closing run summary. `<runid>` is
  `<as_of>-<timestamp>`. Grep / `jq` it, e.g. only the LLM calls:
```
python -c "import json,sys; [print(json.loads(l)['service'], json.loads(l)['model'], json.loads(l).get('total_tokens'), json.loads(l).get('latency_ms')) for l in open(sys.argv[1]) if json.loads(l)['event']=='llm_call']" logs/trace-<runid>.jsonl

python -c "import json; [print('ERR:', e.get('error'), '| RESP:', repr(e.get('response'))[:300]) for e in map(json.loads, open('logs/trace-<runid>.jsonl')) if e.get('event')=='llm_call' and 'summary' in (e.get('system') or '').lower()]"

python -c "import json,sys; [print(json.loads(l).get('service'), json.loads(l).get('model'), json.loads(l).get('total_tokens')) for l in open(sys.argv[1]) if json.loads(l).get('event')=='llm_call']" logs/trace-<runid>.jsonl
```
- Per-component logs — `main.log`, `coordinator.log`, `model_client.log`, `trace.log`
  (each line also still flows into the combined `sector_rotation_agent.log`).
- MCP server logs — `fred_server.log`, `yfin_server.log`. The FRED and yfinance tool
  servers run as separate stdio subprocesses (they speak JSON-RPC over stdout), so each
  logs to its own file with propagation disabled — keeping stdout clean for the protocol.
  They record every external API call (series / tickers, dates, result counts, latency)
  and any per-series / per-ticker failures. These files appear only after a run that
  actually spawns the servers.
- Seeding runs (`historical_analogs`, `fed_narrative_rag`) log their progress — documents
  fetched, snapshots / chunks upserted, and the seeded `leading_index` norm-stats — into
  the combined `sector_rotation_agent.log`.
- Full LLM prompts + responses go to `model_client.log` at DEBUG. At the default INFO
  level you get a one-line summary per call (service / model / tokens / latency / finish);
  the full text is always in the trace JSONL. For full prompts in the logs too:
```
$env:LOGGING_LEVEL="DEBUG"; uv run python -m sector_rotation_agent.main --q "..." --ao "2026-06-15"
```

---

## Running Tests
```
uv sync

uv run ruff check

uv run pytest tests/test_historical_analogs.py            # one file

uv run pytest tests/test_historical_analogs.py::test_find_analogs_returns_self_as_nearest   # one test

uv run pytest -k vectorize                                # any test whose name matches

uv run pytest -s                                          # show print() output (the live hypotheses test prints its results)
```