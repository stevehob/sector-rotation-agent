"""
historical_analogs.py

RAG / long-term memory layer for the agent (spec Sections 3.1, 6.1).

Two halves that MUST agree with each other:

    seed_store()              writes historical macro snapshots into ChromaDB
    find_historical_analogs() reads the most similar ones back at runtime

They share the vector representation and normalization, so they live in one
module. If the seed side and the query side vectorize differently — different
indicator order, or different normalization — cosine similarity becomes
meaningless and every analog is garbage. That symmetry is the single most
important thing to get right here.

KEY IDEA — the snapshot vector IS the embedding
-----------------------------------------------
A macro snapshot is already a numeric vector (normalized indicators), so we hand
that vector to ChromaDB directly via `embeddings=` / `query_embeddings=` and skip
text embedding entirely. (sentence-transformers would only matter if you later
store TEXT, like analyst reports.) So matching macro environments needs no
embedding model — just consistent normalization on both sides.

STUB — scaffolding and contract provided; you implement the bodies. Suggested
order: _vectorize -> norm-stats helpers -> _get_collection -> seed_store ->
find_historical_analogs. Build seed + query together and test the round-trip
early (seed a few rows, query one back, check similarity is sane).

Dependencies: `uv add chromadb`.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from fredapi import Fred
import yfinance as yfin
import chromadb as cdb
import pandas as pd
import logging
from typing import cast
import json

from sector_rotation_agent.classify_regime_tot import MacroSnapshot
import sector_rotation_agent.constants as const

# ---- Set up logging   --------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared representation (used by BOTH seed and query)
# --------------------------------------------------------------------------- #

def _vectorize(indicators: dict[str, float], norm_stats: dict) -> list[float]:
    """
    Turn an indicators dict into a normalized vector — the SHARED representation.

      - Normalize each with norm_stats (z-score: (x - mean) / std). Per-indicator
        stats put everything on a comparable scale so one large-magnitude series
        (e.g. rates) doesn't dominate the cosine distance.
      - Return list[float] of length len(INDICATOR_KEYS).

    Called by BOTH seed_store and find_historical_analogs. They must produce
    identical vectors for identical inputs, so keep ALL the logic here — one copy
    to keep correct.
    """
    return [
        (indicators[key] - norm_stats[key]["mean"]) / norm_stats[key]["std"]
        for key in const.INDICATOR_KEYS
    ]


def _clean(value) -> float | None:
    """Coerce a yfinance scalar to a JSON-safe value: NaN -> None, numpy float -> float.

    yfinance fields arrive as numpy floats or NaN; NaN is not valid JSON, so it
    becomes None. A non-scalar (e.g. a Series, which means the extraction picked
    up the wrong shape) also becomes None rather than emitting junk.
    """
    if value is None:
        return None
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _compute_norm_stats(history: list[dict]) -> dict:
    """
    Per-indicator normalization stats (mean/std) over the full history.

    Computed ONCE at seed time and persisted, so the query side normalizes the
    live snapshot against the same baseline the stored vectors used.
    Expected output format:
        {
            "fed_funds_rate":    { "mean": 2.41, "std": 1.87 },
            "cpi_inflation":     { "mean": 2.55, "std": 1.42 },
            // ... one entry per INDICATOR_KEY
        }
    """
    # history input is one row per month, one column per indicator
    # restrict to the 8 that get vectorized
    df = pd.DataFrame([row["indicators"] for row in history])[list(const.INDICATOR_KEYS)]
    stats = df.agg(["mean", "std"]).to_dict()        # {"fed_funds_rate": {"mean":.., "std":..}, ...}
    # go through and ensure that a) values are float and b) std != 0 and != NaN
    for key in const.INDICATOR_KEYS:
        stats[key]["mean"] = float(stats[key]["mean"])
        s = stats[key]["std"]
        stats[key]["std"]  = 1.0 if (s == 0 or pd.isna(s)) else float(s)   # guard zero/NaN std
    
    return stats



def _save_norm_stats(stats: dict) -> None:
    """
        Save norm_stats to JSON file
    """
    # make sure target directory exists; if not, create
    file_path = Path(const.NORM_STATS_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(stats, json_file, indent=4, ensure_ascii=False)
    except Exception as e:  # could get fancy looking for FileNotFoundError, PermissionError, JSONDecodeError
        logger.critical(f"Unexpected Error: {e} attempting to save file: {const.NORM_STATS_PATH}")
        raise  # re-throw

    return


def _load_norm_stats() -> dict:
    """
        Read back saved norm_stats to JSON and convert to dict
    """
    try:
        with open(const.NORM_STATS_PATH, "r", encoding="utf-8") as json_file:
            output = json.load(json_file)
    except FileNotFoundError:
        logger.critical(f"JSON file {const.NORM_STATS_PATH} not found. History seed must not have run.")
        raise RuntimeError("_load_norm_stats failed")
    except json.JSONDecodeError:
        logger.critical(f"File {const.NORM_STATS_PATH} is not valid JSON")
        raise RuntimeError("_load_norm_stats failed")
    except Exception as e:
        logger.critical(f"Unknown error {e} loading file: {const.NORM_STATS_PATH}")
        raise RuntimeError("_load_norm_stats failed")

    if not isinstance(output, dict):
        logger.critical(f"JSON file {const.NORM_STATS_PATH} loaded, but is not a valid output")
        raise RuntimeError("_load_norm_stats failed")
    
    return output


def _label_regime(indicators: dict[str, float]) -> const.Regime:
    """
    Rules-based regime label for a historical month (no ML, no LLM).

    Relies on a fixed list of indicators - which it is because it's a defined constant
    INDICATOR_KEYS = (
      "fed_funds_rate",       <-- 0
      "cpi_inflation",        <-- 1
      "pce_inflation",        <-- 2
      "unemployment",         <-- 3
      "yield_spread_10_2",    <-- 4
      "gdp_growth",           <-- 5
      "ism_pmi",              <-- 6
      "leading_index")        <-- 7

    In addition to the base indicators, the calling function added 3mo trends
      specifically for this computation:
          "unemployment_chg_3m"       <-- 8
          "leading_index_slope"       <-- 9

    Returns:
      Regime: EARLY_CYCLE / MID_CYCLE / LATE_CYCLE / CONTRACTION

    """
    LABEL_INPUTS = (*const.INDICATOR_KEYS, const.UNEMPLOYMENT_CHANGE_SERIES, const.LEADING_INDEX_CHANGE_SERIES)

    # since I'm comparing values that came from Pandas,
    # check for nan : because nan < 0 → returns False;  None would be TypeError
    if any(pd.isna(indicators[k]) for k in LABEL_INPUTS):
      logger.error("_label_regime missing/NaN inputs among %s", list(LABEL_INPUTS))
      raise ValueError("_label_regime missing inputs: for k in LABEL_INPUTS")
    else:
      gdp_growth = indicators.get(LABEL_INPUTS[5])
      emp_growth = indicators.get(LABEL_INPUTS[8])
      curve = indicators.get(LABEL_INPUTS[4])
      leading_index = indicators.get(LABEL_INPUTS[7])
      cpi_inflation = indicators.get(LABEL_INPUTS[1])

      # simple heuristic
      if (gdp_growth < 0 or curve < 0) and emp_growth > 0:                      # pyright: ignore[reportOptionalOperand]
          return const.Regime.CONTRACTION
      elif emp_growth < 0 and leading_index > const.LEADING_INDEX_NEUTRAL and curve > const.STEEP_CURVE:  # pyright: ignore[reportOptionalOperand]
          return const.Regime.EARLY_CYCLE
      elif curve < const.FLAT_CURVE and cpi_inflation > const.HOT_INFLATION:    # pyright: ignore[reportOptionalOperand]
          return const.Regime.LATE_CYCLE
      else:
          return const.Regime.MID_CYCLE
    


# --------------------------------------------------------------------------- #
# Collection accessor
# --------------------------------------------------------------------------- #
"""
The ToT evaluates branches in parallel, so this must not build a fresh
PersistentClient per call — concurrent client startup on one store races
and fails tenant validation."""
_collection = None
_collection_path: str | None = None
_collection_lock = threading.Lock()

def _get_collection():
    """
    Open (or create) the persistent ChromaDB collection.
        _get_collection() will be invoked in parallel by multiple branches in ToT
            Therefore, need to make it a 'shared' object that each branch can use.
            (Bug hit during initial testing)
        
        Note: cosine space means query() returns cosine DISTANCE; you convert that to
           a 0..1 similarity in find_historical_analogs.

        Bug fix: added a persistent _collection_path, since _collection is global it
           broke for test cases that needed to not be bound to the one global 'real' client.
           But, this could be a problem in future too if more datasets are added.
    """
    # make _collection global (module-level) scope
    global _collection, _collection_path
    path = str(const.STORE_PATH)
    if _collection is None or _collection_path != path:     # no collection, so create one
        with _collection_lock:      # take a lock to isolate
            if _collection is None or _collection_path != path:     # double-check nobody else created while grabbing the lock
                client = cdb.PersistentClient(path=str(const.STORE_PATH))     # now create the client and collection
                _collection = client.get_or_create_collection(
                    name=const.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                _collection_path = path
    return _collection



# --------------------------------------------------------------------------- #
# Seed side (run once, offline)
# --------------------------------------------------------------------------- #

def seed_store(history: list[dict]) -> None:
    """
    Populate ChromaDB with historical macro snapshots. Run once.

    `history` is a list of monthly records you assemble upstream from
    get_macro_indicators + get_sector_performance, each shaped like:

        {
          "date": "2008-10",
          "indicators": {"fed_funds_rate": .., "cpi_inflation": .., ...},
          "subsequent_sector_returns": {"XLK": -0.12, "XLU": 0.03, ...},  # forward 6m (default)
          "subsequent_returns_by_horizon": {3: {...}, 6: {...}, 12: {...}},  # per-horizon
        }

    """
    
    # compute and save the normalized stats
    logger.info("Seeding analog store: %d historical snapshot(s)", len(history))
    norm_stats = _compute_norm_stats(history)
    _save_norm_stats(norm_stats)
    # Log leading_index stats explicitly: a stale norm_stats.json here (after a
    # SUPPORTED_SERIES change) silently collapses similarity to zero confidence, so
    # surfacing the distribution at seed time makes that class of bug diagnosable.
    li = norm_stats.get("leading_index", {})
    logger.info("Computed norm_stats over %d indicators (leading_index mean=%.4f std=%.4f)",
                len(norm_stats), li.get("mean", float("nan")), li.get("std", float("nan")))

    # open a Chroma DB collection
    collection = _get_collection()

    # vectorize each row, batch-add to Chroma
    id_list: list[str] = []
    embeddings_list: list[list[float]] = []
    metadatas_list: list[dict] = []

    for row in history:
        vector = _vectorize(row["indicators"], norm_stats)
        # make sure vector is complete
        assert len(vector) == len(const.INDICATOR_KEYS)
        # get the precomputed label
        regime = row["regime"]     # Note this is the enum value, not a string
        # add all to lists
        id_list.append(row["date"])
        embeddings_list.append(vector)
        metadatas_list.append({
            "date": row["date"],
            "regime": regime.value,
            "subsequent_sector_returns": json.dumps(row["subsequent_sector_returns"]),
            # forward returns at every seeded horizon, keyed by month count (JSON object
            # keys are strings; the query side casts them back to int). The flat field
            # above is kept as the default/legacy slice so older readers still work.
            "subsequent_returns_by_horizon": json.dumps(row["subsequent_returns_by_horizon"]),
        })
    # end for
    # ignoring type issues, .upsert has it's own internal validator to prevent bad things
    # using .upsert instead of .add because it's itempotent (in case we run multiple times)
    collection.upsert(
        ids=id_list,
        embeddings=embeddings_list, # type: ignore
        metadatas=metadatas_list    # type: ignore
    )
    logger.info("Analog store seeded: upserted %d snapshot(s)", len(id_list))
    # verify the data
    logger.debug(f"Result of collection.upsert was {collection.peek(limit=1)}")


def build_seed_history() -> list[dict]:
    # FRED fetch -> resample monthly -> YoY features -> join sector forward returns
    logger.info("Building seed history from FRED + yfinance (start=%s)", const.HISTORY_SEED_START)
    key=os.getenv("FRED_API_KEY")
    if not key:
        logger.error("FRED_API_KEY is not set; cannot build seed history")
        raise RuntimeError("Missing API key for FredAPI")
    
    # Get historical data from FredAPI
    fred_data: dict[str, pd.Series] = {}
    fred = Fred(api_key=key)
    for name, series_id in const.SUPPORTED_SERIES.items():
        fred_data[name] = fred.get_series(
            series_id=series_id,
            observation_start=const.HISTORY_SEED_START)
    logger.info("Fetched %d FRED series for seed history", len(fred_data))

    history_df = pd.DataFrame(fred_data)
    """
    in history_df, pandas aligns series on the union of their date indices.
      - Since T10Y2Y is daily, that union is a daily index with thousands of rows
      - Every monthly series is NaN on all the daily-only dates
      - GDP is NaN on almost everything
    """ 
    # Resample to a 12-month index
    monthly_df = history_df.resample("ME").last()          # one row per month-end,  intentionally not using .mean()
    monthly_df["gdp_growth"] = monthly_df["gdp_growth"].ffill()   # carry the quarterly print forward

    # Compute YoY values
    monthly_df[const.CPI_INFLATION_SERIES] = monthly_df['cpi'].pct_change(12) * 100       # make it readable percent, not fraction
    monthly_df["pce_inflation"] = monthly_df["pce_inflation"].pct_change(12) * 100  # make it readable percent, not fraction

    # Add computed columns of change rate - since the data is absolute values
    monthly_df[const.UNEMPLOYMENT_CHANGE_SERIES] = monthly_df["unemployment"].diff(3)
    monthly_df[const.LEADING_INDEX_CHANGE_SERIES] = monthly_df["leading_index"].diff(3)


    # ----- Now go get YFinance data ---------------------------------------------
    # get the raw data from yfinance
    logger.info("Fetching %d sector ETFs from yfinance for seed history", len(const.SECTOR_ETFS_LIST))
    yfin_data = yfin.download(    # pyright: ignore[reportMissingTypeStubs]
        const.SECTOR_ETFS_LIST, 
        start=const.HISTORY_SEED_START, 
        interval="1mo", 
        auto_adjust=True, 
        progress=False)
    
    # make sure we got back a dataset
    if yfin_data is None or yfin_data.empty:
        logger.error("yfinance returned no data for sector ETFs from %s", const.HISTORY_SEED_START)
        raise RuntimeError(f"yfinance returned no data for sector ETFs starting from {const.HISTORY_SEED_START}")
    
    # yfin returns open, high, low, etc.  I just want Close
    yfin_close = cast(pd.DataFrame, yfin_data["Close"])
    
    # Forward returns at EACH seeded horizon, per sector, per month: close.shift(-h)/close - 1.
    # The default-horizon frame drives the inner join below (so the seeded months match
    # what a single-horizon seed produced); other horizons are looked up per row, and a
    # horizon with no row yet for a recent month yields None there.
    ## Fred returns month end and YFinance month begin -> shift to month-end to match macro_df.
    fwd_by_h: dict[int, pd.DataFrame] = {}
    for h in const.ANALOG_HORIZONS_MONTHS:
        f = yfin_close.shift(-h) / yfin_close - 1
        f.index = f.index + pd.offsets.MonthEnd(0)
        fwd_by_h[h] = f
    default_fwd = fwd_by_h[const.ANALOG_DEFAULT_HORIZON_MONTHS].dropna(how="all")  # trim no-outcome tail


    # replace the macro_df line with one that requires the trends to be present too:
    extended_indicator_keys = const.INDICATOR_KEYS + (const.UNEMPLOYMENT_CHANGE_SERIES, const.LEADING_INDEX_CHANGE_SERIES)
    macro_df = monthly_df.dropna(subset=list(extended_indicator_keys))   # complete vectors AND trends only

    # ------------------  Join sector forward returns  -----------------------------------------
    # now combine the Fred data with the YFin data
    combined = macro_df.join(default_fwd, how="inner")             # default-horizon returns drive the join
    assert len(combined) > 0, "combined DataFrame is empty"  # need to ensure the join succeeded

    
    # create the final output format of the combined dataset
    history = []
    # the date field comes as a hashtable, so cast it string for use in creating the output in the right format
    date_strs = cast(pd.DatetimeIndex, combined.index).strftime("%Y-%m")
    for date_str, (ts, row) in zip(date_strs, combined.iterrows()):
        indicators = {k: float(row[k]) for k in extended_indicator_keys}
        returns    = {tkr: _clean(row[tkr]) for tkr in const.SECTOR_ETFS}   # default-horizon slice
        # forward returns at every seeded horizon for this month (None where a horizon
        # has no row yet, e.g. the longest horizon's most recent months)
        returns_by_h = {
            h: {tkr: (_clean(frame.at[ts, tkr]) if (ts in frame.index and tkr in frame.columns) else None)
                for tkr in const.SECTOR_ETFS}
            for h, frame in fwd_by_h.items()
        }
        history.append({
            "date": date_str,
            "indicators": indicators,
            "subsequent_sector_returns": returns,             # default (6m) slice -- legacy field
            "subsequent_returns_by_horizon": returns_by_h,    # {3: {...}, 6: {...}, 12: {...}}
            "regime": _label_regime(indicators),   # determine regime (using the added unemployment_chg_3m & leading_index_slope)
        })

    # if we got here, everything worked
    logger.info("Built seed history: %d monthly snapshot(s)", len(history))
    return history

# --------------------------------------------------------------------------- #
# Query side (the tool classify_regime_tot calls)
# --------------------------------------------------------------------------- #
def _analog_date_cutoff(as_of: str, months_back: int) -> str:
    """Latest analog month ("YYYY-MM") whose `months_back`-month forward-return
    window had fully closed on or before `as_of`.

    Keeps retrieval point-in-time: we only match history whose OUTCOMES were
    realized by the as-of date, so a backtest can't peek at the future, and live
    runs simply skip the most-recent months whose forward returns don't exist yet.
    `as_of` is an ISO string ("YYYY-MM-DD" or "YYYY-MM")."""
    idx = int(as_of[:4]) * 12 + (int(as_of[5:7]) - 1) - months_back   # 0-based month index
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"

def find_historical_analogs(
    snapshot: MacroSnapshot,
    n: int,
    regime_filter: const.Regime | None = None,
) -> list[dict]:
    """
    Retrieve the `n` historical periods most similar to `snapshot`.

    This is classify_regime_tot's AnalogFinder dependency. Each branch calls it
    with its own regime_filter so a hypothesis is tested only against history of
    that regime.

    Returns
    -------
    list[dict]
        Up to `n` analogs; may be shorter or empty if the regime filter is narrow.
        score_branch already handles thin/empty analog sets (it discounts them).
        
        Here's what the output should look like:

        results = {
            "ids":       [["2008-10", "2001-03", "1990-08"]],          # note: a list holding ONE inner list
            "distances": [[0.0123,     0.0440,     0.0710]],            # cosine DISTANCE, ascending (nearest first)
            "metadatas": [[
                {"date": "2008-10", "regime": "contraction", "subsequent_sector_returns": "{\"XLK\": -0.22, ...}"},
                {"date": "2001-03", "regime": "contraction", "subsequent_sector_returns": "{\"XLK\": -0.05, ...}"},
                {"date": "1990-08", "regime": "contraction", "subsequent_sector_returns": "{\"XLK\":  0.01, ...}"},
            ]],
            "embeddings": None,                                          # not returned unless you ask
            "documents":  [[None, None, None]],
            "included":   ["metadatas", "distances", "documents"],
        }
    """
    # get the existing (seeded history) normalized stats
    norm_stats = _load_norm_stats()

    # now take the input data and vectorize it (same as history was)
    vector = _vectorize(snapshot.indicators, norm_stats)

    # get the Chroma DB
    collection = _get_collection()

    # Point-in-time guard. Only match analogs whose forward-outcome window had fully
    # closed by the snapshot's as_of, so a backtest can't see returns from after the
    # decision date. Filtered in Python rather than Chroma's `where` because string
    # range operators ($lte on "YYYY-MM") aren't supported across all Chroma versions;
    # "YYYY-MM" is fixed-width and sorts lexically, so a plain string compare is chronological.
    cutoff = _analog_date_cutoff(snapshot.as_of, const.ANALOG_DEFAULT_HORIZON_MONTHS)
    # Only the regime goes in Chroma's `where`; the date cutoff is applied in the loop
    # below. Chroma's $lte is numeric-only and rejects the "YYYY-MM" string, so the
    # point-in-time filter can't live in the where clause.
    where = {"regime": regime_filter.value} if regime_filter else None
    # over-fetch: the date filter below drops rows; regime-filtered sets are small anyway
    results = collection.query(query_embeddings=vector, n_results=n * 5, where=where)  # type: ignore

    metadatas = results["metadatas"]
    distances = results["distances"]
    if metadatas is None or distances is None:
        logger.debug("results of collection.query() returned either no metadatas or distances")
        return []

    outputs = []
    for meta, dist in zip(metadatas[0], distances[0]):
        if str(meta["date"]) > cutoff:          # outcome window extends past as_of -> skip
            continue
        default_returns = json.loads(meta["subsequent_sector_returns"])  # type: ignore
        raw_by_h = meta.get("subsequent_returns_by_horizon")
        if raw_by_h:
            by_horizon = {int(h): v for h, v in json.loads(raw_by_h).items()}  # type: ignore
        else:
            by_horizon = {const.ANALOG_DEFAULT_HORIZON_MONTHS: default_returns}
        outputs.append({
            "date": meta["date"],
            "similarity": max(0.0, min(1.0, 1.0 - dist)),
            "regime": meta["regime"],
            "subsequent_sector_returns": default_returns,
            "subsequent_returns_by_horizon": by_horizon,
        })
        if len(outputs) >= n:                   # enough after filtering
            break

    logger.debug("Analog query: %d result(s) for n=%d, regime_filter=%s, cutoff<=%s",
                 len(outputs), n, regime_filter.value if regime_filter else "any", cutoff)
    return outputs

## This is a run-once executor to build the historical dataset
if __name__ == "__main__":
    print("Are you sure you want to build the history store?")
    print("This may take a while ..")
    
    choice = input("Type 'yes' to execute:")
    
    if choice.lower() == "yes":
      seed_store(build_seed_history())
