"""
test_historical_analogs.py

Tests for the RAG / memory layer (src/sector_rotation_agent/historical_analogs.py).

Status when written:
  * _label_regime and build_seed_history are IMPLEMENTED -> real assertions that
    should pass (build_seed_history's positive case is Integration-gated since it
    hits FRED + Yahoo).
  * the remaining helpers are STUBS (raise NotImplementedError). Their tests are
    written against the documented contract and marked xfail(strict=False), so the
    suite stays green now and each test flips to XPASS the moment you implement its
    target -- that's your signal to delete the xfail marker.

Conventions mirror the FRED/yfin suites: offline unit tests always run;
network-touching tests gate on TEST_MODE == "Integration". Paths are redirected
into tmp so tests never touch the real data/ directory.

One positive + one negative test per function. A catalog of further cases to fill
in later sits at the bottom of the file.
"""

from __future__ import annotations

import pytest

import json
import os

import sector_rotation_agent.historical_analogs as ha
import sector_rotation_agent.constants as const
from sector_rotation_agent.classify_regime_tot import MacroSnapshot

INTEGRATION_ONLY = pytest.mark.skipif(
    os.getenv("TEST_MODE") != "Integration",
    reason="Hits FRED + Yahoo; Integration only",
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect the module's persistent paths into tmp so nothing touches data/."""
    monkeypatch.setattr(const, "STORE_PATH", tmp_path / "chroma")
    monkeypatch.setattr(const, "NORM_STATS_PATH", tmp_path / "norm_stats.json")
    return tmp_path


def complete_indicators(**overrides) -> dict[str, float]:
    """A finite, fully-populated dict for _label_regime: the 8 indicators + 2 trends."""
    d = {k: 0.0 for k in const.INDICATOR_KEYS}
    d[const.UNEMPLOYMENT_CHANGE_SERIES] = 0.0
    d[const.LEADING_INDEX_CHANGE_SERIES] = 0.0
    d.update(overrides)
    return d


@pytest.fixture
def sample_history() -> list[dict]:
    """Five monthly rows with deliberately DISTINCT indicator patterns.

    Distinct (non-scalar-multiple) rows matter for the cosine round-trip: rows that
    are scalar multiples of each other share a direction and can't be told apart by
    cosine similarity. These five do not, so a seeded month is uniquely its own
    nearest neighbour.
    """
    raw = [
        (0.5, 1.2, 1.1, 6.0, -0.3, -1.0, 12000.0, -0.5),
        (0.2, 1.0, 0.9, 9.0,  2.2,  3.5, 11000.0,  1.2),
        (4.5, 3.4, 3.2, 3.6,  0.1,  2.0, 13000.0,  0.8),
        (2.5, 2.0, 1.9, 4.0,  0.6,  2.5, 12500.0,  1.0),
        (5.0, 4.0, 3.8, 3.5, -0.2,  1.0, 13200.0,  0.3),
    ]
    regimes = [
        const.Regime.CONTRACTION, const.Regime.EARLY_CYCLE, const.Regime.LATE_CYCLE,
        const.Regime.MID_CYCLE, const.Regime.LATE_CYCLE,
    ]
    history = []
    for i, (vals, reg) in enumerate(zip(raw, regimes)):
        # forward returns at each seeded horizon, scaled by horizon so the slices differ;
        # the default-horizon slice is mirrored into subsequent_sector_returns (as the
        # store does), so the legacy field stays consistent with the per-horizon map.
        by_h = {
            h: {t: round(0.001 * (i + 1) * h, 4) for t in const.SECTOR_ETFS_LIST}
            for h in const.ANALOG_HORIZONS_MONTHS
        }
        history.append({
            "date": f"2010-{i + 1:02d}",
            "indicators": dict(zip(const.INDICATOR_KEYS, map(float, vals))),
            "subsequent_sector_returns": by_h[const.ANALOG_DEFAULT_HORIZON_MONTHS],
            "subsequent_returns_by_horizon": by_h,
            "regime": reg,
        })
    return history

@pytest.fixture
def sample_as_of() -> str:
    return "2011-06-01"

# --------------------------------------------------------------------------- #
# _label_regime
# --------------------------------------------------------------------------- #
def test_label_regime_classifies_contraction():
    # GDP negative AND unemployment rising -> contraction
    ind = complete_indicators(gdp_growth=-1.0, unemployment_chg_3m=0.3)
    assert ha._label_regime(ind) is const.Regime.CONTRACTION


def test_label_regime_raises_on_missing_value():
    # A nan in any required input must fail loudly, not silently mislabel.
    ind = complete_indicators(unemployment_chg_3m=float("nan"))
    with pytest.raises(ValueError):
        ha._label_regime(ind)

# --------------------------------------------------------------------------- #
# _vectorize
# --------------------------------------------------------------------------- #
def test_vectorize_zscores_in_indicator_order():
    norm = {k: {"mean": 1.0, "std": 2.0} for k in const.INDICATOR_KEYS}
    ind = {k: 5.0 for k in const.INDICATOR_KEYS}
    vec = ha._vectorize(ind, norm)
    assert len(vec) == len(const.INDICATOR_KEYS)
    assert all(v == pytest.approx((5.0 - 1.0) / 2.0) for v in vec)


def test_vectorize_missing_indicator_raises():
    norm = {k: {"mean": 0.0, "std": 1.0} for k in const.INDICATOR_KEYS}
    incomplete = {k: 1.0 for k in const.INDICATOR_KEYS[:-1]}  # drop the last key
    with pytest.raises(KeyError):
        ha._vectorize(incomplete, norm)


# --------------------------------------------------------------------------- #
# _compute_norm_stats
# --------------------------------------------------------------------------- #
def test_compute_norm_stats_shape_and_mean(sample_history):
    stats = ha._compute_norm_stats(sample_history)
    for k in const.INDICATOR_KEYS:
        assert "mean" in stats[k] and "std" in stats[k]
    expected = sum(r["indicators"]["fed_funds_rate"] for r in sample_history) / len(sample_history)
    assert stats["fed_funds_rate"]["mean"] == pytest.approx(expected)


def test_compute_norm_stats_guards_zero_std():
    # Constant series -> std must be guarded so _vectorize never divides by zero.
    history = [{"indicators": {k: 1.0 for k in const.INDICATOR_KEYS}} for _ in range(3)]
    stats = ha._compute_norm_stats(history)
    assert stats["fed_funds_rate"]["std"] != 0


# --------------------------------------------------------------------------- #
# _save_norm_stats
# --------------------------------------------------------------------------- #
def test_save_norm_stats_writes_readable_json(isolated_paths):
    stats = {"fed_funds_rate": {"mean": 1.0, "std": 2.0}}
    ha._save_norm_stats(stats)
    assert const.NORM_STATS_PATH.exists()                                                  # type: ignore
    assert json.loads(const.NORM_STATS_PATH.read_text())["fed_funds_rate"]["mean"] == 1.0  # type: ignore


def test_save_norm_stats_creates_missing_parent_dir(tmp_path, monkeypatch):
    nested = tmp_path / "data" / "norm_stats.json"  # parent does not exist yet
    monkeypatch.setattr(const, "NORM_STATS_PATH", nested)
    ha._save_norm_stats({"x": {"mean": 0.0, "std": 1.0}})
    assert nested.exists()


# --------------------------------------------------------------------------- #
# _load_norm_stats
# --------------------------------------------------------------------------- #
def test_load_norm_stats_roundtrips_what_was_saved(isolated_paths):
    stats = {"fed_funds_rate": {"mean": 2.5, "std": 1.5}}
    ha._save_norm_stats(stats)
    assert ha._load_norm_stats() == stats


def test_load_norm_stats_missing_file_raises(isolated_paths):
    # Never seeded -> querying must fail clearly rather than silently misbehave.
    with pytest.raises(Exception):
        ha._load_norm_stats()


# --------------------------------------------------------------------------- #
# _get_collection
# --------------------------------------------------------------------------- #
def test_get_collection_returns_usable_collection(isolated_paths):
    col = ha._get_collection()
    assert col.name == const.COLLECTION_NAME       # type: ignore
    assert callable(col.add) and callable(col.query)


def test_get_collection_empty_query_returns_nothing(isolated_paths):
    col = ha._get_collection()
    res = col.query(query_embeddings=[[0.0] * len(const.INDICATOR_KEYS)], n_results=3)
    assert res["ids"] == [[]]


# --------------------------------------------------------------------------- #
# seed_store
# --------------------------------------------------------------------------- #
def test_seed_store_populates_collection(isolated_paths, sample_history):
    ha.seed_store(sample_history)
    col = ha._get_collection()
    assert col.count() == len(sample_history)
    assert const.NORM_STATS_PATH.exists()      # type: ignore


def test_seed_store_missing_regime_raises(isolated_paths, sample_history):
    bad = [dict(sample_history[0])]
    del bad[0]["regime"]  # seed_store expects a precomputed regime on every row
    with pytest.raises(KeyError):
        ha.seed_store(bad)


# --------------------------------------------------------------------------- #
# build_seed_history
# --------------------------------------------------------------------------- #
def test_build_seed_history_requires_fred_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ha.build_seed_history()


@INTEGRATION_ONLY
def test_build_seed_history_returns_rows():
    history = ha.build_seed_history()
    assert isinstance(history, list) and len(history) > 0
    row = history[0]
    assert set(row) >= {"date", "indicators", "subsequent_sector_returns", "subsequent_returns_by_horizon", "regime"}
    assert set(row["indicators"]) >= set(const.INDICATOR_KEYS)  # superset: the stored dict also carries the 2 trend features
    assert isinstance(row["regime"], const.Regime)


# --------------------------------------------------------------------------- #
# find_historical_analogs  -- the seed<->query round trip
# --------------------------------------------------------------------------- #
def test_find_analogs_returns_self_as_nearest(isolated_paths, sample_history, sample_as_of):
    ha.seed_store(sample_history)
    seeded = sample_history[0]
    #snap = MacroSnapshot(as_of=seeded["date"], indicators=seeded["indicators"])
    snap = MacroSnapshot(sample_as_of, indicators=seeded["indicators"])
    analogs = ha.find_historical_analogs(snap, n=3)
    assert isinstance(analogs, list) and len(analogs) >= 1
    top = analogs[0]
    assert set(top) >= {"date", "similarity", "regime", "subsequent_sector_returns"}
    assert 0.0 <= top["similarity"] <= 1.0
    assert top["date"] == seeded["date"]
    assert top["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert isinstance(top["subsequent_sector_returns"], dict)  # decoded back from JSON


def test_find_analogs_regime_filter_with_no_match_is_empty(isolated_paths):
    rows = [{
        "date": f"2010-0{i + 1}",
        "indicators": {k: float(i + 1) for k in const.INDICATOR_KEYS},
        "subsequent_sector_returns": {t: 0.0 for t in const.SECTOR_ETFS_LIST},
        "subsequent_returns_by_horizon": {h: {t: 0.0 for t in const.SECTOR_ETFS_LIST} for h in const.ANALOG_HORIZONS_MONTHS},
        "regime": const.Regime.MID_CYCLE,
    } for i in range(3)]
    ha.seed_store(rows)
    snap = MacroSnapshot(as_of="2010-02", indicators=rows[1]["indicators"])
    analogs = ha.find_historical_analogs(snap, n=3, regime_filter=const.Regime.CONTRACTION)
    assert analogs == []


def test_find_analogs_roundtrips_per_horizon_returns(isolated_paths, sample_history, sample_as_of):
    """Per-horizon returns survive the seed<->query round trip with INT keys (JSON
    stringifies object keys), and the slices differ as seeded (3m != 12m)."""
    ha.seed_store(sample_history)
    seeded = sample_history[0]
    #snap = MacroSnapshot(as_of=seeded["date"], indicators=seeded["indicators"])
    snap = MacroSnapshot(sample_as_of, indicators=seeded["indicators"])
    top = ha.find_historical_analogs(snap, n=1)[0]
    rbh = top["subsequent_returns_by_horizon"]
    assert set(rbh) == set(const.ANALOG_HORIZONS_MONTHS)   # int keys preserved
    assert all(isinstance(h, int) for h in rbh)
    assert rbh[3]["XLK"] != rbh[12]["XLK"]                 # distinct per-horizon slices


def test_find_analogs_legacy_store_without_by_horizon(isolated_paths):
    """A store seeded before per-horizon returns existed (metadata has no
    subsequent_returns_by_horizon) still reads back: the default horizon maps to the flat
    returns, so old stores keep working until re-seeded."""
    ha._save_norm_stats({k: {"mean": 0.0, "std": 1.0} for k in const.INDICATOR_KEYS})
    col = ha._get_collection()
    flat = {t: 0.03 for t in const.SECTOR_ETFS_LIST}
    col.upsert(
        ids=["2010-01"],
        embeddings=[[1.0] * len(const.INDICATOR_KEYS)],
        metadatas=[{
            "date": "2010-01",
            "regime": const.Regime.MID_CYCLE.value,
            "subsequent_sector_returns": json.dumps(flat),
        }],
    )
    # as_of is + 1yr from the data upsert above to cover the 6-month validation
    snap = MacroSnapshot(as_of="2011-01", indicators={k: 1.0 for k in const.INDICATOR_KEYS})
    analogs = ha.find_historical_analogs(snap, n=1)
    assert analogs
    assert analogs[0]["subsequent_returns_by_horizon"] == {const.ANALOG_DEFAULT_HORIZON_MONTHS: flat}


# =========================================================================== #
# ADDITIONAL TEST CASES TO IMPLEMENT LATER
# =========================================================================== #
# _vectorize:
#   - order sensitivity: swapping two indicator values changes the right vector slots
#   - std == 0 path produces no inf/nan (depends on _compute_norm_stats guard)
#   - extra keys in `indicators` are ignored (only INDICATOR_KEYS consumed)
#   - a nan indicator value -> defined behavior (raise, or documented sentinel)
#
# _compute_norm_stats:
#   - single-row history (std == 0) is guarded
#   - values match numpy mean/std exactly
#   - ignores keys not in INDICATOR_KEYS
#
# _save_/_load_norm_stats:
#   - save overwrites an existing file
#   - float precision survives the round trip
#   - load on corrupt/invalid JSON -> clear error
#
# _label_regime:
#   - parametrize all four regimes (early / mid / late / contraction)
#   - boundary values exactly at STEEP_CURVE / FLAT_CURVE / HOT_INFLATION (strict > vs >=)
#   - CPI vs PCE: a row with CPI hot but PCE cool -- currently lands MID, not LATE,
#     because the late branch reads index [2] (pce_inflation), not [1] (cpi). See note.
#   - a MISSING key (vs nan) -> KeyError, distinct from the nan -> ValueError path
#
# _get_collection:
#   - idempotent: two calls see the same data (get_or_create, same count)
#   - cosine space is actually configured (metadata)
#
# seed_store:
#   - adds are batched (single add call), not one per row
#   - metadata round-trips: regime stored as .value, returns as a JSON string
#   - re-seeding the same ids -> defined upsert/duplicate behavior
#
# build_seed_history:
#   - XLRE / XLC inception: pre-2015 / pre-2018 forward returns come back as None
#   - YoY lead-in NaNs are trimmed (no row with a nan indicator survives)
#   - the trailing 6 months (no forward window) are dropped
#   - cpi_inflation is a PERCENT (matches HOT_INFLATION units), not a fraction
#   - combined frame is non-empty (guards an ME/MS index-convention mismatch)
#
# find_historical_analogs:
#   - n caps the number of analogs returned
#   - results are ordered by descending similarity
#   - regime_filter returns ONLY rows of that regime
#   - empty store -> []
#   - cosine distance -> similarity conversion is correct (1 - distance)
#
# cross-cutting:
#   - seed and query vectorize identically (the round trip is the canary)
#   - query reuses the persisted norm_stats, not a freshly recomputed one
