"""
book_features (docs 27, 32) regressions.

- 2026-11-25: it hard-coded TradeData/Bybit/BTCUSDT/book-{date}.csv instead of asking
  market_data, so the move to .csv.gz left it unable to find any file.
- drawdown_z is the single feature doc 32 tests; it must be causal and match its definition.
"""
import gzip
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "book_features", Path(__file__).resolve().parents[1] / "experiments" / "belief_precheck" / "book_features.py")
bf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bf)


def _write_day(tmp_path, n=600, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1_705_276_800_000  # 2024-01-15 00:00 UTC
    ts = t0 + 30_000 * np.arange(1, n + 1)
    mid = 40_000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    book = {"timestamp": ts}
    for i in range(3):
        book[f"b{i}_p"], book[f"b{i}_q"] = mid - 0.05 - i, 1.0 + i
    for i in range(3):
        book[f"a{i}_p"], book[f"a{i}_q"] = mid + 0.05 + i, 1.5 + i
    bpath, tpath = tmp_path / "book.csv.gz", tmp_path / "trades.csv.gz"
    with gzip.open(bpath, "wt") as f:
        pd.DataFrame(book).to_csv(f, index=False)
    trades = pd.DataFrame({"transact_time": ts - 5_000, "price": mid, "quantity": 0.1,
                           "is_buyer_maker": np.arange(n) % 2 == 0})
    with gzip.open(tpath, "wt") as f:
        trades.to_csv(f, index=False)
    return bpath, tpath, mid


def test_build_reads_files_through_market_data(tmp_path, monkeypatch):
    bpath, tpath, mid = _write_day(tmp_path)
    monkeypatch.setattr(bf, "book_path", lambda d: str(bpath))
    monkeypatch.setattr(bf, "trades_path", lambda d: str(tpath))
    df = bf.build("2024-01-15")
    assert len(df) == len(mid)
    np.testing.assert_allclose(df["mid"], mid, rtol=1e-12)
    assert np.isfinite(df[bf.feature_columns(df)].to_numpy()).all()


def test_drawdown_z_matches_its_definition():
    rng = np.random.default_rng(1)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 1000)))
    r = np.concatenate([[0.0], np.diff(np.log(mid))])
    sd = np.array([np.sqrt(np.mean(r[max(0, k - 239):k + 1] ** 2)) for k in range(len(r))]) + 1e-12
    expected = np.log(mid / np.maximum.accumulate(mid)) / (sd * np.sqrt(240))
    np.testing.assert_allclose(bf.drawdown_z(mid), expected, rtol=1e-9, atol=1e-12)


def test_drawdown_z_is_zero_at_new_highs_and_negative_after_a_fall():
    mid = np.concatenate([np.linspace(100, 110, 300), np.linspace(110, 104, 300)])
    dz = bf.drawdown_z(mid)
    assert np.all(dz[:300] == 0)
    assert np.all(dz[301:] < 0)


def test_drawdown_z_is_causal():
    rng = np.random.default_rng(2)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 800)))
    changed = mid.copy()
    changed[500:] *= 0.9  # a future crash
    np.testing.assert_array_equal(bf.drawdown_z(mid)[:500], bf.drawdown_z(changed)[:500])


@pytest.mark.skipif(not os.path.exists(bf.book_path("2024-01-15")), reason="requires the dataset's book files")
def test_build_on_real_day():
    df = bf.build("2024-01-15")
    assert len(df) == 2880
    assert np.isfinite(df[bf.feature_columns(df)].to_numpy()).all()
    assert df.drawdown_z.max() == 0 and df.drawdown_z.min() < 0
