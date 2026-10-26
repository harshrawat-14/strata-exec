"""Protocol v3.1 order sizing: share of a strictly-prior 20-day ADV on the dataset's own venue."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from evaluation.dates import SEVERITY_RANK_V2
from evaluation.liquidity import adv, adv_from_daily_volume, order_btc
from market_data import DATASETS, book_path, dataset

needs_volume = pytest.mark.skipif(not os.path.exists(dataset().daily_volume_path()),
                                  reason="requires the dataset's daily-volume.csv")


@needs_volume
def test_adv_uses_only_the_20_days_before_the_date():
    vol = pd.read_csv(dataset().daily_volume_path()).set_index("date").volume_btc.sort_index()
    date = "2024-08-05"  # the crash day's own volume must not leak into its ADV
    expected = vol[vol.index < date].tail(20).mean()
    assert adv_from_daily_volume(date) == pytest.approx(expected)
    assert adv_from_daily_volume(date) < vol[date] / 3


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_committed_table_covers_every_research_date(name):
    table = pd.read_csv(DATASETS[name].adv_table_path()).set_index("date").adv20_btc
    assert set(table.index) == set(SEVERITY_RANK_V2)
    assert (table > 50_000).all() and (table < 2_000_000).all()


def test_sizing_follows_the_active_dataset(monkeypatch):
    bybit = adv("2024-01-15")
    monkeypatch.setenv("STRATAEXEC_DATASET", "binance_btcusdt_perp_bookdepth")
    assert adv("2024-01-15") != bybit


def test_order_is_a_share_of_adv():
    assert order_btc("2024-01-15", 0.01) == pytest.approx(0.01 * adv("2024-01-15"))


@pytest.mark.skipif(not os.path.exists(book_path("2024-01-15")), reason="requires the dataset's book files")
def test_env_receives_the_scaled_order():
    from environment import StrataExecEnv
    env = StrataExecEnv(mode="counterfactual", lob_date="2024-01-15", agg_date="2024-01-15",
                        n_state_dims=16, order_share_adv=0.01)
    i = env.cmd.index("--btc-target")
    assert float(env.cmd[i + 1]) == pytest.approx(order_btc("2024-01-15", 0.01))
    with pytest.raises(ValueError):
        StrataExecEnv(mode="synthetic", order_share_adv=0.01)
