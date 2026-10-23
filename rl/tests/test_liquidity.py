"""Protocol v3.1 order sizing: share of a strictly-prior 20-day ADV."""
from __future__ import annotations

import glob
import os

import pandas as pd
import pytest

from evaluation.dates import SEVERITY_RANK_V2
from evaluation.liquidity import ADV_TABLE, adv, adv_from_candles, order_btc

needs_candles = pytest.mark.skipif(not glob.glob("TradeData/Klines/BTCUSDT-1d-*.csv"),
                                   reason="requires TradeData/Klines/")


@needs_candles
def test_adv_uses_only_the_20_days_before_the_date():
    k = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("TradeData/Klines/BTCUSDT-1d-*.csv"))])
    k["date"] = pd.to_datetime(k.open_time, unit="ms").dt.strftime("%Y-%m-%d")
    vol = k.set_index("date").volume.sort_index()
    date = "2024-08-05"  # 1.28M BTC traded that day: must not leak into its own ADV
    expected = vol[vol.index < date].tail(20).mean()
    assert adv_from_candles(date) == pytest.approx(expected)
    assert adv_from_candles(date) < vol[date] / 3


def test_committed_table_covers_every_research_date():
    table = pd.read_csv(ADV_TABLE).set_index("date").adv20_btc
    assert set(table.index) == set(SEVERITY_RANK_V2)
    assert (table > 50_000).all() and (table < 2_000_000).all()


def test_order_is_a_share_of_adv():
    assert order_btc("2024-01-15", 0.01) == pytest.approx(0.01 * adv("2024-01-15"))


@pytest.mark.skipif(not os.path.exists("TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv"),
                    reason="requires TradeData/")
def test_env_receives_the_scaled_order():
    from environment import StrataExecEnv
    env = StrataExecEnv(mode="counterfactual", lob_date="2024-01-15", agg_date="2024-01-15",
                        n_state_dims=16, order_share_adv=0.01)
    i = env.cmd.index("--btc-target")
    assert float(env.cmd[i + 1]) == pytest.approx(order_btc("2024-01-15", 0.01))
    with pytest.raises(ValueError):
        StrataExecEnv(mode="synthetic", order_share_adv=0.01)
