"""
Order sizing relative to the market (protocol v3.1; docs/methodology/evaluation-protocol-v3.md).

Institutional orders are sized as a share of average daily volume (ADV), not a
fixed coin amount: the same coin amount is a different task on a 50k-BTC day
and a 500k-BTC day. ADV is the mean volume of the 20 days strictly BEFORE the
date (no look-ahead), on the active dataset's own venue (rl/market_data.py) --
an order is sized against the market it trades in.

Daily volumes: <dataset root>/daily-volume.csv (rl/experiments/fetch_daily_volume.py).
A table for the research dates is committed per dataset
(rl/results/adv20_<dataset>.csv) so tests and evaluation do not need raw data.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from market_data import dataset

ADV_WINDOW_DAYS = 20


@lru_cache(maxsize=4)
def _daily_volume(path: str) -> pd.Series:
    if not Path(path).exists():
        raise FileNotFoundError(f"no daily volumes at {path}; run rl/experiments/fetch_daily_volume.py")
    return pd.read_csv(path).set_index("date").volume_btc.sort_index()


def adv_from_daily_volume(date: str, window: int = ADV_WINDOW_DAYS) -> float:
    """Mean daily volume (BTC) over the `window` days strictly before `date`."""
    vol = _daily_volume(dataset().daily_volume_path())
    prior = vol[vol.index < date].tail(window)
    if len(prior) < window:
        raise ValueError(f"only {len(prior)} days of volume before {date}; need {window}")
    return float(prior.mean())


@lru_cache(maxsize=4)
def _adv_table(path: str) -> dict:
    if Path(path).exists():
        return pd.read_csv(path).set_index("date").adv20_btc.to_dict()
    return {}


def adv(date: str) -> float:
    """20-day trailing ADV for `date`: from the committed table, else from daily volumes."""
    table = _adv_table(dataset().adv_table_path())
    return float(table[date]) if date in table else adv_from_daily_volume(date)


def order_btc(date: str, share_of_adv: float) -> float:
    """Order size in BTC for an order of `share_of_adv` x ADV on `date`."""
    return share_of_adv * adv(date)
