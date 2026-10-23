"""
Order sizing relative to the market (protocol v3.1; docs/methodology/evaluation-protocol-v3.md).

Institutional orders are sized as a share of average daily volume (ADV),
not a fixed coin amount: 50,000 BTC was 3.9% of one date's volume and 39% of
another's, so "the same order" was a different task on each date. ADV here is
the mean volume of the 20 days strictly BEFORE the date (no look-ahead), from
Binance's daily candles for BTCUSDT perpetual futures -- validated against our
own trade-by-trade sums to within 0.005% on all 25 dates.

The impact model's own volume setting (--adv-btc) is deliberately NOT changed:
its per-date calibration rescales the impact coefficient to absorb it exactly
(verified: identical impact at 240k and 1.28M), so only the order size matters.

Candles: TradeData/Klines/BTCUSDT-1d-YYYY-MM.csv (rl/experiments/fetch_daily_klines.py).
A table for the 25 research dates is kept in rl/results/adv20.csv so tests and
evaluation do not need the raw candles.
"""
import glob
from functools import lru_cache
from pathlib import Path

import pandas as pd

ADV_WINDOW_DAYS = 20
ADV_TABLE = "rl/results/adv20.csv"
KLINES_GLOB = "TradeData/Klines/BTCUSDT-1d-*.csv"


@lru_cache(maxsize=1)
def _daily_volume() -> pd.Series:
    files = sorted(glob.glob(KLINES_GLOB))
    if not files:
        raise FileNotFoundError(f"no daily candles at {KLINES_GLOB}; run rl/experiments/fetch_daily_klines.py")
    k = pd.concat([pd.read_csv(f) for f in files])
    k["date"] = pd.to_datetime(k.open_time, unit="ms").dt.strftime("%Y-%m-%d")
    return k.drop_duplicates("date").set_index("date").volume.sort_index()


def adv_from_candles(date: str, window: int = ADV_WINDOW_DAYS) -> float:
    """Mean daily volume (BTC) over the `window` days strictly before `date`."""
    vol = _daily_volume()
    prior = vol[vol.index < date].tail(window)
    if len(prior) < window:
        raise ValueError(f"only {len(prior)} days of candles before {date}; need {window}")
    return float(prior.mean())


@lru_cache(maxsize=1)
def _adv_table() -> dict:
    if Path(ADV_TABLE).exists():
        return pd.read_csv(ADV_TABLE).set_index("date").adv20_btc.to_dict()
    return {}


def adv(date: str) -> float:
    """20-day trailing ADV for `date`: from the committed table, else from candles."""
    table = _adv_table()
    return float(table[date]) if date in table else adv_from_candles(date)


def order_btc(date: str, share_of_adv: float) -> float:
    """Order size in BTC for an order of `share_of_adv` x ADV on `date`."""
    return share_of_adv * adv(date)
