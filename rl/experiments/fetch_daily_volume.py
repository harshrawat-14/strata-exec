"""
Download Bybit daily volume (BTCUSDT perpetual, 2023-12 -> 2024-12) for ADV
sizing, then write the 20-day trailing ADV table for the research dates.

    python rl/experiments/fetch_daily_volume.py
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import pandas as pd

URL = ("https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}"
       "&interval=D&start={start}&end={end}&limit=1000")
FIRST, LAST = "2023-12-01", "2024-12-31"


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main():
    from evaluation.dates import SEVERITY_RANK_V2
    from evaluation.liquidity import adv_from_daily_volume
    from market_data import DATASETS

    ds = DATASETS["bybit_btcusdt_perp"]
    with urllib.request.urlopen(URL.format(symbol=ds.symbol, start=_ms(FIRST), end=_ms(LAST)), timeout=60) as r:
        payload = json.load(r)
    if payload.get("retCode") != 0:
        raise RuntimeError(payload)
    rows = [dict(date=pd.to_datetime(int(k[0]), unit="ms").strftime("%Y-%m-%d"), volume_btc=float(k[5]))
            for k in payload["result"]["list"]]
    vol = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    Path(ds.daily_volume_path()).parent.mkdir(parents=True, exist_ok=True)
    vol.to_csv(ds.daily_volume_path(), index=False)
    print(f"{len(vol)} days, {vol.date.iloc[0]} -> {vol.date.iloc[-1]}")
    table = pd.DataFrame([dict(date=d, adv20_btc=adv_from_daily_volume(d)) for d in sorted(SEVERITY_RANK_V2)])
    table.to_csv(ds.adv_table_path(), index=False)
    print(table.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
