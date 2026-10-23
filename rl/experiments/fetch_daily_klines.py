"""
Download Binance daily candles (BTCUSDT perpetual futures) used for ADV sizing,
then write the 20-day trailing ADV table for the 25 research dates.

    python rl/experiments/fetch_daily_klines.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import pandas as pd

MONTHS = [f"{y}-{m:02d}" for y, m in [(2023, 12)] + [(2024, i) for i in range(1, 13)]]
URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-{month}.zip"
OUT_DIR = Path("TradeData/Klines")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        target = OUT_DIR / f"BTCUSDT-1d-{month}.csv"
        if not target.exists():
            with urllib.request.urlopen(URL.format(month=month), timeout=60) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(OUT_DIR)
    from evaluation.dates import SEVERITY_RANK_V2
    from evaluation.liquidity import ADV_TABLE, adv_from_candles
    rows = [dict(date=d, adv20_btc=adv_from_candles(d)) for d in sorted(SEVERITY_RANK_V2)]
    pd.DataFrame(rows).to_csv(ADV_TABLE, index=False)
    print(pd.DataFrame(rows).round(0).to_string(index=False))


if __name__ == "__main__":
    main()
