"""Real traded volume per date (Binance BTCUSDT perpetual futures aggTrades), by regime."""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import pandas as pd


def _volume(date):
    df = pd.read_csv(f"TradeData/AggTrades/BTCUSDT-aggTrades-{date}.csv", usecols=["price", "quantity"])
    return date, float(df.quantity.sum()), float((df.price * df.quantity).sum()), len(df)


def main():
    from evaluation.dates import HELD_OUT_DATES_V2, SEVERITY_RANK_V2, TRAINING_DATES_V3, VALIDATION_DATES_V3
    with Pool(6) as pool:
        rows = pool.map(_volume, sorted(SEVERITY_RANK_V2))
    df = pd.DataFrame(rows, columns=["date", "volume_btc", "notional_usd", "trades"])
    df["regime"] = df.date.map(lambda d: SEVERITY_RANK_V2[d][1])
    df["split"] = df.date.map(lambda d: "train" if d in TRAINING_DATES_V3 else "validation" if d in VALIDATION_DATES_V3 else "held-out")
    df["order_share_50k"] = 50_000 / df.volume_btc
    df.to_csv("rl/results/daily_volume.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.assign(volume_btc=df.volume_btc.round(0), notional_bn=(df.notional_usd / 1e9).round(1),
                    order_share=(100 * df.order_share_50k).round(1))
          [["date", "regime", "split", "volume_btc", "notional_bn", "trades", "order_share"]].to_string(index=False))
    print("\nmedian by regime:")
    print(df.groupby("regime").volume_btc.agg(["median", "min", "max", "count"]).round(0).to_string())


if __name__ == "__main__":
    main()
