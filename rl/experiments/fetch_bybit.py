"""
Download and convert Bybit BTCUSDT perpetual data for the research dates (era 2).

Book: Bybit's free ob500 archive (full 500-level snapshot + tick-by-tick deltas)
is replayed and the complete 500-level book is written every 30 s (UTC-aligned)
in the simulator's wide format. Trades: Bybit's public trade archive, converted
to transact_time / price / quantity / is_buyer_maker (taker "Sell" -> buyer was
the maker). Raw downloads are deleted after conversion.

    python rl/experiments/fetch_bybit.py                 # all research dates
    python rl/experiments/fetch_bybit.py 2024-01-15      # selected dates
"""
import csv
import gzip
import io
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import orjson

BOOK_URL = "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/{date}_BTCUSDT_ob500.data.zip"
TRADES_URL = "https://public.bybit.com/trading/BTCUSDT/BTCUSDT{date}.csv.gz"
LEVELS = 500
INTERVAL_MS = 30_000
RAW_DIR = Path("TradeData/BybitRaw")


def _download(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(target)
    return target


def _day_start_ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _emit(writer, ts: int, bids: dict, asks: dict) -> None:
    b = sorted(bids.items(), key=lambda kv: -float(kv[0]))[:LEVELS]
    a = sorted(asks.items(), key=lambda kv: float(kv[0]))[:LEVELS]
    row = [ts]
    for side in (b, a):
        flat = [x for level in side for x in level]
        row += flat + [""] * (2 * LEVELS - len(flat))
    writer.writerow(row)


def convert_book(raw_zip: Path, out_csv: Path, date: str) -> int:
    """Replay snapshot + deltas; write the book as of every 30 s boundary."""
    start = _day_start_ms(date)
    end = start + 86_400_000
    header = (["timestamp"] + [f"b{i}_{k}" for i in range(LEVELS) for k in ("p", "q")]
              + [f"a{i}_{k}" for i in range(LEVELS) for k in ("p", "q")])
    bids, asks, have_book, rows = {}, {}, False, 0
    next_boundary = start + INTERVAL_MS
    tmp = out_csv.with_suffix(".csv.part")
    with zipfile.ZipFile(raw_zip) as z, open(tmp, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        name = z.namelist()[0]
        with z.open(name) as fin:
            for line in io.BufferedReader(fin):
                msg = orjson.loads(line)
                ts = msg["ts"]
                while have_book and ts > next_boundary and next_boundary <= end:
                    _emit(writer, next_boundary, bids, asks)  # state as of the boundary
                    rows += 1
                    next_boundary += INTERVAL_MS
                data = msg["data"]
                if msg["type"] == "snapshot":
                    bids = {p: q for p, q in data["b"]}
                    asks = {p: q for p, q in data["a"]}
                    have_book = True
                    while next_boundary < ts:  # nothing to emit before the first book
                        next_boundary += INTERVAL_MS
                    continue
                for side, book in (("b", bids), ("a", asks)):
                    for p, q in data.get(side, []):
                        if float(q) == 0.0:
                            book.pop(p, None)
                        else:
                            book[p] = q
        while have_book and next_boundary <= end - INTERVAL_MS:
            _emit(writer, next_boundary, bids, asks)
            rows += 1
            next_boundary += INTERVAL_MS
    tmp.rename(out_csv)
    return rows


def convert_trades(raw_gz: Path, out_csv: Path) -> int:
    rows = 0
    tmp = out_csv.with_suffix(".csv.part")
    with gzip.open(raw_gz, "rt", newline="") as fin, open(tmp, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["transact_time", "price", "quantity", "is_buyer_maker"])
        for r in reader:
            writer.writerow([int(round(float(r["timestamp"]) * 1000)), r["price"], r["size"],
                             "true" if r["side"] == "Sell" else "false"])
            rows += 1
    tmp.rename(out_csv)
    return rows


def fetch(date: str, keep_raw: bool = False) -> None:
    from market_data import DATASETS
    ds = DATASETS["bybit_btcusdt_perp"]
    book_out, trades_out = Path(ds.book_path(date)), Path(ds.trades_path(date))
    book_out.parent.mkdir(parents=True, exist_ok=True)
    if not trades_out.exists():
        raw = _download(TRADES_URL.format(date=date), RAW_DIR / f"BTCUSDT{date}.csv.gz")
        n = convert_trades(raw, trades_out)
        print(f"{date}: {n:,} trades", flush=True)
        if not keep_raw:
            raw.unlink()
    if not book_out.exists():
        raw = _download(BOOK_URL.format(date=date), RAW_DIR / f"{date}_BTCUSDT_ob500.data.zip")
        n = convert_book(raw, book_out, date)
        print(f"{date}: {n} book snapshots", flush=True)
        if not keep_raw:
            raw.unlink()


def main():
    from evaluation.dates import SEVERITY_RANK_V2
    dates = sys.argv[1:] or sorted(SEVERITY_RANK_V2)
    for d in dates:
        fetch(d)


if __name__ == "__main__":
    main()
