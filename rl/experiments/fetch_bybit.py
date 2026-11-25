"""
Download and convert Bybit linear perpetual data (era 2) for any dataset registered in
market_data.DATASETS with venue "bybit" (BTCUSDT by default).

Book: Bybit's free ob500 archive (full 500-level snapshot + tick-by-tick deltas)
is replayed and the complete 500-level book is written every 30 s (UTC-aligned)
in the simulator's wide format. Trades: Bybit's public trade archive, converted
to transact_time / price / quantity / is_buyer_maker (taker "Sell" -> buyer was
the maker). Output is gzip-compressed CSV (about 5x smaller). Raw downloads are
deleted after conversion.

    python rl/experiments/fetch_bybit.py                 # all research dates, BTCUSDT
    python rl/experiments/fetch_bybit.py 2024-01-15      # selected dates
    python rl/experiments/fetch_bybit.py --dataset bybit_solusdt_perp 2024-01-15
"""
import csv
import gzip
import http.client
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import orjson

BOOK_URL = "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{date}_{symbol}_ob500.data.zip"
TRADES_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"
DEFAULT_DATASET = "bybit_btcusdt_perp"
LEVELS = 500
INTERVAL_MS = 30_000
RAW_DIR = Path("TradeData/BybitRaw")


def _download(url: str, target: Path, attempts: int = 6) -> Path:
    """Download to target, resuming a partial .part file after a dropped connection."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resumed = have and r.status == 206  # 200: server ignored Range
                # full size: Content-Range "bytes a-b/total" on 206, Content-Length on 200
                total = int(r.headers["Content-Range"].rsplit("/", 1)[1]) if resumed \
                    else int(r.headers["Content-Length"])
                with open(tmp, "ab" if resumed else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            # a connection can also close cleanly mid-file; only the full size counts as done
            if tmp.stat().st_size != total:
                raise OSError(f"incomplete: {tmp.stat().st_size} of {total} bytes")
            tmp.rename(target)
            return target
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            if attempt == attempts - 1:
                raise
            print(f"retry {attempt + 1} for {url}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
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
    tmp = out_csv.with_name(out_csv.name + ".part")
    with zipfile.ZipFile(raw_zip) as z, gzip.open(tmp, "wt", newline="") as fout:
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
    tmp = out_csv.with_name(out_csv.name + ".part")
    with gzip.open(raw_gz, "rt", newline="") as fin, gzip.open(tmp, "wt", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["transact_time", "price", "quantity", "is_buyer_maker"])
        for r in reader:
            writer.writerow([int(round(float(r["timestamp"]) * 1000)), r["price"], r["size"],
                             "true" if r["side"] == "Sell" else "false"])
            rows += 1
    tmp.rename(out_csv)
    return rows


def fetch(date: str, keep_raw: bool = False, dataset: str = DEFAULT_DATASET) -> None:
    from market_data import DATASETS
    ds = DATASETS[dataset]
    assert ds.venue == "bybit", f"{dataset} is not a Bybit dataset"
    sym = ds.symbol
    book_out, trades_out = Path(ds.book_path(date)), Path(ds.trades_path(date))
    book_out.parent.mkdir(parents=True, exist_ok=True)
    if not trades_out.exists():
        raw = _download(TRADES_URL.format(symbol=sym, date=date), RAW_DIR / f"{sym}{date}.csv.gz")
        n = convert_trades(raw, trades_out)
        print(f"{sym} {date}: {n:,} trades", flush=True)
        if not keep_raw:
            raw.unlink()
    if not book_out.exists():
        raw = _download(BOOK_URL.format(symbol=sym, date=date), RAW_DIR / f"{date}_{sym}_ob500.data.zip")
        n = convert_book(raw, book_out, date)
        print(f"{sym} {date}: {n} book snapshots", flush=True)
        if not keep_raw:
            raw.unlink()


def main():
    from evaluation.dates import SEVERITY_RANK_V2
    args = sys.argv[1:]
    dataset = DEFAULT_DATASET
    if args[:1] == ["--dataset"]:
        dataset, args = args[1], args[2:]
    for d in args or sorted(SEVERITY_RANK_V2):
        fetch(d, dataset=dataset)


if __name__ == "__main__":
    main()
