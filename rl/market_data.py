"""
Where each market dataset's files live (era 2; docs/history/era-1-binance-bookdepth.md).

Every consumer asks this module for a date's order-book and trade files instead
of hard-coding a venue's paths, so changing venue or asset is a registry entry,
not a code search. The active dataset comes from STRATAEXEC_DATASET (default
below).

Files are in the simulator's venue-neutral formats (plain or .gz CSV; every reader,
pandas and the Rust simulator, decompresses .gz by file name):
  book   -- one row per snapshot: timestamp (unix ms), b0_p, b0_q, ..., a0_p, a0_q, ...
  trades -- transact_time (unix ms), price, quantity, is_buyer_maker
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    name: str
    venue: str
    symbol: str
    root: str
    book_pattern: str
    trades_pattern: str
    # Daily volume per calendar day (date, volume_btc), for ADV-relative sizing.
    daily_volume_file: str = "daily-volume.csv"

    def book_path(self, date: str) -> str:
        return os.path.join(self.root, self.book_pattern.format(date=date))

    def trades_path(self, date: str) -> str:
        return os.path.join(self.root, self.trades_pattern.format(date=date))

    def daily_volume_path(self) -> str:
        return os.path.join(self.root, self.daily_volume_file)

    def adv_table_path(self) -> str:
        """Committed 20-day ADV table for the research dates (no raw data needed)."""
        return f"rl/results/adv20_{self.name}.csv"


DATASETS = {
    # Era 2: Bybit's real 500-level L2 book, sampled every 30 s, and Bybit trades.
    "bybit_btcusdt_perp": Dataset(
        name="bybit_btcusdt_perp", venue="bybit", symbol="BTCUSDT",
        root="TradeData/Bybit/BTCUSDT",
        book_pattern="book-{date}.csv.gz", trades_pattern="trades-{date}.csv.gz"),
    # Less liquid Bybit perpetuals, same format: do the BTC findings (docs 26-32) carry over?
    "bybit_solusdt_perp": Dataset(
        name="bybit_solusdt_perp", venue="bybit", symbol="SOLUSDT",
        root="TradeData/Bybit/SOLUSDT",
        book_pattern="book-{date}.csv.gz", trades_pattern="trades-{date}.csv.gz"),
    "bybit_dogeusdt_perp": Dataset(
        name="bybit_dogeusdt_perp", venue="bybit", symbol="DOGEUSDT",
        root="TradeData/Bybit/DOGEUSDT",
        book_pattern="book-{date}.csv.gz", trades_pattern="trades-{date}.csv.gz"),
    # Era 1 (retired, data deleted 2026-10-26): Binance bookDepth bands + aggTrades.
    "binance_btcusdt_perp_bookdepth": Dataset(
        name="binance_btcusdt_perp_bookdepth", venue="binance", symbol="BTCUSDT",
        root="TradeData",
        book_pattern="BookDepth/BTCUSDT-bookDepth-{date}.csv",
        trades_pattern="AggTrades/BTCUSDT-aggTrades-{date}.csv"),
}
DEFAULT_DATASET = "bybit_btcusdt_perp"


def dataset() -> Dataset:
    name = os.environ.get("STRATAEXEC_DATASET", DEFAULT_DATASET)
    if name not in DATASETS:
        raise ValueError(f"unknown STRATAEXEC_DATASET {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def book_path(date: str) -> str:
    return dataset().book_path(date)


def trades_path(date: str) -> str:
    return dataset().trades_path(date)


def has_date(date: str) -> bool:
    return os.path.exists(book_path(date)) and os.path.exists(trades_path(date))


def results_tag() -> str:
    """Suffix for result files: '' for the default dataset (BTC outputs keep their names),
    else '_' + the lower-case symbol, e.g. '_solusdt' (doc 34)."""
    d = dataset()
    return "" if d.name == DEFAULT_DATASET else "_" + d.symbol.lower()


def use_dataset_from_argv(argv: list) -> list:
    """Remove '--dataset NAME' from argv, make NAME the active dataset (inherited by worker
    processes) and return the remaining arguments."""
    args = list(argv)
    if "--dataset" in args:
        i = args.index("--dataset")
        if i + 1 >= len(args):
            raise ValueError("--dataset needs a name")
        name = args[i + 1]
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
        os.environ["STRATAEXEC_DATASET"] = name
        del args[i:i + 2]
    return args
