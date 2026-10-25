"""
Where each market dataset's files live (era 2; docs/history/era-1-binance-bookdepth.md).

Every consumer asks this module for a date's order-book and trade files instead
of hard-coding a venue's paths, so changing venue or asset is a registry entry,
not a code search. The active dataset comes from STRATAEXEC_DATASET (default
below).

Files are in the simulator's venue-neutral formats:
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

    def book_path(self, date: str) -> str:
        return os.path.join(self.root, self.book_pattern.format(date=date))

    def trades_path(self, date: str) -> str:
        return os.path.join(self.root, self.trades_pattern.format(date=date))


DATASETS = {
    # Era 2: Bybit's real 500-level L2 book, sampled every 30 s, and Bybit trades.
    "bybit_btcusdt_perp": Dataset(
        name="bybit_btcusdt_perp", venue="bybit", symbol="BTCUSDT",
        root="TradeData/Bybit/BTCUSDT",
        book_pattern="book-{date}.csv", trades_pattern="trades-{date}.csv"),
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
