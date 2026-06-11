"""
LOB CSV validation and preview generation.
Handles BookDepth and AggTrades CSV formats from Binance.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd

# Expected column prefixes for BookDepth CSV (Binance format)
LOB_REQUIRED_COLS = {"timestamp_ms", "price_bid_1", "qty_bid_1", "price_ask_1", "qty_ask_1"}
AGG_REQUIRED_COLS = {"timestamp_ms", "price", "quantity_btc", "is_sell_aggressor"}


class FileValidationError(Exception):
    pass


def validate_lob_csv(file_bytes: bytes) -> tuple[int, dict[str, Any]]:
    """
    Validate a BookDepth CSV and return (row_count, preview_dict).
    Raises FileValidationError on bad format.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), nrows=5000)
    except Exception as e:
        raise FileValidationError(f"Could not parse CSV: {e}")

    # Normalize column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = LOB_REQUIRED_COLS - set(df.columns)
    if missing:
        raise FileValidationError(
            f"BookDepth CSV missing required columns: {missing}. "
            f"Found: {list(df.columns[:8])}"
        )

    # Compute preview from first row
    first = df.iloc[0]
    try:
        best_bid = float(first["price_bid_1"])
        best_ask = float(first["price_ask_1"])
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / mid_price * 10_000
    except Exception as e:
        raise FileValidationError(f"Could not compute preview stats: {e}")

    n_rows = len(df)

    # Try to get timestamp range
    ts_first = ts_last = None
    if "timestamp_ms" in df.columns:
        try:
            ts_first = str(pd.to_datetime(df["timestamp_ms"].iloc[0], unit="ms"))
            ts_last = str(pd.to_datetime(df["timestamp_ms"].iloc[-1], unit="ms"))
        except Exception:
            pass

    preview = {
        "mid_price": round(mid_price, 4),
        "best_bid": round(best_bid, 4),
        "best_ask": round(best_ask, 4),
        "spread_bps": round(spread_bps, 4),
        "timestamp_first": ts_first,
        "timestamp_last": ts_last,
    }
    return n_rows, preview


def validate_agg_trades_csv(file_bytes: bytes) -> tuple[int, dict[str, Any]]:
    """
    Validate an AggTrades CSV and return (row_count, preview_dict).
    Raises FileValidationError on bad format.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), nrows=5000)
    except Exception as e:
        raise FileValidationError(f"Could not parse CSV: {e}")

    df.columns = [c.strip().lower() for c in df.columns]

    missing = AGG_REQUIRED_COLS - set(df.columns)
    if missing:
        raise FileValidationError(
            f"AggTrades CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    n_rows = len(df)
    ts_first = ts_last = None
    try:
        ts_first = str(pd.to_datetime(df["timestamp_ms"].iloc[0], unit="ms"))
        ts_last = str(pd.to_datetime(df["timestamp_ms"].iloc[-1], unit="ms"))
    except Exception:
        pass

    preview = {
        "n_trades": n_rows,
        "timestamp_first": ts_first,
        "timestamp_last": ts_last,
    }
    return n_rows, preview


def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYY-MM-DD from Binance-style filename like BTCUSDT-bookDepth-2024-01-15.csv"""
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def get_lob_depth_preview(file_path: str | Path, n_snapshots: int = 50) -> list[dict]:
    """
    Read first N snapshots from a BookDepth CSV for the order book visualization.
    Returns list of {step, bid_levels: [{price, qty}], ask_levels: [{price, qty}]}.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path, nrows=n_snapshots)
        df.columns = [c.strip().lower() for c in df.columns]
        result = []
        for i, row in df.iterrows():
            bid_levels = []
            ask_levels = []
            for level in range(1, 6):
                bp_col = f"price_bid_{level}"
                bq_col = f"qty_bid_{level}"
                ap_col = f"price_ask_{level}"
                aq_col = f"qty_ask_{level}"
                if bp_col in row and bq_col in row:
                    bid_levels.append({
                        "price": float(row[bp_col]) if pd.notna(row[bp_col]) else None,
                        "qty": float(row[bq_col]) if pd.notna(row[bq_col]) else None,
                    })
                if ap_col in row and aq_col in row:
                    ask_levels.append({
                        "price": float(row[ap_col]) if pd.notna(row[ap_col]) else None,
                        "qty": float(row[aq_col]) if pd.notna(row[aq_col]) else None,
                    })
            result.append({
                "step": i,
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
            })
        return result
    except Exception:
        return []


def get_spread_series(file_path: str | Path, every_n: int = 10) -> list[dict]:
    """
    Compute mid price and spread series for the preview chart.
    Decimated by every_n to keep payload small.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.iloc[::every_n].reset_index(drop=True)
        result = []
        for i, row in df.iterrows():
            try:
                bid = float(row["price_bid_1"])
                ask = float(row["price_ask_1"])
                mid = (bid + ask) / 2
                spread_bps = (ask - bid) / mid * 10_000
                result.append({
                    "step": i,
                    "mid_price": round(mid, 2),
                    "spread_bps": round(spread_bps, 4),
                })
            except Exception:
                continue
        return result
    except Exception:
        return []
