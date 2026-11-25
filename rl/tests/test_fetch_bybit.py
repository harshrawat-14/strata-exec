"""
Downloader regressions (doc 30, 2026-11-24), against a local HTTP server that misbehaves
the way Bybit's did:

- connections that drop mid-file (2 days failed: no retry);
- connections that close cleanly mid-file. http.client then returns EOF instead of raising,
  so a stream that simply ended was saved as a finished file (two half-length zips).
"""
import gzip
import importlib.util
import io
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import orjson
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_bybit", Path(__file__).resolve().parents[1] / "experiments" / "fetch_bybit.py")
fb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fb)


class _Server:
    """Serves `files` (path -> bytes). Each path's first `faults[path]` responses stop after
    half the remaining body and close cleanly; Range requests get 206 unless ignore_range."""

    def __init__(self, files, faults=None, ignore_range=False):
        self.files, self.faults, self.ignore_range = files, dict(faults or {}), ignore_range
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                outer.requests.append((self.path, rng))
                start = int(rng.split("=")[1].rstrip("-")) if rng and not outer.ignore_range else 0
                part = body[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(part)))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                self.end_headers()
                if outer.faults.get(self.path, 0) > 0:
                    outer.faults[self.path] -= 1
                    self.wfile.write(part[: len(part) // 2])
                    self.close_connection = True  # clean close, short of Content-Length
                    return
                self.wfile.write(part)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(fb.time, "sleep", lambda s: None)


BODY = os.urandom(200_000)


# ── unit: _download ──────────────────────────────────────────────────────────

def test_clean_close_mid_file_is_resumed_not_saved_as_complete(tmp_path, no_sleep):
    srv = _Server({"/f": BODY}, faults={"/f": 2})
    try:
        out = fb._download(srv.url + "/f", tmp_path / "f.zip")
    finally:
        srv.close()
    assert out.read_bytes() == BODY
    assert not (tmp_path / "f.zip.part").exists()
    # resumed from the partial file each time, not restarted
    assert [r for _, r in srv.requests][0] is None
    assert all(r and r.startswith("bytes=") for _, r in srv.requests[1:])
    assert len(srv.requests) == 3


def test_server_ignoring_range_restarts_the_file(tmp_path, no_sleep):
    srv = _Server({"/f": BODY}, faults={"/f": 1}, ignore_range=True)
    try:
        out = fb._download(srv.url + "/f", tmp_path / "f.zip")
    finally:
        srv.close()
    assert out.read_bytes() == BODY  # 200 on a resume overwrites; no duplicated prefix


def test_persistent_truncation_raises_and_leaves_no_final_file(tmp_path, no_sleep):
    srv = _Server({"/f": BODY}, faults={"/f": 99})
    try:
        with pytest.raises(OSError, match="incomplete"):
            fb._download(srv.url + "/f", tmp_path / "f.zip", attempts=3)
    finally:
        srv.close()
    assert not (tmp_path / "f.zip").exists()


def test_existing_partial_from_an_earlier_run_is_resumed(tmp_path, no_sleep):
    (tmp_path / "f.zip.part").write_bytes(BODY[:12_345])
    srv = _Server({"/f": BODY})
    try:
        out = fb._download(srv.url + "/f", tmp_path / "f.zip")
    finally:
        srv.close()
    assert out.read_bytes() == BODY
    assert srv.requests == [("/f", "bytes=12345-")]


# ── unit: incomplete raw files fail loudly at conversion ────────────────────

DATE = "2024-01-15"


def _book_zip(date=DATE):
    t0 = fb._day_start_ms(date)
    lines = [orjson.dumps({"type": "snapshot", "ts": t0 + 1_000,
                           "data": {"b": [["100.0", "1.5"], ["99.5", "2"]], "a": [["100.5", "1"]]}})]
    for k in range(1, 6):
        lines.append(orjson.dumps({"type": "delta", "ts": t0 + k * 45_000,
                                   "data": {"b": [["100.0", str(1 + k)]], "a": [["100.5", "0"], ["101", "3"]]}}))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{date}_BTCUSDT_ob500.data", b"\n".join(lines) + b"\n")
    return buf.getvalue()


def _trades_gz(date=DATE):
    t0 = fb._day_start_ms(date) / 1000
    rows = "timestamp,symbol,side,size,price\n" + "".join(
        f"{t0 + i * 7.5},BTCUSDT,{'Sell' if i % 3 else 'Buy'},0.0{i + 1},{100 + i}\n" for i in range(400))
    return gzip.compress(rows.encode())


def test_truncated_book_zip_fails_conversion(tmp_path):
    raw = tmp_path / "b.zip"
    z = _book_zip()
    raw.write_bytes(z[: len(z) // 2])
    with pytest.raises(zipfile.BadZipFile):
        fb.convert_book(raw, tmp_path / "book.csv.gz", DATE)
    assert not (tmp_path / "book.csv.gz").exists()


def test_truncated_trades_gz_fails_conversion(tmp_path):
    raw = tmp_path / "t.csv.gz"
    g = _trades_gz()
    raw.write_bytes(g[: len(g) // 2])
    with pytest.raises(EOFError):
        fb.convert_trades(raw, tmp_path / "trades.csv.gz")
    assert not (tmp_path / "trades.csv.gz").exists()


# ── integration: fetch() through a flaky server matches a clean fetch ───────

def _fetch_via(server_faults, tmp_path, monkeypatch, sub):
    files = {f"/book/{DATE}.zip": _book_zip(), f"/trades/{DATE}.csv.gz": _trades_gz()}
    srv = _Server(files, faults=server_faults)
    work = tmp_path / sub
    work.mkdir()
    monkeypatch.chdir(work)  # dataset root and RAW_DIR are relative paths
    monkeypatch.setattr(fb, "BOOK_URL", srv.url + "/book/{date}.zip")
    monkeypatch.setattr(fb, "TRADES_URL", srv.url + "/trades/{date}.csv.gz")
    try:
        fb.fetch(DATE)
    finally:
        srv.close()
    root = work / "TradeData/Bybit/BTCUSDT"
    assert not any((work / "TradeData/BybitRaw").iterdir())  # raw deleted after conversion
    return (pd.read_csv(root / f"book-{DATE}.csv.gz"), pd.read_csv(root / f"trades-{DATE}.csv.gz"))


def test_fetch_through_flaky_connections_equals_clean_fetch(tmp_path, monkeypatch, no_sleep):
    clean_book, clean_trades = _fetch_via({}, tmp_path, monkeypatch, "clean")
    flaky_book, flaky_trades = _fetch_via(
        {f"/book/{DATE}.zip": 2, f"/trades/{DATE}.csv.gz": 2}, tmp_path, monkeypatch, "flaky")
    pd.testing.assert_frame_equal(clean_book, flaky_book)
    pd.testing.assert_frame_equal(clean_trades, flaky_trades)
    assert len(clean_book) == 2879  # first snapshot at 00:00:01: boundaries 00:00:30 .. 23:59:30
    assert len(clean_trades) == 400


def test_fetch_other_symbol_uses_its_urls_and_dataset_root(tmp_path, monkeypatch, no_sleep):
    files = {f"/SOLUSDT/book/{DATE}.zip": _book_zip(), f"/SOLUSDT/trades/{DATE}.csv.gz": _trades_gz()}
    srv = _Server(files)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fb, "BOOK_URL", srv.url + "/{symbol}/book/{date}.zip")
    monkeypatch.setattr(fb, "TRADES_URL", srv.url + "/{symbol}/trades/{date}.csv.gz")
    try:
        fb.fetch(DATE, dataset="bybit_solusdt_perp")
    finally:
        srv.close()
    root = tmp_path / "TradeData/Bybit/SOLUSDT"
    assert (root / f"book-{DATE}.csv.gz").exists() and (root / f"trades-{DATE}.csv.gz").exists()
    assert not (tmp_path / "TradeData/Bybit/BTCUSDT").exists()
