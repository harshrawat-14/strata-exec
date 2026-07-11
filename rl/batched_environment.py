"""
Python client for rl-env's batched mode (Phase S, ADR-006).

Speaks Arrow RecordBatch IPC over a Unix domain socket to a single rl-env
process running N environment instances (--batch-size N --socket-path PATH),
instead of N separate JSON-over-stdio subprocesses. See
docs/system_design/ADR.md#adr-006 for the transport decision and
docs/system_design/phase_s_baseline.md for the measurement that motivated it.

This is additive: StrataExecEnv (environment.py) and its JSON-over-stdio
protocol are unchanged and still used by every existing single-episode call
site (rl/evaluate.py etc.), where the transport was never the bottleneck.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


class BatchedStrataExecEnv:
    """N environment instances stepped together in one rl-env process."""

    def __init__(
        self,
        batch_size: int,
        mode: str = "synthetic",
        lob_date: Optional[str] = None,
        agg_date: Optional[str] = None,
        total_inventory: float = 1_000_000.0,
        total_steps: int = 2880,
        sigma_ref: float = 0.20,
        gamma: float = 0.005,
        btc_target: float = 50_000.0,
        fixed_steps: bool = True,
        fixed_size: bool = True,
        no_time_features: bool = True,
        connect_timeout_s: float = 5.0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.batch_size = batch_size
        self.n_state_dims = 12 if no_time_features else 16

        if lob_date:
            lob_path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv"
            if not os.path.exists(lob_path):
                raise FileNotFoundError(f"Historical LOB data file not found: {lob_path}")
        if agg_date:
            agg_path = f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv"
            if not os.path.exists(agg_path):
                raise FileNotFoundError(f"AggTrades data file not found: {agg_path}")

        self.socket_path = os.path.join(
            tempfile.gettempdir(),
            f"rl_env_batch_{os.getpid()}_{id(self)}.sock",
        )
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        cmd = [
            "./target/release/rl-env",
            "--rl-mode", mode,
            "--batch-size", str(batch_size),
            "--socket-path", self.socket_path,
            "--btc-target", str(btc_target),
        ]
        if lob_date:
            cmd += ["--lob-data", f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv"]
        if agg_date:
            cmd += ["--agg-trades", f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv"]
        if fixed_steps:
            cmd.append("--fixed-steps")
        if fixed_size:
            cmd.append("--fixed-size")
        if no_time_features:
            cmd.append("--no-time-features")

        self.process = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)

        self.sock: Optional[socket.socket] = None
        deadline = time.monotonic() + connect_timeout_s
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.socket_path)
                self.sock = s
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.02)
        if self.sock is None:
            self.process.terminate()
            raise RuntimeError(
                f"failed to connect to rl-env batch server at {self.socket_path} "
                f"within {connect_timeout_s}s"
            )

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("rl-env batch server closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    def _send_and_recv(self, table: pa.Table) -> pa.RecordBatch:
        # Length-prefixed framing (4-byte little-endian length + exactly that
        # many raw Arrow-IPC-stream bytes), matching the Rust server
        # (run_batch_server in rl_env.rs). Necessary because pyarrow's
        # high-level stream reader reads ahead past one message's EOS marker
        # when given a live socket shared across multiple request/response
        # exchanges, corrupting the second and subsequent messages -- Arrow's
        # stream format assumes "the whole file/stream is one Arrow stream,"
        # not independent messages interleaved on one persistent connection.
        # Serializing to an in-memory buffer first and framing it explicitly
        # sidesteps that: pyarrow parses a bounded byte range with nothing
        # after it, so there is nothing to over-read.
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()
        payload = sink.getvalue().to_pybytes()

        self.sock.sendall(struct.pack("<I", len(payload)) + payload)

        resp_len = struct.unpack("<I", self._recv_exact(4))[0]
        resp_bytes = self._recv_exact(resp_len)

        reader = ipc.open_stream(pa.BufferReader(resp_bytes))
        batch = reader.read_next_batch()
        return batch

    def _extract_states(self, batch: pa.RecordBatch) -> np.ndarray:
        # Zero-copy buffer views, not a per-element parse -- the point of
        # the Arrow transport (ADR-006).
        cols = [
            batch.column(f"s{i}").to_numpy(zero_copy_only=True)
            for i in range(self.n_state_dims)
        ]
        return np.stack(cols, axis=1).astype(np.float32)

    def reset_batch(self, seeds: list[int]) -> np.ndarray:
        if len(seeds) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} seeds, got {len(seeds)}")
        table = pa.table({"seed": pa.array(seeds, type=pa.uint64())})
        batch = self._send_and_recv(table)
        return self._extract_states(batch)

    def step_batch(self, actions: list[int]):
        if len(actions) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} actions, got {len(actions)}")
        table = pa.table({"action": pa.array(actions, type=pa.int32())})
        batch = self._send_and_recv(table)
        states = self._extract_states(batch)
        rewards = batch.column("reward").to_numpy(zero_copy_only=True)
        dones = batch.column("done").to_numpy(zero_copy_only=False)
        return states, rewards, dones

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
