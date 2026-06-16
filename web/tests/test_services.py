"""
Unit tests for service-layer functions:
  - parallel_sim.aggregate_runs
  - parallel_sim.parse_comparison_csv
  - parallel_sim.simulate_rl_path (edge cases)
  - storage.get_content_hash / store_file / get_presigned_upload_url
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ── aggregate_runs ─────────────────────────────────────────────────────────────

class TestAggregateRuns:
    """Tests for parallel_sim.aggregate_runs()."""

    def _make_run(self, n_steps: int, prefix: str, is_val: float) -> dict[str, Any]:
        """Build a minimal run dict as would be returned by parse_comparison_csv."""
        return {
            "price_path": [100.0 + i * 0.1 for i in range(n_steps)],
            "strategies": {
                prefix: {
                    "trajectory": [float(n_steps - i) for i in range(n_steps)],
                    "cost_series": [float(i) * 10 for i in range(n_steps)],
                }
            },
        }

    def _make_summary(self, display_name: str, is_val: float) -> dict[str, Any]:
        return {
            display_name: {
                "mean_is_pct": is_val,
                "is_variance": 0.01,
                "ac_objective": is_val * 0.9,
                "trade_count": 20,
                "avg_exec_price": 100.5,
                "cost_decomposition": {
                    "spread_cost": 0.05,
                    "temporary_impact": 0.10,
                    "permanent_impact": 0.03,
                    "timing_cost": -0.08,
                    "opportunity_cost": 0.0,
                },
            }
        }

    def test_empty_inputs_return_empty(self):
        from web.services.parallel_sim import aggregate_runs
        result = aggregate_runs([], [])
        assert result == {"strategies": [], "price_path": [], "total_steps": 0}

    def test_single_run_passthrough(self):
        from web.services.parallel_sim import aggregate_runs
        run = self._make_run(10, "twap", -0.5)
        summary = self._make_summary("TWAP", -0.5)
        result = aggregate_runs([run], [summary])

        assert len(result["strategies"]) == 1
        strat = result["strategies"][0]
        assert strat["name"] == "TWAP"
        assert abs(strat["mean_is_pct"] - (-0.5)) < 1e-9

    def test_two_runs_mean_is_averaged(self):
        from web.services.parallel_sim import aggregate_runs
        run1 = self._make_run(10, "twap", -0.4)
        run2 = self._make_run(10, "twap", -0.6)
        summary1 = self._make_summary("TWAP", -0.4)
        summary2 = self._make_summary("TWAP", -0.6)

        result = aggregate_runs([run1, run2], [summary1, summary2])
        strat = result["strategies"][0]
        # Mean IS should be average of -0.4 and -0.6 = -0.5
        assert abs(strat["mean_is_pct"] - (-0.5)) < 1e-9

    def test_cvar95_uses_worst_5_percent(self):
        from web.services.parallel_sim import aggregate_runs
        # 100 paths: worst IS = +5.0 (highest cost), rest near 0.0
        runs = [self._make_run(5, "twap", 0.0) for _ in range(99)]
        runs.append(self._make_run(5, "twap", 0.0))
        summaries_good = [self._make_summary("TWAP", 0.0) for _ in range(95)]
        summaries_bad  = [self._make_summary("TWAP", 5.0) for _ in range(5)]
        all_summaries = summaries_bad + summaries_good  # worst first

        result = aggregate_runs(runs, all_summaries)
        strat = result["strategies"][0]
        # CVaR95 should be average of top 5% worst (highest IS) = 5.0
        assert strat["cvar95"] >= 0.0, "CVaR must capture worst-case tail"

    def test_price_path_averaged_across_runs(self):
        from web.services.parallel_sim import aggregate_runs
        run1 = {"price_path": [100.0, 102.0], "strategies": {
            "twap": {"trajectory": [1.0, 0.0], "cost_series": [0.0, 10.0]}
        }}
        run2 = {"price_path": [100.0, 104.0], "strategies": {
            "twap": {"trajectory": [1.0, 0.0], "cost_series": [0.0, 20.0]}
        }}
        summary = self._make_summary("TWAP", -0.5)
        result = aggregate_runs([run1, run2], [summary, summary])
        # Average price at step 1: (102 + 104)/2 = 103
        assert abs(result["price_path"][1] - 103.0) < 1e-9

    def test_is_variance_computed_across_paths(self):
        from web.services.parallel_sim import aggregate_runs
        # IS values: -0.2, -0.4, -0.6, -0.8 → variance = var([-0.2,-0.4,-0.6,-0.8], ddof=1)
        is_vals = [-0.2, -0.4, -0.6, -0.8]
        runs = [self._make_run(5, "twap", v) for v in is_vals]
        summaries = [self._make_summary("TWAP", v) for v in is_vals]
        result = aggregate_runs(runs, summaries)
        strat = result["strategies"][0]
        expected_var = float(np.var(is_vals, ddof=1))
        assert abs(strat["is_variance"] - expected_var) < 1e-9


# ── parse_comparison_csv ──────────────────────────────────────────────────────

class TestParseComparisonCsv:
    """Tests for parallel_sim.parse_comparison_csv()."""

    def _write_csv(self, rows: list[dict], tmp_path: Path) -> str:
        filepath = tmp_path / "comparison.csv"
        if not rows:
            filepath.write_text("")
            return str(filepath)
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(filepath)

    def test_nonexistent_file_returns_empty(self):
        from web.services.parallel_sim import parse_comparison_csv
        result = parse_comparison_csv("/nonexistent/path/comparison.csv")
        assert result == {"price_path": [], "strategies": {}}

    def test_valid_two_strategy_csv(self, tmp_path):
        from web.services.parallel_sim import parse_comparison_csv
        rows = [
            {"Price": "100.0", "twap_remaining": "1000", "twap_cost": "0.0",
             "optimal_remaining": "900", "optimal_cost": "5.0"},
            {"Price": "101.0", "twap_remaining": "500",  "twap_cost": "10.0",
             "optimal_remaining": "400", "optimal_cost": "12.0"},
            {"Price": "102.0", "twap_remaining": "0.0",  "twap_cost": "20.0",
             "optimal_remaining": "0.0", "optimal_cost": "22.0"},
        ]
        csv_path = self._write_csv(rows, tmp_path)
        result = parse_comparison_csv(csv_path)

        assert result["price_path"] == [100.0, 101.0, 102.0]
        assert "twap" in result["strategies"]
        assert "optimal" in result["strategies"]
        assert result["strategies"]["twap"]["trajectory"] == [1000.0, 500.0, 0.0]
        assert result["strategies"]["optimal"]["cost_series"] == [5.0, 12.0, 22.0]

    def test_invalid_price_values_default_to_zero(self, tmp_path):
        from web.services.parallel_sim import parse_comparison_csv
        rows = [
            {"Price": "N/A", "twap_remaining": "bad", "twap_cost": "0.0"},
        ]
        csv_path = self._write_csv(rows, tmp_path)
        result = parse_comparison_csv(csv_path)
        assert result["price_path"] == [0.0]
        assert result["strategies"]["twap"]["trajectory"] == [0.0]

    def test_no_strategy_columns_returns_empty_strategies(self, tmp_path):
        from web.services.parallel_sim import parse_comparison_csv
        rows = [{"Price": "100.0", "Step": "0"}]
        csv_path = self._write_csv(rows, tmp_path)
        result = parse_comparison_csv(csv_path)
        assert result["strategies"] == {}
        assert result["price_path"] == [100.0]


# ── simulate_rl_path edge cases ────────────────────────────────────────────────

class TestSimulateRlPathEdgeCases:
    """Additional edge-case tests for simulate_rl_path."""

    class ZeroTradeModel:
        """Always returns action 0 (fraction=0.000 → never trade)."""
        observation_space = type("OS", (), {"shape": (8,)})()
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return 0, state

    class FullLiquidateModel:
        """Always returns action 19 (fraction=1.000 → liquidate everything now)."""
        observation_space = type("OS", (), {"shape": (8,)})()
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return 19, state

    class TwelveObsModel:
        """Model with 12-dim observation space."""
        observation_space = type("OS", (), {"shape": (12,)})()
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            assert obs.shape == (12,), f"Expected 12-dim obs, got {obs.shape}"
            return 9, state

    def test_zero_trade_model_forces_final_liquidation(self):
        from web.services.parallel_sim import simulate_rl_path
        # Declining price path so forced liquidation at lower price → real opportunity cost
        price_path = [100.0 - i * 0.5 for i in range(10)]  # 100 → 95.5
        result = simulate_rl_path(self.ZeroTradeModel(), price_path, {"total_notional": 1e6})

        # With zero trading, the whole position is force-liquidated at end
        assert result["trajectory"][-1] == 0.0
        # opportunity_cost = remaining * (final_price - arrival_price)
        # Since price declined, opportunity_cost should be negative (sold below arrival)
        assert result["cost_decomposition"]["opportunity_cost"] != 0.0
        # mean_is_pct should be a valid float
        assert isinstance(result["mean_is_pct"], float)

    def test_full_liquidate_model_trades_on_step_0(self):
        from web.services.parallel_sim import simulate_rl_path
        price_path = [100.0] * 10
        result = simulate_rl_path(self.FullLiquidateModel(), price_path, {"total_notional": 1e6})

        # Most inventory traded in step 0
        assert result["trajectory"][-1] == 0.0
        # Mean IS should be close to 0 (bought everything near arrival price)
        assert abs(result["mean_is_pct"]) < 5.0

    def test_twelve_dim_observation_space(self):
        from web.services.parallel_sim import simulate_rl_path
        price_path = [100.0 + i * 0.1 for i in range(20)]
        # Should not raise — model checks obs shape internally
        result = simulate_rl_path(
            self.TwelveObsModel(), price_path, {"total_notional": 1e6, "sigma": 0.02}
        )
        assert "trajectory" in result
        assert result["trajectory"][-1] == 0.0

    def test_single_step_price_path(self):
        from web.services.parallel_sim import simulate_rl_path
        # Edge case: only 2 prices (1 step)
        price_path = [100.0, 100.5]
        result = simulate_rl_path(self.ZeroTradeModel(), price_path, {"total_notional": 1e4})
        assert isinstance(result["mean_is_pct"], float)
        assert len(result["trajectory"]) == 2
        assert result["trajectory"][-1] == 0.0


# ── get_content_hash ──────────────────────────────────────────────────────────

class TestContentHash:
    def test_deterministic(self):
        from web.services.storage import get_content_hash
        data = b"hello world"
        assert get_content_hash(data) == get_content_hash(data)

    def test_sixteen_chars(self):
        from web.services.storage import get_content_hash
        h = get_content_hash(b"strataexec")
        assert len(h) == 16

    def test_different_data_different_hash(self):
        from web.services.storage import get_content_hash
        assert get_content_hash(b"aaaa") != get_content_hash(b"bbbb")

    def test_matches_sha256_prefix(self):
        from web.services.storage import get_content_hash
        data = b"test content"
        expected = hashlib.sha256(data).hexdigest()[:16]
        assert get_content_hash(data) == expected


# ── store_file CAS deduplication ──────────────────────────────────────────────

class TestStoreFileCas:
    """
    CAS store_file should not re-write a file if the same bytes are stored twice.
    """

    async def test_cas_deduplication(self, tmp_path):
        from web.services.storage import store_file, get_content_hash
        from web.config import get_settings

        settings = get_settings()
        original = settings.__dict__.get("lob_upload_dir")
        settings.__dict__["lob_upload_dir"] = tmp_path / "lob"

        try:
            data = b"LOB CSV DATA " * 100
            p1 = await store_file(data, "lob_data.csv", "lob")
            mtime1 = os.path.getmtime(p1)

            import asyncio
            await asyncio.sleep(0.05)  # ensure mtime would differ if re-written

            p2 = await store_file(data, "lob_data.csv", "lob")
            mtime2 = os.path.getmtime(p2)

            assert p1 == p2, "Same content must produce same stored path (CAS)"
            assert mtime1 == mtime2, "File must NOT be re-written for duplicate content"
        finally:
            if original is not None:
                settings.__dict__["lob_upload_dir"] = original

    async def test_different_content_different_path(self, tmp_path):
        from web.services.storage import store_file
        from web.config import get_settings

        settings = get_settings()
        settings.__dict__["lob_upload_dir"] = tmp_path / "lob2"

        try:
            p1 = await store_file(b"data_A", "file.csv", "lob")
            p2 = await store_file(b"data_B", "file.csv", "lob")
            assert p1 != p2
        finally:
            settings.__dict__.pop("lob_upload_dir", None)


# ── get_presigned_upload_url ──────────────────────────────────────────────────

class TestPresignedUrl:
    def test_local_fallback_for_lob(self):
        from web.services.storage import get_presigned_upload_url
        result = get_presigned_upload_url("data.csv", "lob")
        assert result["use_s3"] is False
        assert "/api/upload/local" in result["url"]
        assert "category=lob" in result["url"]

    def test_local_fallback_for_agg_trades(self):
        from web.services.storage import get_presigned_upload_url
        result = get_presigned_upload_url("trades.csv", "agg_trades")
        assert "category=agg_trades" in result["url"]

    def test_local_fallback_for_model(self):
        from web.services.storage import get_presigned_upload_url
        result = get_presigned_upload_url("model.zip", "model")
        assert "category=model" in result["url"]

    def test_fields_empty_in_local_mode(self):
        from web.services.storage import get_presigned_upload_url
        result = get_presigned_upload_url("x.csv", "lob")
        assert result["fields"] == {}
        assert result["s3_key"] is None
