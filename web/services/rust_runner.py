"""
Rust binary subprocess management.

Both research-sim and rl-env are invoked via asyncio subprocesses.
This module handles process lifecycle, argument construction, and output parsing.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from web.config import get_settings

settings = get_settings()


class RustRunnerError(Exception):
    pass


class RustRunner:
    """Manages Rust binary subprocesses for simulation and RL evaluation."""

    def __init__(self):
        self.research_sim = str(settings.research_sim_path)
        self.rl_env = str(settings.rl_env_path)
        self._verify_binaries()

    def _verify_binaries(self):
        for name, path in [("research-sim", self.research_sim), ("rl-env", self.rl_env)]:
            if not os.path.isfile(path):
                raise RustRunnerError(
                    f"Binary '{name}' not found at '{path}'. "
                    f"Run 'cargo build --release' first."
                )
            if not os.access(path, os.X_OK):
                raise RustRunnerError(f"Binary '{name}' at '{path}' is not executable.")

    # ── research-sim ──────────────────────────────────────────────────────────

    async def run_simulation(
        self,
        n_paths: int,
        model: str = "gbm",
        lob_path: str | None = None,
        agg_path: str | None = None,
        include_regime_ac: bool = False,
        params: dict[str, Any] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run research-sim and return parsed results dict.
        Calls on_progress(completed_paths, total_paths) as the binary emits progress events.
        """
        cmd = [
            self.research_sim,
            "--paths", str(n_paths),
            "--model", model,
            "--progress",  # emit progress events to stdout
        ]
        if lob_path:
            cmd.extend(["--lob-data", lob_path])
        if agg_path:
            cmd.extend(["--agg-trades", agg_path])
        if include_regime_ac:
            cmd.append("--include-regime-ac")
        if params:
            if "sigma" in params and params["sigma"] is not None:
                cmd.extend(["--sigma", str(params["sigma"])])
            if "eta" in params and params["eta"] is not None:
                cmd.extend(["--eta", str(params["eta"])])
            if "lambda" in params and params["lambda"] is not None:
                cmd.extend(["--lambda", str(params["lambda"])])
            if "total_notional" in params and params["total_notional"] is not None:
                cmd.extend(["--notional", str(params["total_notional"])])
            if "horizon_steps" in params and params["horizon_steps"] is not None:
                cmd.extend(["--horizon", str(params["horizon_steps"])])

        # Run from the project root so relative paths (TradeData/, results/) resolve correctly
        cwd = Path(settings.rust_binary_path).parent.parent  # target/release -> project root
        if not cwd.exists():
            cwd = Path(".")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        # Stream stdout line by line to capture progress events
        assert proc.stdout is not None
        assert proc.stderr is not None

        async def read_stdout():
            async for line in proc.stdout:
                decoded = line.decode(errors="replace").rstrip()
                stdout_lines.append(decoded)
                # Progress events look like: "Progress: 42/500"
                if decoded.startswith("Progress:") and on_progress:
                    try:
                        parts = decoded.split(":")[-1].strip().split("/")
                        completed, total = int(parts[0]), int(parts[1])
                        on_progress(completed, total)
                    except Exception:
                        pass

        async def read_stderr():
            async for line in proc.stderr:
                stderr_lines.append(line.decode(errors="replace").rstrip())

        await asyncio.gather(read_stdout(), read_stderr())
        await proc.wait()

        if proc.returncode != 0:
            err = "\n".join(stderr_lines[-20:])
            raise RustRunnerError(
                f"research-sim exited with code {proc.returncode}.\nStderr:\n{err}"
            )

        # Parse the results CSV written to results/comparison.csv
        results_csv = cwd / "results" / "comparison.csv"
        if not results_csv.exists():
            raise RustRunnerError(
                f"research-sim completed but results/comparison.csv not found at {results_csv}"
            )

        return self._parse_comparison_csv(str(results_csv))

    def _parse_comparison_csv(self, csv_path: str) -> dict[str, Any]:
        """
        Parse results/comparison.csv into a structured dict.
        
        CSV format:
        Step,Price,twap_remaining,twap_cost,heuristic_remaining,heuristic_cost,
             optimal_remaining,optimal_cost,adaptiveoptimal_remaining,adaptiveoptimal_cost
        """
        df_rows: list[dict] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                df_rows.append(row)

        if not df_rows:
            return {"strategies": [], "price_path": []}

        # Detect strategy columns from header
        sample = df_rows[0]
        strategy_map: dict[str, str] = {}  # display_name -> col_prefix
        for key in sample.keys():
            if key.endswith("_remaining") and key != "":
                prefix = key[: -len("_remaining")]
                display = {
                    "twap": "TWAP",
                    "heuristic": "Heuristic",
                    "optimal": "Optimal (AC)",
                    "adaptiveoptimal": "Adaptive Optimal",
                    "regimeac": "RegimeAC",
                }.get(prefix.lower(), prefix)
                strategy_map[display] = prefix

        price_path = []
        trajectories: dict[str, list[float]] = {name: [] for name in strategy_map}
        costs: dict[str, list[float]] = {name: [] for name in strategy_map}

        # Decimate to at most 500 points for frontend performance
        step_count = len(df_rows)
        step = max(1, step_count // 500)

        for i, row in enumerate(df_rows):
            if i % step == 0 or i == step_count - 1:
                try:
                    price_path.append(float(row["Price"]))
                except (KeyError, ValueError):
                    price_path.append(0.0)

                for display_name, prefix in strategy_map.items():
                    try:
                        trajectories[display_name].append(
                            float(row.get(f"{prefix}_remaining", 0))
                        )
                        costs[display_name].append(
                            float(row.get(f"{prefix}_cost", 0))
                        )
                    except (KeyError, ValueError):
                        trajectories[display_name].append(0.0)
                        costs[display_name].append(0.0)

        summary_path = Path(csv_path).parent / "summary.json"
        summary_data = {}
        if summary_path.exists():
            try:
                with open(summary_path) as sf:
                    summary_data = json.load(sf)
            except Exception:
                pass

        def clean_name(s: str) -> str:
            return s.lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "")

        strategies_out = []
        for display_name in strategy_map:
            traj = trajectories[display_name]
            cost_series = costs[display_name]

            # Try to fetch matching statistics from summary_data
            stats = summary_data.get(display_name)
            if not stats:
                for k, v in summary_data.items():
                    ck = clean_name(k)
                    cd = clean_name(display_name)
                    if ck == cd or ck in cd or cd in ck:
                        stats = v
                        break

            mean_is_pct = stats.get("mean_is_pct") if stats else None
            is_variance = stats.get("is_variance") if stats else None
            cvar95 = stats.get("cvar95") if stats else None
            ac_objective = stats.get("ac_objective") if stats else None
            cost_decomp = stats.get("cost_decomposition", {}) if stats else {}
            trade_count = stats.get("trade_count") if stats else None
            avg_exec_price = stats.get("avg_exec_price") if stats else None

            strategies_out.append({
                "name": display_name,
                "trajectory": traj,
                "cost_series": cost_series,
                "mean_is_pct": mean_is_pct,
                "is_variance": is_variance,
                "cvar95": cvar95,
                "ac_objective": ac_objective,
                "cost_decomposition": cost_decomp,
                "trade_count": trade_count,
                "avg_exec_price": avg_exec_price,
            })

        return {
            "strategies": strategies_out,
            "price_path": price_path,
            "total_steps": step_count,
        }

    # ── rl-env ────────────────────────────────────────────────────────────────

    async def start_rl_env(
        self,
        mode: str = "synthetic",
        lob_date: str | None = None,
        agg_date: str | None = None,
        btc_target: float = 50000.0,
        fixed_steps: bool = True,
        fixed_size: bool = True,
        adversarial: bool = False,
    ) -> asyncio.subprocess.Process:
        """
        Launch rl-env process for interactive use (stdin/stdout JSON protocol).
        Returns the process handle — caller is responsible for communicating and terminating.
        """
        cwd = Path(settings.rust_binary_path).parent.parent
        if not cwd.exists():
            cwd = Path(".")

        cmd = [self.rl_env, "--rl-mode", mode]

        if adversarial:
            cmd.append("--adversarial")

        if lob_date and mode in ("historical", "counterfactual"):
            cmd.extend([
                "--lob-data",
                f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv",
                "--btc-target", str(btc_target),
            ])

        if agg_date and mode in ("historical", "counterfactual"):
            cmd.extend([
                "--agg-trades",
                f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv",
            ])

        if fixed_steps:
            cmd.append("--fixed-steps")
        if fixed_size:
            cmd.append("--fixed-size")

        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(cwd),
        )

    async def run_sweep(
        self,
        n_paths: int,
        model: str = "gbm",
        on_progress: Callable[[int, int], None] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Run research-sim with --experiments flag to produce sweep CSVs."""
        cmd = [
            self.research_sim,
            "--paths", str(n_paths),
            "--model", model,
            "--experiments",
            "--progress",
        ]
        cwd = Path(settings.rust_binary_path).parent.parent
        if not cwd.exists():
            cwd = Path(".")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        if job_id:
            from web.services.job_registry import register_subprocess
            await register_subprocess(job_id, proc)

        try:
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            async def read_stdout():
                async for line in proc.stdout:
                    decoded = line.decode(errors="replace").rstrip()
                    stdout_lines.append(decoded)

            async def read_stderr():
                async for line in proc.stderr:
                    stderr_lines.append(line.decode(errors="replace").rstrip())

            await asyncio.gather(read_stdout(), read_stderr())
            await proc.wait()
        finally:
            if job_id:
                from web.services.job_registry import unregister_subprocess
                await unregister_subprocess(job_id, proc)

        if proc.returncode != 0:
            err = "\n".join(stderr_lines[-20:])
            raise RustRunnerError(
                f"research-sim --experiments exited with code {proc.returncode}.\nStderr:\n{err}"
            )

        # Load the sweep CSVs
        results = {}
        for dim in ["volatility", "horizon", "impact", "trade_chunks"]:
            csv_path = cwd / "results" / f"sweep_{dim}.csv"
            if csv_path.exists():
                results[dim] = self._parse_sweep_csv(str(csv_path))

        return results

    def _parse_sweep_csv(self, csv_path: str) -> list[dict]:
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: self._try_float(v) for k, v in row.items()})
        return rows

    @staticmethod
    def _try_float(v: str) -> float | str:
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


# Singleton
_runner: RustRunner | None = None


def get_rust_runner() -> RustRunner:
    global _runner
    if _runner is None:
        _runner = RustRunner()
    return _runner
