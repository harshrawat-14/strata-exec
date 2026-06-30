"""
web/demo_middleware.py

Thin demo-mode layer for Render deployment.
Intercepts simulation and evaluation requests and serves
pre-computed results. Everything else passes through unchanged.

Activated by environment variable: DEMO_MODE=true
When DEMO_MODE is not set or false: this middleware is a no-op.
"""

import json
import os
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from typing import Callable

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEMO_DIR = Path(__file__).parent.parent / "demo_results"


class DemoModeMiddleware(BaseHTTPMiddleware):
    """
    Intercepts API requests in demo mode and returns pre-computed results.
    Does nothing when DEMO_MODE=false (local development).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Pass through immediately when not in demo mode
        if not DEMO_MODE:
            return await call_next(request)

        path = request.url.path
        method = request.method

        # ── Manifest endpoint ──────────────────────────────────────────
        # Returns what's available in demo mode
        if path == "/api/demo/manifest" and method == "GET":
            return self._serve_json(DEMO_DIR / "manifest.json")

        # ── Simulation endpoint ────────────────────────────────────────
        if path == "/api/simulate" and method == "POST":
            return await self._handle_simulate(request)

        # ── Evaluation endpoint ────────────────────────────────────────
        if path == "/api/evaluate" and method == "POST":
            return await self._handle_evaluate(request)

        # ── Job state endpoint ─────────────────────────────────────────
        # Demo mode: jobs complete "instantly" (pre-computed)
        if path.startswith("/api/jobs/") and path.endswith("/state"):
            job_id = path.split("/")[-2]
            return self._serve_demo_job_state(job_id)

        # ── Progress event stream endpoint ──────────────────────────────
        if path.startswith("/api/jobs/") and path.endswith("/progress"):
            job_id = path.split("/")[-2]
            if job_id.startswith("demo_"):
                return self._serve_demo_progress_stream(job_id)

        # ── Simulation result endpoint ─────────────────────────────────
        if path.startswith("/api/compare/") and method == "GET":
            job_id = path.split("/")[-1]
            if job_id.startswith("demo_"):
                return self._serve_demo_simulation_result(job_id)

        # ── Evaluation result endpoint ─────────────────────────────────
        if path.startswith("/api/evaluate/result/") and method == "GET":
            job_id = path.split("/")[-1]
            if job_id.startswith("demo_"):
                return self._serve_demo_evaluation_result(job_id)

        # ── Upload models endpoint (needed for RL Evaluation page) ────────
        if path == "/api/upload/models" and method == "GET":
            return self._serve_demo_models()

        # ── Upload files endpoint (needed for Simulator page) ────────────
        if path.startswith("/api/upload/files") and method == "GET":
            return JSONResponse([])

        # ── Other upload endpoints ────────────────────────────────────────
        if path.startswith("/api/upload/") and method == "POST":
            return self._run_locally_response(
                feature=path.split("/")[-1],
                message=self._upload_message(path),
                run_locally=self._upload_run_locally(path),
            )

        # ── Everything else passes through ─────────────────────────────
        # Frontend assets, health checks, WebSocket, etc.
        return await call_next(request)

    async def _handle_simulate(self, request: Request) -> Response:
        """Match simulation request to a pre-computed preset."""
        try:
            body = await request.json()
        except Exception:
            return await self._demo_not_found("simulate")

        preset_key = self._build_sim_key(body)
        preset_file = DEMO_DIR / "simulation" / f"{preset_key}.json"

        if preset_file.exists():
            data = json.loads(preset_file.read_text())
            # Return in the same format the existing route would return
            # IMPORTANT: use the full preset_key (no truncation!) so that
            # _serve_demo_simulation_result can reconstruct the filename.
            return JSONResponse({
                "job_id": f"demo_{preset_key}",
                "status": "complete",
                "results": data["results"],
                "preset_key": preset_key,
                "demo_mode": True,
            })

        # No exact match — find the closest available preset file and serve it
        return self._serve_closest_preset(preset_key, body)

    async def _handle_evaluate(self, request: Request) -> Response:
        """Match evaluation request to a pre-computed date result."""
        try:
            body = await request.json()
        except Exception:
            return await self._demo_not_found("evaluate")

        dates = body.get("dates", [])

        if not dates:
            # Return all pre-computed evaluation dates
            return self._serve_json(DEMO_DIR / "evaluation" / "all_dates.json")

        # Single date or multiple dates
        if len(dates) == 1:
            date = dates[0]
            date_file = DEMO_DIR / "evaluation" / f"{date}.json"
            if date_file.exists():
                data = json.loads(date_file.read_text())
                return JSONResponse({
                    "job_id": f"demo_eval_{date}",
                    "status": "complete",
                    "results": [data],
                    "demo_mode": True,
                })
            else:
                return self._run_locally_response(
                    feature="custom_evaluation_date",
                    message=f"Evaluation for {date} is not pre-computed.",
                    run_locally=(
                        f"python rl/evaluate.py "
                        f"--passive-model rl/models/ppo_lstm_v5_adaptive_best/best_model "
                        f"--date {date} --n-state-dims 12"
                    ),
                )

        # Multiple dates — return all that are available
        results = []
        missing = []
        for date in dates:
            date_file = DEMO_DIR / "evaluation" / f"{date}.json"
            if date_file.exists():
                results.append(json.loads(date_file.read_text()))
            else:
                missing.append(date)

        response = {
            "job_id": "demo_eval_multi",
            "status": "complete",
            "results": results,
            "demo_mode": True,
        }
        if missing:
            response["unavailable_dates"] = missing
            response["run_locally_for_missing"] = (
                f"python rl/evaluate.py "
                f"--passive-model rl/models/ppo_lstm_v5_adaptive_best/best_model "
                f"--include-test-dates --n-state-dims 12"
            )
        return JSONResponse(response)

    def _serve_closest_preset(self, requested_key: str, body: dict) -> Response:
        """Find the closest available preset file and return it."""
        sim_dir = DEMO_DIR / "simulation"
        if not sim_dir.exists():
            return self._suggest_presets(body)

        available = list(sim_dir.glob("*.json"))
        if not available:
            return self._suggest_presets(body)

        # Parse requested key parts
        parts = requested_key.split("__")
        req_model = parts[0] if len(parts) > 0 else "garch"
        req_sigma = float(parts[1]) if len(parts) > 1 else 0.20
        req_lam   = float(parts[2]) if len(parts) > 2 else 0.001
        req_not   = float(parts[3]) if len(parts) > 3 else 50000
        req_hz    = float(parts[4]) if len(parts) > 4 else 500

        def score(path):
            """Lower = better match."""
            p = path.stem.split("__")
            if len(p) < 5:
                return float('inf')
            model_match = 0 if p[0] == req_model else 2
            try:
                sigma_d = abs(float(p[1]) - req_sigma)
                lam_d   = abs(float(p[2]) - req_lam)
                not_d   = abs(float(p[3]) - req_not) / 50000
                hz_d    = abs(float(p[4]) - req_hz) / 100
            except ValueError:
                return float('inf')
            return model_match + sigma_d + lam_d + not_d + hz_d

        closest = min(available, key=score)
        data = json.loads(closest.read_text())
        preset_key = closest.stem
        return JSONResponse({
            "job_id": f"demo_{preset_key}",
            "status": "complete",
            "results": data["results"],
            "preset_key": preset_key,
            "demo_mode": True,
            "note": f"Showing closest preset: {preset_key}",
        })

    def _serve_demo_models(self) -> Response:
        """Return hard-coded demo models for the RL Evaluation page."""
        return JSONResponse([
            {
                "model_id": "ppo_lstm_v5_adaptive_best",
                "name": "ppo_lstm_v5_adaptive_best",
                "size_bytes": 3800000,
                "uploaded_at": "2024-11-01T10:00:00Z",
                "demo": True,
            },
        ])

    def _build_sim_key(self, body: dict) -> str:
        """
        Build the preset key from request params.
        Matches the key format used in SIMULATION_PRESETS.
        """
        params = body.get("params", {})
        model = body.get("price_model", "garch")
        sigma = params.get("sigma", 0.20)
        lam = params.get("lambda", 0.001)
        notional = params.get("total_notional", 50000.0)
        horizon = params.get("horizon_steps", 500)

        # Round to nearest preset value to handle floating point
        sigma = self._nearest(sigma, [0.10, 0.20, 0.40])
        lam = self._nearest(lam, [0.0001, 0.001, 0.01])
        notional = self._nearest(notional, [10000, 50000, 100000])
        horizon = self._nearest(horizon, [200, 500])

        return f"{model}__{sigma}__{lam}__{int(notional)}__{int(horizon)}"

    def _nearest(self, value: float, options: list) -> float:
        """Return the nearest value from options list."""
        return min(options, key=lambda x: abs(x - value))

    def _serve_demo_job_state(self, job_id: str) -> Response:
        """Return instant 'complete' state for demo jobs."""
        if not job_id.startswith("demo_"):
            return JSONResponse({"error": "Job not found"}, status_code=404)
        response_data = {
            "job_id": job_id,
            "status": "complete",
            "progress": 100,
            "demo_mode": True,
        }
        if job_id.startswith("demo_eval_"):
            response_data["results"] = {"date_results": []}
        return JSONResponse(response_data)

    def _serve_demo_progress_stream(self, job_id: str) -> Response:
        """Stream complete event for demo jobs."""
        from starlette.responses import StreamingResponse
        import asyncio

        async def event_generator():
            # First send running status
            yield f"data: {json.dumps({'type': 'status', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.1)
            # Send 100% progress
            yield f"data: {json.dumps({'type': 'progress', 'completed': 100, 'total': 100})}\n\n"
            await asyncio.sleep(0.1)
            # Send complete status
            results_url = f"/api/compare/{job_id}"
            if job_id.startswith("demo_eval_"):
                results_url = f"/api/evaluate/result/{job_id}"
            yield f"data: {json.dumps({'type': 'complete', 'job_id': job_id, 'results_url': results_url})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            }
        )

    def _serve_demo_simulation_result(self, job_id: str) -> Response:
        """Return pre-computed simulation results."""
        from datetime import datetime
        preset_key = job_id.replace("demo_", "", 1)
        preset_file = DEMO_DIR / "simulation" / f"{preset_key}.json"

        if not preset_file.exists():
            return JSONResponse({"error": f"Pre-computed simulation result not found for {preset_key}"}, status_code=404)

        data = json.loads(preset_file.read_text())
        results = data["results"]

        return JSONResponse({
            "job_id": job_id,
            "status": "complete",
            "strategies": results.get("strategies", []),
            "price_path": results.get("price_path", []),
            "params_used": data.get("params", {}),
            "duration_seconds": 1.5,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })

    def _normalize_date_result(self, raw: dict) -> dict:
        """
        Remap field names from precomputed JSON to what EvaluationResultsPanel expects.
        Stored: rl_is, rl_std, rl_ci_lower, rl_ci_upper, rl_cvar95
        Expected: mean_is_pct, std_is, ci_lower, ci_upper, cvar95
        """
        return {
            **raw,
            "mean_is_pct": raw.get("rl_is"),
            "std_is":      raw.get("rl_std"),
            "ci_lower":    raw.get("rl_ci_lower"),
            "ci_upper":    raw.get("rl_ci_upper"),
            "cvar95":      raw.get("rl_cvar95"),
        }

    def _serve_demo_evaluation_result(self, job_id: str) -> Response:
        """Return pre-computed evaluation results with normalised field names."""
        from datetime import datetime
        if job_id == "demo_eval_multi":
            all_dates_file = DEMO_DIR / "evaluation" / "all_dates.json"
            if not all_dates_file.exists():
                return JSONResponse({"error": "Pre-computed all-dates evaluation result not found"}, status_code=404)
            data = json.loads(all_dates_file.read_text())
            date_results = [self._normalize_date_result(d) for d in data.get("dates", [])]
            return JSONResponse({
                "job_id": job_id,
                "status": "complete",
                "model_name": "ppo_lstm_v5_adaptive",
                "date_results": date_results,
                "synthetic_is": -0.45,
                "duration_seconds": 2.5,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
        else:
            date = job_id.replace("demo_eval_", "", 1)
            date_file = DEMO_DIR / "evaluation" / f"{date}.json"

            if not date_file.exists():
                return JSONResponse({"error": f"Pre-computed evaluation result not found for {date}"}, status_code=404)

            raw = json.loads(date_file.read_text())
            dr = self._normalize_date_result(raw)
            return JSONResponse({
                "job_id": job_id,
                "status": "complete",
                "model_name": "ppo_lstm_v5_adaptive",
                "date_results": [dr],
                "synthetic_is": -0.45,
                "duration_seconds": 0.1,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })

    def _suggest_presets(self, body: dict) -> Response:
        """No matching preset — tell user what IS available."""
        manifest_file = DEMO_DIR / "manifest.json"
        available = []
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            available = manifest.get("simulation_presets", [])

        return JSONResponse({
            "demo_mode": True,
            "error": "no_matching_preset",
            "message": (
                "These exact parameters are not pre-computed in demo mode. "
                "Select one of the available presets below, or run locally "
                "for custom parameter values."
            ),
            "available_presets": available,
            "run_locally": (
                "cargo run --release --bin research-sim -- "
                "--paths 500 --model garch --sigma 0.35 ..."
            ),
        }, status_code=200)

    def _run_locally_response(
        self, feature: str, message: str, run_locally: str
    ) -> Response:
        return JSONResponse({
            "demo_mode": True,
            "error": "feature_unavailable_in_demo",
            "feature": feature,
            "message": message,
            "run_locally": run_locally,
            "docs": "https://github.com/harshrawat-14/strata-exec#local-setup",
        }, status_code=200)

    def _serve_json(self, path: Path) -> Response:
        if not path.exists():
            return JSONResponse(
                {"error": "Pre-computed data not found", "demo_mode": True},
                status_code=404
            )
        return JSONResponse(json.loads(path.read_text()))

    def _upload_message(self, path: str) -> str:
        if "lob" in path:
            return ("Upload your own Binance LOB data for custom dates. "
                    "Run locally to use this feature.")
        if "model" in path:
            return ("Upload your own trained RL model for evaluation. "
                    "Run locally to use this feature.")
        return "This upload feature requires running locally."

    def _upload_run_locally(self, path: str) -> str:
        if "lob" in path:
            return ("python scripts/download_data.py --date 2024-09-15 "
                    "# then restart the server")
        if "model" in path:
            return ("python rl/train.py --timesteps 3000000 "
                    "--mode counterfactual ...")
        return "See README for local setup instructions."

    async def _demo_not_found(self, endpoint: str) -> Response:
        return JSONResponse(
            {"error": f"Invalid request to {endpoint}", "demo_mode": True},
            status_code=400
        )
