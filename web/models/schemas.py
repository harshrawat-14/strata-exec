"""
Pydantic v2 request / response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | complete | failed
    websocket_url: str


# ── Simulation ────────────────────────────────────────────────────────────────

class SimParams(BaseModel):
    sigma: float = Field(0.02, ge=0.001, le=2.0, description="Annual volatility (fraction)")
    eta: float = Field(0.001, ge=0.0, le=1.0, description="Temporary impact coefficient")
    lambda_: float = Field(1e-4, ge=0.0, le=1.0, alias="lambda", description="Risk-aversion parameter")
    total_notional: float = Field(1_000_000.0, ge=1000.0, description="Total notional to liquidate ($)")
    horizon_steps: int = Field(500, ge=10, le=2880, description="Number of trading steps")

    model_config = {"populate_by_name": True}


class SimulationRequest(BaseModel):
    price_model: str = Field("gbm", pattern="^(gbm|garch)$")
    strategies: list[str] = Field(
        default=["twap", "heuristic", "optimal", "adaptive"],
        min_length=1,
    )
    n_paths: int = Field(100, ge=1, le=2000)
    params: SimParams = Field(default_factory=SimParams)
    lob_file_id: str | None = None
    agg_file_id: str | None = None

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, v: list[str]) -> list[str]:
        valid = {"twap", "heuristic", "optimal", "adaptive"}
        for s in v:
            if s not in valid:
                raise ValueError(f"Unknown strategy '{s}'. Valid: {valid}")
        return v


class StrategyResult(BaseModel):
    name: str
    mean_is_pct: float | None = None
    is_variance: float | None = None
    cvar95: float | None = None
    ac_objective: float | None = None
    trajectory: list[float] = Field(default_factory=list)
    cost_decomposition: dict[str, float] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    job_id: str
    status: str
    strategies: list[StrategyResult] = Field(default_factory=list)
    price_path: list[float] = Field(default_factory=list)
    params_used: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float | None = None
    created_at: datetime | None = None


# ── RL Evaluation ─────────────────────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    model_id: str
    dates: list[str] = Field(min_length=1, max_length=11)
    n_episodes: int = Field(50, ge=1, le=200)
    compare_with: list[str] = Field(default=["optimal", "adaptive"])

    @field_validator("dates")
    @classmethod
    def validate_dates(cls, v: list[str]) -> list[str]:
        import re
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in v:
            if not pattern.match(d):
                raise ValueError(f"Invalid date format '{d}'. Expected YYYY-MM-DD")
        return v


class DateResult(BaseModel):
    date: str
    regime: str
    mean_is_pct: float | None = None
    std_is: float | None = None
    cvar95: float | None = None
    forced_liquidation_rate: float | None = None
    action_distribution: dict[str, float] = Field(default_factory=dict)
    mean_action: float | None = None
    action_entropy: float | None = None
    # Baselines
    static_optimal_is: float | None = None
    adaptive_optimal_is: float | None = None
    twap_is: float | None = None
    heuristic_is: float | None = None
    # Stats
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    significantly_better: bool | None = None


class EvaluationResult(BaseModel):
    job_id: str
    status: str
    model_name: str | None = None
    date_results: list[DateResult] = Field(default_factory=list)
    synthetic_is: float | None = None
    duration_seconds: float | None = None
    created_at: datetime | None = None


# ── Upload ────────────────────────────────────────────────────────────────────

class LOBPreview(BaseModel):
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    timestamp_first: str | None = None
    timestamp_last: str | None = None


class UploadedFileInfo(BaseModel):
    file_id: str
    file_type: str
    original_name: str
    date_str: str | None = None
    n_rows: int | None = None
    file_size_bytes: int
    preview: LOBPreview | None = None
    created_at: datetime


class UploadedModelInfo(BaseModel):
    model_id: str
    name: str
    original_name: str
    file_size_bytes: int
    is_builtin: bool
    created_at: datetime


# ── Parameter Sweep ───────────────────────────────────────────────────────────

class SweepRequest(BaseModel):
    sweep_dimension: str = Field(..., pattern="^(volatility|horizon|impact|slices)$")
    grid_values: list[float] = Field(min_length=2, max_length=20)
    n_paths: int = Field(100, ge=1, le=500)
    strategies: list[str] = Field(default=["twap", "optimal", "adaptive"])


class SweepCell(BaseModel):
    dimension_value: float
    strategy: str
    mean_is_pct: float
    is_variance: float
    cvar95: float


class SweepResult(BaseModel):
    job_id: str
    status: str
    sweep_dimension: str
    grid_values: list[float]
    cells: list[SweepCell] = Field(default_factory=list)
    created_at: datetime | None = None


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_simulations: int
    total_evaluations: int
    available_lob_files: int
    available_models: int
    available_dates: list[str]
    recent_jobs: list[dict[str, Any]]
