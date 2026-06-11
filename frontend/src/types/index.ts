/**
 * TypeScript interfaces mirroring Pydantic schemas.
 */

// ── Jobs ──────────────────────────────────────────────────────────────────────
export interface JobStatus {
  job_id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  websocket_url: string
}

export type JobStatusValue = 'queued' | 'running' | 'complete' | 'failed'

// ── WebSocket messages ────────────────────────────────────────────────────────
export type WsMessage =
  | { type: 'progress'; completed: number; total: number }
  | { type: 'status'; status: JobStatusValue }
  | { type: 'complete'; job_id: string; results_url: string }
  | { type: 'error'; message: string }

// ── Simulation ────────────────────────────────────────────────────────────────
export interface SimParams {
  sigma: number
  eta: number
  lambda: number
  total_notional: number
  horizon_steps: number
}

export interface SimulationRequest {
  price_model: 'gbm' | 'garch'
  strategies: string[]
  n_paths: number
  params: SimParams
  lob_file_id?: string
  agg_file_id?: string
}

export interface StrategyResult {
  name: string
  mean_is_pct: number | null
  is_variance: number | null
  cvar95: number | null
  ac_objective: number | null
  trajectory: number[]
  cost_decomposition: Record<string, number>
}

export interface SimulationResult {
  job_id: string
  status: JobStatusValue
  strategies: StrategyResult[]
  price_path: number[]
  params_used: Record<string, unknown>
  duration_seconds: number | null
  created_at: string | null
}

// ── RL Evaluation ─────────────────────────────────────────────────────────────
export interface EvaluationRequest {
  model_id: string
  dates: string[]
  n_episodes: number
  compare_with: string[]
}

export interface DateResult {
  date: string
  regime: string
  mean_is_pct: number | null
  std_is: number | null
  cvar95: number | null
  forced_liquidation_rate: number | null
  action_distribution: Record<string, number>
  mean_action: number | null
  action_entropy: number | null
  static_optimal_is: number | null
  adaptive_optimal_is: number | null
  twap_is: number | null
  heuristic_is: number | null
  p_value: number | null
  ci_lower: number | null
  ci_upper: number | null
  significantly_better: boolean | null
}

export interface EvaluationResult {
  job_id: string
  status: JobStatusValue
  model_name: string | null
  date_results: DateResult[]
  synthetic_is: number | null
  duration_seconds: number | null
  created_at: string | null
}

// ── Upload ────────────────────────────────────────────────────────────────────
export interface LOBPreview {
  mid_price: number
  best_bid: number
  best_ask: number
  spread_bps: number
  timestamp_first: string | null
  timestamp_last: string | null
}

export interface UploadedFileInfo {
  file_id: string
  file_type: 'lob' | 'agg_trades'
  original_name: string
  date_str: string | null
  n_rows: number | null
  file_size_bytes: number
  preview: LOBPreview | null
  created_at: string
}

export interface UploadedModelInfo {
  model_id: string
  name: string
  original_name: string
  file_size_bytes: number
  is_builtin: boolean
  created_at: string
}

// ── Parameter Sweep ───────────────────────────────────────────────────────────
export interface SweepRequest {
  sweep_dimension: 'volatility' | 'horizon' | 'impact' | 'slices'
  grid_values: number[]
  n_paths: number
  strategies: string[]
}

export interface SweepCell {
  dimension_value: number
  strategy: string
  mean_is_pct: number
  is_variance: number
  cvar95: number
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
export interface AvailableDate {
  date: string
  regime: string
  source: 'builtin' | 'uploaded'
  file_id: string | null
}

export interface RecentJob {
  job_id: string
  type: string
  status: JobStatusValue
  label: string
  created_at: string | null
  duration_seconds: number | null
}

export interface DashboardStats {
  total_simulations: number
  total_evaluations: number
  available_lob_files: number
  available_models: number
  available_dates: AvailableDate[]
  recent_jobs: RecentJob[]
}

export interface StrategyInfo {
  id: string
  name: string
  description: string
  color: string
}

// ── Depth preview ─────────────────────────────────────────────────────────────
export interface OrderLevel {
  price: number | null
  qty: number | null
}
export interface DepthSnapshot {
  step: number
  bid_levels: OrderLevel[]
  ask_levels: OrderLevel[]
}
export interface SpreadPoint {
  step: number
  mid_price: number
  spread_bps: number
}
export interface LobDepthPreview {
  depth_snapshots: DepthSnapshot[]
  spread_series: SpreadPoint[]
}
