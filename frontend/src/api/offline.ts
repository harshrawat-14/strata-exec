/**
 * Static fallbacks for when the engine is offline. Everything here comes from files exported from
 * the pinned tag by scripts/export_site_snapshot.py.
 */
import { fetchSimulationResult } from './client'
import type { SimulationResult } from '../types'
import sweeps from '../data/pinned/sweeps.json'

interface PresetFile {
  preset_key: string
  params: Record<string, unknown>
  results: { strategies: SimulationResult['strategies']; price_path: number[] }
  generated_at?: string
}

/** Maps a stored preset to what /api/compare returns (same mapping as web/demo_middleware.py). */
export function presetToResult(file: PresetFile): SimulationResult {
  return {
    job_id: `demo_${file.preset_key}`,
    status: 'complete',
    strategies: file.results.strategies ?? [],
    price_path: file.results.price_path ?? [],
    params_used: file.params ?? {},
    duration_seconds: null,
    created_at: file.generated_at ?? null,
  }
}

/** A precomputed simulation: from the API when it is up, else from the static copy. */
export async function loadSimulationPreset(presetKey: string, fetchImpl: typeof fetch = fetch): Promise<SimulationResult> {
  try {
    return await fetchSimulationResult(`demo_${presetKey}`)
  } catch {
    const res = await fetchImpl(`/demo/simulation/${presetKey}.json`)
    if (!res.ok) throw new Error(`Precomputed scenario ${presetKey} not found`)
    return presetToResult(await res.json())
  }
}

export type SweepDimension = 'volatility' | 'horizon' | 'impact' | 'slices'

/** The pinned GARCH sweeps (500 paths per point), shaped like a finished sweep job. */
export function pinnedSweepResult(dimension: string) {
  const key = dimension === 'slices' ? 'trade_chunks' : dimension
  const rows = (sweeps as Record<string, { ParameterValue: number }[]>)[key] ?? []
  return {
    sweep_dimension: dimension,
    grid_values: [...new Set(rows.map(r => r.ParameterValue))],
    duration_seconds: undefined,
    sweep_data: sweeps,
    precomputed: true,
  }
}
