/**
 * RL Evaluation page — Demo Mode
 * Shows 4 pre-computed date cards. Clicking one fetches & displays the result instantly.
 * No job queue, no SSE, no spinning progress — just direct fetch from demo endpoint.
 */

import { useState, useCallback } from 'react'
import { BrainCircuit } from 'lucide-react'
import { fetchEvaluationResult } from '../api/client'
import { EmptyState } from '../components/ui'
import EvaluationResultsPanel from '../components/EvaluationResultsPanel'
import type { EvaluationResult } from '../types'

/* ── Pre-computed dates — must match demo_results/evaluation/ filenames ─────── */
interface DateScenario {
  date: string
  regime: string
  label: string
  description: string
  tag: string
  tagColor: string
}

const DATE_SCENARIOS: DateScenario[] = [
  {
    date: '2024-01-15',
    regime: 'Calm bull',
    label: 'Jan 15 — Calm Bull',
    description: 'Low volatility, steady uptrend. The agent can afford to execute patiently, capturing liquidity across the full horizon with minimal price impact.',
    tag: 'CALM',
    tagColor: '#10B981',
  },
  {
    date: '2024-04-15',
    regime: 'BTC breakout',
    label: 'Apr 15 — BTC Breakout',
    description: 'Rapid breakout phase with momentum and spread widening. The agent must balance speed (avoiding adverse drift) against impact (thin liquidity on the ask).',
    tag: 'BREAKOUT',
    tagColor: '#3B82F6',
  },
  {
    date: '2024-06-10',
    regime: 'Quiet consolidation',
    label: 'Jun 10 — Quiet Consolidation',
    description: 'Market in a tight range with very low realized volatility. Ideal conditions for TWAP-style execution; the RL agent captures micro-structure alpha tactically.',
    tag: 'QUIET',
    tagColor: '#8B5CF6',
  },
  {
    date: '2024-08-05',
    regime: 'Crash — Yen unwind',
    label: 'Aug 05 — Crash Day',
    description: 'Yen carry trade unwind — sudden 5%+ drop. Critical test: the agent must liquidate aggressively to avoid holding depreciating inventory while the order book collapses.',
    tag: 'CRASH',
    tagColor: '#EF4444',
  },
]

/* ── Hardcoded model (demo) ──────────────────────────────────────────────────── */
const DEMO_MODEL = {
  id: 'smoke_v5_final',
  name: 'SMOKE-V5 — Strategic Market Order Execution',
}

export default function RLEvaluation() {
  const [activeDate, setActiveDate]   = useState<string | null>(null)
  const [loading, setLoading]         = useState(false)
  const [result, setResult]           = useState<EvaluationResult | null>(null)
  const [error, setError]             = useState<string | null>(null)

  const handleSelectDate = useCallback(async (scenario: DateScenario) => {
    if (loading) return
    setActiveDate(scenario.date)
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Direct fetch — no job queue needed in demo mode
      const res = await fetchEvaluationResult(`demo_eval_${scenario.date}`)
      setResult(res)
    } catch (err: any) {
      setError(err?.message || 'Failed to load evaluation result')
    } finally {
      setLoading(false)
    }
  }, [loading])

  return (
    <div className="flex flex-col lg:flex-row gap-6 animate-fade-in">

      {/* ── Left: Date selector ───────────────────────────────────────────── */}
      <div className="w-full lg:w-72 lg:flex-shrink-0 space-y-3">
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-1">
            <BrainCircuit size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="label-text">RL Evaluation</h2>
          </div>
          <p className="text-[10px] font-mono mb-4" style={{ color: 'var(--text-muted)' }}>
            Click a date to load pre-computed results instantly
          </p>

          {/* Model badge */}
          <div className="mb-4 p-2.5 rounded-lg" style={{ background: 'var(--input-bg)', border: '1px solid var(--divider)' }}>
            <div className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Model</div>
            <div className="text-[11px] font-mono font-semibold mt-0.5" style={{ color: 'var(--text)' }}>
              {DEMO_MODEL.name}
            </div>
          </div>

          {/* Date cards */}
          <div className="space-y-2.5">
            {DATE_SCENARIOS.map(scenario => {
              const isActive  = activeDate === scenario.date
              const isLoading = isActive && loading
              return (
                <button
                  key={scenario.date}
                  id={`eval-date-${scenario.date}`}
                  onClick={() => handleSelectDate(scenario)}
                  disabled={loading}
                  className="w-full text-left rounded-xl p-3.5 transition-all duration-200 space-y-1.5"
                  style={{
                    border: '1px solid',
                    borderColor: isActive ? scenario.tagColor + '60' : 'var(--card-border)',
                    background: isActive ? scenario.tagColor + '10' : 'var(--input-bg)',
                    opacity: loading && !isActive ? 0.5 : 1,
                    cursor: loading ? 'wait' : 'pointer',
                    transform: isActive ? 'scale(1.01)' : 'scale(1)',
                    boxShadow: isActive ? `0 0 0 1px ${scenario.tagColor}30` : 'none',
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: scenario.tagColor }}
                      />
                      <div>
                        <div className="text-xs font-bold font-mono" style={{ color: 'var(--text)' }}>
                          {scenario.label}
                        </div>
                        <div className="text-[9px] font-mono mt-0.5 uppercase" style={{ color: 'var(--text-muted)' }}>
                          {scenario.regime}
                        </div>
                      </div>
                    </div>
                    <span
                      className="text-[8px] font-mono font-bold uppercase px-1.5 py-0.5 rounded flex-shrink-0"
                      style={{ background: scenario.tagColor + '20', color: scenario.tagColor }}
                    >
                      {isLoading ? '...' : scenario.tag}
                    </span>
                  </div>

                  {isLoading && (
                    <div className="flex items-center gap-1.5 text-[9px] font-mono" style={{ color: scenario.tagColor }}>
                      <div className="w-2.5 h-2.5 border border-current border-t-transparent rounded-full animate-spin" />
                      Loading results…
                    </div>
                  )}
                  {isActive && !loading && result && (
                    <div className="text-[9px] font-mono" style={{ color: scenario.tagColor }}>
                      ✓ Results loaded
                    </div>
                  )}
                </button>
              )
            })}
          </div>

          {/* Description of selected date */}
          {activeDate && (
            <div className="mt-4 p-3 rounded-lg text-[10px] font-mono leading-relaxed" style={{ background: 'var(--input-bg)', color: 'var(--text-sub)', border: '1px solid var(--divider)' }}>
              {DATE_SCENARIOS.find(s => s.date === activeDate)?.description}
            </div>
          )}
        </div>

        {/* Demo mode notice */}
        <div className="glass-card p-3 text-[9px] font-mono" style={{ color: 'var(--text-muted)', borderLeft: '2px solid var(--active-fill)' }}>
          <span className="font-bold" style={{ color: 'var(--active-fill)' }}>DEMO MODE</span>
          {' '}— Results are pre-computed from 30 episodes each. Run locally to evaluate custom dates.
        </div>
      </div>

      {/* ── Right: Results ────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-4">

        {/* Empty state */}
        {!result && !loading && !error && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <EmptyState
              icon={<BrainCircuit size={28} />}
              title="Select an evaluation date"
              description="Click one of the market regime dates on the left to load pre-computed RL evaluation results instantly"
            />
          </div>
        )}

        {/* Loading */}
        {loading && !result && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <div className="text-center space-y-3">
              <div className="w-8 h-8 border-2 border-[var(--active-fill)] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Loading evaluation…</p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-card p-6 text-center">
            <p className="text-sm font-mono text-red-400">{error}</p>
            <p className="text-[10px] font-mono mt-2" style={{ color: 'var(--text-muted)' }}>
              Make sure the backend is running with DEMO_MODE=true
            </p>
          </div>
        )}

        {/* Results panel — existing component, unchanged */}
        {result && <EvaluationResultsPanel result={result} />}
      </div>
    </div>
  )
}
