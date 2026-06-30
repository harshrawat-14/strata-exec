/**
 * Simulator page — Demo Mode
 * Shows 3 pre-computed scenario cards. Clicking one fetches the result instantly.
 * The full results panel (charts, tables, tabs) is unchanged.
 */

import { useState, useCallback, useEffect } from 'react'
import { clsx } from 'clsx'
import { fetchSimulationResult } from '../api/client'
import { PriceChart } from '../components/TrajectoryChart'
import { EmptyState } from '../components/ui'
import SimulationResultsPanel, {
  TrajectoryPanel,
  CostBreakdownPanel,
  MetricsTablePanel,
} from '../components/SimulationResultsPanel'
import type { SimulationResult } from '../types'

/* ── Dark mode hook ──────────────────────────────────────────────────────────── */
function useDarkMode() {
  const [isDark, setIsDark] = useState(() =>
    typeof window !== 'undefined' ? document.documentElement.classList.contains('dark') : true
  )
  useEffect(() => {
    const obs = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains('dark'))
    )
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return isDark
}

/* ── Demo Scenarios ─────────────────────────────────────────────────────────── */
interface Scenario {
  id: string
  label: string
  subtitle: string
  description: string
  model: string
  sigma: string
  notional: string
  horizon: string
  tag: string
  tagColor: string
  presetKey: string   // matches demo_results/simulation/{presetKey}.json
}

const SCENARIOS: Scenario[] = [
  {
    id: 'garch_volatile',
    label: 'GARCH — Volatile Market',
    subtitle: 'High-sigma regime with vol clustering',
    description: 'GARCH(1,1) process with σ=20%, λ=0.001. Simulates a volatile day where volatility begets volatility — typical of crypto market stress periods.',
    model: 'GARCH',
    sigma: '20%',
    notional: '$50,000',
    horizon: '500 steps',
    tag: 'VOLATILE',
    tagColor: '#F59E0B',
    presetKey: 'garch__0.20__0.001__50000__500',
  },
  {
    id: 'gbm_stable',
    label: 'GBM — Stable Market',
    subtitle: 'Constant volatility baseline',
    description: 'Geometric Brownian Motion with σ=20%, λ=0.001. Classic Black-Scholes environment — the industry benchmark for optimal execution strategy comparison.',
    model: 'GBM',
    sigma: '20%',
    notional: '$50,000',
    horizon: '500 steps',
    tag: 'STABLE',
    tagColor: '#10B981',
    presetKey: 'gbm__0.20__0.001__50000__500',
  },
  {
    id: 'garch_crash',
    label: 'GARCH — Flash Crash',
    subtitle: 'Extreme vol + tight horizon',
    description: 'GARCH with σ=40%, λ=0.01 compressed to 200 steps. Simulates a flash-crash scenario where the agent must liquidate rapidly against adverse price movement.',
    model: 'GARCH',
    sigma: '40%',
    notional: '$50,000',
    horizon: '200 steps',
    tag: 'CRASH',
    tagColor: '#EF4444',
    presetKey: 'garch__0.40__0.01__50000__200',
  },
]

type TabKey = 'dashboard' | 'price' | 'trajectory' | 'cost' | 'table'

export default function Simulator() {
  const isDark = useDarkMode()

  const [activeScenario, setActiveScenario] = useState<string | null>(null)
  const [loading, setLoading]               = useState(false)
  const [result, setResult]                 = useState<SimulationResult | null>(null)
  const [error, setError]                   = useState<string | null>(null)
  const [activeTab, setActiveTab]           = useState<TabKey>('dashboard')

  const handleSelectScenario = useCallback(async (scenario: Scenario) => {
    if (loading) return
    setActiveScenario(scenario.id)
    setLoading(true)
    setError(null)
    setResult(null)
    setActiveTab('dashboard')
    try {
      // Fetch directly from the demo endpoint — no job queue needed
      const res = await fetchSimulationResult(`demo_${scenario.presetKey}`)
      setResult(res)
    } catch (err: any) {
      setError(err?.message || 'Failed to load scenario')
    } finally {
      setLoading(false)
    }
  }, [loading])

  const tabs: { key: TabKey; label: string; emoji: string }[] = [
    { key: 'dashboard',  label: 'Dashboard',      emoji: '⊞' },
    { key: 'trajectory', label: 'Trajectory',     emoji: '↗' },
    { key: 'cost',       label: 'Cost Breakdown', emoji: '＄' },
    { key: 'table',      label: 'Metrics Table',  emoji: '≡' },
    { key: 'price',      label: 'Price Path',     emoji: '∿' },
  ]

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full animate-fade-in">

      {/* ── Left: Scenario Selector ───────────────────────────────────────── */}
      <div className="w-full lg:w-72 lg:flex-shrink-0 space-y-3">
        <div className="glass-card p-5">
          <h2 className="label-text mb-1">Market Scenario</h2>
          <p className="text-[10px] font-mono mb-4" style={{ color: 'var(--text-muted)' }}>
            Select a pre-computed scenario to view results instantly
          </p>

          <div className="space-y-2.5">
            {SCENARIOS.map(scenario => {
              const isActive  = activeScenario === scenario.id
              const isLoading = isActive && loading
              return (
                <button
                  key={scenario.id}
                  id={`scenario-${scenario.id}`}
                  onClick={() => handleSelectScenario(scenario)}
                  disabled={loading}
                  className="w-full text-left rounded-xl p-3.5 transition-all duration-200 space-y-2"
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
                  {/* Header row */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full flex-shrink-0 mt-0.5"
                        style={{ background: scenario.tagColor }}
                      />
                      <div>
                        <div className="text-xs font-bold font-mono" style={{ color: 'var(--text)' }}>
                          {scenario.label}
                        </div>
                        <div className="text-[9px] font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>
                          {scenario.subtitle}
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

                  {/* Params row */}
                  <div className="flex flex-wrap gap-1.5 pt-1" style={{ borderTop: '1px solid var(--divider)' }}>
                    {[
                      ['Model', scenario.model],
                      ['σ', scenario.sigma],
                      ['Notional', scenario.notional],
                      ['Horizon', scenario.horizon],
                    ].map(([k, v]) => (
                      <span key={k} className="text-[9px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--card-border)', color: 'var(--text-sub)' }}>
                        {k}: <strong>{v}</strong>
                      </span>
                    ))}
                  </div>

                  {/* Spinner or checkmark */}
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

          {/* Description of selected scenario */}
          {activeScenario && (
            <div className="mt-4 p-3 rounded-lg text-[10px] font-mono leading-relaxed" style={{ background: 'var(--input-bg)', color: 'var(--text-sub)', border: '1px solid var(--divider)' }}>
              {SCENARIOS.find(s => s.id === activeScenario)?.description}
            </div>
          )}
        </div>

        {/* Demo mode notice */}
        <div className="glass-card p-3 text-[9px] font-mono" style={{ color: 'var(--text-muted)', borderLeft: '2px solid var(--active-fill)' }}>
          <span className="font-bold" style={{ color: 'var(--active-fill)' }}>DEMO MODE</span>
          {' '}— Results are pre-computed. Run locally with the Rust simulator for custom parameters.
        </div>
      </div>

      {/* ── Right: Results Panel ──────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-4">

        {/* Empty state */}
        {!result && !loading && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <EmptyState
              icon="📊"
              title="Select a scenario"
              description="Click one of the three market scenarios on the left to load pre-computed simulation results instantly"
            />
          </div>
        )}

        {/* Loading spinner */}
        {loading && !result && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <div className="text-center space-y-3">
              <div className="w-8 h-8 border-2 border-[var(--active-fill)] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Loading scenario…</p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-card p-6 text-center">
            <p className="text-sm font-mono text-red-400">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Tab bar */}
            <div className="flex flex-wrap gap-1.5 pb-2" style={{ borderBottom: '1px solid var(--divider)' }}>
              {tabs.map(t => (
                <button
                  key={t.key}
                  onClick={() => setActiveTab(t.key)}
                  className={clsx(
                    'flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-[11px] font-mono font-semibold uppercase tracking-wider transition-all',
                    activeTab === t.key
                      ? 'bg-[var(--active-fill)] text-[var(--active-text)] shadow-sm'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--card-hover)]'
                  )}
                >
                  <span className="opacity-70" style={{ fontFamily: 'sans-serif', fontSize: 13 }}>{t.emoji}</span>
                  {t.label}
                </button>
              ))}
            </div>

            <div className="glass-card p-6">
              {activeTab === 'dashboard'  && <SimulationResultsPanel result={result} isDark={isDark} />}
              {activeTab === 'trajectory' && <TrajectoryPanel result={result} isDark={isDark} />}
              {activeTab === 'cost'       && <CostBreakdownPanel result={result} isDark={isDark} />}
              {activeTab === 'table'      && <MetricsTablePanel result={result} />}
              {activeTab === 'price'      && (
                <div>
                  <p className="text-[10px] font-mono uppercase mb-4" style={{ color: 'var(--text-muted)' }}>
                    Simulated price path ({result.price_path.length} steps)
                  </p>
                  <PriceChart pricePoints={result.price_path} />
                </div>
              )}
            </div>

            {/* Meta bar */}
            <div className="flex items-center gap-4 text-[10px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>
              <span>Job: <span className="font-semibold">{result.job_id.slice(0, 12)}…</span></span>
              {result.duration_seconds && <span>Duration: {result.duration_seconds.toFixed(1)}s</span>}
              <span>{result.strategies.length} strategies · {result.price_path.length} steps</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
