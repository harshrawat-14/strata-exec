/**
 * Simulator page — parameters + live-progress Monte Carlo simulation.
 * All styling via CSS variables so light/dark modes are consistent.
 */

import { useState, useCallback, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Play } from 'lucide-react'
import { clsx } from 'clsx'

import { startSimulation, fetchSimulationResult, fetchUploadedFiles, cancelJob } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { PriceChart } from '../components/TrajectoryChart'
import { JobStatusBadge, EmptyState } from '../components/ui'
import SimulationResultsPanel, {
  TrajectoryPanel,
  CostBreakdownPanel,
  MetricsTablePanel,
} from '../components/SimulationResultsPanel'
import { LiveProgressDashboard } from '../components/LiveProgressDashboard'
import type { JobStatusValue, SimulationResult, WsMessage } from '../types'

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

const STRATEGY_OPTIONS = [
  { id: 'twap',     label: 'TWAP',        desc: 'Uniform time-slicing' },
  { id: 'heuristic',label: 'Heuristic',   desc: 'Vol-adaptive chunking' },
  { id: 'optimal',  label: 'Optimal (AC)',desc: 'Almgren-Chriss optimal' },
  { id: 'adaptive', label: 'Adaptive AC', desc: 'Real-time recalibration' },
]

type TabKey = 'dashboard' | 'price' | 'trajectory' | 'cost' | 'table'

export default function Simulator() {
  const isDark = useDarkMode()

  // ── Form state ─────────────────────────────────────────────────────────────
  const [model,        setModel]        = useState<'gbm' | 'garch'>('gbm')
  const [strategies,   setStrategies]   = useState<string[]>(['twap','heuristic','optimal','adaptive'])
  const [nPaths,       setNPaths]       = useState(100)
  const [sigma,        setSigma]        = useState(0.02)
  const [eta,          setEta]          = useState(0.001)
  const [lambda,       setLambda]       = useState(0.0001)
  const [totalNotional, setTotalNotional] = useState(1_000_000)
  const [horizonSteps, setHorizonSteps] = useState(500)
  const [lobFileId,    setLobFileId]    = useState('')

  // ── Job state ──────────────────────────────────────────────────────────────
  const [jobId,          setJobId]          = useState<string | null>(() => sessionStorage.getItem('activeJobId_simulation'))
  const [jobStatus,      setJobStatus]      = useState<JobStatusValue | null>(null)
  const [progress,       setProgress]       = useState({ completed: 0, total: 0 })
  const [partialResults, setPartialResults] = useState<Record<string, any>>({})
  const [result,         setResult]         = useState<SimulationResult | null>(null)
  const [activeTab,      setActiveTab]      = useState<TabKey>('dashboard')
  const [jobStartedAt,   setJobStartedAt]   = useState<number | undefined>(undefined)

  // ── Session Storage Sync ───────────────────────────────────────────────────
  useEffect(() => {
    if (jobId) {
      sessionStorage.setItem('activeJobId_simulation', jobId)
    } else {
      sessionStorage.removeItem('activeJobId_simulation')
    }
  }, [jobId])

  // ── Files ──────────────────────────────────────────────────────────────────
  const { data: lobFiles } = useQuery({
    queryKey: ['lob-files'],
    queryFn: () => fetchUploadedFiles('lob'),
  })


  // ── Start simulation ───────────────────────────────────────────────────────
  const simMut = useMutation({
    mutationFn: startSimulation,
    onSuccess: (data) => {
      setJobId(data.job_id)
      setJobStatus('queued')
      setProgress({ completed: 0, total: nPaths })
      setPartialResults({})
      setResult(null)
      setJobStartedAt(Date.now())
    },
  })

  // ── WebSocket progress ─────────────────────────────────────────────────────
  const handleWsMessage = useCallback(async (msg: WsMessage) => {
    if (msg.started_at) {
      setJobStartedAt(msg.started_at)
    }

    if (msg.type === 'paths_update') {
      setProgress({ completed: msg.paths_done ?? 0, total: msg.paths_total ?? nPaths })
      setPartialResults(msg.partial_results ?? {})
      setJobStatus('running')
    } else if (msg.type === 'progress') {
      setProgress({ completed: msg.completed, total: msg.total })
      setJobStatus('running')
    } else if (msg.type === 'status') {
      setJobStatus(msg.status)
    } else if (msg.type === 'complete' && jobId) {
      setJobStatus('complete')
      const res = await fetchSimulationResult(jobId)
      setResult(res)
    } else if (msg.type === 'error') {
      setJobStatus('failed')
    }
  }, [jobId, nPaths])

  useWebSocket({
    jobId,
    onMessage: handleWsMessage,
    enabled: !!jobId && jobStatus !== 'complete' && jobStatus !== 'failed',
  })

  const handleRun = () => {
    simMut.mutate({
      price_model: model,
      strategies,
      n_paths: nPaths,
      params: { sigma, eta, lambda, total_notional: totalNotional, horizon_steps: horizonSteps },
      lob_file_id: lobFileId || undefined,
      include_rl: false,
    })
  }

  const handleCancel = async (id: string) => {
    try {
      await cancelJob(id)
      setJobStatus('failed')
    } catch (err) {
      console.error("Failed to cancel simulation", err)
    }
  }

  const toggleStrategy = (id: string) =>
    setStrategies(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id])

  const isRunning = jobStatus === 'queued' || jobStatus === 'running'

  const tabs: { key: TabKey; label: string; emoji: string }[] = [
    { key: 'dashboard',  label: 'Dashboard',    emoji: '⊞' },
    { key: 'trajectory', label: 'Trajectory',   emoji: '↗' },
    { key: 'cost',       label: 'Cost Breakdown', emoji: '＄' },
    { key: 'table',      label: 'Metrics Table', emoji: '≡' },
    { key: 'price',      label: 'Price Path',    emoji: '∿' },
  ]

  return (
    <div className="flex gap-6 h-full animate-fade-in">

      {/* ── Left panel ────────────────────────────────────────────────────── */}
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="glass-card p-5 space-y-5">
          <h2 className="label-text">Simulation Parameters</h2>

          {/* Price model */}
          <div className="space-y-2">
            <label className="label-text">Price Model</label>
            <div className="flex gap-2">
              {(['gbm', 'garch'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => setModel(m)}
                  className="flex-1 py-2 rounded-lg text-xs font-mono uppercase tracking-wider transition-all"
                  style={{
                    border: '1px solid',
                    borderColor: model === m ? 'var(--active-fill)' : 'var(--card-border)',
                    background: model === m ? 'var(--active-fill)' : 'transparent',
                    color: model === m ? 'var(--active-text)' : 'var(--text-sub)',
                    fontWeight: model === m ? 700 : 500,
                  }}
                >
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
            <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
              {model === 'gbm' ? 'GBM — constant volatility' : 'GARCH(1,1) — vol clustering'}
            </p>
          </div>

          {/* Strategies */}
          <div className="space-y-2">
            <label className="label-text">Strategies</label>
            <div className="space-y-1.5">
              {STRATEGY_OPTIONS.map(s => (
                <button
                  key={s.id}
                  onClick={() => toggleStrategy(s.id)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all"
                  style={{
                    border: '1px solid',
                    borderColor: strategies.includes(s.id) ? 'var(--active-fill)' : 'var(--card-border)',
                    background: strategies.includes(s.id) ? 'var(--active-fill)' : 'transparent',
                    color: strategies.includes(s.id) ? 'var(--active-text)' : 'var(--text-sub)',
                  }}
                >
                  <div className="text-left font-mono">
                    <div className="font-semibold text-xs uppercase tracking-wide">{s.label}</div>
                    <div className="text-[9px] opacity-65 mt-0.5">{s.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>


          {/* Sliders */}
          <div className="space-y-4">
            <SliderField label="Paths"       value={nPaths}       min={10}    max={2000}  step={10}     onChange={setNPaths}       display={nPaths.toString()} />
            <SliderField label="Horizon"     value={horizonSteps} min={50}    max={2880}  step={50}     onChange={setHorizonSteps} display={horizonSteps.toString()} />
            <SliderField label="Volatility σ" value={sigma}       min={0.005} max={0.5}   step={0.005}  onChange={setSigma}        display={`${(sigma * 100).toFixed(1)}%`} />
            <SliderField label="Impact η"    value={eta}          min={0.0001} max={0.05} step={0.0001} onChange={setEta}          display={eta.toFixed(4)} />
            <SliderField label="Risk λ"      value={lambda}       min={1e-6}  max={1e-3}  step={1e-6}   onChange={setLambda}       display={lambda.toExponential(1)} />
            <SliderField label="Total Notional" value={totalNotional} min={10000} max={10000000} step={10000} onChange={setTotalNotional} display={totalNotional.toLocaleString()} />
          </div>

          {/* LOB file */}
          {lobFiles && lobFiles.length > 0 && (
            <div className="space-y-2">
              <label className="label-text">Historical LOB Data (optional)</label>
              <select
                value={lobFileId}
                onChange={e => setLobFileId(e.target.value)}
                className="input-field text-xs font-mono uppercase"
              >
                <option value="">None — synthetic prices</option>
                {lobFiles.map(f => (
                  <option key={f.file_id} value={f.file_id}>{f.date_str || f.original_name}</option>
                ))}
              </select>
            </div>
          )}

          {/* Run */}
          <button
            onClick={handleRun}
            disabled={isRunning || strategies.length === 0}
            className="btn-primary w-full"
            id="run-simulation-btn"
          >
            {isRunning ? (
              <>
                <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                Running…
              </>
            ) : (
              <>
                <Play size={12} fill="currentColor" />
                Run Simulation
              </>
            )}
          </button>

          {/* Stop Button */}
          {isRunning && jobId && (
            <button
              onClick={() => handleCancel(jobId)}
              className="w-full py-2 px-4 rounded-lg text-xs font-mono font-semibold uppercase tracking-wider transition-all border border-red-500 hover:bg-red-500/10 text-red-500 flex items-center justify-center gap-1.5 mt-2"
              id="stop-simulation-btn"
            >
              <span className="w-2.5 h-2.5 bg-red-500 rounded-sm animate-pulse" />
              Stop Simulation
            </button>
          )}

          {/* Status */}
          {jobStatus && (
            <div className="flex items-center justify-between pt-3" style={{ borderTop: '1px solid var(--divider)' }}>
              <span className="text-[10px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Status</span>
              <JobStatusBadge status={jobStatus} />
            </div>
          )}
        </div>
      </div>

      {/* ── Right panel ───────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-4">
        {!result && !isRunning && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <EmptyState
              icon="📊"
              title="No results yet"
              description="Configure parameters on the left and click Run Simulation"
            />
          </div>
        )}

        {(result || isRunning) && (
          <>
            {/* Tabs — only show when results are ready */}
            {result && (
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
            )}

            {/* Live progress dashboard while running */}
            {isRunning && !result && (
              <LiveProgressDashboard
                jobType="simulation"
                jobId={jobId}
                jobStatus={jobStatus}
                progress={progress}
                startedAt={jobStartedAt}
                partialResults={partialResults}
              />
            )}

            {/* Results */}
            {result && (
              <div className="glass-card p-6">
                {activeTab === 'dashboard' && (
                  <SimulationResultsPanel result={result} isDark={isDark} />
                )}
                {activeTab === 'trajectory' && (
                  <TrajectoryPanel result={result} isDark={isDark} />
                )}
                {activeTab === 'cost' && (
                  <CostBreakdownPanel result={result} isDark={isDark} />
                )}
                {activeTab === 'table' && (
                  <MetricsTablePanel result={result} />
                )}
                {activeTab === 'price' && (
                  <div>
                    <p className="text-[10px] font-mono uppercase mb-4" style={{ color: 'var(--text-muted)' }}>
                      Simulated price path ({result.price_path.length} steps)
                    </p>
                    <PriceChart pricePoints={result.price_path} />
                  </div>
                )}
              </div>
            )}

            {/* Meta info */}
            {result && (
              <div className="flex items-center gap-4 text-[10px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>
                <span>Job: <span className="font-semibold">{result.job_id.slice(0, 8)}…</span></span>
                {result.duration_seconds && <span>Duration: {result.duration_seconds.toFixed(1)}s</span>}
                <span>{result.strategies.length} strategies · {result.price_path.length} steps</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/* ── Slider field ─────────────────────────────────────────────────────────────── */
function SliderField({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; display: string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <label className="label-text">{label}</label>
        <span className="text-[10px] font-mono font-semibold" style={{ color: 'var(--text)' }}>{display}</span>
      </div>
      <div className="relative h-1.5 rounded-full" style={{ background: 'var(--card-border)' }}>
        <div
          className="absolute left-0 top-0 h-full rounded-full"
          style={{ width: `${pct}%`, background: 'var(--active-fill)' }}
        />
        <input
          type="range"
          min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer"
          style={{ height: '6px' }}
        />
      </div>
    </div>
  )
}

