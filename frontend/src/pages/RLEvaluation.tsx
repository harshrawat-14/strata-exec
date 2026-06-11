/**
 * RL Evaluation page — model selector, date picker, results table + charts.
 * All styling via CSS variables.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { BrainCircuit, Play } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'

import { startEvaluation, fetchEvaluationResult, fetchUploadedModels } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { JobStatusBadge, ProgressBar, Skeleton, EmptyState, IS } from '../components/ui'
import type { JobStatusValue, EvaluationResult, WsMessage } from '../types'

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

const KNOWN_DATES = [
  { date: '2024-01-15', regime: 'Calm bull' },
  { date: '2024-03-05', regime: 'BTC breakout' },
  { date: '2024-06-10', regime: 'Quiet consolidation' },
  { date: '2024-08-05', regime: 'Crash — Yen unwind' },
  { date: '2024-11-06', regime: 'Post-election surge' },
]

export default function RLEvaluation() {
  const isDark = useDarkMode()
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDates, setSelectedDates] = useState<string[]>(['2024-01-15', '2024-08-05'])
  const [nEpisodes,     setNEpisodes]     = useState(30)

  const [jobId,     setJobId]     = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatusValue | null>(null)
  const [progress,  setProgress]  = useState({ completed: 0, total: 0 })
  const [result,    setResult]    = useState<EvaluationResult | null>(null)
  const jobStartedAt = useRef<number | undefined>(undefined)

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['models'],
    queryFn: fetchUploadedModels,
  })

  if (models?.length && !selectedModel) setSelectedModel(models[0].model_id)

  const evalMut = useMutation({
    mutationFn: startEvaluation,
    onSuccess: (data) => {
      setJobId(data.job_id)
      setJobStatus('queued')
      setProgress({ completed: 0, total: nEpisodes * selectedDates.length })
      setResult(null)
      jobStartedAt.current = Date.now()
    },
  })

  const handleWsMessage = useCallback(async (msg: WsMessage) => {
    if (msg.type === 'progress') {
      setProgress({ completed: msg.completed, total: msg.total })
      setJobStatus('running')
    } else if (msg.type === 'status') {
      setJobStatus(msg.status)
    } else if (msg.type === 'complete' && jobId) {
      setJobStatus('complete')
      const res = await fetchEvaluationResult(jobId)
      setResult(res)
    } else if (msg.type === 'error') {
      setJobStatus('failed')
    }
  }, [jobId])

  useWebSocket({
    jobId,
    onMessage: handleWsMessage,
    enabled: jobStatus === 'queued' || jobStatus === 'running',
  })

  const isRunning = jobStatus === 'queued' || jobStatus === 'running'

  const toggleDate = (d: string) =>
    setSelectedDates(prev => prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d])

  const handleRun = () => {
    if (!selectedModel || !selectedDates.length) return
    evalMut.mutate({
      model_id: selectedModel,
      dates: selectedDates,
      n_episodes: nEpisodes,
      compare_with: ['optimal', 'adaptive'],
    })
  }

  return (
    <div className="flex gap-6 animate-fade-in">

      {/* ── Config panel ──────────────────────────────────────────────────── */}
      <div className="w-72 flex-shrink-0">
        <div className="glass-card p-5 space-y-5">
          <div className="flex items-center gap-2">
            <BrainCircuit size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="label-text">RL Evaluation</h2>
          </div>

          {/* Model selector */}
          <div className="space-y-2">
            <label className="label-text">RL Model</label>
            {modelsLoading ? (
              <Skeleton className="h-10" />
            ) : (
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
                className="input-field text-xs font-mono uppercase"
                id="model-selector"
              >
                <option value="">Select a model…</option>
                {models?.map(m => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name}{m.is_builtin ? ' ★' : ''} ({(m.file_size_bytes / 1024).toFixed(0)} KB)
                  </option>
                ))}
              </select>
            )}
            <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>★ = built-in models</p>
          </div>

          {/* Date selection */}
          <div className="space-y-2">
            <label className="label-text">Evaluation Dates</label>
            <div className="space-y-1.5">
              {KNOWN_DATES.map(({ date, regime }) => {
                const active = selectedDates.includes(date)
                return (
                  <button
                    key={date}
                    onClick={() => toggleDate(date)}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-mono uppercase transition-all text-left"
                    style={{
                      border: '1px solid',
                      borderColor: active ? 'var(--active-fill)' : 'var(--card-border)',
                      background: active ? 'var(--active-fill)' : 'transparent',
                      color: active ? 'var(--active-text)' : 'var(--text-muted)',
                    }}
                  >
                    <div
                      className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                      style={{ background: active ? 'var(--active-text)' : 'var(--text-muted)', opacity: active ? 1 : 0.3 }}
                    />
                    <div className="text-left">
                      <div className="font-semibold">{date}</div>
                      <div className="text-[9px] opacity-65 mt-0.5">{regime}</div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Episodes slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between">
              <label className="label-text">Episodes per Date</label>
              <span className="text-[10px] font-mono font-semibold" style={{ color: 'var(--text)' }}>{nEpisodes}</span>
            </div>
            <div className="relative h-1.5 rounded-full" style={{ background: 'var(--card-border)' }}>
              <div
                className="absolute left-0 top-0 h-full rounded-full"
                style={{
                  width: `${((nEpisodes - 5) / 195) * 100}%`,
                  background: 'var(--active-fill)',
                }}
              />
              <input
                type="range" min={5} max={200} step={5} value={nEpisodes}
                onChange={e => setNEpisodes(parseInt(e.target.value))}
                className="absolute inset-0 w-full opacity-0 cursor-pointer"
                style={{ height: '6px' }}
              />
            </div>
            <p className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>Higher = more accurate</p>
          </div>

          <button
            onClick={handleRun}
            disabled={isRunning || !selectedModel || !selectedDates.length}
            className="btn-primary w-full"
          >
            {isRunning ? (
              <>
                <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                Evaluating…
              </>
            ) : (
              <>
                <Play size={12} fill="currentColor" />
                Evaluate Model
              </>
            )}
          </button>

          {isRunning && (
            <ProgressBar
              completed={progress.completed}
              total={progress.total || nEpisodes * selectedDates.length}
              label="Episodes"
              startedAt={jobStartedAt.current}
            />
          )}

          {jobStatus && <JobStatusBadge status={jobStatus} />}
        </div>
      </div>

      {/* ── Results panel ─────────────────────────────────────────────────── */}
      <div className="flex-1 space-y-4">
        {!result && !isRunning && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <EmptyState
              icon={<BrainCircuit size={28} />}
              title="No evaluation results"
              description="Select a model and dates, then click Evaluate Model"
            />
          </div>
        )}

        {isRunning && !result && (
          <div className="glass-card p-6 space-y-6">
            <div className="p-4 rounded-xl" style={{ background: 'var(--card-hover)' }}>
              <ProgressBar
                completed={progress.completed}
                total={progress.total || nEpisodes * selectedDates.length}
                label="Evaluating episodes"
                startedAt={jobStartedAt.current}
              />
            </div>
            {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
          </div>
        )}

        {result && (
          <>
            {/* Header card */}
            <div className="glass-card p-5">
              <h2 className="text-sm font-semibold font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text)' }}>
                {result.model_name}
              </h2>
              <p className="text-[10px] font-mono uppercase mb-5" style={{ color: 'var(--text-muted)' }}>
                {result.date_results.length} dates · {nEpisodes} episodes each
                {result.duration_seconds && ` · ${result.duration_seconds.toFixed(1)}s`}
              </p>

              {/* IS comparison table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--divider)' }}>
                      {['Date', 'Regime', 'TWAP', 'Heuristic', 'Optimal AC', 'Adaptive AC', 'RL Agent', 'P-value', 'Better?'].map(h => (
                        <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.date_results.map(dr => (
                      <tr
                        key={dr.date}
                        style={{ borderBottom: '1px solid var(--divider)' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                      >
                        <td className="px-3 py-3 font-mono text-xs" style={{ color: 'var(--text)' }}>{dr.date}</td>
                        <td className="px-3 py-3 text-xs font-mono uppercase" style={{ color: 'var(--text-muted)' }}>{dr.regime}</td>
                        <td className="px-3 py-3"><IS value={dr.twap_is} /></td>
                        <td className="px-3 py-3"><IS value={dr.heuristic_is} /></td>
                        <td className="px-3 py-3"><IS value={dr.static_optimal_is} /></td>
                        <td className="px-3 py-3"><IS value={dr.adaptive_optimal_is} /></td>
                        <td className="px-3 py-3 font-bold"><IS value={dr.mean_is_pct} /></td>
                        <td className="px-3 py-3 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                          {dr.p_value !== null ? dr.p_value.toFixed(3) : '—'}
                        </td>
                        <td className="px-3 py-3">
                          {dr.significantly_better === null
                            ? <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>—</span>
                            : dr.significantly_better
                            ? <span className="badge-success text-[9px]">✓ Yes</span>
                            : <span className="badge-neutral text-[9px]">✗ No</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Action distribution charts */}
            {result.date_results.some(dr => Object.keys(dr.action_distribution).length > 0) && (
              <div className="glass-card p-5">
                <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
                  Action Distribution (Fraction of episode steps)
                </h3>
                <div className="grid grid-cols-2 gap-6">
                  {result.date_results.map(dr => {
                    const dist = Object.entries(dr.action_distribution)
                      .map(([a, f]) => ({ action: parseInt(a), fraction: f }))
                      .sort((a, b) => a.action - b.action)
                    return (
                      <div key={dr.date} className="space-y-2">
                        <div className="text-[10px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>
                          {dr.date} — {dr.regime}
                        </div>
                        <ResponsiveContainer width="100%" height={80}>
                          <BarChart data={dist} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                            <XAxis dataKey="action" tickLine={false} />
                            <YAxis hide />
                            <Tooltip
                              contentStyle={{
                                background: 'var(--card)',
                                border: '1px solid var(--card-border)',
                                borderRadius: 6,
                                fontSize: 10,
                                fontFamily: 'JetBrains Mono, monospace',
                                color: 'var(--text)',
                              }}
                              formatter={(v: any) => [`${(Number(v) * 100).toFixed(1)}%`, 'Fraction']}
                            />
                            <Bar dataKey="fraction" radius={[2, 2, 0, 0]}>
                              {dist.map(entry => (
                                <Cell
                                  key={entry.action}
                                  fill={isDark
                                    ? `rgba(255,255,255,${0.15 + (entry.action / 19) * 0.6})`
                                    : `rgba(0,0,0,${0.1 + (entry.action / 19) * 0.6})`}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
