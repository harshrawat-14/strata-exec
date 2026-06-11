/**
 * Parameter Sweep page — grid configuration + sensitivity charts.
 * All styling via CSS variables.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { BarChart3, Play, Download } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'
import { startSweep, fetchSweepResult } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { JobStatusBadge, ProgressBar, Skeleton, EmptyState } from '../components/ui'
import type { JobStatusValue, WsMessage } from '../types'

const DIMENSION_OPTIONS = [
  { id: 'volatility', label: 'Volatility σ',    desc: 'Sweep annual vol (0.05→0.40)', presets: [0.05, 0.10, 0.15, 0.20, 0.30, 0.40] },
  { id: 'horizon',    label: 'Horizon (steps)',  desc: 'Sweep trading horizon',         presets: [100, 200, 500, 1000, 2000, 2880] },
  { id: 'impact',     label: 'Impact η',         desc: 'Sweep temporary impact',        presets: [0.0005, 0.001, 0.003, 0.005, 0.01] },
  { id: 'slices',     label: 'Trade Chunks',     desc: 'Number of execution slices',    presets: [10, 25, 50, 100, 200, 500] },
]

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

export default function ParameterSweep() {
  const isDark = useDarkMode()

  const [dimension,   setDimension]   = useState('volatility')
  const [gridValues,  setGridValues]  = useState([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
  const [nPaths,      setNPaths]      = useState(100)
  const [strategies,  setStrategies]  = useState(['twap', 'optimal', 'adaptive'])
  const [customInput, setCustomInput] = useState('')

  const [jobId,     setJobId]     = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatusValue | null>(null)
  const [progress,  setProgress]  = useState({ completed: 0, total: 0 })
  const [result,    setResult]    = useState<any>(null)
  const jobStartedAt = useRef<number | undefined>(undefined)

  const selectedDim = DIMENSION_OPTIONS.find(d => d.id === dimension)

  const getStrategyStyle = (name: string) => {
    const k = name.toLowerCase()
    if (k.includes('twap'))     return { stroke: isDark ? '#888' : '#aaa', dash: '3 3' }
    if (k.includes('heuristic'))return { stroke: isDark ? '#aaa' : '#777', dash: '6 3' }
    if (k.includes('optimal') && !k.includes('adaptive'))
                                return { stroke: isDark ? '#fff' : '#000', dash: '0' }
    if (k.includes('adaptive')) return { stroke: isDark ? '#ddd' : '#444', dash: '8 3 2 3' }
    return                             { stroke: isDark ? '#bbb' : '#555', dash: '0' }
  }

  const sweepMut = useMutation({
    mutationFn: startSweep,
    onSuccess: (data: any) => {
      setJobId(data.job_id)
      setJobStatus('queued')
      setProgress({ completed: 0, total: gridValues.length })
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
      const res = await fetchSweepResult(jobId)
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

  const handleRun = () => {
    sweepMut.mutate({ sweep_dimension: dimension, grid_values: gridValues, n_paths: nPaths, strategies })
  }

  const isRunning = jobStatus === 'queued' || jobStatus === 'running'

  const handleCustomGrid = () => {
    try {
      const vals = customInput.split(',').map(s => parseFloat(s.trim())).filter(Boolean)
      if (vals.length >= 2) setGridValues(vals)
    } catch {}
  }

  const toggleStrategy = (id: string) =>
    setStrategies(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id])

  const sensitivityData = result?.sweep_data
    ? buildSensitivityData(result.sweep_data, dimension, strategies)
    : null

  return (
    <div className="flex gap-6 animate-fade-in">

      {/* ── Config ───────────────────────────────────────────────────── */}
      <div className="w-72 flex-shrink-0">
        <div className="glass-card p-5 space-y-5">
          <div className="flex items-center gap-2">
            <BarChart3 size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="label-text">Parameter Sweep</h2>
          </div>

          {/* Dimension selector */}
          <div className="space-y-2">
            <label className="label-text">Sweep Dimension</label>
            <div className="space-y-1">
              {DIMENSION_OPTIONS.map(d => {
                const active = dimension === d.id
                return (
                  <button
                    key={d.id}
                    onClick={() => { setDimension(d.id); setGridValues(d.presets) }}
                    className="w-full flex items-start gap-2.5 px-3 py-2.5 rounded-lg text-xs text-left transition-all"
                    style={{
                      border: '1px solid',
                      borderColor: active ? 'var(--active-fill)' : 'var(--card-border)',
                      background: active ? 'var(--active-fill)' : 'transparent',
                      color: active ? 'var(--active-text)' : 'var(--text-muted)',
                    }}
                  >
                    <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                      style={{ background: active ? 'var(--active-text)' : 'var(--text-muted)', opacity: active ? 1 : 0.3 }} />
                    <div className="text-left font-mono">
                      <div className="font-semibold">{d.label}</div>
                      <div className="text-[9px] opacity-65 mt-0.5">{d.desc}</div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Grid values */}
          <div className="space-y-2">
            <label className="label-text">Grid Values</label>
            <div className="flex flex-wrap gap-1.5">
              {gridValues.map((v, i) => (
                <span
                  key={i}
                  onClick={() => setGridValues(gridValues.filter((_, j) => j !== i))}
                  className="badge-neutral font-mono text-[10px] cursor-pointer transition-all"
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  title="Click to remove"
                >
                  {v}
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="0.10, 0.20, 0.30"
                value={customInput}
                onChange={e => setCustomInput(e.target.value)}
                className="input-field text-xs flex-1 font-mono"
              />
              <button onClick={handleCustomGrid} className="btn-secondary text-xs px-3 font-mono">Set</button>
            </div>
          </div>

          {/* Strategies */}
          <div className="space-y-2">
            <label className="label-text">Strategies</label>
            <div className="flex flex-wrap gap-1.5">
              {['twap', 'heuristic', 'optimal', 'adaptive'].map(s => {
                const active = strategies.includes(s)
                return (
                  <button
                    key={s}
                    onClick={() => toggleStrategy(s)}
                    className="strategy-pill text-[10px]"
                    style={{
                      borderColor: active ? 'var(--active-fill)' : 'var(--card-border)',
                      background: active ? 'var(--active-fill)' : 'transparent',
                      color: active ? 'var(--active-text)' : 'var(--text-muted)',
                    }}
                  >
                    {s}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Paths slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between">
              <label className="label-text">Paths per Config</label>
              <span className="text-[10px] font-mono font-semibold" style={{ color: 'var(--text)' }}>{nPaths}</span>
            </div>
            <div className="relative h-1.5 rounded-full" style={{ background: 'var(--card-border)' }}>
              <div
                className="absolute left-0 top-0 h-full rounded-full"
                style={{ width: `${((nPaths - 10) / 490) * 100}%`, background: 'var(--active-fill)' }}
              />
              <input
                type="range" min={10} max={500} step={10} value={nPaths}
                onChange={e => setNPaths(parseInt(e.target.value))}
                className="absolute inset-0 w-full opacity-0 cursor-pointer"
                style={{ height: '6px' }}
              />
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={isRunning || gridValues.length < 2 || strategies.length === 0}
            className="btn-primary w-full"
          >
            {isRunning
              ? <><div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />Running…</>
              : <><Play size={12} className="fill-current" />Run Sweep ({gridValues.length} configs)</>
            }
          </button>

          {isRunning && (
            <ProgressBar
              completed={progress.completed}
              total={progress.total || gridValues.length}
              label="Configurations"
              startedAt={jobStartedAt.current}
            />
          )}

          {jobStatus && <JobStatusBadge status={jobStatus} />}
        </div>
      </div>

      {/* ── Results ──────────────────────────────────────────────────── */}
      <div className="flex-1 space-y-4">
        {!result && !isRunning && (
          <div className="glass-card flex items-center justify-center" style={{ height: 500 }}>
            <EmptyState
              icon={<BarChart3 size={28} />}
              title="No sweep results"
              description="Configure a sweep grid and click Run Sweep"
            />
          </div>
        )}

        {isRunning && !result && (
          <div className="glass-card p-6 space-y-5">
            <div className="p-4 rounded-xl" style={{ background: 'var(--card-hover)' }}>
              <ProgressBar
                completed={progress.completed}
                total={progress.total || gridValues.length}
                label={`Sweeping ${selectedDim?.label}`}
                startedAt={jobStartedAt.current}
              />
            </div>
            {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
          </div>
        )}

        {result && (
          <>
            {/* Header */}
            <div className="glass-card p-5 flex justify-between items-center">
              <div>
                <h2 className="text-base font-bold tracking-tight capitalize" style={{ color: 'var(--text)' }}>
                  {result.sweep_dimension} Sensitivity Analysis
                </h2>
                <p className="text-xs font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  {result.grid_values?.length} grid points · {result.duration_seconds?.toFixed(1)}s
                </p>
              </div>
              <button onClick={() => downloadSweepJson(result)} className="btn-secondary text-xs font-mono">
                <Download size={13} />
                Download JSON
              </button>
            </div>

            {/* Sensitivity chart */}
            {sensitivityData && (
              <div className="glass-card p-5">
                <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
                  Mean IS% vs {selectedDim?.label}
                </h3>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={sensitivityData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" tickLine={false} />
                    <YAxis tickLine={false} axisLine={false} tickFormatter={v => `${v.toFixed(2)}%`} />
                    <Tooltip
                      formatter={(v: any) => [`${Number(v).toFixed(4)}%`, '']}
                      contentStyle={{
                        background: 'var(--card)',
                        border: '1px solid var(--card-border)',
                        borderRadius: 8,
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                        color: 'var(--text)',
                      }}
                    />
                    <Legend />
                    {strategies.map(s => {
                      const style = getStrategyStyle(s)
                      return (
                        <Line key={s} type="monotone" dataKey={s}
                          stroke={style.stroke} strokeDasharray={style.dash} strokeWidth={1.5}
                          dot={{ r: 3, fill: style.stroke, strokeWidth: 0 }}
                          activeDot={{ r: 4, strokeWidth: 0 }}
                        />
                      )
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Raw data */}
            {result.sweep_data && (
              <div className="glass-card p-5">
                <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
                  Raw Sweep Data
                </h3>
                <div className="overflow-x-auto custom-scrollbar">
                  <pre className="text-[10px] font-mono whitespace-pre-wrap" style={{ color: 'var(--text-sub, var(--text-muted))' }}>
                    {JSON.stringify(result.sweep_data, null, 2).slice(0, 2000)}
                    {JSON.stringify(result.sweep_data, null, 2).length > 2000 ? '\n… (truncated)' : ''}
                  </pre>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function buildSensitivityData(sweepData: any, dimension: string, strategies: string[]): any[] {
  const dimKey = dimension === 'slices' ? 'trade_chunks' : dimension
  const rows = sweepData[dimKey] || []
  if (!rows.length) return []
  const grouped: Record<number, Record<string, number>> = {}
  for (const row of rows) {
    const x = row[dimKey] ?? row['volatility'] ?? row['horizon'] ?? row['impact'] ?? 0
    if (!grouped[x]) grouped[x] = {}
    for (const s of strategies) {
      const col = `${s}_mean_is_pct`
      if (row[col] !== undefined) grouped[x][s] = row[col]
      for (const k of Object.keys(row)) {
        if (k.toLowerCase().includes(s) && k.includes('IS')) grouped[x][s] = row[k]
      }
    }
  }
  return Object.entries(grouped)
    .map(([x, vals]) => ({ x: parseFloat(x), ...vals }))
    .sort((a, b) => a.x - b.x)
}

function downloadSweepJson(result: any) {
  const json = JSON.stringify(result.sweep_data || result, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `sweep_${result.sweep_dimension}_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
}
