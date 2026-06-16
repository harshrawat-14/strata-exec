/**
 * RL Evaluation page — model selector, date picker, results table + charts.
 * All styling via CSS variables.
 */

import { useState, useCallback, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { BrainCircuit, Play } from 'lucide-react'
import { startEvaluation, fetchEvaluationResult, fetchUploadedModels, cancelJob } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { JobStatusBadge, ProgressBar, Skeleton, EmptyState } from '../components/ui'
import EvaluationResultsPanel from '../components/EvaluationResultsPanel'
import { LiveProgressDashboard } from '../components/LiveProgressDashboard'
import type { JobStatusValue, EvaluationResult, WsMessage } from '../types'



const KNOWN_DATES = [
  { date: '2024-01-15', regime: 'Calm bull' },
  { date: '2024-03-05', regime: 'BTC breakout' },
  { date: '2024-06-10', regime: 'Quiet consolidation' },
  { date: '2024-08-05', regime: 'Crash — Yen unwind' },
  { date: '2024-11-06', regime: 'Post-election surge' },
]

export default function RLEvaluation() {
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDates, setSelectedDates] = useState<string[]>(['2024-01-15', '2024-08-05'])
  const [nEpisodes,     setNEpisodes]     = useState(30)

  const [jobId,     setJobId]     = useState<string | null>(() => sessionStorage.getItem('activeJobId_evaluation'))
  const [jobStatus, setJobStatus] = useState<JobStatusValue | null>(null)
  const [progress,  setProgress]  = useState({ completed: 0, total: 0 })
  const [result,    setResult]    = useState<EvaluationResult | null>(null)
  const [completedDates, setCompletedDates] = useState<any[]>([])
  const [jobStartedAt,   setJobStartedAt]   = useState<number | undefined>(undefined)

  useEffect(() => {
    if (jobId) {
      sessionStorage.setItem('activeJobId_evaluation', jobId)
    } else {
      sessionStorage.removeItem('activeJobId_evaluation')
    }
  }, [jobId])

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['models'],
    queryFn: fetchUploadedModels,
  })

  const filteredModelsList = (models || [])
    .filter(m => {
      const name = m.name.toLowerCase()
      return name === 'smoke_v5_final' || name === 'ppo_lstm_v5_adaptive_best'
    })
    .map(m => {
      const name = m.name.toLowerCase()
      let cleanName = m.name
      if (name === 'smoke_v5_final') {
        cleanName = 'SMOKE-V5 (Impact-Robust Liquidator)'
      } else if (name === 'ppo_lstm_v5_adaptive_best') {
        cleanName = 'PPO-LSTM (Regime-Adaptive Liquidator)'
      }
      return { ...m, name: cleanName }
    })

  useEffect(() => {
    if (filteredModelsList.length && !selectedModel) {
      const timer = setTimeout(() => {
        setSelectedModel(filteredModelsList[0].model_id)
      }, 0)
      return () => clearTimeout(timer)
    }
  }, [filteredModelsList, selectedModel])


  const evalMut = useMutation({
    mutationFn: startEvaluation,
    onSuccess: (data) => {
      setJobId(data.job_id)
      setJobStatus('queued')
      setProgress({ completed: 0, total: selectedDates.length })
      setCompletedDates([])
      setResult(null)
      setJobStartedAt(Date.now())
    },
  })

  const handleWsMessage = useCallback(async (msg: WsMessage) => {
    if (msg.started_at) {
      setJobStartedAt(msg.started_at)
    }

    if (msg.type === 'progress') {
      setProgress({ completed: msg.completed, total: msg.total })
      setJobStatus('running')
    } else if (msg.type === 'date_complete') {
      setCompletedDates(prev => {
        if (prev.some(d => d.date === msg.date)) return prev
        return [...prev, msg]
      })
      setProgress({ completed: msg.dates_done, total: msg.dates_total })
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
    enabled: !!jobId && jobStatus !== 'complete' && jobStatus !== 'failed',
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

  const handleCancel = async (id: string) => {
    try {
      await cancelJob(id)
      setJobStatus('failed')
    } catch (err) {
      console.error("Failed to cancel RL evaluation", err)
    }
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
                className="input-field text-xs font-semibold"
                id="model-selector"
              >
                <option value="">Select a model…</option>
                {filteredModelsList.map(m => (
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

          {/* Stop Button */}
          {isRunning && jobId && (
            <button
              onClick={() => handleCancel(jobId)}
              className="w-full py-2 px-4 rounded-lg text-xs font-mono font-semibold uppercase tracking-wider transition-all border border-red-500 hover:bg-red-500/10 text-red-500 flex items-center justify-center gap-1.5 mt-2"
              id="stop-evaluation-btn"
            >
              <span className="w-2.5 h-2.5 bg-red-500 rounded-sm animate-pulse" />
              Stop Evaluation
            </button>
          )}

          {isRunning && (
            <ProgressBar
              completed={progress.completed}
              total={progress.total || nEpisodes * selectedDates.length}
              label="Episodes"
              startedAt={jobStartedAt}
            />
          )}

          {jobStatus && <JobStatusBadge status={jobStatus} />}
        </div>
      </div>

      {/* ── Results panel ─────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-4">
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
          <LiveProgressDashboard
            jobType="evaluation"
            jobId={jobId}
            jobStatus={jobStatus}
            progress={progress}
            startedAt={jobStartedAt}
            completedDates={completedDates}
          />
        )}

        {result && (
          <EvaluationResultsPanel result={result} />
        )}
      </div>
    </div>
  )
}
