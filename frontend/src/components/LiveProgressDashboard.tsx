import { useEffect, useState, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import { Check, Clock, TrendingUp, AlertTriangle } from 'lucide-react'

export interface LiveProgressProps {
  jobType: 'simulation' | 'evaluation' | 'sweep'
  jobId: string | null
  jobStatus: string | null
  progress: { completed: number; total: number }
  startedAt?: number
  partialResults?: any // simulation: Record<string, { mean_is: number; cost_series: number[] }>
  completedDates?: any[] // evaluation: array of completed dates results
}

export function LiveProgressDashboard({
  jobType,
  jobId,
  jobStatus,
  progress,
  startedAt,
  partialResults = {},
  completedDates = []
}: LiveProgressProps) {
  const [elapsed, setElapsed] = useState(0)
  const [eta, setEta] = useState<number | null>(null) // in seconds
  const [rate, setRate] = useState<number | null>(null) // items / sec
  const initialTime = useRef(startedAt ?? Date.now())

  // Sync initial started time if it updates
  useEffect(() => {
    if (startedAt) {
      initialTime.current = startedAt
    } else {
      initialTime.current = Date.now()
    }
  }, [startedAt])

  const progressRef = useRef(progress)
  const jobStatusRef = useRef(jobStatus)

  useEffect(() => {
    progressRef.current = progress
    jobStatusRef.current = jobStatus
  }, [progress, jobStatus])

  // Timer loop for elapsed and ETA (every 1 second)
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now()
      const elapsedSec = (now - initialTime.current) / 1000
      setElapsed(elapsedSec)

      const comp = progressRef.current.completed
      const tot = progressRef.current.total

      if (comp > 0 && tot > 0 && comp < tot) {
        const currentRate = comp / elapsedSec
        setRate(currentRate)
        const remaining = tot - comp
        setEta(remaining / currentRate)
      } else if (comp >= tot) {
        setEta(null)
        setRate(null)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  // Progress ring SVG details
  const completed = progress.completed
  const total = progress.total || 100
  const pct = Math.min(100, Math.round((completed / total) * 100))

  const radius = 50
  const stroke = 6
  const normalizedRadius = radius - stroke * 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (pct / 100) * circumference

  // Progress color based on completion percentage
  let progressColor = '#F59E0B' // amber
  if (pct >= 33 && pct < 66) {
    progressColor = '#3B82F6' // blue
  } else if (pct >= 66) {
    progressColor = '#10B981' // green
  }

  // Format elapsed time string
  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return m > 0 ? `${m} min ${s} sec` : `${s} sec`
  };

  // Convert partialResults for Recharts LineChart
  const getChartData = () => {
    const chartData: any[] = []
    const keys = Object.keys(partialResults)
    if (keys.length === 0) return []

    // Find the max length of cost series
    const len = partialResults[keys[0]]?.cost_series?.length || 0
    for (let i = 0; i < len; i++) {
      const point: any = { step: i }
      for (const k of keys) {
        if (partialResults[k]?.cost_series) {
          point[k] = partialResults[k].cost_series[i]
        }
      }
      chartData.push(point)
    }
    return chartData
  };

  const chartData = getChartData()

  // Determine path stroke styles based on path confidence count
  const pathsDone = progress.completed
  let strokeWidth = 1.0
  let strokeOpacity = 0.3
  if (pathsDone >= 50 && pathsDone < 200) {
    strokeWidth = 1.8
    strokeOpacity = 0.6
  } else if (pathsDone >= 200) {
    strokeWidth = 2.5
    strokeOpacity = 1.0
  }

  const STRATEGY_INFO = {
    twap: { label: 'TWAP', color: '#94A3B8' },
    heuristic: { label: 'Heuristic', color: '#F59E0B' },
    optimal: { label: 'AC Optimal', color: '#3B82F6' },
    adaptive: { label: 'AdaptiveAC', color: '#A78BFA' },
    rl: { label: 'RL Agent', color: '#10B981' }
  };

  return (
    <div className="space-y-6 w-full animate-fade-in">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* PANEL 1: Circular Progress Ring */}
        <div className="glass-card p-5 flex flex-col items-center justify-center text-center">
          <div className="relative w-36 h-36 flex items-center justify-center mb-3">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                className="text-slate-700/35"
                strokeWidth={stroke}
                stroke="currentColor"
                fill="transparent"
                r={normalizedRadius}
                cx="50"
                cy="50"
              />
              <circle
                stroke={progressColor}
                strokeWidth={stroke}
                strokeDasharray={circumference + ' ' + circumference}
                style={{ strokeDashoffset, transition: 'stroke-dashoffset 300ms ease-in-out' }}
                strokeLinecap="round"
                fill="transparent"
                r={normalizedRadius}
                cx="50"
                cy="50"
              />
            </svg>
            <div className="absolute text-center">
              <span className="text-3xl font-bold font-mono text-slate-100">{pct}%</span>
              <p className="text-[9px] font-mono uppercase tracking-wider text-slate-400 mt-0.5">Completed</p>
            </div>
          </div>
          <h4 className="text-xs font-mono font-semibold uppercase text-slate-300">
            Job: <span className="text-amber-500 font-bold">{jobId?.slice(0, 8) || '...'}</span>
          </h4>
          <p className="text-[10px] font-mono text-slate-400 mt-1 uppercase">
            Status: <span className="animate-pulse font-bold">{jobStatus || 'running'}</span>
          </p>
        </div>

        {/* PANEL 4: Running Statistics & ETA */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-4">
            <Clock size={14} className="text-slate-400" />
            <h3 className="label-text">Execution Timer & Rates</h3>
          </div>
          <div className="space-y-3.5">
            <div className="flex items-center justify-between border-b border-slate-750 pb-2">
              <span className="text-xs text-slate-400 font-mono uppercase">Elapsed Time</span>
              <span className="text-sm font-mono font-bold text-slate-200">{formatDuration(elapsed)}</span>
            </div>
            <div className="flex items-center justify-between border-b border-slate-750 pb-2">
              <span className="text-xs text-slate-400 font-mono uppercase">Estimated ETA</span>
              <span className="text-sm font-mono font-bold text-slate-200">
                {eta !== null ? formatDuration(eta) : 'Estimating...'}
              </span>
            </div>
            <div className="flex items-center justify-between border-b border-slate-750 pb-2">
              <span className="text-xs text-slate-400 font-mono uppercase">Items Processed</span>
              <span className="text-sm font-mono font-bold text-slate-200">
                {completed.toLocaleString()} / {total.toLocaleString()} {jobType === 'evaluation' ? 'dates' : 'paths'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono uppercase">Processing Speed</span>
              <span className="text-sm font-mono font-bold text-slate-200">
                {rate !== null ? `${rate.toFixed(1)} items/s` : 'Calculating...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* PANEL 2: Live Metric Cards for RL Evaluations */}
      {jobType === 'evaluation' && (
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="label-text">Date Regimes Completed</h3>
            <span className="text-[10px] font-mono text-slate-400">{completedDates.length} completed</span>
          </div>
          
          {completedDates.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-6 text-slate-500 font-mono text-xs border border-dashed border-slate-700 rounded-lg">
              <TrendingUp size={16} className="mb-2 opacity-50" />
              Waiting for first date simulation results...
            </div>
          ) : (
            <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar pr-1">
              {completedDates.map((d, index) => {
                const isBetter = d.improvement_pp >= 0
                return (
                  <div
                    key={d.date + index}
                    className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-3.5 rounded-lg border border-slate-700 bg-slate-800/40 hover:border-slate-600 transition-all animate-fade-in"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-bold text-slate-100">{d.date}</span>
                        <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-400">
                          {d.regime}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs font-mono mt-1 text-slate-300">
                        <span>RL IS: <strong className="text-slate-100">{d.rl_is.toFixed(3)}%</strong></span>
                        <span>vs AC: <strong className="text-slate-100">{d.ac_is.toFixed(3)}%</strong></span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3 mt-2 sm:mt-0">
                      <div className="flex items-center gap-1.5 font-mono text-xs">
                        <span className="text-slate-400">Improvement:</span>
                        <span className={`font-bold ${isBetter ? 'text-emerald-400' : 'text-red-400'}`}>
                          {isBetter ? '+' : ''}{d.improvement_pp.toFixed(3)}pp
                        </span>
                        <span>{isBetter ? '🟢' : '🔴'}</span>
                      </div>
                      <div className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                        <Check size={11} strokeWidth={2.5} />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* PANEL 3: Live IS Chart for Simulations */}
      {jobType === 'simulation' && (
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="label-text">Live Trajectory Convergence</h3>
              <p className="text-[10px] font-mono text-slate-500 uppercase mt-0.5">
                Path count: <strong className="text-slate-300">{progress.completed}</strong> · Line solidity adapts to sample size
              </p>
            </div>
            {progress.completed < 50 && (
              <span className="flex items-center gap-1 text-[9px] font-mono text-amber-500 uppercase bg-amber-500/10 px-1.5 py-0.5 rounded">
                <AlertTriangle size={10} />
                Low confidence path lines
              </span>
            )}
          </div>

          {chartData.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-slate-500 font-mono text-xs border border-dashed border-slate-700 rounded-lg">
              <TrendingUp size={20} className="mb-2 opacity-50 animate-bounce" />
              Accumulating paths to render trajectory lines...
            </div>
          ) : (
            <div className="w-full h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="step" hide />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 9, fill: '#64748B' }} />
                  <Tooltip
                    contentStyle={{
                      background: '#0F172A',
                      border: '1px solid #334155',
                      borderRadius: 8,
                      fontSize: 10,
                      color: '#F8FAFC'
                    }}
                    labelFormatter={(label) => `Execution Step: ${label}`}
                  />
                  {Object.entries(STRATEGY_INFO).map(([key, info]) => {
                    // Only render line if the strategy exists in partialResults
                    if (!partialResults[key]) return null;
                    return (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        name={info.label}
                        stroke={info.color}
                        strokeWidth={strokeWidth}
                        strokeOpacity={strokeOpacity}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
