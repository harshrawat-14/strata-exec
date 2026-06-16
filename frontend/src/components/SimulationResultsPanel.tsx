import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, ErrorBar, Cell,
  LineChart, Line
} from 'recharts'
import { ArrowUpDown, Info } from 'lucide-react'
import type { SimulationResult, StrategyResult } from '../types'
import { IS } from './ui'

interface SimulationResultsPanelProps {
  result: SimulationResult
  isDark: boolean
}

// Helper to resolve colors for the 5 strategies consistently
export const getStrategyColor = (name: string) => {
  const n = name.toLowerCase()
  if (n.includes('twap')) return '#94A3B8'        // Slate
  if (n.includes('heuristic')) return '#F59E0B'   // Amber
  if (n.includes('optimal (ac)') || n.includes('static') || n === 'optimal' || (n.includes('optimal') && !n.includes('adaptive'))) {
    return '#3B82F6' // Blue
  }
  if (n.includes('adaptive')) return '#A78BFA'    // Violet
  if (n.includes('rl') || n.includes('ppo')) return '#10B981' // Emerald
  return '#6B7280' // default fallback
}

// Helper to standardise labels for the 5 strategies
export const getStrategyLabel = (name: string) => {
  const n = name.toLowerCase()
  if (n.includes('twap')) return 'TWAP'
  if (n.includes('heuristic')) return 'Heuristic'
  if (n.includes('optimal (ac)') || n.includes('static') || n === 'optimal') return 'AC Optimal'
  if (n.includes('adaptive')) return 'AdaptiveAC'
  if (n.includes('rl') || n.includes('ppo')) return 'RL Agent ✦'
  return name
}

// Helper to convert cost keys to labels
const getCostLabel = (key: string) => {
  switch (key) {
    case 'spread_cost': return 'Spread Cost'
    case 'temporary_impact': return 'Temporary Impact'
    case 'permanent_impact': return 'Permanent Impact'
    case 'timing_cost': return 'Timing Cost'
    case 'opportunity_cost': return 'Opportunity Cost'
    default: return key
  }
}

// Palette for cost decomposition components
const COST_COLORS: Record<string, string> = {
  spread_cost: '#64748B',      // Slate
  temporary_impact: '#3B82F6', // Blue
  permanent_impact: '#EF4444', // Rose/Red
  timing_cost: '#F59E0B',      // Amber
  opportunity_cost: '#10B981', // Emerald
}

type SortKey = 'name' | 'mean_is_pct' | 'is_variance' | 'cvar95' | 'ac_objective' | 'trade_count' | 'avg_exec_price' | 'savings'

export default function SimulationResultsPanel({ result, isDark }: SimulationResultsPanelProps) {
  const [sortKey, setSortKey] = useState<SortKey>('mean_is_pct')
  const [sortAsc, setSortAsc] = useState(true)

  // 1. Parse number of paths for CI calculations (standard error)
  const nPaths = (result.params_used?.n_paths as number) || 100

  // 2. Locate TWAP mean for savings comparisons
  const twapStrat = result.strategies.find(s => s.name.toLowerCase().includes('twap'))
  const twapMean = twapStrat ? (twapStrat.mean_is_pct ?? 0) : 0

  // 3. Prepare dataset for the Grouped Bar Chart of IS Means + 95% CI
  const isMeansData = result.strategies.map(s => {
    const mean = s.mean_is_pct ?? 0
    const variance = s.is_variance ?? 0
    const stdDev = Math.sqrt(variance)
    const stdError = stdDev / Math.sqrt(nPaths)
    const ciError = 1.96 * stdError
    const ciLower = mean - ciError
    const ciUpper = mean + ciError

    return {
      name: s.name,
      label: getStrategyLabel(s.name),
      mean,
      variance,
      stdDev,
      ci_error: [ciError, ciError], // formatted for Recharts ErrorBar error bounds [low, high]
      ci_lower: ciLower,
      ci_upper: ciUpper,
      color: getStrategyColor(s.name)
    }
  })

  // 4. Prepare dataset for the Execution Trajectory Line Chart
  const maxLen = Math.max(...result.strategies.map((s) => s.trajectory.length))
  const trajectoryData = Array.from({ length: maxLen }, (_, i) => {
    const pt: any = { step: i }
    for (const s of result.strategies) {
      pt[s.name] = s.trajectory[i] ?? null
    }
    return pt
  })

  // 5. Prepare dataset for the Stacked Cost Decomposition Chart
  const costDecompData = result.strategies.map(s => {
    const decomp = s.cost_decomposition || {}
    return {
      name: getStrategyLabel(s.name),
      spread_cost: decomp.spread_cost || 0,
      temporary_impact: decomp.temporary_impact || 0,
      permanent_impact: decomp.permanent_impact || 0,
      timing_cost: decomp.timing_cost || 0,
      opportunity_cost: decomp.opportunity_cost || 0,
    }
  })

  // 6. Handle strategy comparison table sorting
  const getSortVal = (s: StrategyResult, key: SortKey) => {
    if (key === 'name') return getStrategyLabel(s.name)
    if (key === 'savings') {
      const mean = s.mean_is_pct ?? 0
      return twapMean - mean
    }
    return s[key as keyof StrategyResult] ?? Infinity
  }

  const sortedStrategies = [...result.strategies].sort((a, b) => {
    const av = getSortVal(a, sortKey)
    const bv = getSortVal(b, sortKey)

    if (typeof av === 'string' && typeof bv === 'string') {
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
    }
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number)
  })

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(true)
    }
  }

  // Header helpers
  const Th = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none border-b border-[var(--divider)]"
      style={{ color: 'var(--text-muted)' }}
      onClick={() => toggleSort(k)}
    >
      <span className="flex items-center gap-1 hover:text-[var(--text)] transition-colors">
        {label}
        <ArrowUpDown size={11} className="opacity-40" />
      </span>
    </th>
  )

  // Custom tooltips
  const MeanISTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null
    const data = payload[0].payload
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[200px]">
        <div className="font-semibold text-black dark:text-white uppercase mb-2 flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full" style={{ background: data.color }} />
          {data.label}
        </div>
        <div className="flex justify-between gap-4">
          <span>Mean IS%:</span>
          <span className="font-bold">{data.mean >= 0 ? '+' : ''}{data.mean.toFixed(3)}%</span>
        </div>
        <div className="flex justify-between gap-4">
          <span>95% CI:</span>
          <span className="font-medium text-black/60 dark:text-white/60">
            [{data.ci_lower.toFixed(3)}%, {data.ci_upper.toFixed(3)}%]
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span>Variance:</span>
          <span className="font-medium text-black/60 dark:text-white/60">
            {data.variance.toFixed(6)}
          </span>
        </div>
        <div className="text-[9px] text-amber-500/90 leading-normal max-w-[220px] pt-1.5 border-t border-black/5 dark:border-white/5 mt-1.5">
          * Negative IS = sold above arrival price (gain).
        </div>
      </div>
    )
  }

  const TrajectoryTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[200px]">
        <div className="text-black/30 dark:text-white/30 mb-2 uppercase font-semibold">Step {label}</div>
        {payload.map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: getStrategyColor(p.dataKey) }} />
              <span>{getStrategyLabel(p.dataKey)}</span>
            </div>
            <span className="font-bold">
              {p.value !== null ? `${Math.round(p.value).toLocaleString()}` : '—'}
            </span>
          </div>
        ))}
      </div>
    )
  }

  const CostDecompTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null
    const total = payload.reduce((sum: number, p: any) => sum + (p.value || 0), 0)
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[220px]">
        <div className="text-black/40 dark:text-white/40 mb-2 uppercase font-bold">{label}</div>
        {payload.map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ background: COST_COLORS[p.dataKey] }} />
              <span>{getCostLabel(p.dataKey)}</span>
            </div>
            <span className="font-semibold">${p.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
          </div>
        ))}
        <div className="border-t border-dashed border-black/10 dark:border-white/10 pt-1.5 mt-1.5 flex justify-between font-bold">
          <span>Total Slippage Cost:</span>
          <span>${total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
      {/* Defs block to hold glow filters */}
      <svg width="0" height="0" className="absolute">
        <defs>
          <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      {/* ── Left Column ────────────────────────────────────────────────────── */}
      <div className="space-y-6">
        {/* IS Mean Grouped Bar Chart */}
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>
              Mean Implementation Shortfall & 95% CI
            </h3>
            <div className="flex items-center gap-1 text-[10px] font-mono text-amber-500/90">
              <Info size={11} />
              <span>Lower IS is better</span>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={isMeansData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
              <XAxis 
                dataKey="label" 
                tickLine={false} 
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
              />
              <YAxis 
                tickLine={false} 
                axisLine={false} 
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
                label={{ value: 'Shortfall (IS %)', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
              />
              <Tooltip content={<MeanISTooltip />} />
              <ReferenceLine 
                y={0} 
                stroke="#EF4444" 
                strokeDasharray="4 4" 
                label={{ value: 'Arrival Price', position: 'right', fill: '#EF4444', fontSize: 8, fontFamily: 'monospace', fontWeight: 600 }} 
              />
              <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
                {isMeansData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
                {/* 95% CI Error bars */}
                <ErrorBar 
                  dataKey="ci_error" 
                  width={6} 
                  stroke={isDark ? '#E2E8F0' : '#475569'} 
                  strokeWidth={1.5} 
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Depletion curve / Execution Trajectory Chart */}
        <div className="glass-card p-5 space-y-4">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>
            Execution Inventory Trajectory
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={trajectoryData} margin={{ top: 10, right: 16, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
              <XAxis 
                dataKey="step" 
                tickLine={false} 
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
                label={{ value: 'TRADING STEP', position: 'insideBottomRight', offset: -5, style: { fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
              />
              <YAxis 
                tickLine={false} 
                axisLine={false} 
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
                label={{ value: 'Remaining Shares', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
              />
              <Tooltip content={<TrajectoryTooltip />} />
              <Legend 
                verticalAlign="bottom" 
                height={32}
                iconType="circle"
                wrapperStyle={{ fontSize: 9, fontFamily: 'monospace', textTransform: 'uppercase' }}
                formatter={(value) => getStrategyLabel(value)}
              />
              {/* x=25% reference line to show rapid initial execution */}
              <ReferenceLine 
                x={Math.round(maxLen * 0.25)} 
                stroke={isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'} 
                strokeDasharray="3 3" 
                label={{ value: '25% Horizon', position: 'top', fill: 'var(--text-muted)', fontSize: 8, fontFamily: 'monospace' }} 
              />
              {result.strategies.map((s) => {
                const color = getStrategyColor(s.name)
                const isRL = s.name.toLowerCase().includes('rl') || s.name.toLowerCase().includes('ppo')
                return (
                  <Line
                    key={s.name}
                    type="monotone"
                    dataKey={s.name}
                    stroke={color}
                    strokeWidth={isRL ? 2.5 : 1.5}
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 0, fill: color }}
                    connectNulls
                    filter={isRL ? 'url(#glow-emerald)' : undefined}
                  />
                )
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Right Column ───────────────────────────────────────────────────── */}
      <div className="space-y-6">
        {/* Strategy comparison table */}
        <div className="glass-card p-5 space-y-4">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>
            Strategy Performance Metric Summary
          </h3>
          
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)]" style={{ color: 'var(--text-muted)' }}>
                    Strategy
                  </th>
                  <Th k="mean_is_pct" label="Mean IS%" />
                  <Th k="is_variance" label="Std Dev" />
                  <Th k="cvar95" label="CVaR95" />
                  <Th k="ac_objective" label="AC Obj" />
                  <Th k="savings" label="vs TWAP Savings" />
                </tr>
              </thead>
              <tbody>
                {sortedStrategies.map((s) => {
                  const color = getStrategyColor(s.name)
                  const label = getStrategyLabel(s.name)
                  
                  // Compute std deviation from variance
                  const stdDev = s.is_variance !== null ? Math.sqrt(s.is_variance) : null
                  
                  // Compute savings vs TWAP (expressed in percentage points)
                  const diff = s.name.toLowerCase().includes('twap') ? 0 : twapMean - (s.mean_is_pct ?? 0)
                  const savingsUsd = diff * 1_000_000 / 100 // based on $1M notional

                  const isRL = s.name.toLowerCase().includes('rl') || s.name.toLowerCase().includes('ppo')

                  return (
                    <tr
                      key={s.name}
                      style={{ borderBottom: '1px solid var(--divider)', transition: 'background 0.15s ease' }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                    >
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2">
                          <span 
                            className="w-2 h-2 rounded-full flex-shrink-0" 
                            style={{ 
                              background: color,
                              boxShadow: isRL ? '0 0 6px #10B981' : 'none' 
                            }} 
                          />
                          <span className={`font-mono text-xs ${isRL ? 'font-bold text-emerald-500' : 'text-[var(--text)]'}`}>
                            {label}
                          </span>
                        </div>
                      </td>
                      <td className="px-3 py-3"><IS value={s.mean_is_pct} /></td>
                      <td className="px-3 py-3 text-xs font-mono text-[var(--text-muted)]">
                        {stdDev !== null ? `${stdDev.toFixed(3)}%` : '—'}
                      </td>
                      <td className="px-3 py-3"><IS value={s.cvar95} /></td>
                      <td className="px-3 py-3 text-xs font-mono text-[var(--text-muted)]">
                        {s.ac_objective !== null ? s.ac_objective.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-3 font-mono text-xs">
                        {s.name.toLowerCase().includes('twap') ? (
                          <span className="text-[var(--text-muted)] opacity-50">—</span>
                        ) : diff > 0 ? (
                          <span className="text-emerald-500 font-semibold">
                            +{diff.toFixed(2)}% (+${Math.round(savingsUsd).toLocaleString()})
                          </span>
                        ) : diff < 0 ? (
                          <span className="text-rose-500">
                            {diff.toFixed(2)}% (-${Math.round(Math.abs(savingsUsd)).toLocaleString()})
                          </span>
                        ) : (
                          <span className="text-[var(--text-muted)]">0.00%</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Grouped Cost Decomposition Chart — one color per cost component */}
        <div className="glass-card p-5 space-y-4">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>
            Slippage Cost Decomposition ($)
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={costDecompData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }} barCategoryGap="20%" barGap={2}>
              <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
              <XAxis 
                dataKey="name" 
                tickLine={false} 
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
              />
              <YAxis 
                tickLine={false} 
                axisLine={false} 
                tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                tick={{ fontSize: 9, fontFamily: 'monospace' }} 
                label={{ value: 'Cost (USD $)', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
              />
              <Tooltip content={<CostDecompTooltip />} />
              <Legend 
                verticalAlign="bottom" 
                height={36}
                iconType="rect"
                wrapperStyle={{ fontSize: 8, fontFamily: 'monospace' }}
                formatter={(value) => getCostLabel(value).toUpperCase()}
              />
              <ReferenceLine y={0} stroke={isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'} strokeDasharray="3 3" />
              {/* Each component is its own grouped bar — gives clear distinct colors per component */}
              <Bar dataKey="spread_cost" fill={COST_COLORS.spread_cost} radius={[2, 2, 0, 0]} maxBarSize={18} />
              <Bar dataKey="temporary_impact" fill={COST_COLORS.temporary_impact} radius={[2, 2, 0, 0]} maxBarSize={18} />
              <Bar dataKey="permanent_impact" fill={COST_COLORS.permanent_impact} radius={[2, 2, 0, 0]} maxBarSize={18} />
              <Bar dataKey="timing_cost" fill={COST_COLORS.timing_cost} radius={[2, 2, 0, 0]} maxBarSize={18} />
              <Bar dataKey="opportunity_cost" fill={COST_COLORS.opportunity_cost} radius={[2, 2, 0, 0]} maxBarSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ── Named panel exports for individual tabs ───────────────────────────────────

interface PanelProps { result: SimulationResult; isDark: boolean }

/** Full-width Execution Inventory Trajectory chart tab */
export function TrajectoryPanel({ result, isDark }: PanelProps) {
  const maxLen = Math.max(...result.strategies.map((s) => s.trajectory.length))
  const trajectoryData = Array.from({ length: maxLen }, (_, i) => {
    const pt: any = { step: i }
    for (const s of result.strategies) pt[s.name] = s.trajectory[i] ?? null
    return pt
  })
  const TrajectoryTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[200px]">
        <div className="text-black/30 dark:text-white/30 mb-2 uppercase font-semibold">Step {label}</div>
        {payload.map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: getStrategyColor(p.dataKey) }} />
              <span>{getStrategyLabel(p.dataKey)}</span>
            </div>
            <span className="font-bold">{p.value !== null ? Math.round(p.value).toLocaleString() : '—'}</span>
          </div>
        ))}
      </div>
    )
  }
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>Execution Inventory Trajectory</h3>
        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{result.strategies.length} strategies · {maxLen} steps</span>
      </div>
      <ResponsiveContainer width="100%" height={440}>
        <LineChart data={trajectoryData} margin={{ top: 10, right: 24, bottom: 24, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
          <XAxis dataKey="step" tickLine={false} tick={{ fontSize: 10, fontFamily: 'monospace' }}
            label={{ value: 'TRADING STEP', position: 'insideBottomRight', offset: -5, style: { fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'monospace' } }} />
          <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} tick={{ fontSize: 10, fontFamily: 'monospace' }}
            label={{ value: 'Remaining Shares', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'monospace' } }} />
          <Tooltip content={<TrajectoryTooltip />} />
          <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: 10, fontFamily: 'monospace', textTransform: 'uppercase' }} formatter={(v) => getStrategyLabel(v)} />
          <ReferenceLine x={Math.round(maxLen * 0.25)} stroke={isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'} strokeDasharray="3 3"
            label={{ value: '25% Horizon', position: 'top', fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} />
          {result.strategies.map((s) => {
            const color = getStrategyColor(s.name)
            const isRL = s.name.toLowerCase().includes('rl') || s.name.toLowerCase().includes('ppo')
            return <Line key={s.name} type="monotone" dataKey={s.name} stroke={color} strokeWidth={isRL ? 2.5 : 1.5} dot={false} activeDot={{ r: 5, strokeWidth: 0, fill: color }} connectNulls />
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Full-width Cost Breakdown grouped bar chart tab */
export function CostBreakdownPanel({ result, isDark }: PanelProps) {
  const C: Record<string, string> = {
    spread_cost: '#64748B', temporary_impact: '#3B82F6',
    permanent_impact: '#EF4444', timing_cost: '#F59E0B', opportunity_cost: '#10B981',
  }
  const lbl = (k: string) => ({ spread_cost: 'Spread Cost', temporary_impact: 'Temp Impact',
    permanent_impact: 'Perm Impact', timing_cost: 'Timing Cost', opportunity_cost: 'Opp. Cost' }[k] ?? k)
  const data = result.strategies.map(s => {
    const d = s.cost_decomposition || {}
    return { name: getStrategyLabel(s.name), spread_cost: d.spread_cost || 0, temporary_impact: d.temporary_impact || 0,
      permanent_impact: d.permanent_impact || 0, timing_cost: d.timing_cost || 0, opportunity_cost: d.opportunity_cost || 0 }
  })
  const CostTip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const total = payload.reduce((s: number, p: any) => s + (p.value || 0), 0)
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[220px]">
        <div className="font-bold mb-2 uppercase text-black/50 dark:text-white/50">{label}</div>
        {payload.map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-sm" style={{ background: C[p.dataKey] }} /><span>{lbl(p.dataKey)}</span></div>
            <span className="font-semibold">${p.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          </div>
        ))}
        <div className="border-t border-dashed border-black/10 dark:border-white/10 pt-1.5 mt-1.5 flex justify-between font-bold">
          <span>Total:</span><span>${total.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        </div>
      </div>
    )
  }
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>Slippage Cost Decomposition</h3>
        <div className="flex items-center gap-3 flex-wrap">
          {Object.entries(C).map(([key, color]) => (
            <span key={key} className="flex items-center gap-1 text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>
              <span className="w-2 h-2 rounded-sm" style={{ background: color }} />{lbl(key)}
            </span>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={420}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 24, left: 20 }} barCategoryGap="20%" barGap={2}>
          <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
          <XAxis dataKey="name" tickLine={false} tick={{ fontSize: 10, fontFamily: 'monospace' }} />
          <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} tick={{ fontSize: 10, fontFamily: 'monospace' }}
            label={{ value: 'Cost (USD $)', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'monospace' } }} />
          <Tooltip content={<CostTip />} />
          <Legend verticalAlign="bottom" height={36} iconType="rect" wrapperStyle={{ fontSize: 9, fontFamily: 'monospace' }} formatter={(v) => lbl(v).toUpperCase()} />
          <ReferenceLine y={0} stroke={isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'} strokeDasharray="3 3" />
          <Bar dataKey="spread_cost" fill={C.spread_cost} radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="temporary_impact" fill={C.temporary_impact} radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="permanent_impact" fill={C.permanent_impact} radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="timing_cost" fill={C.timing_cost} radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="opportunity_cost" fill={C.opportunity_cost} radius={[3, 3, 0, 0]} maxBarSize={22} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Full-width sortable metrics table with cost decomp columns */
export function MetricsTablePanel({ result }: { result: SimulationResult }) {
  const [sortKey, setSortKey] = useState<'name' | 'mean_is_pct' | 'is_variance' | 'cvar95' | 'ac_objective' | 'trade_count'>('mean_is_pct')
  const [sortAsc, setSortAsc] = useState(true)
  const twapMean = result.strategies.find(s => s.name.toLowerCase().includes('twap'))?.mean_is_pct ?? 0
  const toggleSort = (k: typeof sortKey) => { if (sortKey === k) setSortAsc(!sortAsc); else { setSortKey(k); setSortAsc(true) } }
  const sorted = [...result.strategies].sort((a, b) => {
    const av = sortKey === 'name' ? getStrategyLabel(a.name) : (a[sortKey as keyof typeof a] ?? Infinity)
    const bv = sortKey === 'name' ? getStrategyLabel(b.name) : (b[sortKey as keyof typeof b] ?? Infinity)
    if (typeof av === 'string' && typeof bv === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number)
  })
  const Th = ({ k, label }: { k: typeof sortKey; label: string }) => (
    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none border-b border-[var(--divider)] whitespace-nowrap"
      style={{ color: 'var(--text-muted)' }} onClick={() => toggleSort(k)}>
      <span className="flex items-center gap-1 hover:text-[var(--text)] transition-colors">{label} <ArrowUpDown size={11} className="opacity-40" /></span>
    </th>
  )
  const fmtUsd = (v: number) => `$${Math.round(v).toLocaleString()}`
  return (
    <div className="space-y-4">
      <h3 className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--text-sub)' }}>Full Strategy Performance Metrics</h3>
      <div className="overflow-x-auto custom-scrollbar rounded-lg border border-[var(--divider)]">
        <table className="w-full text-sm">
          <thead style={{ background: 'var(--card-hover)' }}>
            <tr>
              <Th k="name" label="Strategy" />
              <Th k="mean_is_pct" label="Mean IS%" />
              <Th k="is_variance" label="Std Dev" />
              <Th k="cvar95" label="CVaR95" />
              <Th k="ac_objective" label="AC Obj." />
              <Th k="trade_count" label="Trades" />
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>vs TWAP</th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>Spread</th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>Temp Impact</th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>Perm Impact</th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider border-b border-[var(--divider)] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>Timing Cost</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => {
              const color = getStrategyColor(s.name); const label = getStrategyLabel(s.name)
              const isRL = label.includes('RL')
              const stdDev = s.is_variance !== null ? Math.sqrt(s.is_variance) : null
              const diff = s.name.toLowerCase().includes('twap') ? 0 : twapMean - (s.mean_is_pct ?? 0)
              const d = s.cost_decomposition || {}
              return (
                <tr key={s.name} style={{ borderBottom: '1px solid var(--divider)' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color, boxShadow: isRL ? '0 0 8px #10B981' : 'none' }} />
                      <span className={`font-mono text-xs ${isRL ? 'font-bold text-emerald-500' : 'text-[var(--text)]'}`}>{label}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3"><IS value={s.mean_is_pct} /></td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{stdDev !== null ? `${stdDev.toFixed(3)}%` : '—'}</td>
                  <td className="px-4 py-3"><IS value={s.cvar95} /></td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{s.ac_objective !== null ? s.ac_objective.toFixed(4) : '—'}</td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{s.trade_count !== null ? Math.round(s.trade_count) : '—'}</td>
                  <td className="px-4 py-3 font-mono text-xs">
                    {s.name.toLowerCase().includes('twap') ? <span className="text-[var(--text-muted)] opacity-50">baseline</span>
                      : diff > 0 ? <span className="text-emerald-500 font-semibold">+{diff.toFixed(2)}%</span>
                      : <span className="text-rose-500">{diff.toFixed(2)}%</span>}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{fmtUsd(d.spread_cost ?? 0)}</td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{fmtUsd(d.temporary_impact ?? 0)}</td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{fmtUsd(d.permanent_impact ?? 0)}</td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: (d.timing_cost ?? 0) < 0 ? '#10B981' : '#F59E0B' }}>
                    {fmtUsd(d.timing_cost ?? 0)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] font-mono uppercase text-center" style={{ color: 'var(--text-muted)' }}>
        * Negative IS% = sold above arrival price (gain) · Cost columns based on $1M notional
      </p>
    </div>
  )
}
