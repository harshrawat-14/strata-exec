/**
 * Results tour — the pinned PPO-LSTM results, served from a frozen snapshot.
 * Works with the backend offline: nothing here calls the API.
 */

import { Fragment, useMemo, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  ScatterChart, Scatter, ZAxis, Legend,
} from 'recharts'
import {
  ArrowUpDown, BarChart2, BrainCircuit, ChevronDown, ChevronUp, Layers, ShieldCheck, TrendingDown,
  TrendingUp,
} from 'lucide-react'
import { MetricTile } from '../components/ui'
import { useBackendStatus } from '../hooks/useBackendStatus'
import { SystemDownBanner } from '../components/SystemDownBanner'
import {
  PINNED, REGIMES, REGIME_COLORS, STRATEGY_COLORS, edgeVsAc, episodeHistogram, filterDates, strategyMeansByRegime,
  summariseByRegime,
} from '../data/pinned/results'
import type { DateResult, Regime, Split, StrategyKey } from '../data/pinned/results'

const SPLITS: { id: Split | 'all'; label: string }[] = [
  { id: 'all', label: 'All days' },
  { id: 'train', label: 'Training' },
  { id: 'held_out', label: 'Held-out' },
]
const REGIME_FILTERS: (Regime | 'all')[] = ['all', 'crash', 'normal', 'calm']
const HIST_KEYS: StrategyKey[] = ['ac_optimal', 'adaptive_ac', 'rl']

const tooltipStyle = {
  background: 'var(--card)',
  border: '1px solid var(--card-border)',
  borderRadius: 8,
  fontSize: 10,
  fontFamily: 'JetBrains Mono, monospace',
  color: 'var(--text)',
}
const axisTick = { fill: 'var(--text-muted)', fontSize: 9 }

const pp = (v: number, digits = 2) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`
const pct = (v: number, digits = 3) => `${pp(v, digits)}%`
const label = (k: StrategyKey) => PINNED.strategies.find(s => s.key === k)!.label

type SortKey = 'date' | 'edge' | 'rl' | 'ac_optimal'

function Segmented<T extends string>({ value, options, onChange, ariaLabel }: {
  value: T
  options: { id: T; label: string }[]
  onChange: (v: T) => void
  ariaLabel: string
}) {
  return (
    <div role="group" aria-label={ariaLabel} className="inline-flex flex-wrap gap-1 p-1 rounded-lg"
      style={{ background: 'var(--input-bg)', border: '1px solid var(--divider)' }}>
      {options.map(o => {
        const on = o.id === value
        return (
          <button key={o.id} type="button" aria-pressed={on} onClick={() => onChange(o.id)}
            className="px-3 py-1.5 rounded-md text-[10px] font-mono font-semibold uppercase tracking-wider"
            style={{
              background: on ? 'var(--active-fill)' : 'transparent',
              color: on ? 'var(--active-text)' : 'var(--text-muted)',
            }}>
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

function SortHead({ k, current, asc, onSort, children }: {
  k: SortKey
  current: SortKey
  asc: boolean
  onSort: (k: SortKey) => void
  children: React.ReactNode
}) {
  return (
    <th className="px-3 py-2 text-right">
      <button type="button" onClick={() => onSort(k)}
        className="inline-flex items-center gap-1 text-[9px] font-mono font-bold uppercase tracking-wider"
        style={{ color: current === k ? 'var(--text)' : 'var(--text-muted)' }}>
        {children}
        {current === k ? (asc ? <ChevronUp size={10} /> : <ChevronDown size={10} />) : <ArrowUpDown size={10} />}
      </button>
    </th>
  )
}

function RegimeTag({ regime }: { regime: Regime }) {
  return (
    <span className="text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded"
      style={{ background: REGIME_COLORS[regime] + '20', color: REGIME_COLORS[regime] }}>
      {regime}
    </span>
  )
}

function ChartCard({ icon, title, sub, children }: {
  icon: React.ReactNode
  title: string
  sub?: string
  children: React.ReactNode
}) {
  return (
    <div className="glass-card p-5 space-y-4 min-w-0">
      <div>
        <div className="flex items-center gap-2">
          <span style={{ color: 'var(--text-muted)' }}>{icon}</span>
          <h3 className="label-text">{title}</h3>
        </div>
        {sub && <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
      </div>
      <div className="w-full h-[280px]">{children}</div>
    </div>
  )
}

export default function ResultsTour() {
  const backend = useBackendStatus()
  const [split, setSplit] = useState<Split | 'all'>('all')
  const [regime, setRegime] = useState<Regime | 'all'>('all')
  const [sortKey, setSortKey] = useState<SortKey>('date')
  const [sortAsc, setSortAsc] = useState(true)
  const [expanded, setExpanded] = useState<string | null>('2024-08-05')

  const dates = useMemo(() => filterDates(PINNED.dates, split, regime), [split, regime])
  const bySplit = useMemo(() => filterDates(PINNED.dates, split, 'all'), [split])
  const summary = summariseByRegime(bySplit)
  const crash = summary.find(s => s.regime === 'crash')
  const quiet = summary.filter(s => s.regime !== 'crash')
  const quietDays = quiet.reduce((a, s) => a + s.days, 0)
  const quietWins = quiet.reduce((a, s) => a + s.wins, 0)
  const episodes = bySplit.length * PINNED.strategies.length * PINNED.protocol.episodes_per_strategy

  const edgeData = dates.map(d => ({
    name: d.date.slice(5), date: d.date, edge: edgeVsAc(d), regime: d.regime, split: d.split, event: d.event_detail,
  }))
  const regimeData = strategyMeansByRegime(dates)
  const scatterData = REGIMES.filter(r => dates.some(d => d.regime === r)).map(r => ({
    regime: r,
    points: dates.filter(d => d.regime === r).map(d => ({
      severity: d.strategies.ac_optimal.mean, edge: edgeVsAc(d), date: d.date,
    })),
  }))
  const ablation = PINNED.ablation.filter(a =>
    (split === 'all' || a.split === split) && (regime === 'all' || a.regime === regime))
    .map(a => ({ name: a.date.slice(5), '12-dim': a.delta12, '16-dim': a.delta16 }))

  const sorted = [...dates].sort((a, b) => {
    const v = (d: DateResult) => sortKey === 'date' ? d.date : sortKey === 'edge' ? edgeVsAc(d)
      : d.strategies[sortKey].mean
    const [x, y] = [v(a), v(b)]
    const c = typeof x === 'string' ? x.localeCompare(y as string) : (x as number) - (y as number)
    return sortAsc ? c : -c
  })
  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortAsc(!sortAsc)
    else { setSortKey(k); setSortAsc(k === 'date') }
  }

  const crashAll = PINNED.dates.filter(d => d.regime === 'crash')
  const beatsAdaptive = crashAll.filter(d => d.strategies.rl.mean > d.strategies.adaptive_ac.mean).length

  return (
    <div className="space-y-8 animate-fade-in">
      {backend === 'offline' && <SystemDownBanner compact />}

      {/* Header */}
      <div className="flex flex-col lg:flex-row items-start lg:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
            Results <span className="text-gradient-electric">Tour</span>
          </h1>
          <p className="mt-1 text-xs font-mono uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {PINNED.model.name} · {PINNED.dates.length} trading days · {PINNED.strategies.length} strategies ·{' '}
            {PINNED.protocol.episodes_per_strategy} episodes each
          </p>
          <p className="mt-3 text-sm max-w-2xl leading-relaxed" style={{ color: 'var(--text-sub)' }}>
            A recurrent PPO agent sells a large BTC order against five classical schedules, on real Binance
            order flow replayed through StrataExec's Rust order-book simulator. Training and held-out days
            are scored the same way, on the same episode starts.
          </p>
        </div>
        <div className="p-3 rounded-lg w-full lg:w-auto" style={{ background: 'var(--card)', border: '1px solid var(--card-border)' }}>
          <div className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Pinned model</div>
          <div className="text-[11px] font-mono font-semibold mt-0.5" style={{ color: 'var(--text)' }}>
            {PINNED.model.name} · {PINNED.model.variant}
          </div>
          <div className="text-[9px] font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {PINNED.model.algorithm} · {PINNED.model.pinned_commit.slice(0, 7)}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <Segmented ariaLabel="Data split" value={split} onChange={setSplit}
          options={SPLITS.map(s => ({
            id: s.id, label: `${s.label} · ${filterDates(PINNED.dates, s.id, 'all').length}`,
          }))} />
        <Segmented ariaLabel="Regime" value={regime} onChange={setRegime}
          options={REGIME_FILTERS.map(r => ({ id: r, label: r === 'all' ? 'All regimes' : r }))} />
      </div>

      {/* Stat tiles (by split; the regime filter narrows the charts and table) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile label="Crash days won" value={crash ? `${crash.wins} / ${crash.days}` : '—'}
          sub={crash ? `vs AC Optimal · best ${pp(crash.bestEdge)} pp` : 'no crash days in view'}
          icon={<TrendingUp size={14} />} />
        <MetricTile label="Mean edge, crash days" value={crash ? `${pp(crash.meanEdge)} pp` : '—'}
          sub="RL − AC Optimal, IS" icon={<BrainCircuit size={14} />} />
        <MetricTile label="Calm & normal days won" value={`${quietWins} / ${quietDays}`}
          sub="AC Optimal is hard to beat here" icon={<TrendingDown size={14} />} />
        <MetricTile label="Episodes scored" value={episodes.toLocaleString()}
          sub={`${bySplit.length} days × ${PINNED.strategies.length} strategies × ${PINNED.protocol.episodes_per_strategy}`}
          icon={<Layers size={14} />} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard icon={<BarChart2 size={14} />} title="RL edge over AC Optimal, by day (pp)"
          sub="Above zero: the RL agent sold at a better average price. Faded bars are training days.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={edgeData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
              <XAxis dataKey="name" tickLine={false} axisLine={false} tick={axisTick} interval={0} angle={-45}
                textAnchor="end" height={40} />
              <YAxis tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => pp(v, 1)} />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                formatter={(v) => [`${pp(Number(v), 3)} pp`, 'RL − AC']}
                labelFormatter={(_, p) => {
                  const d = p?.[0]?.payload
                  return d ? `${d.date} · ${d.regime} · ${d.split === 'train' ? 'training' : 'held-out'}` : ''
                }} />
              <Bar isAnimationActive={false} dataKey="edge" radius={[2, 2, 0, 0]}>
                {edgeData.map(d => (
                  <Cell key={d.date} fill={REGIME_COLORS[d.regime]} fillOpacity={d.split === 'train' ? 0.45 : 1} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard icon={<BarChart2 size={14} />} title="Strategy performance by regime (mean IS %)"
          sub="Average over the days in view. Higher is better.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={regimeData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
              <XAxis dataKey="regime" tickLine={false} axisLine={false} tick={axisTick} />
              <YAxis tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => `${pp(v, 1)}%`} />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                formatter={(v) => [pct(Number(v)), '']} />
              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
              {PINNED.strategies.map(s => (
                <Bar isAnimationActive={false} key={s.key} dataKey={s.label} fill={STRATEGY_COLORS[s.key]} radius={[2, 2, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard icon={<TrendingDown size={14} />} title="The edge grows with how hard the day is"
          sub="x: AC Optimal's IS on the day (further left = more costly day). y: RL − AC Optimal.">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" />
              <XAxis type="number" dataKey="severity" name="AC Optimal IS" tickLine={false} axisLine={false}
                tick={axisTick} tickFormatter={v => `${pp(v, 0)}%`} />
              <YAxis type="number" dataKey="edge" name="RL − AC" tickLine={false} axisLine={false} tick={axisTick}
                tickFormatter={v => pp(v, 1)} />
              <ZAxis range={[60, 60]} />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }}
                formatter={(v, n) => [n === 'AC Optimal IS' ? pct(Number(v)) : `${pp(Number(v), 3)} pp`, n]} />
              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
              {scatterData.map(s => (
                <Scatter isAnimationActive={false} key={s.regime} name={s.regime} data={s.points} fill={REGIME_COLORS[s.regime]} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard icon={<ShieldCheck size={14} />} title="Does the richer state help? 12-dim vs 16-dim"
          sub={`RL − AC Optimal (pp) for two otherwise identical agents, on the ${PINNED.ablation.length}-day ablation set.`}>
          {ablation.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ablation} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tick={axisTick} interval={0} angle={-45}
                  textAnchor="end" height={40} />
                <YAxis tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => pp(v, 1)} />
                <ReferenceLine y={0} stroke="var(--text-muted)" />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                  formatter={(v, n) => [`${pp(Number(v), 3)} pp`, n]} />
                <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
                <Bar isAnimationActive={false} dataKey="12-dim" fill="#94A3B8" radius={[2, 2, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="16-dim" fill={STRATEGY_COLORS.rl} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
              No ablation days match this filter
            </div>
          )}
        </ChartCard>
      </div>

      {/* Per-day table */}
      <div className="glass-card p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
          <div>
            <h3 className="label-text">Every day, every strategy (mean IS %)</h3>
            <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
              Click a day to see how its {PINNED.protocol.episodes_per_strategy} episodes spread.
            </p>
          </div>
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{dates.length} days</span>
        </div>
        <div className="overflow-x-auto custom-scrollbar">
          <table className="w-full text-xs font-mono min-w-[880px]">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--divider)' }}>
                <th className="px-3 py-2 text-left">
                  <button type="button" onClick={() => toggleSort('date')}
                    className="inline-flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider"
                    style={{ color: sortKey === 'date' ? 'var(--text)' : 'var(--text-muted)' }}>
                    Day {sortKey === 'date' ? (sortAsc ? <ChevronUp size={10} /> : <ChevronDown size={10} />) : <ArrowUpDown size={10} />}
                  </button>
                </th>
                <th className="px-3 py-2 text-left text-[9px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Set</th>
                <th className="px-3 py-2 text-left text-[9px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Regime</th>
                <th className="px-3 py-2 text-right text-[9px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>TWAP</th>
                <th className="px-3 py-2 text-right text-[9px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Heuristic</th>
                <SortHead k="ac_optimal" current={sortKey} asc={sortAsc} onSort={toggleSort}>AC Optimal</SortHead>
                <th className="px-3 py-2 text-right text-[9px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Adaptive AC</th>
                <SortHead k="rl" current={sortKey} asc={sortAsc} onSort={toggleSort}>RL Agent</SortHead>
                <SortHead k="edge" current={sortKey} asc={sortAsc} onSort={toggleSort}>RL − AC</SortHead>
              </tr>
            </thead>
            <tbody>
              {sorted.map(d => {
                const edge = edgeVsAc(d)
                const open = expanded === d.date
                const best = Math.max(...PINNED.strategies.map(s => d.strategies[s.key].mean))
                return (
                  <Fragment key={d.date}>
                    <tr onClick={() => setExpanded(open ? null : d.date)} aria-expanded={open}
                      className="cursor-pointer hover:bg-[var(--card-hover)]"
                      style={{ borderBottom: '1px solid var(--divider)' }}>
                      <td className="px-3 py-2.5 font-semibold" style={{ color: 'var(--text)' }}>
                        <span className="inline-flex items-center gap-1.5">
                          {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}{d.date}
                        </span>
                      </td>
                      <td className="px-3 py-2.5">
                        <span className={d.split === 'held_out' ? 'badge-info' : 'badge-neutral'}>
                          {d.split === 'held_out' ? 'Held-out' : 'Training'}
                        </span>
                      </td>
                      <td className="px-3 py-2.5"><RegimeTag regime={d.regime} /></td>
                      {PINNED.strategies.map(s => {
                        const v = d.strategies[s.key].mean
                        return (
                          <td key={s.key} className="px-3 py-2.5 text-right"
                            style={{
                              color: v === best ? 'var(--text)' : 'var(--text-sub)',
                              fontWeight: v === best ? 700 : 400,
                            }}>
                            {pct(v)}
                          </td>
                        )
                      })}
                      <td className="px-3 py-2.5 text-right font-bold"
                        style={{ color: edge > 0 ? STRATEGY_COLORS.rl : 'var(--text-muted)' }}>
                        {pp(edge, 3)}
                      </td>
                    </tr>
                    {open && (
                      <tr style={{ borderBottom: '1px solid var(--divider)', background: 'var(--input-bg)' }}>
                        <td colSpan={9} className="p-4">
                          <DayDetail d={d} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Caveats */}
      <div className="glass-card p-5 space-y-4">
        <h3 className="label-text">Read before citing</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs leading-relaxed" style={{ color: 'var(--text-sub)' }}>
          <p>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>A crash-day specialist. </span>
            The agent wins by selling earlier when prices fall. On calm and normal days AC Optimal beat it on
            all but one day.
          </p>
          <p>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>Correlated days. </span>
            Three of the four held-out crash days fall in one week (the August 2024 yen-carry unwind), so the
            held-out crash wins come from two separate events.
          </p>
          <p>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>Other baselines. </span>
            Against Adaptive AC the RL agent was better on {beatsAdaptive} of {crashAll.length} crash days. Its
            tail-risk (CVaR) edge is its mean edge carried into the tail, not a separate effect.
          </p>
        </div>
      </div>
    </div>
  )
}

function DayDetail({ d }: { d: DateResult }) {
  const hist = episodeHistogram(d, HIST_KEYS)
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="space-y-2">
        <div className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Event</div>
        <div className="text-xs" style={{ color: 'var(--text)' }}>{d.event_detail}</div>
        <table className="w-full text-[10px] font-mono mt-2">
          <thead>
            <tr style={{ color: 'var(--text-muted)' }}>
              <th className="text-left py-1 font-normal">Strategy</th>
              <th className="text-right py-1 font-normal">Std</th>
              <th className="text-right py-1 font-normal">CVaR 95</th>
            </tr>
          </thead>
          <tbody>
            {PINNED.strategies.map(s => (
              <tr key={s.key} style={{ color: s.key === 'rl' ? STRATEGY_COLORS.rl : 'var(--text-sub)' }}>
                <td className="py-0.5">{s.label}</td>
                <td className="py-0.5 text-right">{d.strategies[s.key].std.toFixed(3)}</td>
                <td className="py-0.5 text-right">{pct(d.strategies[s.key].cvar95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="lg:col-span-2 min-w-0">
        <div className="text-[9px] font-mono uppercase mb-2" style={{ color: 'var(--text-muted)' }}>
          Episode IS distribution (count of {d.strategies.rl.episodes.length} episodes per bin)
        </div>
        <div className="h-[180px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hist} margin={{ top: 4, right: 8, bottom: 4, left: -16 }} barGap={0}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
              <XAxis dataKey="bin" tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => `${v}%`} />
              <YAxis tickLine={false} axisLine={false} tick={axisTick} allowDecimals={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                labelFormatter={v => `IS ≈ ${v}%`} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {HIST_KEYS.map(k => (
                <Bar isAnimationActive={false} key={k} dataKey={k} name={label(k)} fill={STRATEGY_COLORS[k]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
