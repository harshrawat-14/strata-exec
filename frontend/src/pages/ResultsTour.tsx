/**
 * Results — the pinned PPO-LSTM results, served from a frozen snapshot.
 * Works with the backend offline: nothing here calls the API.
 */

import { Fragment, useMemo, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  ScatterChart, Scatter, ZAxis, Legend,
} from 'recharts'
import {
  ArrowUpDown, BarChart2, BookOpen, BrainCircuit, ChevronDown, ChevronUp, Compass, Layers, ShieldCheck,
  TrendingDown, TrendingUp,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { MetricTile } from '../components/ui'
import { useBackendStatus } from '../hooks/useBackendStatus'
import { SystemDownBanner } from '../components/SystemDownBanner'
import { useTour } from '../tour/context'
import {
  PINNED, REGIMES, REGIME_COLORS, STRATEGY_COLORS, edgeVsAc, filterDates, strategyMeansByRegime,
  summariseByRegime,
} from '../data/pinned/results'
import type { DateResult, Regime } from '../data/pinned/results'
import { ChartCard, DayDetail, LearnMore, RegimeTag, Segmented } from '../components/pinned/PinnedViews'
import { REGIME_INFO, axisTick, pct, pp, tooltipStyle } from '../components/pinned/format'

const REGIME_FILTERS: (Regime | 'all')[] = ['all', 'crash', 'normal', 'calm']

type SortKey = 'date' | 'edge' | 'rl' | 'ac_optimal'

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

const TH = 'px-3 py-2 text-[9px] font-bold uppercase tracking-wider'

export default function ResultsTour() {
  const backend = useBackendStatus()
  const { start } = useTour()
  const navigate = useNavigate()
  const [regime, setRegime] = useState<Regime | 'all'>('all')
  const [sortKey, setSortKey] = useState<SortKey>('date')
  const [sortAsc, setSortAsc] = useState(true)
  const [expanded, setExpanded] = useState<string | null>('2024-08-05')

  // Headline tiles describe all 25 days and do not move with the regime filter.
  const summary = summariseByRegime(PINNED.dates)
  const crash = summary.find(s => s.regime === 'crash')!
  const quiet = summary.filter(s => s.regime !== 'crash')
  const quietDays = quiet.reduce((a, s) => a + s.days, 0)
  const quietWins = quiet.reduce((a, s) => a + s.wins, 0)
  const unseenCrash = PINNED.dates.filter(d => d.regime === 'crash' && d.split === 'held_out')
  const unseenCrashWins = unseenCrash.filter(d => edgeVsAc(d) > 0).length
  const episodes = PINNED.dates.length * PINNED.strategies.length * PINNED.protocol.episodes_per_strategy

  const dates = useMemo(() => filterDates(PINNED.dates, 'all', regime), [regime])
  const edgeData = dates.map(d => ({
    name: d.date.slice(5), date: d.date, edge: edgeVsAc(d), regime: d.regime, split: d.split,
  }))
  const regimeData = strategyMeansByRegime(dates)
  const scatterData = REGIMES.filter(r => dates.some(d => d.regime === r)).map(r => ({
    regime: r,
    points: dates.filter(d => d.regime === r).map(d => ({
      severity: d.strategies.ac_optimal.mean, edge: edgeVsAc(d), date: d.date,
    })),
  }))
  const ablation = PINNED.ablation.filter(a => regime === 'all' || a.regime === regime)
    .map(a => ({ name: a.date.slice(5), '12-dim': a.delta12, '16-dim': a.delta16 }))
  const regimeCount = (r: Regime | 'all') => filterDates(PINNED.dates, 'all', r).length

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
            Research <span className="text-gradient-electric">Results</span>
          </h1>
          <p className="mt-1 text-xs font-mono uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {PINNED.model.name} · {PINNED.dates.length} trading days · {PINNED.strategies.length} strategies ·{' '}
            {PINNED.protocol.episodes_per_strategy} episodes each
          </p>
          <p className="mt-3 text-sm max-w-2xl leading-relaxed" style={{ color: 'var(--text-sub)' }}>
            A recurrent PPO agent sells a large BTC order against four classical schedules, on real Binance
            order flow replayed through StrataExec's Rust order-book simulator. Every strategy trades the same
            episodes on the same days.
          </p>
          <div className="flex flex-wrap gap-2 mt-4">
            <button type="button" onClick={start} className="btn-primary"><Compass size={12} /> Start the tour</button>
            <button type="button" onClick={() => navigate('/learn')} className="btn-secondary">
              <BookOpen size={12} /> How it works
            </button>
          </div>
        </div>
        <div className="p-3 rounded-lg w-full lg:w-auto" style={{ background: 'var(--card)', border: '1px solid var(--card-border)' }}>
          <div className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Model</div>
          <div className="text-[11px] font-mono font-semibold mt-0.5" style={{ color: 'var(--text)' }}>
            {PINNED.model.name} · {PINNED.model.variant}
          </div>
          <div className="text-[9px] font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {PINNED.model.algorithm} · {PINNED.model.pinned_commit.slice(0, 7)}
          </div>
        </div>
      </div>

      {/* Headline tiles: fixed, all 25 days */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" data-tour="results-headline">
        <MetricTile label="Crash days won" value={`${crash.wins} / ${crash.days}`}
          sub={`vs AC Optimal · ${unseenCrashWins} of ${unseenCrash.length} never seen in training`}
          icon={<TrendingUp size={14} />} />
        <MetricTile label="Mean edge, crash days" value={`${pp(crash.meanEdge)} pp`}
          sub={`RL − AC Optimal · best ${pp(crash.bestEdge)} pp`} icon={<BrainCircuit size={14} />} />
        <MetricTile label="Calm & normal days won" value={`${quietWins} / ${quietDays}`}
          sub="AC Optimal is hard to beat here" icon={<TrendingDown size={14} />} />
        <MetricTile label="Episodes scored" value={episodes.toLocaleString()}
          sub={`${PINNED.dates.length} days × ${PINNED.strategies.length} strategies × ${PINNED.protocol.episodes_per_strategy}`}
          icon={<Layers size={14} />} />
      </div>

      {/* Regime filter + what the regime means */}
      <div className="space-y-3" data-tour="regime-filter">
        <Segmented ariaLabel="Regime" value={regime} onChange={setRegime}
          options={REGIME_FILTERS.map(r => ({ id: r, label: `${REGIME_INFO[r].title} · ${regimeCount(r)}` }))} />
        <div className="glass-card px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <p className="text-xs leading-relaxed max-w-3xl" style={{ color: 'var(--text-sub)' }}>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>{REGIME_INFO[regime].title}. </span>
            {REGIME_INFO[regime].text}
          </p>
          <LearnMore to="/learn#regimes" />
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard tour="edge-chart" icon={<BarChart2 size={14} />} title="RL edge over AC Optimal, by day (pp)"
          sub="Above zero: the RL agent sold at a better average price. Faded bars: days the model trained on.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={edgeData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
              <XAxis dataKey="name" tickLine={false} axisLine={false} tick={axisTick} interval={0} angle={-45}
                textAnchor="end" height={40} />
              <YAxis tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => pp(v, 1)} />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                formatter={v => [`${pp(Number(v), 3)} pp`, 'RL − AC']}
                labelFormatter={(_, p) => {
                  const d = p?.[0]?.payload
                  return d ? `${d.date} · ${d.regime}${d.split === 'train' ? ' · training day' : ''}` : ''
                }} />
              <Bar isAnimationActive={false} dataKey="edge" radius={[2, 2, 0, 0]}>
                {edgeData.map(d => (
                  <Cell key={d.date} fill={REGIME_COLORS[d.regime]} fillOpacity={d.split === 'train' ? 0.4 : 1} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard tour="strategy-chart" icon={<BarChart2 size={14} />} title="Strategy performance by regime (mean IS %)"
          sub="Average over the days in view. Higher is better.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={regimeData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" vertical={false} />
              <XAxis dataKey="regime" tickLine={false} axisLine={false} tick={axisTick} />
              <YAxis tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => `${pp(v, 1)}%`} />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                formatter={v => [pct(Number(v)), '']} />
              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
              {PINNED.strategies.map(s => (
                <Bar isAnimationActive={false} key={s.key} dataKey={s.label} fill={STRATEGY_COLORS[s.key]} radius={[2, 2, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard tour="severity-chart" icon={<TrendingDown size={14} />} title="The edge grows with how hard the day is"
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

        <ChartCard tour="ablation-chart" icon={<ShieldCheck size={14} />} title="Does the richer state help? 12-dim vs 16-dim"
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
      <div className="glass-card p-5 space-y-4" data-tour="day-table">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
          <div>
            <h3 className="label-text">Every day, every strategy (mean IS %)</h3>
            <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
              Click a day to see how its {PINNED.protocol.episodes_per_strategy} episodes spread. Bold: best strategy on the day.
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
                <th className={`${TH} text-left`} style={{ color: 'var(--text-muted)' }}>Regime</th>
                <th className={`${TH} text-left`} style={{ color: 'var(--text-muted)' }}>Training day</th>
                <th className={`${TH} text-right`} style={{ color: 'var(--text-muted)' }}>TWAP</th>
                <th className={`${TH} text-right`} style={{ color: 'var(--text-muted)' }}>Heuristic</th>
                <SortHead k="ac_optimal" current={sortKey} asc={sortAsc} onSort={toggleSort}>AC Optimal</SortHead>
                <th className={`${TH} text-right`} style={{ color: 'var(--text-muted)' }}>Adaptive AC</th>
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
                      <td className="px-3 py-2.5"><RegimeTag regime={d.regime} /></td>
                      <td className="px-3 py-2.5" style={{ color: 'var(--text-muted)' }}>
                        {d.split === 'train' ? <span className="badge-neutral">Yes</span> : '—'}
                      </td>
                      {PINNED.strategies.map(s => {
                        const v = d.strategies[s.key].mean
                        return (
                          <td key={s.key} className="px-3 py-2.5 text-right"
                            style={{ color: v === best ? 'var(--text)' : 'var(--text-sub)', fontWeight: v === best ? 700 : 400 }}>
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
                        <td colSpan={9} className="p-4"><DayDetail d={d} /></td>
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
        <div className="flex items-center justify-between gap-2">
          <h3 className="label-text">Read before citing</h3>
          <LearnMore to="/learn#limits" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs leading-relaxed" style={{ color: 'var(--text-sub)' }}>
          <p>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>A crash-day specialist. </span>
            The agent wins by selling earlier when prices fall. On calm and normal days AC Optimal beat it on
            all but one day.
          </p>
          <p>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>Correlated days. </span>
            Several crash days fall in the same week (the July 2024 German-government sales, the August 2024
            yen-carry unwind), so they are fewer independent events than days.
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
