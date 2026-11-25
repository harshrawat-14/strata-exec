/**
 * Building blocks for pages that show the pinned PPO-LSTM results (Results, RL Eval).
 */
import { Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { ArrowRight } from 'lucide-react'
import { PINNED, REGIME_COLORS, STRATEGY_COLORS, episodeHistogram } from '../../data/pinned/results'
import type { DateResult, Regime, StrategyKey } from '../../data/pinned/results'
import { axisTick, pct, strategyLabel, tooltipStyle } from './format'

export function LearnMore({ to, label = 'Learn more' }: { to: string; label?: string }) {
  return (
    <Link to={to} className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold uppercase tracking-wider"
      style={{ color: 'var(--text)' }}>
      {label} <ArrowRight size={10} />
    </Link>
  )
}

export function Segmented<T extends string>({ value, options, onChange, ariaLabel }: {
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

export function RegimeTag({ regime }: { regime: Regime }) {
  return (
    <span className="text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded"
      style={{ background: REGIME_COLORS[regime] + '20', color: REGIME_COLORS[regime] }}>
      {regime}
    </span>
  )
}

export function ChartCard({ icon, title, sub, children, tour, height = 280 }: {
  icon: React.ReactNode
  title: string
  sub?: string
  children: React.ReactNode
  tour?: string
  height?: number
}) {
  return (
    <div className="glass-card p-5 space-y-4 min-w-0" data-tour={tour}>
      <div>
        <div className="flex items-center gap-2">
          <span style={{ color: 'var(--text-muted)' }}>{icon}</span>
          <h3 className="label-text">{title}</h3>
        </div>
        {sub && <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
      </div>
      <div className="w-full" style={{ height }}>{children}</div>
    </div>
  )
}

const HIST_KEYS: StrategyKey[] = ['ac_optimal', 'adaptive_ac', 'rl']

/** One day: event, per-strategy dispersion and tail, and how the episodes spread. */
export function DayDetail({ d }: { d: DateResult }) {
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
              <th className="text-right py-1 font-normal">Mean</th>
              <th className="text-right py-1 font-normal">Std</th>
              <th className="text-right py-1 font-normal">CVaR 95</th>
            </tr>
          </thead>
          <tbody>
            {PINNED.strategies.map(s => (
              <tr key={s.key} style={{ color: s.key === 'rl' ? STRATEGY_COLORS.rl : 'var(--text-sub)' }}>
                <td className="py-0.5">{s.label}</td>
                <td className="py-0.5 text-right">{pct(d.strategies[s.key].mean)}</td>
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
                <Bar key={k} isAnimationActive={false} dataKey={k} name={strategyLabel(k)} fill={STRATEGY_COLORS[k]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
