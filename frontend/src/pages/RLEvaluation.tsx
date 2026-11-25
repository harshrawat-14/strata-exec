/**
 * RL Evaluation — the pinned PPO-LSTM agent, one day at a time.
 * Pick any of the evaluated days to see every strategy on it, from the frozen snapshot.
 */

import { useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { BarChart2, BrainCircuit, ShieldAlert, Target, TrendingUp } from 'lucide-react'
import { MetricTile } from '../components/ui'
import { PINNED, REGIMES, STRATEGY_COLORS, edgeVsAc } from '../data/pinned/results'
import { ChartCard, DayDetail, LearnMore, RegimeTag } from '../components/pinned/PinnedViews'
import { REGIME_INFO, axisTick, pct, pp, tooltipStyle } from '../components/pinned/format'

export default function RLEvaluation() {
  const [date, setDate] = useState('2024-08-05')
  const day = PINNED.dates.find(d => d.date === date)!
  const edge = edgeVsAc(day)
  const ranked = PINNED.strategies
    .map(s => ({ name: s.label, key: s.key, value: day.strategies[s.key].mean }))
    .sort((a, b) => b.value - a.value)
  const rank = ranked.findIndex(r => r.key === 'rl') + 1

  return (
    <div className="flex flex-col lg:flex-row gap-6 animate-fade-in" data-tour="rl-eval">
      {/* Day picker */}
      <div className="w-full lg:w-72 lg:flex-shrink-0 space-y-3">
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-1">
            <BrainCircuit size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="label-text">RL Evaluation</h2>
          </div>
          <p className="text-[10px] font-mono mb-4" style={{ color: 'var(--text-muted)' }}>
            Pick a day. {PINNED.protocol.episodes_per_strategy} episodes per strategy.
          </p>
          <div className="mb-4 p-2.5 rounded-lg" style={{ background: 'var(--input-bg)', border: '1px solid var(--divider)' }}>
            <div className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Model</div>
            <div className="text-[11px] font-mono font-semibold mt-0.5" style={{ color: 'var(--text)' }}>
              {PINNED.model.name} · {PINNED.model.variant}
            </div>
          </div>
          <div className="space-y-4 max-h-[60vh] overflow-y-auto custom-scrollbar pr-1">
            {REGIMES.map(r => (
              <div key={r} className="space-y-1.5">
                <div className="text-[9px] font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                  {REGIME_INFO[r].title}
                </div>
                {PINNED.dates.filter(d => d.regime === r).map(d => {
                  const on = d.date === date
                  const e = edgeVsAc(d)
                  return (
                    <button key={d.date} type="button" onClick={() => setDate(d.date)} aria-pressed={on}
                      className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left"
                      style={{
                        border: '1px solid', borderColor: on ? 'var(--card-border-hover)' : 'var(--divider)',
                        background: on ? 'var(--card-hover)' : 'transparent',
                      }}>
                      <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text)' }}>
                        {d.date}
                        {d.split === 'train' && (
                          <span className="ml-1.5 text-[8px] uppercase" style={{ color: 'var(--text-muted)' }}>train</span>
                        )}
                      </span>
                      <span className="text-[10px] font-mono" style={{ color: e > 0 ? STRATEGY_COLORS.rl : 'var(--text-muted)' }}>
                        {pp(e)}
                      </span>
                    </button>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Selected day */}
      <div className="flex-1 min-w-0 space-y-6">
        <div className="glass-card p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold font-mono" style={{ color: 'var(--text)' }}>{day.date}</h1>
              <RegimeTag regime={day.regime} />
              {day.split === 'train' && <span className="badge-neutral">Training day</span>}
            </div>
            <p className="text-xs mt-1" style={{ color: 'var(--text-sub)' }}>{day.event_detail}</p>
          </div>
          <LearnMore to="/learn#evaluation" label="How this was scored" />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          <MetricTile label="RL agent IS" value={pct(day.strategies.rl.mean)} sub="mean of 50 episodes"
            icon={<BrainCircuit size={14} />} />
          <MetricTile label="AC Optimal IS" value={pct(day.strategies.ac_optimal.mean)} sub="Almgren–Chriss schedule"
            icon={<Target size={14} />} />
          <MetricTile label="RL − AC Optimal" value={`${pp(edge, 3)} pp`}
            sub={edge > 0 ? 'RL sold at a better price' : 'AC sold at a better price'} icon={<TrendingUp size={14} />} />
          <MetricTile label="RL rank on the day" value={`${rank} / ${ranked.length}`}
            sub={`worst 5% (CVaR) ${pct(day.strategies.rl.cvar95)}`} icon={<ShieldAlert size={14} />} />
        </div>

        <ChartCard icon={<BarChart2 size={14} />} title="Every strategy on this day (mean IS %)" height={240}
          sub="Higher is better. Same 50 episode starts for every strategy.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ranked} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--divider)" horizontal={false} />
              <XAxis type="number" tickLine={false} axisLine={false} tick={axisTick} tickFormatter={v => `${pp(v, 1)}%`} />
              <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={96}
                tick={{ fill: 'var(--text-sub)', fontSize: 10 }} />
              <ReferenceLine x={0} stroke="var(--text-muted)" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--card-hover)' }}
                formatter={v => [pct(Number(v)), 'mean IS']} />
              <Bar isAnimationActive={false} dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={22}>
                {ranked.map(r => <Cell key={r.key} fill={STRATEGY_COLORS[r.key]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="glass-card p-5">
          <DayDetail d={day} />
        </div>
      </div>
    </div>
  )
}
