/**
 * Dashboard while the engine is offline: what StrataExec is and where to look, from static data.
 */
import { useNavigate } from 'react-router-dom'
import {
  ArrowRight, BarChart3, BookOpen, BrainCircuit, CalendarDays, Compass, FlaskConical, Layers, Upload,
} from 'lucide-react'
import { MetricTile } from './ui'
import { PINNED, STRATEGY_COLORS, edgeVsAc } from '../data/pinned/results'
import { RegimeTag } from './pinned/PinnedViews'
import { pp } from './pinned/format'

const AREAS = [
  { to: '/results', icon: Compass, title: 'Results', text: 'The PPO-LSTM agent against four classical schedules on 25 real trading days.' },
  { to: '/simulator', icon: FlaskConical, title: 'Simulator', text: 'Monte Carlo strategy comparison on GBM, GARCH and jump price paths.' },
  { to: '/evaluate', icon: BrainCircuit, title: 'RL Eval', text: 'Every strategy on any single day, episode by episode.' },
  { to: '/sweep', icon: BarChart3, title: 'Parameter Sweep', text: 'How cost responds to volatility, horizon, impact and slicing.' },
  { to: '/upload', icon: Upload, title: 'Data Upload', text: 'Bring your own order-book data or trained model (engine required).' },
  { to: '/learn', icon: BookOpen, title: 'Learn', text: 'How it all works, in depth, with links to the source documents.' },
]

export function DashboardShowcase() {
  const navigate = useNavigate()
  const episodes = PINNED.dates.length * PINNED.strategies.length * PINNED.protocol.episodes_per_strategy
  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
          Research <span className="text-gradient-electric">Dashboard</span>
        </h1>
        <p className="mt-1 text-xs font-mono uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          Execution strategy benchmarking · Monte Carlo · RL evaluation
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile label="Trading days evaluated" value={PINNED.dates.length} sub="Binance BTC-USDT perp, 2024" icon={<CalendarDays size={14} />} />
        <MetricTile label="Strategies compared" value={PINNED.strategies.length} sub="4 classical + PPO-LSTM" icon={<BrainCircuit size={14} />} />
        <MetricTile label="Episodes scored" value={episodes.toLocaleString()} sub="same starts for every strategy" icon={<Layers size={14} />} />
        <MetricTile label="Simulation presets" value={6} sub="GBM · GARCH · jumps" icon={<FlaskConical size={14} />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="col-span-1 lg:col-span-2 glass-card p-6">
          <h2 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-5" style={{ color: 'var(--text-muted)' }}>
            Explore
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {AREAS.map(({ to, icon: Icon, title, text }) => (
              <button key={to} type="button" onClick={() => navigate(to)}
                className="glass-card-hover p-4 text-left flex gap-3 items-start">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ border: '1px solid var(--card-border)', color: 'var(--text-muted)' }}>
                  <Icon size={14} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold flex items-center gap-1.5" style={{ color: 'var(--text)' }}>
                    {title} <ArrowRight size={12} style={{ opacity: 0.4 }} />
                  </div>
                  <div className="text-xs mt-1 leading-5" style={{ color: 'var(--text-muted)' }}>{text}</div>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
            Evaluated days
          </h2>
          <div className="space-y-1 custom-scrollbar max-h-96 overflow-y-auto">
            {PINNED.dates.map(d => {
              const e = edgeVsAc(d)
              return (
                <button key={d.date} type="button" onClick={() => navigate('/evaluate')}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left hover:bg-[var(--card-hover)]"
                  style={{ border: '1px solid var(--divider)' }}>
                  <span className="flex items-center gap-2">
                    <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text)' }}>{d.date}</span>
                    <RegimeTag regime={d.regime} />
                  </span>
                  <span className="text-[10px] font-mono" style={{ color: e > 0 ? STRATEGY_COLORS.rl : 'var(--text-muted)' }}>
                    {pp(e)} pp
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
