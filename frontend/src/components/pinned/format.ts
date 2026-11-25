/** Formatting and copy shared by the pinned-results views. */
import { PINNED } from '../../data/pinned/results'
import type { Regime, StrategyKey } from '../../data/pinned/results'

export const tooltipStyle = {
  background: 'var(--card)',
  border: '1px solid var(--card-border)',
  borderRadius: 8,
  fontSize: 10,
  fontFamily: 'JetBrains Mono, monospace',
  color: 'var(--text)',
}
export const axisTick = { fill: 'var(--text-muted)', fontSize: 9 }

/** Signed number with a real minus sign: +1.41, −0.25. */
export const pp = (v: number, digits = 2) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`
export const pct = (v: number, digits = 3) => `${pp(v, digits)}%`
export const strategyLabel = (k: StrategyKey) => PINNED.strategies.find(s => s.key === k)!.label

export const REGIME_INFO: Record<Regime | 'all', { title: string; text: string }> = {
  all: {
    title: 'All regimes',
    text: 'Every evaluated day, 25 in all. Days are sorted into regimes by how costly they were for the AC Optimal schedule, split into thirds.',
  },
  crash: {
    title: 'Crash',
    text: 'The costliest third: sharp sell-offs such as the August 2024 yen-carry unwind. Prices fall while you sell, so selling earlier pays, and this is where the agent wins.',
  },
  normal: {
    title: 'Normal',
    text: 'The middle third: moderate moves with no clear direction. A steady schedule like AC Optimal is hard to beat here.',
  },
  calm: {
    title: 'Calm',
    text: 'The cheapest third: flat or rising prices. Waiting costs little or even pays, so the agent’s early selling loses to the classical schedules.',
  },
}
