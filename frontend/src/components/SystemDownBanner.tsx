import { AlertTriangle, ArrowRight } from 'lucide-react'
import { useTour } from '../tour/context'

const AMBER = '#F59E0B'

/**
 * Shown wherever a feature needs the simulation engine and the backend is unreachable.
 * `compact` drops the apology text; `stacked` keeps the button under the text (narrow cards,
 * such as the login card, which also sits outside the app's light/dark theme).
 */
export function SystemDownBanner({ compact = false, stacked = false }: { compact?: boolean; stacked?: boolean }) {
  const { start } = useTour()
  return (
    <div
      role="status"
      className={stacked
        ? 'rounded-xl p-4 flex flex-col gap-4 text-left'
        : 'glass-card p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center gap-4 justify-between'}
      style={stacked
        ? { background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.35)' }
        : { borderColor: 'rgba(245, 158, 11, 0.35)' }}
    >
      <div className="flex items-start gap-3">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ border: '1px solid rgba(245, 158, 11, 0.35)', color: AMBER }}
        >
          <AlertTriangle size={14} />
        </div>
        <div>
          <div className="text-sm font-semibold" style={{ color: stacked ? '#FDE68A' : 'var(--text)' }}>
            The simulation engine is down
          </div>
          {!compact && (
            <p className="text-xs mt-1 max-w-xl leading-relaxed"
              style={{ color: stacked ? 'rgba(253, 230, 138, 0.8)' : 'var(--text-sub)' }}>
              We're sorry for the inconvenience. Live simulations run on our compute backend, which is
              offline right now. How about a guided tour of what we built and what we found? Every number
              in it comes from completed experiments.
            </p>
          )}
        </div>
      </div>
      <button
        onClick={start}
        className={stacked ? 'inline-flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-xs font-semibold uppercase tracking-wider' : 'btn-primary flex-shrink-0'}
        style={stacked ? { background: AMBER, color: '#111827' } : undefined}
      >
        Take the tour
        <ArrowRight size={12} />
      </button>
    </div>
  )
}
