/**
 * Shared UI primitives — all styled via CSS variables (light/dark aware).
 */

import { clsx } from 'clsx'
import { useEffect, useRef, useState } from 'react'
import type { JobStatusValue } from '../types'

// ── Job Status Badge ──────────────────────────────────────────────────────────
export function JobStatusBadge({ status }: { status: JobStatusValue }) {
  const map: Record<JobStatusValue, string> = {
    queued:   'badge-neutral',
    running:  'badge-info',
    complete: 'badge-success',
    failed:   'badge-error',
  }
  const dot: Record<JobStatusValue, string> = {
    queued:  'opacity-30',
    running: 'animate-pulse',
    complete:'opacity-80',
    failed:  'opacity-30',
  }
  return (
    <span className={map[status]}>
      <span
        className={clsx('inline-block w-1.5 h-1.5 rounded-full', dot[status])}
        style={{ background: 'var(--text)' }}
      />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

// ── Skeleton loader ───────────────────────────────────────────────────────────
export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx('skeleton', className)} />
}

// ── Enhanced Progress Bar with ETA ───────────────────────────────────────────
interface ProgressBarProps {
  completed: number
  total: number
  label?: string
  startedAt?: number // Date.now() when job started
}

export function ProgressBar({ completed, total, label, startedAt }: ProgressBarProps) {
  const pct = total > 0 ? Math.min(100, (completed / total) * 100) : 0
  const [eta, setEta] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const mountedAt = useRef(startedAt ?? Date.now())

  useEffect(() => {
    if (startedAt) {
      mountedAt.current = startedAt
    }
  }, [startedAt])

  const completedRef = useRef(completed)
  const totalRef = useRef(total)

  useEffect(() => {
    completedRef.current = completed
    totalRef.current = total
  }, [completed, total])

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now()
      const elapsedSec = (now - mountedAt.current) / 1000
      setElapsed(elapsedSec)

      const comp = completedRef.current
      const tot = totalRef.current

      if (comp > 0 && tot > 0 && comp < tot) {
        const rate = comp / elapsedSec           // paths / sec
        const remaining = tot - comp
        const etaSec = remaining / rate
        if (etaSec < 3600) {
          const m = Math.floor(etaSec / 60)
          const s = Math.round(etaSec % 60)
          setEta(m > 0 ? `~${m}m ${s}s` : `~${s}s`)
        }
      } else if (comp >= tot) {
        setEta(null)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-full space-y-2.5">
      {/* Top row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Pulsing indicator */}
          <span
            className="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
            style={{ background: 'var(--text)' }}
          />
          <span className="text-[10px] font-mono uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {label || 'Progress'}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono" style={{ color: 'var(--text-sub)' }}>
          {eta && (
            <span style={{ color: 'var(--text-muted)' }}>ETA {eta}</span>
          )}
          <span style={{ color: 'var(--text)' }} className="font-semibold">
            {pct.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Progress track */}
      <div
        className="h-1.5 rounded-full overflow-hidden"
        style={{ background: 'var(--card-border)' }}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            background: 'var(--active-fill)',
          }}
        />
      </div>

      {/* Bottom row */}
      <div className="flex justify-between text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>
        <span>{completed.toLocaleString()} / {total.toLocaleString()} paths</span>
        {elapsed > 0 && <span>{elapsed.toFixed(0)}s elapsed</span>}
      </div>
    </div>
  )
}

// ── IS value (implementation shortfall) ──────────────────────────────────────
export function IS({ value, digits = 3 }: { value: number | null; digits?: number }) {
  if (value === null || value === undefined)
    return <span className="font-mono" style={{ color: 'var(--text-muted)' }}>—</span>
  return (
    <span
      className="font-mono"
      style={{
        color: value < 0 ? 'var(--text)' : 'var(--text-muted)',
        fontWeight: value < 0 ? 700 : 400,
      }}
    >
      {value >= 0 ? '+' : ''}{value.toFixed(digits)}%
    </span>
  )
}

// ── Metric tile ───────────────────────────────────────────────────────────────
interface MetricTileProps {
  label: string
  value: React.ReactNode
  sub?: string
  icon?: React.ReactNode
}
export function MetricTile({ label, value, sub, icon }: MetricTileProps) {
  return (
    <div className="stat-card">
      <div className="flex items-start justify-between mb-4">
        <span className="label-text">{label}</span>
        {icon && (
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-xs"
            style={{
              border: '1px solid var(--card-border)',
              color: 'var(--text-muted)',
            }}
          >
            {icon}
          </div>
        )}
      </div>
      <div
        className="text-2xl font-semibold tracking-tight"
        style={{ color: 'var(--text)' }}
      >
        {value}
      </div>
      {sub && (
        <div className="text-[10px] font-mono uppercase mt-2" style={{ color: 'var(--text-muted)' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
export function EmptyState({ icon, title, description, action }: {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      {icon && (
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center text-xl"
          style={{ border: '1px solid var(--card-border)', color: 'var(--text-muted)' }}
        >
          {icon}
        </div>
      )}
      <div>
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-sub)' }}>{title}</h3>
        {description && (
          <p className="text-xs mt-1 max-w-sm" style={{ color: 'var(--text-muted)' }}>{description}</p>
        )}
      </div>
      {action}
    </div>
  )
}

// ── Section header ────────────────────────────────────────────────────────────
export function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-6">
      <h2 className="text-lg font-bold tracking-tight" style={{ color: 'var(--text)' }}>{title}</h2>
      {sub && <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
    </div>
  )
}
