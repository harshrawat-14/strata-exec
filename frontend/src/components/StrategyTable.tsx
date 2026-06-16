/**
 * Strategy comparison table — sortable, monochromatic, CSS variable aware.
 */

import { useState } from 'react'
import { ArrowUpDown } from 'lucide-react'
import type { StrategyResult } from '../types'
import { IS } from './ui'

interface StrategyTableProps { strategies: StrategyResult[] }
type SortKey = 'name' | 'mean_is_pct' | 'is_variance' | 'cvar95' | 'ac_objective' | 'trade_count' | 'avg_exec_price'

const StrategyIndicator = ({ name }: { name: string }) => {
  if (name.includes('TWAP'))
    return <div className="w-2.5 h-2.5 rounded-full border border-dashed flex-shrink-0" style={{ borderColor: 'var(--text-muted)' }} />
  if (name.includes('Heuristic'))
    return <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'var(--text-muted)' }} />
  if (name.includes('Optimal (AC)') || name === 'Optimal')
    return <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'var(--text)' }} />
  return <div className="w-2.5 h-2.5 rounded-full border-2 border-double flex-shrink-0" style={{ borderColor: 'var(--text-sub, var(--text-muted))' }} />
}

export function StrategyTable({ strategies }: StrategyTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('mean_is_pct')
  const [sortAsc, setSortAsc] = useState(true)

  const sorted = [...strategies].sort((a, b) => {
    const av = a[sortKey] ?? Infinity
    const bv = b[sortKey] ?? Infinity
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number)
  })

  const toggle = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  const Th = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none"
      style={{ color: 'var(--text-muted)' }}
      onClick={() => toggle(k)}
    >
      <span className="flex items-center gap-1">
        {label}
        <ArrowUpDown size={11} style={{ opacity: 0.4 }} />
      </span>
    </th>
  )

  return (
    <div className="overflow-x-auto custom-scrollbar">
      <table className="w-full">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--divider)' }}>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Strategy
            </th>
            <Th k="mean_is_pct"   label="Mean IS%" />
            <Th k="is_variance"   label="IS Variance" />
            <Th k="cvar95"        label="CVaR 95%" />
            <Th k="ac_objective"  label="AC Objective" />
            <Th k="trade_count"   label="Trade Count" />
            <Th k="avg_exec_price" label="Avg Exec Price" />
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Cost Breakdown
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((s) => {
            const breakdown = s.cost_decomposition || {}
            const total = Math.abs(Object.values(breakdown).reduce((a, b) => a + b, 0)) || 1
            return (
              <tr
                key={s.name}
                style={{ borderBottom: '1px solid var(--divider)', background: 'transparent', transition: 'background 0.15s ease' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <td className="px-4 py-3.5">
                  <div className="flex items-center gap-2.5">
                    <StrategyIndicator name={s.name} />
                    <span className="font-semibold text-sm" style={{ color: 'var(--text)' }}>{s.name}</span>
                  </div>
                </td>
                <td className="px-4 py-3.5"><IS value={s.mean_is_pct} /></td>
                <td className="px-4 py-3.5 text-sm font-mono" style={{ color: 'var(--text-muted)' }}>
                  {s.is_variance !== null ? s.is_variance.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-3.5"><IS value={s.cvar95} /></td>
                <td className="px-4 py-3.5 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>
                  {s.ac_objective !== null ? s.ac_objective?.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-3.5 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>
                  {s.trade_count !== null && s.trade_count !== undefined ? s.trade_count.toFixed(1) : '—'}
                </td>
                <td className="px-4 py-3.5 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>
                  {s.avg_exec_price !== null && s.avg_exec_price !== undefined ? s.avg_exec_price.toFixed(2) : '—'}
                </td>
                <td className="px-4 py-3.5">
                  {Object.keys(breakdown).length > 0 ? (
                    <div className="flex gap-0.5 h-4 rounded-sm overflow-hidden w-36" style={{ border: '1px solid var(--card-border)' }}>
                      {Object.entries(breakdown).map(([k, v], idx) => {
                        const pct = (Math.abs(v) / total) * 100
                        // Monochromatic gradient from light to dark
                        const opacity = 0.15 + (idx / 5) * 0.7
                        return (
                          <div
                            key={k}
                            style={{
                              width: `${pct}%`,
                              background: `rgba(128,128,128,${opacity})`,
                              height: '100%',
                            }}
                            title={`${k}: ${v.toFixed(4)}`}
                          />
                        )
                      })}
                    </div>
                  ) : (
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)', opacity: 0.4 }}>—</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
