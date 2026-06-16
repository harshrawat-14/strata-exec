/**
 * Execution trajectory chart — inventory remaining over time for all strategies.
 * Styled in stark monochrome to match B&W mode.
 */

import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import type { StrategyResult } from '../types'

function useDarkMode() {
  const [isDark, setIsDark] = useState(() =>
    typeof window !== 'undefined' ? document.documentElement.classList.contains('dark') : true
  )

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'))
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  return isDark
}

interface TrajectoryChartProps {
  strategies: StrategyResult[]
  pricePoints?: number[]
  showPrice?: boolean
}

interface DataPoint {
  step: number
  [key: string]: number | null
}

export function TrajectoryChart({ strategies, pricePoints = [], showPrice = false }: TrajectoryChartProps) {
  const isDark = useDarkMode()

  if (!strategies.length) return null

  const maxLen = Math.max(...strategies.map((s) => s.trajectory.length))

  // Build combined data array
  const data: DataPoint[] = Array.from({ length: maxLen }, (_, i) => {
    const pt: DataPoint = { step: i }
    for (const s of strategies) {
      pt[s.name] = s.trajectory[i] ?? null
    }
    if (showPrice && pricePoints.length) {
      pt['_price'] = pricePoints[i] ?? null
    }
    return pt
  })

  // Monochrome styling configuration
  const STRATEGY_STYLE: Record<string, { stroke: string; dash: string; width: number }> = {
    'TWAP': { stroke: isDark ? '#555555' : '#c0c0c0', dash: '3 3', width: 1.5 },
    'Heuristic': { stroke: isDark ? '#888888' : '#888888', dash: '6 3', width: 1.5 },
    'Optimal (AC)': { stroke: isDark ? '#ffffff' : '#000000', dash: '0', width: 2 },
    'Adaptive Optimal': { stroke: isDark ? '#cccccc' : '#555555', dash: '8 3 2 3', width: 1.5 },
    'RegimeAC': { stroke: isDark ? '#e0e0e0' : '#333333', dash: '2 1', width: 1.5 },
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-48">
        <div className="text-black/30 dark:text-white/30 mb-2 uppercase">Step {label}</div>
        {payload.map((p: any) => (
          <div key={p.dataKey} className="flex justify-between gap-4">
            <span>{p.dataKey}</span>
            <span className="font-semibold">
              {p.value !== null ? `${p.value.toLocaleString('en', { maximumFractionDigits: 1 })}` : '—'}
            </span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="step"
          tickLine={false}
          label={{ value: 'STEP', position: 'insideBottomRight', dy: 10, fontSize: 10, fill: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)' }}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          verticalAlign="bottom"
          height={36}
          iconType="plainline"
        />
        {strategies.map((s) => {
          const style = STRATEGY_STYLE[s.name] || { stroke: isDark ? '#ffffff' : '#000000', dash: '0', width: 1.5 }
          return (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              stroke={style.stroke}
              strokeDasharray={style.dash}
              strokeWidth={style.width}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0, fill: style.stroke }}
              connectNulls
            />
          )
        })}
        <ReferenceLine y={0} stroke={isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'} strokeDasharray="4 4" />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Price path chart ──────────────────────────────────────────────────────────

interface PriceChartProps {
  pricePoints: number[]
}

const PriceTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="glass-card p-3 text-[11px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white min-w-[150px]">
      <div className="text-black/30 dark:text-white/30 mb-2 uppercase font-semibold">Step {label}</div>
      <div className="flex justify-between gap-4">
        <span>Price:</span>
        <span className="font-bold">${payload[0].value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
      </div>
    </div>
  )
}

export function PriceChart({ pricePoints }: PriceChartProps) {
  const isDark = useDarkMode()
  if (!pricePoints.length) return null

  const data = pricePoints.map((p, i) => ({ step: i, price: p }))

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 10, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={isDark ? 0.08 : 0.15} />
        <XAxis
          dataKey="step"
          tickLine={false}
          tick={{ fontSize: 9, fontFamily: 'monospace' }}
          label={{ value: 'STEP', position: 'insideBottomRight', offset: -5, style: { fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          domain={['auto', 'auto']}
          tickFormatter={(v) => `$${v.toFixed(0)}`}
          tick={{ fontSize: 9, fontFamily: 'monospace' }}
          label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'monospace' } }}
        />
        <Tooltip content={<PriceTooltip />} />
        <Line
          type="monotone"
          dataKey="price"
          stroke={isDark ? '#ffffff' : '#000000'}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3, fill: isDark ? '#ffffff' : '#000000' }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
