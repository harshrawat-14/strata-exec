/**
 * Order book depth visualisation — horizontal bid/ask bars at price levels.
 * Rebuilt in stark solid/outline monochrome styling.
 */

import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid } from 'recharts'
import type { DepthSnapshot, SpreadPoint } from '../types'

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

interface OrderBookVizProps {
  snapshot: DepthSnapshot | null
}

export function OrderBookViz({ snapshot }: OrderBookVizProps) {
  const isDark = useDarkMode()

  if (!snapshot) {
    return (
      <div className="flex items-center justify-center h-48 text-black/20 dark:text-white/20 text-xs font-mono uppercase tracking-wider">
        Upload LOB data to see depth
      </div>
    )
  }

  // Build chart data — bids negative (left), asks positive (right)
  const data = [
    ...snapshot.bid_levels
      .filter((l) => l.price !== null && l.qty !== null)
      .reverse()
      .map((l) => ({
        price: `${l.price?.toFixed(2)}`,
        bid: -(l.qty ?? 0),
        ask: 0,
        side: 'bid',
      })),
    ...snapshot.ask_levels
      .filter((l) => l.price !== null && l.qty !== null)
      .map((l) => ({
        price: `${l.price?.toFixed(2)}`,
        bid: 0,
        ask: l.qty ?? 0,
        side: 'ask',
      })),
  ]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const isBid = payload.find((p: any) => p.name === 'bid' && p.value !== 0)
    return (
      <div className="glass-card p-3 text-[10px] font-mono space-y-1 bg-white dark:bg-black border border-black/10 dark:border-white/10 text-black dark:text-white">
        <div className="text-black/30 dark:text-white/30 uppercase">Price: ${label}</div>
        {payload.map((p: any) => {
          if (p.value === 0) return null
          return (
            <div key={p.name} className="flex gap-4 justify-between">
              <span className="font-semibold uppercase">{isBid ? 'Bid' : 'Ask'}</span>
              <span className="font-mono">{Math.abs(p.value).toFixed(4)} BTC</span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 0, right: 8, bottom: 0, left: 60 }}
      >
        <XAxis
          type="number"
          tickLine={false}
          tickFormatter={(v) => Math.abs(v).toFixed(2)}
        />
        <YAxis
          type="category"
          dataKey="price"
          tickLine={false}
          width={55}
        />
        <Tooltip content={<CustomTooltip />} />
        {/* Bids: Solid black/white */}
        <Bar dataKey="bid" stackId="a" fill={isDark ? '#ffffff' : '#000000'} radius={[0, 0, 0, 0]} />
        {/* Asks: Stark Outline border */}
        <Bar dataKey="ask" stackId="a" fill="transparent" stroke={isDark ? '#ffffff' : '#000000'} strokeWidth={1} radius={[0, 0, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Spread over time chart ────────────────────────────────────────────────────

interface SpreadChartProps {
  series: SpreadPoint[]
}

export function SpreadChart({ series }: SpreadChartProps) {
  const isDark = useDarkMode()
  if (!series.length) return null

  return (
    <ResponsiveContainer width="100%" height={120}>
      <LineChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" tickLine={false} />
        <YAxis
          dataKey="spread_bps"
          tickLine={false}
          axisLine={false}
          width={30}
          tickFormatter={(v) => `${v.toFixed(1)}`}
        />
        <Tooltip
          contentStyle={{
            background: isDark ? '#000000' : '#ffffff',
            border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.1)',
            borderRadius: 6,
            fontSize: 10,
            fontFamily: 'JetBrains Mono, monospace'
          }}
          formatter={(v: any) => [`${Number(v).toFixed(3)} bps`, 'Spread']}
        />
        <Line
          type="monotone"
          dataKey="spread_bps"
          stroke={isDark ? '#ffffff' : '#000000'}
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
