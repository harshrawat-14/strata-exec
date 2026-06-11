/**
 * Sidebar navigation — fixed left rail with page links and theme toggle.
 * Uses CSS variable --sidebar so it's one shade distinct from the page background.
 */

import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, FlaskConical, BrainCircuit,
  Upload, BarChart3, Sun, Moon
} from 'lucide-react'
import { clsx } from 'clsx'

const NAV_ITEMS = [
  { to: '/',          icon: LayoutDashboard, label: 'Dashboard'   },
  { to: '/simulator', icon: FlaskConical,    label: 'Simulator'   },
  { to: '/evaluate',  icon: BrainCircuit,    label: 'RL Eval'     },
  { to: '/upload',    icon: Upload,          label: 'Data Upload' },
  { to: '/sweep',     icon: BarChart3,       label: 'Param Sweep' },
]

export function Sidebar() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    (localStorage.getItem('theme') as 'light' | 'dark') || 'dark'
  )

  useEffect(() => {
    const root = window.document.documentElement
    if (theme === 'dark') root.classList.add('dark')
    else root.classList.remove('dark')
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark')

  return (
    <aside
      className="fixed left-0 top-0 h-screen w-60 flex flex-col z-40"
      style={{
        background: 'var(--sidebar)',
        borderRight: '1px solid var(--divider)',
      }}
    >
      <div className="flex flex-col h-full p-6">

        {/* Logo */}
        <div className="flex items-center gap-3 py-2 mb-8">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center font-bold font-mono text-sm"
            style={{
              background: 'var(--active-fill)',
              color: 'var(--active-text)',
            }}
          >
            S
          </div>
          <div>
            <div className="text-sm font-bold tracking-tight" style={{ color: 'var(--text)' }}>
              StrataExec
            </div>
            <div className="text-[9px] font-semibold tracking-widest uppercase font-mono" style={{ color: 'var(--text-muted)' }}>
              Research Platform
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-1">
          <div className="section-title">Navigation</div>
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-[11px] font-mono font-medium tracking-wide uppercase transition-all duration-150',
                  isActive
                    ? 'font-semibold'
                    : 'hover:opacity-90'
                )
              }
              style={({ isActive }) => isActive
                ? { background: 'var(--active-fill)', color: 'var(--active-text)' }
                : { color: 'var(--text-muted)', background: 'transparent' }
              }
            >
              {({ isActive }) => (
                <>
                  <Icon size={14} style={{ opacity: isActive ? 1 : 0.5 }} />
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom */}
        <div className="mt-auto space-y-4">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-[10px] font-mono font-semibold tracking-widest uppercase transition-all duration-150"
            style={{
              border: '1px solid var(--card-border)',
              color: 'var(--text-sub)',
              background: 'transparent',
            }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--card-border-hover)')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--card-border)')}
          >
            <span className="flex items-center gap-2">
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </span>
          </button>

          {/* System Stack card */}
          <div
            className="rounded-xl p-4"
            style={{ background: 'var(--card)', border: '1px solid var(--card-border)' }}
          >
            <div className="section-title mb-3">System Stack</div>
            {[
              { label: 'Rust Simulator', detail: 'GBM / GARCH' },
              { label: 'FastAPI Backend', detail: 'Async SQLite' },
              { label: 'RL Evaluation',  detail: 'PPO-LSTM' },
            ].map(({ label, detail }) => (
              <div key={label} className="py-1">
                <div className="text-[10px] font-semibold" style={{ color: 'var(--text)' }}>{label}</div>
                <div className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{detail}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  )
}
