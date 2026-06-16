/**
 * Sidebar navigation — fixed left rail with page links and theme toggle.
 * Uses CSS variable --sidebar so it's one shade distinct from the page background.
 */

import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, FlaskConical, BrainCircuit,
  Upload, BarChart3, Sun, Moon, LogOut,
  ChevronLeft, ChevronRight
} from 'lucide-react'
import { clsx } from 'clsx'

const NAV_ITEMS = [
  { to: '/',          icon: LayoutDashboard, label: 'Dashboard'   },
  { to: '/simulator', icon: FlaskConical,    label: 'Simulator'   },
  { to: '/evaluate',  icon: BrainCircuit,    label: 'RL Eval'     },
  { to: '/upload',    icon: Upload,          label: 'Data Upload' },
  { to: '/sweep',     icon: BarChart3,       label: 'Param Sweep' },
]

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
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
      className={clsx(
        "sticky left-0 top-0 h-screen flex flex-col z-40 transition-all duration-300 flex-shrink-0",
        collapsed ? "w-16" : "w-60"
      )}
      style={{
        background: 'var(--sidebar)',
        borderRight: '1px solid var(--divider)',
      }}
    >
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-6 w-6 h-6 rounded-full flex items-center justify-center border z-50 transition-all duration-150"
        style={{
          background: 'var(--card)',
          borderColor: 'var(--card-border)',
          color: 'var(--text-sub)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          cursor: 'pointer'
        }}
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = 'var(--card-border-hover)'
          e.currentTarget.style.color = 'var(--text)'
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = 'var(--card-border)'
          e.currentTarget.style.color = 'var(--text-sub)'
        }}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      <div className={clsx("flex flex-col h-full", collapsed ? "p-3" : "p-6")}>

        {/* Logo */}
        <div className={clsx("flex items-center gap-3 py-2 mb-8", collapsed && "justify-center")}>
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center font-bold font-mono text-sm flex-shrink-0"
            style={{
              background: 'var(--active-fill)',
              color: 'var(--active-text)',
            }}
          >
            S
          </div>
          {!collapsed && (
            <div>
              <div className="text-sm font-bold tracking-tight" style={{ color: 'var(--text)' }}>
                StrataExec
              </div>
              <div className="text-[9px] font-semibold tracking-widest uppercase font-mono" style={{ color: 'var(--text-muted)' }}>
                Research Platform
              </div>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-1">
          {!collapsed && <div className="section-title">Navigation</div>}
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 py-2.5 rounded-lg text-[11px] font-mono font-medium tracking-wide uppercase transition-all duration-150',
                  collapsed ? 'justify-center px-0' : 'px-3.5',
                  isActive
                    ? 'font-semibold'
                    : 'hover:opacity-90'
                )
              }
              style={({ isActive }) => isActive
                ? { background: 'var(--active-fill)', color: 'var(--active-text)' }
                : { color: 'var(--text-sub)', background: 'transparent' }
              }
              title={collapsed ? label : undefined}
            >
              {({ isActive }) => (
                <>
                  <Icon size={14} style={{ opacity: isActive ? 1 : 0.65 }} className="flex-shrink-0" />
                  {!collapsed && label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom */}
        <div className="mt-auto space-y-4">
          {/* Logout button */}
          <button
            onClick={() => {
              localStorage.removeItem('strataexec_token')
              window.location.href = '/login'
            }}
            className={clsx(
              "w-full flex items-center rounded-lg text-[10px] font-mono font-semibold tracking-widest uppercase transition-all duration-150",
              collapsed ? "justify-center p-2.5" : "justify-between px-3.5 py-2.5"
            )}
            style={{
              border: '1px solid var(--card-border)',
              color: '#fca5a5',
              background: 'rgba(239, 68, 68, 0.03)',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.2)'
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.08)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--card-border)'
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.03)'
            }}
            title={collapsed ? "Logout" : undefined}
          >
            <span className="flex items-center gap-2">
              <LogOut size={14} className="flex-shrink-0" />
              {!collapsed && 'Logout'}
            </span>
          </button>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className={clsx(
              "w-full flex items-center rounded-lg text-[10px] font-mono font-semibold tracking-widest uppercase transition-all duration-150",
              collapsed ? "justify-center p-2.5" : "justify-between px-3.5 py-2.5"
            )}
            style={{
              border: '1px solid var(--card-border)',
              color: 'var(--text-sub)',
              background: 'transparent',
            }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--card-border-hover)')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--card-border)')}
            title={collapsed ? (theme === 'dark' ? 'Light Mode' : 'Dark Mode') : undefined}
          >
            <span className="flex items-center gap-2">
              {theme === 'dark' ? <Sun size={14} className="flex-shrink-0" /> : <Moon size={14} className="flex-shrink-0" />}
              {!collapsed && (theme === 'dark' ? 'Light Mode' : 'Dark Mode')}
            </span>
          </button>

          {/* System Stack card */}
          {!collapsed && (
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
          )}
        </div>
      </div>
    </aside>
  )
}
