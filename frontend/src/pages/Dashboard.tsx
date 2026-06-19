/**
 * Dashboard page — overview stats, recent jobs, quick-actions.
 */

import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Activity, BarChart3, Database, BrainCircuit,
  ArrowRight, FlaskConical, Upload, TrendingDown
} from 'lucide-react'
import { fetchDashboard } from '../api/client'
import { JobStatusBadge, MetricTile, Skeleton, EmptyState } from '../components/ui'
import type { RecentJob } from '../types'

export default function Dashboard() {
  const navigate = useNavigate()
  const { data, isLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 10_000,
  })

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
            Research <span className="text-gradient-electric">Dashboard</span>
          </h1>
          <p className="mt-1 text-xs font-mono uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            Execution strategy benchmarking · Monte Carlo · RL evaluation
          </p>
        </div>
        <div className="flex flex-wrap gap-2 w-full sm:w-auto">
          <button onClick={() => navigate('/simulator')} className="btn-primary flex-1 sm:flex-initial">
            <FlaskConical size={12} />
            Run Simulation
          </button>
          <button onClick={() => navigate('/upload')} className="btn-secondary flex-1 sm:flex-initial">
            <Upload size={12} />
            Upload Data
          </button>
        </div>
      </div>

      {/* Stat tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28" />)
        ) : (
          <>
            <MetricTile label="Simulations Run"  value={data?.total_simulations ?? 0}  sub="Monte Carlo paths"  icon={<Activity size={14} />} />
            <MetricTile label="RL Evaluations"   value={data?.total_evaluations ?? 0}   sub="across regimes"     icon={<BrainCircuit size={14} />} />
            <MetricTile label="Available Data"   value={data?.available_lob_files ?? 0} sub="LOB snapshots"      icon={<Database size={14} />} />
            <MetricTile label="RL Models"        value={data?.available_models ?? 0}    sub="built-in + uploaded" icon={<BarChart3 size={14} />} />
          </>
        )}
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Recent jobs */}
        <div className="col-span-1 lg:col-span-2 glass-card p-6">
          <h2 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-5" style={{ color: 'var(--text-muted)' }}>
            Recent Jobs
          </h2>

          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12" />)}
            </div>
          ) : !data?.recent_jobs?.length ? (
            <EmptyState
              icon={<Activity size={24} />}
              title="No jobs yet"
              description="Run a simulation or RL evaluation to get started"
              action={
                <button onClick={() => navigate('/simulator')} className="btn-primary">
                  <FlaskConical size={12} />
                  Start Simulation
                </button>
              }
            />
          ) : (
            <div className="space-y-1">
              {(data.recent_jobs as RecentJob[]).map((job) => (
                <div
                  key={job.job_id}
                  onClick={() => navigate(
                    job.type === 'simulation'
                      ? `/simulator?job=${job.job_id}`
                      : `/evaluate?job=${job.job_id}`
                  )}
                  className="flex items-center justify-between px-4 py-3 rounded-lg cursor-pointer group transition-all duration-150"
                  style={{ background: 'transparent' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--card-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                >
                  <div className="flex items-center gap-3">
                    <JobStatusBadge status={job.status} />
                    <div>
                      <div className="text-sm font-medium" style={{ color: 'var(--text)' }}>{job.label}</div>
                      <div className="text-[10px] font-mono uppercase" style={{ color: 'var(--text-muted)' }}>
                        {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {job.duration_seconds && (
                      <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                        {job.duration_seconds.toFixed(1)}s
                      </span>
                    )}
                    <ArrowRight size={14} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Available dates + quick actions */}
        <div className="glass-card p-6 flex flex-col gap-6">
          <div>
            <h2 className="text-[10px] font-mono font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
              Available Dates
            </h2>
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-9" />)}
              </div>
            ) : (
              <div className="space-y-1 custom-scrollbar max-h-72 overflow-y-auto">
                {data?.available_dates?.map((d) => (
                  <div
                    key={d.date}
                    className="flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-150 cursor-default"
                    style={{ border: '1px solid var(--divider)' }}
                    onMouseEnter={e => {
                      e.currentTarget.style.background = 'var(--card-hover)'
                      e.currentTarget.style.borderColor = 'var(--card-border)'
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.background = 'transparent'
                      e.currentTarget.style.borderColor = 'var(--divider)'
                    }}
                  >
                    <div>
                      <div className="text-xs font-mono font-semibold" style={{ color: 'var(--text)' }}>{d.date}</div>
                      <div className="text-[9px] font-mono uppercase tracking-wider mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        {d.regime}
                      </div>
                    </div>
                    <span className="badge-neutral text-[9px]">{d.source}</span>
                  </div>
                ))}
                {!data?.available_dates?.length && (
                  <div className="text-xs text-center py-4 font-mono" style={{ color: 'var(--text-muted)' }}>
                    No LOB data files found
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Quick actions */}
          <div>
            <div className="section-title">Quick Actions</div>
            {[
              { label: 'Compare strategies', icon: TrendingDown, to: '/simulator' },
              { label: 'Evaluate RL model',  icon: BrainCircuit, to: '/evaluate' },
              { label: 'Upload your data',   icon: Upload,       to: '/upload' },
            ].map(({ label, icon: Icon, to }) => (
              <button
                key={to}
                onClick={() => navigate(to)}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[10px] font-mono font-semibold tracking-wider uppercase transition-all duration-150"
                style={{ color: 'var(--text-muted)', background: 'transparent' }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'var(--card-hover)'
                  e.currentTarget.style.color = 'var(--text)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = 'var(--text-muted)'
                }}
              >
                <Icon size={14} style={{ opacity: 0.5 }} />
                {label}
                <ArrowRight size={12} className="ml-auto" style={{ opacity: 0.3 }} />
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
