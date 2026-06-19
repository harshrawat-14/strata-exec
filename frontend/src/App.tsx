import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Simulator from './pages/Simulator'
import RLEvaluation from './pages/RLEvaluation'
import DataUpload from './pages/DataUpload'
import ParameterSweep from './pages/ParameterSweep'
import Login from './pages/Login'
import { Menu } from 'lucide-react'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30_000 },
  },
})

function ProtectedRoute() {
  const token = localStorage.getItem('strataexec_token')
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return <Outlet />
}

function AppLayout() {
  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem('strataexec_sidebar_collapsed') === 'true'
  })
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  // Auto-close sidebar on mobile when navigating to a new route
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const toggleSidebar = () => {
    setCollapsed(prev => {
      const next = !prev
      localStorage.setItem('strataexec_sidebar_collapsed', String(next))
      return next
    })
  }

  return (
    <div className="flex flex-col md:flex-row min-h-screen overflow-x-hidden w-full">
      {/* Mobile Top Header */}
      <header
        className="md:hidden flex items-center justify-between px-5 py-4 sticky top-0 z-40"
        style={{
          background: 'var(--sidebar)',
          borderBottom: '1px solid var(--divider)',
        }}
      >
        <div className="flex items-center gap-3">
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
            <div className="text-[8px] font-semibold tracking-widest uppercase font-mono" style={{ color: 'var(--text-muted)' }}>
              Research Platform
            </div>
          </div>
        </div>
        <button
          onClick={() => setMobileOpen(true)}
          className="p-2 rounded-lg transition-colors"
          style={{ color: 'var(--text-sub)' }}
          aria-label="Open navigation menu"
        >
          <Menu size={20} />
        </button>
      </header>

      {/* Mobile Backdrop overlay */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="md:hidden fixed inset-0 z-40 bg-black/40 backdrop-blur-xs transition-opacity duration-300"
        />
      )}

      {/* Sidebar (drawer on mobile, sticky on desktop) */}
      <Sidebar
        collapsed={collapsed}
        onToggle={toggleSidebar}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      {/* Main scrollable area */}
      <main
        className="flex-1 min-w-0 min-h-screen p-4 sm:p-6 md:p-8 transition-all duration-300"
        style={{ 
          background: 'var(--page)',
        }}
      >
        <div className="max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public Route */}
          <Route path="/login" element={<Login />} />

          {/* Protected Routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route path="/"         element={<Dashboard />} />
              <Route path="/simulator" element={<Simulator />} />
              <Route path="/evaluate"  element={<RLEvaluation />} />
              <Route path="/upload"    element={<DataUpload />} />
              <Route path="/sweep"     element={<ParameterSweep />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

