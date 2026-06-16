import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Simulator from './pages/Simulator'
import RLEvaluation from './pages/RLEvaluation'
import DataUpload from './pages/DataUpload'
import ParameterSweep from './pages/ParameterSweep'
import Login from './pages/Login'

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

  const toggleSidebar = () => {
    setCollapsed(prev => {
      const next = !prev
      localStorage.setItem('strataexec_sidebar_collapsed', String(next))
      return next
    })
  }

  return (
    <div className="flex min-h-screen overflow-x-hidden">
      <Sidebar collapsed={collapsed} onToggle={toggleSidebar} />

      {/* Main scrollable area */}
      <main
        className="flex-1 min-w-0 min-h-screen p-8 transition-all duration-300"
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
