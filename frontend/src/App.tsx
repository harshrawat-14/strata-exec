/**
 * Root application — routing + React Query.
 * Layout: Sidebar (fixed) + main (scrollable, layered bg).
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Simulator from './pages/Simulator'
import RLEvaluation from './pages/RLEvaluation'
import DataUpload from './pages/DataUpload'
import ParameterSweep from './pages/ParameterSweep'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30_000 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {/* Outer shell — bg is set via CSS variable --page on body */}
        <div className="flex min-h-screen">
          <Sidebar />

          {/* Main scrollable area */}
          <main
            className="flex-1 ml-60 min-h-screen p-8"
            style={{ background: 'var(--page)' }}
          >
            <div className="relative max-w-7xl mx-auto">
              <Routes>
                <Route path="/"         element={<Dashboard />} />
                <Route path="/simulator" element={<Simulator />} />
                <Route path="/evaluate"  element={<RLEvaluation />} />
                <Route path="/upload"    element={<DataUpload />} />
                <Route path="/sweep"     element={<ParameterSweep />} />
              </Routes>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
