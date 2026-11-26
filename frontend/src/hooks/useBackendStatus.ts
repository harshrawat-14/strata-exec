import { useQuery } from '@tanstack/react-query'

export type BackendStatus = 'checking' | 'online' | 'offline'

const BASE = import.meta.env.VITE_API_URL || ''
const TIMEOUT_MS = 6_000

/**
 * Probes the API's health endpoint. Any network error, timeout, non-2xx answer or body other than
 * the API's own `{"status": "ok"}` counts as offline. The body check matters on Vercel: an
 * unproxied path falls through to index.html with a 200.
 */
export async function probeBackend(fetchImpl: typeof fetch = fetch, timeoutMs = TIMEOUT_MS): Promise<boolean> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetchImpl(`${BASE}/health`, { signal: ctrl.signal, cache: 'no-store' })
    if (!res.ok) return false
    const body = await res.json()
    return body?.status === 'ok'
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

export function useBackendStatus(): BackendStatus {
  const { data, isPending } = useQuery({
    queryKey: ['backend-health'],
    queryFn: () => probeBackend(),
    staleTime: 60_000,
    refetchInterval: 60_000,
    retry: false,
  })
  if (isPending) return 'checking'
  return data ? 'online' : 'offline'
}

function hasSession(): boolean {
  try {
    return !!localStorage.getItem('strataexec_token')
  } catch {
    return false
  }
}

/**
 * Show the pinned, precomputed views. True while the engine is offline, and for visitors without a
 * session even when it is up: the live API answers them with 401s and empty lists.
 */
export function useShowcase(): boolean {
  const backend = useBackendStatus()
  return backend === 'offline' || !hasSession()
}
