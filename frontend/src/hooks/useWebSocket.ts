/**
 * useWebSocket — SSE connection hook with visibility/focus state recovery.
 * Subscribes to /api/jobs/{jobId}/progress via EventSource.
 */

import { useEffect, useRef, useCallback } from 'react'
import type { WsMessage } from '../types'

interface UseWebSocketOptions {
  jobId: string | null
  onMessage: (msg: WsMessage) => void
  enabled?: boolean
}

export function useWebSocket({ jobId, onMessage, enabled = true }: UseWebSocketOptions) {
  const eventSourceRef = useRef<EventSource | null>(null)
  const onMessageRef = useRef(onMessage)
  const connectRef = useRef<() => void>(() => {})

  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  const connect = useCallback(() => {
    if (!jobId || !enabled) return

    if (eventSourceRef.current) {
      if (eventSourceRef.current.readyState === EventSource.OPEN || eventSourceRef.current.readyState === EventSource.CONNECTING) {
        return
      }
      eventSourceRef.current.close()
    }

    const token = localStorage.getItem('strataexec_token') || ''
    const es = new EventSource(`/api/jobs/${jobId}/progress?token=${encodeURIComponent(token)}`)
    eventSourceRef.current = es

    es.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WsMessage
        onMessageRef.current(msg)
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      es.close()
      setTimeout(() => connectRef.current(), 2000)
    }
  }, [jobId, enabled])

  useEffect(() => {
    connectRef.current = connect
  }, [connect])


  const checkStateAndRestore = useCallback(async () => {
    if (!jobId || !enabled) return

    try {
      const token = localStorage.getItem('strataexec_token') || ''
      const state = await fetch(`/api/jobs/${jobId}/state?token=${encodeURIComponent(token)}`).then(r => r.json())
      
      const startedAtMs = state.started_at
        ? Date.parse(state.started_at.endsWith('Z') ? state.started_at : state.started_at + 'Z')
        : undefined

      if (state.status === 'complete') {
        onMessageRef.current({ type: 'status', status: 'complete', started_at: startedAtMs })
        // If results are present, trigger complete event
        const resultsUrl = (state.results && 'date_results' in state.results)
          ? `/api/evaluate/result/${jobId}`
          : `/api/compare/${jobId}`
        onMessageRef.current({ type: 'complete', job_id: jobId, results_url: resultsUrl, started_at: startedAtMs })
      } else if (state.status === 'failed') {
        onMessageRef.current({ type: 'status', status: 'failed', started_at: startedAtMs })
        onMessageRef.current({ type: 'error', message: state.error || 'Job failed', started_at: startedAtMs })
      } else {
        // Status is running or queued
        onMessageRef.current({ type: 'status', status: state.status, started_at: startedAtMs })
        
        // Restore progress values
        if (state.status === 'running' && state.progress !== undefined) {
          if (Array.isArray(state.partial_results)) {
            // Replay completed evaluation dates
            const total = state.dates_total || 5
            state.partial_results.forEach((dr: any, idx: number) => {
              onMessageRef.current({
                type: 'date_complete',
                date: dr.date,
                regime: dr.regime,
                rl_is: dr.rl_is,
                ac_is: dr.ac_is,
                improvement_pp: dr.improvement_pp,
                dates_done: idx + 1,
                dates_total: total,
                started_at: startedAtMs
              })
            })
          } else if (state.partial_results && typeof state.partial_results === 'object' && Object.keys(state.partial_results).length > 0) {
            const completed = Math.round((state.progress / 100) * (state.paths_total || 100))
            onMessageRef.current({
              type: 'paths_update',
              paths_done: completed,
              paths_total: state.paths_total || 100,
              partial_results: state.partial_results,
              started_at: startedAtMs
            })
          } else {
            // General progress percentage fallback
            onMessageRef.current({
              type: 'progress',
              completed: Math.round(state.progress),
              total: 100,
              started_at: startedAtMs
            })
          }
        }
        connect()
      }
    } catch (err) {
      console.error("Error restoring job state", err)
      connect()
    }
  }, [jobId, enabled, connect])

  useEffect(() => {
    checkStateAndRestore()

    // Setup fallback poll interval (every 3 seconds) to ensure status updates
    // occur even if SSE connection is buffered, closed or disconnected.
    const pollInterval = setInterval(() => {
      checkStateAndRestore()
    }, 3000)

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        checkStateAndRestore()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      clearInterval(pollInterval)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [jobId, enabled, checkStateAndRestore])
}
