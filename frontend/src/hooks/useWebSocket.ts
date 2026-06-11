/**
 * useWebSocket — generic WebSocket hook with auto-reconnect.
 * Subscribes to /ws/job/{jobId} and calls onMessage for each event.
 */

import { useEffect, useRef, useCallback } from 'react'
import type { WsMessage } from '../types'

interface UseWebSocketOptions {
  jobId: string | null
  onMessage: (msg: WsMessage) => void
  enabled?: boolean
}

export function useWebSocket({ jobId, onMessage, enabled = true }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (!jobId || !enabled) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/ws/job/${jobId}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WsMessage
        onMessageRef.current(msg)
      } catch {
        // ignore parse errors
      }
    }

    ws.onerror = () => {
      // Silently reconnect after 1s
      setTimeout(connect, 1000)
    }

    ws.onclose = (e) => {
      // Don't reconnect on intentional close (code 1000)
      if (e.code !== 1000 && enabled) {
        setTimeout(connect, 1000)
      }
    }
  }, [jobId, enabled])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000)
        wsRef.current = null
      }
    }
  }, [connect])
}
