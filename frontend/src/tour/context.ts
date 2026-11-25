import { createContext, useContext } from 'react'

export interface TourState {
  active: boolean
  index: number
  start: () => void
  stop: () => void
  next: () => void
  back: () => void
}

export const TourContext = createContext<TourState | null>(null)

export function useTour(): TourState {
  const ctx = useContext(TourContext)
  if (!ctx) throw new Error('useTour must be used inside <TourProvider>')
  return ctx
}
