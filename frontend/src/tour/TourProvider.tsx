import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, X } from 'lucide-react'
import { TOUR_STEPS } from './steps'
import type { TourStep } from './steps'
import { TourContext, useTour } from './context'

/** Holds the tour's position; must sit inside the router so steps can change pages. */
export function TourProvider({ children, steps = TOUR_STEPS }: { children: React.ReactNode; steps?: TourStep[] }) {
  // A link with ?tour=1 (e.g. shared from the site) opens straight into the tour.
  const location = useLocation()
  const [active, setActive] = useState(() => new URLSearchParams(location.search).get('tour') === '1')
  const [index, setIndex] = useState(0)
  const start = useCallback(() => { setIndex(0); setActive(true) }, [])
  const stop = useCallback(() => setActive(false), [])
  const next = useCallback(() => setIndex(i => {
    if (i + 1 >= steps.length) { setActive(false); return i }
    return i + 1
  }), [steps.length])
  const back = useCallback(() => setIndex(i => Math.max(0, i - 1)), [])

  return (
    <TourContext.Provider value={{ active, index, start, stop, next, back }}>
      {children}
      {active && <TourOverlay step={steps[index]} index={index} total={steps.length} />}
    </TourContext.Provider>
  )
}

const CARD_W = 360
const GAP = 14
const PAD = 8

function TourOverlay({ step, index, total }: { step: TourStep; index: number; total: number }) {
  const { next, back, stop } = useTour()
  const navigate = useNavigate()
  const location = useLocation()
  const [rect, setRect] = useState<DOMRect | null>(null)
  const scrolledFor = useRef<string | null>(null)

  useEffect(() => {
    if (location.pathname !== step.route) navigate(step.route)
  }, [step.route, location.pathname, navigate])

  // Follow the target element: it may render late (data, charts) and moves when the page scrolls.
  useEffect(() => {
    let raf = 0
    const key = `${index}:${step.target}`
    const tick = () => {
      const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)
      if (el && scrolledFor.current !== key) {
        scrolledFor.current = key
        el.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
      }
      const r = el?.getBoundingClientRect() ?? null
      setRect(prev => (prev && r && prev.top === r.top && prev.left === r.left && prev.width === r.width
        && prev.height === r.height) || (!prev && !r) ? prev : r)
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [index, step.target])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') stop() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [stop])

  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800
  const w = Math.min(CARD_W, vw - 32)
  let cardStyle: React.CSSProperties
  if (!rect || vw < 640) {
    cardStyle = { left: (vw - w) / 2, bottom: 16 }
  } else if (rect.height > vh * 0.6) {
    cardStyle = { right: 24, bottom: 24 } // a whole page: keep the card out of the way
  } else {
    const below = rect.bottom + GAP + 240 < vh
    const left = Math.min(Math.max(16, rect.left), vw - w - 16)
    cardStyle = below ? { left, top: rect.bottom + PAD + GAP } : { left, bottom: vh - rect.top + PAD + GAP }
  }
  const last = index === total - 1

  return (
    <div className="fixed inset-0 z-[100] pointer-events-none" aria-live="polite">
      {rect ? (
        <div className="fixed rounded-xl transition-all duration-200"
          style={{
            top: rect.top - PAD, left: rect.left - PAD, width: rect.width + 2 * PAD, height: rect.height + 2 * PAD,
            boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.55)', border: '1px solid rgba(255,255,255,0.35)',
          }} />
      ) : (
        <div className="fixed inset-0" style={{ background: 'rgba(0, 0, 0, 0.55)' }} />
      )}
      <div role="dialog" aria-label={`Tour step ${index + 1} of ${total}: ${step.title}`}
        className="fixed glass-card p-5 space-y-3 pointer-events-auto shadow-2xl"
        style={{ ...cardStyle, width: w, maxHeight: '70vh', overflowY: 'auto' }}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[9px] font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
              Tour · {index + 1} / {total}
            </div>
            <h3 className="text-sm font-bold mt-1" style={{ color: 'var(--text)' }}>{step.title}</h3>
          </div>
          <button type="button" onClick={stop} aria-label="Close tour" className="p-1 rounded"
            style={{ color: 'var(--text-muted)' }}>
            <X size={14} />
          </button>
        </div>
        <div className="text-xs leading-relaxed" style={{ color: 'var(--text-sub)' }}>{step.body}</div>
        <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--divider)' }}>
          <div className="h-full" style={{ width: `${((index + 1) / total) * 100}%`, background: 'var(--text)' }} />
        </div>
        <div className="flex items-center justify-between gap-2 pt-1">
          {step.learn ? (
            <Link to={step.learn} onClick={stop}
              className="text-[10px] font-mono font-semibold uppercase tracking-wider underline underline-offset-4"
              style={{ color: 'var(--text)' }}>
              Learn more
            </Link>
          ) : <span />}
          <div className="flex gap-2">
            {index > 0 && (
              <button type="button" onClick={back} className="btn-secondary" aria-label="Previous step">
                <ArrowLeft size={12} />
              </button>
            )}
            <button type="button" onClick={next} className="btn-primary">
              {last ? 'Finish' : 'Next'} {!last && <ArrowRight size={12} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
