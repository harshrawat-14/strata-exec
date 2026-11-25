/**
 * The guided tour: one step per thing a visitor should see, in order. A step points at an
 * element marked `data-tour="<target>"` on `route`; the overlay navigates there first.
 */
import { REGIME_INFO } from '../components/pinned/format'

export interface TourStep {
  route: string
  target: string
  title: string
  body: React.ReactNode
  learn?: string
}

const regimeList = (
  <ul className="space-y-1.5 mt-2">
    {(['all', 'crash', 'normal', 'calm'] as const).map(r => (
      <li key={r}>
        <span className="font-semibold" style={{ color: 'var(--text)' }}>{REGIME_INFO[r].title}: </span>
        {REGIME_INFO[r].text}
      </li>
    ))}
  </ul>
)

export const TOUR_STEPS: TourStep[] = [
  {
    route: '/results', target: 'results-headline', title: 'The headline',
    body: 'A PPO-LSTM agent sold a large BTC order on 25 real trading days against four classical schedules. It beat Almgren–Chriss on every crash day, including 7 it never saw in training, and lost on calm and normal days.',
    learn: '/learn#results',
  },
  {
    route: '/results', target: 'regime-filter', title: 'Market regimes',
    body: <>Filter everything below by the kind of day.{regimeList}</>,
    learn: '/learn#regimes',
  },
  {
    route: '/results', target: 'edge-chart', title: 'Day by day',
    body: 'Each bar is the agent’s implementation shortfall minus AC Optimal’s, in percentage points. Above zero, the agent sold at a better average price. Faded bars are the 7 days it trained on.',
    learn: '/learn#evaluation',
  },
  {
    route: '/results', target: 'strategy-chart', title: 'Five strategies, same conditions',
    body: 'TWAP, a volatility-aware heuristic, Almgren–Chriss Optimal, Adaptive AC and the RL agent all trade the same 50 episodes per day in the same simulator.',
    learn: '/learn#strategies',
  },
  {
    route: '/results', target: 'severity-chart', title: 'Harder days, bigger edge',
    body: 'The worse a day was for AC Optimal, the more the agent gained. Its advantage comes from selling earlier when prices are falling.',
    learn: '/learn#results',
  },
  {
    route: '/results', target: 'ablation-chart', title: 'Does more state help?',
    body: 'Two otherwise identical agents: one sees 12 features, the other 16 (adding time of day and intraday volatility).',
    learn: '/learn#agent',
  },
  {
    route: '/results', target: 'day-table', title: 'Every number',
    body: 'All 25 days and 5 strategies. Sort by any column, and click a day to see the spread of its 50 episodes and the worst-case (CVaR) cost.',
    learn: '/learn#evaluation',
  },
  {
    route: '/simulator', target: 'simulator', title: 'Strategy simulator',
    body: 'The Rust research simulator runs Monte Carlo price paths (GBM, GARCH, jumps) and compares execution strategies on cost, risk and trajectory. While the engine is down, the presets load precomputed runs.',
    learn: '/learn#simulator',
  },
  {
    route: '/evaluate', target: 'rl-eval', title: 'RL evaluation',
    body: 'Pick any of the 25 days to see how the agent and each baseline did on it, episode by episode.',
    learn: '/learn#agent',
  },
  {
    route: '/sweep', target: 'sweep', title: 'Parameter sweeps',
    body: 'How each strategy’s cost moves with volatility, horizon, impact and slicing. 500 Monte Carlo paths per point.',
    learn: '/learn#strategies',
  },
  {
    route: '/upload', target: 'upload', title: 'Bring your own data',
    body: 'With the engine running you can upload order-book data or a trained model and evaluate it in the same pipeline.',
    learn: '/learn#data',
  },
  {
    route: '/learn', target: 'learn-top', title: 'Learn it in depth',
    body: 'The problem, the market data, the simulator, the agent, how it was evaluated and what the results do and do not show, with links to the source documents.',
  },
]
