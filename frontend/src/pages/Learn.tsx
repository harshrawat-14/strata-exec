/**
 * Learn — how StrataExec works, in depth, for anyone curious. Static; no backend.
 * Everything described here is the pinned PPO-LSTM era; source links point at the pinned tag.
 */
import { useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BookOpen, ExternalLink } from 'lucide-react'
import { PINNED, edgeVsAc, summariseByRegime } from '../data/pinned/results'
import { pp } from '../components/pinned/format'

const REPO = 'https://github.com/harshrawat-14/strata-exec'
const src = (path: string) => `${REPO}/blob/${PINNED.model.pinned_tag}/${path}`

const SECTIONS = [
  { id: 'overview', title: 'What StrataExec is' },
  { id: 'problem', title: 'The execution problem' },
  { id: 'strategies', title: 'The five strategies' },
  { id: 'data', title: 'Market data' },
  { id: 'simulator', title: 'The simulators' },
  { id: 'agent', title: 'The RL agent' },
  { id: 'regimes', title: 'Market regimes' },
  { id: 'evaluation', title: 'How it was evaluated' },
  { id: 'results', title: 'What we found' },
  { id: 'limits', title: 'Limits and caveats' },
  { id: 'engineering', title: 'Engineering' },
  { id: 'glossary', title: 'Glossary' },
  { id: 'reading', title: 'Further reading' },
]

const STATE_DIMS: [string, string, string][] = [
  ['0', 'Inventory left', 'Share of the order still to sell'],
  ['1', 'Time left', 'Share of the time budget remaining'],
  ['2', 'Volatility', 'Current volatility relative to a reference'],
  ['3', 'Spread', 'Current half-spread relative to a reference'],
  ['4', 'Depth', 'Liquidity within 1% of mid relative to a reference'],
  ['5', 'Order-flow imbalance', 'Buy vs sell pressure over the last 30 s of trades'],
  ['6', 'Drift', 'Recent price move over the last 10 steps'],
  ['7', 'Slippage', 'How adverse the last fill was'],
  ['8', 'Permanent impact', 'Our accumulated lasting price impact'],
  ['9', 'Temporary impact', 'Our impact that has not yet decayed'],
  ['10', 'Volatility regime', 'Low / normal / high'],
  ['11', 'Liquidity regime', 'Thin / normal / deep'],
  ['12', 'Time of day', 'UTC fraction of the day (16-dim agent only)'],
  ['13', 'Intraday volatility', 'Intraday vs daily volatility (16-dim only)'],
  ['14–15', 'sin / cos of time', 'Cyclical time encoding (16-dim only)'],
]

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h2 id={id} className="text-xl font-bold tracking-tight scroll-mt-24 pt-2" style={{ color: 'var(--text)' }}>
      <a href={`#${id}`} className="hover:underline underline-offset-4" style={{ color: 'var(--text)' }}>{children}</a>
    </h2>
  )
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-sm leading-7" style={{ color: 'var(--text-sub)' }}>{children}</p>
}

function Source({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer"
      className="inline-flex items-center gap-1 text-[11px] font-mono underline underline-offset-4"
      style={{ color: 'var(--text)' }}>
      {children} <ExternalLink size={10} />
    </a>
  )
}

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <pre className="text-xs font-mono p-4 rounded-lg overflow-x-auto leading-6"
      style={{ background: 'var(--input-bg)', border: '1px solid var(--divider)', color: 'var(--text)' }}>
      {children}
    </pre>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="glass-card p-4 space-y-1.5">
      <div className="text-xs font-semibold" style={{ color: 'var(--text)' }}>{title}</div>
      <div className="text-xs leading-6" style={{ color: 'var(--text-sub)' }}>{children}</div>
    </div>
  )
}

export default function Learn() {
  const { hash } = useLocation()
  useEffect(() => {
    if (!hash) return
    document.getElementById(hash.slice(1))?.scrollIntoView?.({ behavior: 'smooth', block: 'start' })
  }, [hash])

  const summary = summariseByRegime(PINNED.dates)
  const crash = summary.find(s => s.regime === 'crash')!
  const normal = summary.find(s => s.regime === 'normal')!
  const calm = summary.find(s => s.regime === 'calm')!
  const trainDays = PINNED.dates.filter(d => d.split === 'train')
  const unseenCrash = PINNED.dates.filter(d => d.regime === 'crash' && d.split === 'held_out')

  return (
    <div className="flex flex-wrap gap-8 animate-fade-in">
      {/* Contents */}
      <nav aria-label="Contents" className="flex-1 basis-56 min-w-0 lg:max-w-[220px]">
        <div className="glass-card p-4 lg:sticky lg:top-6">
          <div className="label-text mb-3">Contents</div>
          <ol className="space-y-1.5 text-xs">
            {SECTIONS.map((s, i) => (
              <li key={s.id}>
                <a href={`#${s.id}`} className="hover:underline underline-offset-4" style={{ color: 'var(--text-sub)' }}>
                  <span className="font-mono mr-2" style={{ color: 'var(--text-muted)' }}>{String(i + 1).padStart(2, '0')}</span>
                  {s.title}
                </a>
              </li>
            ))}
          </ol>
        </div>
      </nav>

      <article className="flex-[999_1_560px] min-w-0 max-w-3xl space-y-6" data-tour="learn-top">
        <header className="space-y-2">
          <div className="flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
            <BookOpen size={14} />
            <span className="text-[10px] font-mono uppercase tracking-widest">Learn</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
            How StrataExec <span className="text-gradient-electric">works</span>
          </h1>
          <P>
            A guide for anyone curious about the project: the trading problem it studies, the market data and
            simulators, the reinforcement-learning agent, how it was tested and what the results do and do
            not show. Each section links to the source document or code that covers it in full.
          </P>
        </header>

        <H2 id="overview">What StrataExec is</H2>
        <P>
          StrataExec is a research platform for <b>optimal execution</b>: how to sell a very large position
          without moving the market against yourself. It has a Rust core with two simulators, a Python
          reinforcement-learning stack that trains against the Rust engine, and this web app for running
          and exploring experiments. The question it asks: can a learned policy beat the textbook
          Almgren–Chriss solution on real market data, and when?
        </P>
        <Source href={src('README.md')}>Project README</Source>

        <H2 id="problem">The execution problem</H2>
        <P>
          An institution wants to sell a large amount of BTC. Selling it all at once walks down the order
          book and fills at worse and worse prices: that is <b>market impact</b>. Selling slowly avoids most
          of the impact, but the price can drift away while you wait: that is <b>timing risk</b>. Every
          execution strategy is a trade-off between the two.
        </P>
        <P>
          The yardstick is <b>implementation shortfall (IS)</b>: how the average price you actually got
          compares with the price when you started (the arrival price).
        </P>
        <Formula>{`IS (%) = (average fill price − arrival price) / arrival price × 100

On this site, higher is better: −0.5% means you sold 0.5% below the arrival price.
"RL − AC" is the difference in IS between two strategies, in percentage points (pp).`}</Formula>

        <H2 id="strategies">The five strategies</H2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Card title="TWAP">Sells an equal slice every step until the deadline. The simplest benchmark; it ignores the market completely.</Card>
          <Card title="Heuristic">TWAP adjusted by conditions: sells faster when volatility is high or the book is deep, slower when it is calm or thin.</Card>
          <Card title="AC Optimal (Almgren–Chriss)">The closed-form optimal schedule that balances expected impact against price risk, x(t) = X·sinh(κ(T−t)) / sinh(κT). Computed once at the start.</Card>
          <Card title="Adaptive AC">Re-solves the Almgren–Chriss schedule every step using the current volatility estimate.</Card>
          <Card title="RL agent (PPO-LSTM)">A neural policy with memory that chooses how much to sell each step from 16 market and inventory features. Learned, not derived.</Card>
        </div>
        <P>
          All five trade through the same simulator, on the same days and the same episode starts, and
          choose from the same set of sell sizes, so differences come from decisions, not plumbing. The AC
          risk aversion was fixed in advance (λ = 10⁻⁶) instead of tuned per day; tuning it on the day being
          scored would let AC peek at the answer.
        </P>
        <Source href={src('docs/phase2r_ac_calibration_design_note.md')}>AC calibration note</Source>

        <H2 id="data">Market data</H2>
        <P>
          Binance BTC-USDT perpetual futures, 2024, from Binance’s public archive. Two feeds are used:
          <b> order-book depth</b>, the cumulative quantity resting within ±1% to ±5% of the mid price about every
          30 seconds, and <b>every individual trade</b> (1–6 million a day), used for volume, order-flow
          imbalance and calibrating impact. The {PINNED.dates.length} evaluated days cover three stress
          episodes (the April 2024 Iran–Israel escalation, the July 2024 German-government BTC sales and Mt. Gox
          fears, and the August 2024 US jobs report and yen-carry unwind) plus isolated days across the year.
        </P>

        <H2 id="simulator">The simulators</H2>
        <P>
          <b>Research simulator.</b> Monte Carlo price paths (geometric Brownian motion, GARCH volatility
          clustering, and jumps) with square-root market impact and decaying transient impact. It compares
          execution strategies over hundreds of paths and powers the Simulator and Parameter Sweep pages.
        </P>
        <P>
          <b>Counterfactual order book.</b> Replaying history cannot tell you what would have happened if you
          had traded, because the recorded book never saw your orders. The counterfactual book replays the
          real Binance book and then removes the liquidity your own sales would have consumed: about 30% of
          each sale’s impact is permanent and about 70% is temporary and decays over time, with a square-root
          size law. Selling aggressively now makes later fills worse, which is what stops a learning agent
          from simply dumping everything at once.
        </P>
        <Source href={src('docs/system_design/LLD.md')}>Low-level design: book and fill algorithm</Source>

        <H2 id="agent">The RL agent</H2>
        <P>
          <b>Algorithm.</b> Recurrent PPO (sb3-contrib) with an LSTM policy: separate 64-unit LSTMs for the
          actor and the critic. The memory lets it react to how the day has unfolded, not just the current
          snapshot. It talks to the Rust environment over newline-delimited JSON.
        </P>
        <P><b>What it sees.</b> 16 numbers each step, scaled to similar ranges:</P>
        <div className="overflow-x-auto">
          <table className="w-full text-xs min-w-[480px]">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--divider)', color: 'var(--text-muted)' }}>
                <th className="text-left py-2 pr-3 font-mono font-normal">#</th>
                <th className="text-left py-2 pr-3 font-mono font-normal">Feature</th>
                <th className="text-left py-2 font-mono font-normal">Meaning</th>
              </tr>
            </thead>
            <tbody>
              {STATE_DIMS.map(([i, name, meaning]) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--divider)' }}>
                  <td className="py-1.5 pr-3 font-mono" style={{ color: 'var(--text-muted)' }}>{i}</td>
                  <td className="py-1.5 pr-3 font-semibold" style={{ color: 'var(--text)' }}>{name}</td>
                  <td className="py-1.5" style={{ color: 'var(--text-sub)' }}>{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <P>
          <b>What it does.</b> Each 30-second step it picks one of 20 actions: sell 0% (wait), 0.5%, 1%, …,
          50%, 75% or 100% of the inventory that remains. Anything unsold at the deadline is force-liquidated.
        </P>
        <P><b>What it is rewarded for.</b> Each fill is scored against the prevailing mid price, weighted by size:</P>
        <Formula>{`reward per fill = (fill price − mid price) × fill quantity / (arrival price × total order)

summed over an episode this is the execution cost; waiting carries a small inventory-risk penalty`}</Formula>
        <P>
          The 12-dim and 16-dim agents are identical except for features 12–15; comparing them shows what
          time-of-day and intraday-volatility information adds.
        </P>
        <Source href={src('rl/environment.py')}>Gym environment</Source>{' '}
        <Source href={src('src/bin/rl_env.rs')}>Rust environment</Source>{' '}
        <Source href={src('docs/phase2r_objective_bias_finding.md')}>What the reward does and does not reward</Source>

        <H2 id="regimes">Market regimes</H2>
        <P>
          Days are sorted by how costly they were for the AC Optimal schedule, which is independent of the RL
          agent, and split into thirds:
        </P>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Card title={`Crash · ${crash.days} days`}>AC lost 1.09% or more. Sharp sell-offs: prices fall while you sell, so selling earlier pays.</Card>
          <Card title={`Normal · ${normal.days} days`}>AC lost between 0.51% and 0.89%. Moderate moves with no clear direction.</Card>
          <Card title={`Calm · ${calm.days} days`}>AC lost 0.48% or less, or gained. Flat or rising prices: waiting costs little or pays.</Card>
        </div>

        <H2 id="evaluation">How it was evaluated</H2>
        <P>
          Every strategy ran <b>{PINNED.protocol.episodes_per_strategy} episodes per day</b> with the same seeds,
          so each comparison is paired: same day, same starts, different decisions. Results are reported per
          day with the spread of episodes and the worst-case 5% (CVaR 95), not just the mean.
        </P>
        <P>
          The agent trained on <b>{trainDays.length} days</b> ({trainDays.map(d => d.date).join(', ')}). The
          other {PINNED.dates.length - trainDays.length} days were never used in training, including{' '}
          {unseenCrash.length} of the {crash.days} crash days. On the Results page, training days are marked
          and shown faded.
        </P>
        <Source href={src('docs/phase3_tail_validation.md')}>Full 25-day evaluation</Source>

        <H2 id="results">What we found</H2>
        <ul className="space-y-2 text-sm leading-7 list-disc pl-5" style={{ color: 'var(--text-sub)' }}>
          <li>
            <b>Crash days: {crash.wins} of {crash.days} won</b> against AC Optimal, by {pp(crash.meanEdge)} pp on
            average and up to {pp(crash.bestEdge)} pp (2024-08-05). {unseenCrash.filter(d => edgeVsAc(d) > 0).length} of
            the {unseenCrash.length} crash days it never trained on are among the wins.
          </li>
          <li><b>Normal days: {normal.wins} of {normal.days}</b> and <b>calm days: {calm.wins} of {calm.days}</b>. The classical schedule is better when the market is not falling.</li>
          <li><b>The edge grows with the severity of the day.</b> The agent’s advantage comes from selling earlier when prices are dropping, not from cheaper execution.</li>
          <li><b>Its worst-case (CVaR) advantage is the same effect.</b> Once the mean difference is removed, its tail is no better than AC’s.</li>
        </ul>
        <P>
          <Link to="/results" className="underline underline-offset-4" style={{ color: 'var(--text)' }}>Explore every number on the Results page</Link>.
        </P>

        <H2 id="limits">Limits and caveats</H2>
        <ul className="space-y-2 text-sm leading-7 list-disc pl-5" style={{ color: 'var(--text-sub)' }}>
          <li><b>A specialist, not a general improvement.</b> It wins crashes and loses calm days; no single strategy here wins everywhere.</li>
          <li><b>Fewer independent events than days.</b> Crash days cluster in a few weeks of stress, so 8 crash days are closer to 4 independent events.</li>
          <li><b>Depth bands, not a full order book.</b> The data records cumulative depth within ±1–5% of mid, so absolute costs are higher than on a real full-depth book. All strategies share this simulator, so comparisons are like for like.</li>
          <li><b>Limited training variety.</b> A later audit found every training episode on a given day started from the same point, so the agent saw less variety than intended.</li>
          <li><b>One market, one year.</b> BTC perpetual futures in 2024. Other assets and periods are untested here.</li>
        </ul>
        <Source href={src('docs/phase2r_severity_expansion.md')}>Severity expansion and counting rules</Source>

        <H2 id="engineering">Engineering</H2>
        <ul className="space-y-2 text-sm leading-7 list-disc pl-5" style={{ color: 'var(--text-sub)' }}>
          <li><b>Rust core</b> with a discrete-event simulator, pluggable strategies behind a common trait, and a risk manager.</li>
          <li><b>Python ↔ Rust over JSON lines</b>, so the two toolchains never share memory or an ABI.</li>
          <li><b>Web stack:</b> React + TypeScript front end; FastAPI back end with Redis job queues and live progress over WebSocket.</li>
          <li><b>Tests gate every push</b> in CI, for Rust, Python and the front end.</li>
        </ul>
        <div className="flex flex-wrap gap-4">
          <Source href={src('docs/system_design/HLD.md')}>High-level design</Source>
          <Source href={src('docs/system_design/ADR.md')}>Architecture decisions</Source>
        </div>

        <H2 id="glossary">Glossary</H2>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-xs leading-6">
          {[
            ['Arrival price', 'The mid price when the order starts. The reference for IS.'],
            ['Implementation shortfall (IS)', 'Average fill price vs arrival price, in % of the order’s value.'],
            ['pp', 'Percentage points: the difference between two percentages.'],
            ['Market impact', 'How much your own trading moves the price against you.'],
            ['TWAP', 'Time-weighted average price: equal slices over time.'],
            ['Almgren–Chriss', 'The classic model of optimal execution (2001), trading off impact against risk.'],
            ['PPO', 'Proximal Policy Optimization, a stable policy-gradient RL algorithm.'],
            ['LSTM', 'A recurrent network layer that carries memory across steps.'],
            ['CVaR 95', 'Mean of the worst 5% of outcomes: the tail risk.'],
            ['Episode', 'One complete sale of the order, from start to deadline.'],
          ].map(([t, d]) => (
            <div key={t}>
              <dt className="font-semibold" style={{ color: 'var(--text)' }}>{t}</dt>
              <dd style={{ color: 'var(--text-sub)' }}>{d}</dd>
            </div>
          ))}
        </dl>

        <H2 id="reading">Further reading</H2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            ['README.md', 'Project overview and engineering decisions'],
            ['docs/system_design/HLD.md', 'High-level design'],
            ['docs/system_design/LLD.md', 'Low-level design: book, fills, state, wire protocol'],
            ['docs/system_design/ADR.md', 'Architecture decision records'],
            ['docs/phase3_tail_validation.md', 'The 25-day, 5-strategy evaluation'],
            ['docs/phase2r_severity_expansion.md', 'Stress-window days and counting rules'],
            ['docs/phase2r_objective_bias_finding.md', 'What the reward rewards'],
            ['rl/evaluate.py', 'Baseline strategies and evaluation harness'],
          ].map(([path, label]) => (
            <a key={path} href={src(path)} target="_blank" rel="noreferrer" className="glass-card-hover p-4 block">
              <div className="text-xs font-semibold flex items-center gap-1.5" style={{ color: 'var(--text)' }}>
                {label} <ExternalLink size={10} />
              </div>
              <div className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>{path}</div>
            </a>
          ))}
        </div>
      </article>
    </div>
  )
}
