# Project history — mistakes, corrections and decisions

A record of what went wrong in StrataExec, how it was found, what it cost, and
what changed because of it. It exists so the same class of mistake is caught
faster the next time, and so any published number can be traced back to the
state of the code that produced it.

| File | What it holds |
|---|---|
| [timeline.md](timeline.md) | Every entry in date order |
| [lessons.md](lessons.md) | The recurring patterns behind the entries, with the checks that now guard against each |
| [era-1-binance-bookdepth.md](era-1-binance-bookdepth.md) | Era 1 (May–Oct 2026): the 25 dates, their regimes and significance, best results, the dead end, and why the simulator's data source changed |

## Entry types

| Tag | Meaning |
|---|---|
| **BUG** | Code computed something other than what was intended |
| **METHOD** | The code was right but the evaluation design was not (leakage, small samples, wrong quantity measured) |
| **CLAIM** | A stated result that was later retracted or narrowed |
| **PROCESS** | A workflow mistake (wrong command, wrong baseline, scope slip) |
| **DECISION** | A deliberate choice that shaped later work, recorded with its reason |

## How to add an entry

Append to `timeline.md` under the right date, newest last:

```
### YYYY-MM-DD · TAG · short title
- **What:** what was wrong, in one or two sentences.
- **Found by:** how it surfaced.
- **Impact:** which results or behaviour it touched (or "none").
- **Fix / status:** what changed, or OPEN.
- **Refs:** commit hashes, docs.
```

If the entry is a new instance of a pattern in `lessons.md`, add it to that
pattern's instance list too. Write entries plainly: say what the mistake was,
including when the mistake was ours.

Commit hashes refer to the history as pushed to `origin/main` on 2026-10-18.
