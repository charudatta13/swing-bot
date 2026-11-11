# Swing trading research bot

This is a research and paper-trading tool, not investment advice.

## Daily use (after 4:30 pm ET)
```
python -m swingbot.daily              # refresh prices, scan, update paper trades, write report
python -m swingbot.daily --backtest   # also re-run the 5-year backtest (it re-runs on its own weekly)
python -m swingbot.daily --refresh    # also re-download earnings/fundamentals
python -m swingbot.daily --allow-stale  # build the report even if today's prices are missing
```
If Yahoo has not published the latest trading day yet, the run stops with `STALE DATA` and writes
nothing, so a report is never a day behind without saying so. GitHub Actions tries again at
8:30 pm ET and 7:00 am ET (before the next open).
Open `reports/latest.html`. A dated copy is saved for every day.

## What's where
| File | Purpose |
|---|---|
| `swingbot/config.py` | Account size, position cap (30%), risk per trade, market filter |
| `swingbot/rules.py` | The 4 swing rules and their exits |
| `swingbot/backtest.py` | Trade simulator, used by both the backtest and the paper ledger |
| `swingbot/daily.py` | Daily pipeline + rule status (Active / Reduced / Paused / Off) |
| `swingbot/quarterly.py` | Q2 vs Q3 2026 study and the computed "driver" text |
| `research/why_notes.json` | News explanations for the big movers (edit or extend freely) |
| `data/paper_signals.csv` | The paper ledger: every logged signal, re-simulated each day |
| `data/q2_q3_analysis.csv` | Full Q2 vs Q3 table for all ~500 stocks |

## How it "learns"
Each rule starts with its backtest result. Every day, the top setups are logged as paper trades and
tracked with real prices. Once a rule has 20+ closed paper trades, the bot reduces it to half size
(average return below half of the backtest) or pauses it (average return negative).
Rules are never re-tuned on a single day's result.

## Requirements
`pip install --user pandas numpy yfinance lxml requests`
