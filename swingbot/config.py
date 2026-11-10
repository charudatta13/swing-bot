"""Settings you can change. Everything else reads from here."""

# --- Account / sizing (used only to size suggestions in the report) ---
ACCOUNT_SIZE = 10_000          # dollars
MAX_POSITION_PCT = 30          # never put more than this % of the account in one stock
RISK_PER_TRADE_PCT = 2         # % of account lost if the stop-loss is hit
MAX_OPEN_POSITIONS = 4         # ~100 / MAX_POSITION_PCT, rounded up

# --- Universe / data ---
HISTORY_START = "2020-06-01"   # warm-up for 200/252-day indicators
BACKTEST_START = "2021-06-01"  # ~5 years of tested trades
COST_ROUND_TRIP = 0.001        # 0.10% slippage + fees per trade

# --- Quarter windows for the Q2 vs Q3 study (calendar quarters, 2026) ---
Q_ENDS = {"Q1": "2026-03-31", "Q2": "2026-06-30", "Q3": "2026-09-30"}

# --- Paper trading ---
MAX_NEW_PAPER_TRADES_PER_DAY = 5
USE_MARKET_FILTER = True       # only go long when SPY is above its 200-day average
