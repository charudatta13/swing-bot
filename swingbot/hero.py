"""Hero watchlist (paper only): volatile market leaders that may double within a year.

Based on research/hero/README.md ("Leader" screen, 20% trailing exit):
  - screen: among the most volatile 20% of S&P 500 stocks (60-day volatility), those up at least
    30% over 6 months and within 10% of their 52-week high; ranked by 6-month gain
  - once a month (first run with data in a new month), free slots in a 4-position sleeve are filled
    with the top-ranked candidates not already held
  - exits: -25% initial stop, then a trailing exit 20% below the best close since entry
    (once that is above the initial stop), or after ~12 months
Most picks are expected to lose; the case for it rests on a few large winners.
"""
import json

import numpy as np
import pandas as pd

from . import config
from . import data as D

LEDGER = D.DATA / "hero_signals.csv"
STATE = D.DATA / "hero_state.json"

VOL_TOP = 0.20          # most volatile 20%
MIN_MOM6 = 0.30         # up at least 30% over 6 months
MAX_BELOW_HIGH = 0.10   # within 10% of the 52-week high
INITIAL_STOP = 0.25
TRAIL = 0.20
MAX_DAYS = 252
SLOTS = 4


def screen(C: pd.DataFrame, H: pd.DataFrame, i: int, tickers: list[str]) -> pd.DataFrame:
    """Candidates on bar i, best first."""
    cols = [t for t in tickers if t in C.columns and t != "SPY"]
    c, h = C[cols], H[cols]
    vol = c.iloc[i - 60:i + 1].pct_change().std() * np.sqrt(252)
    x = pd.DataFrame({
        "vol60": vol,
        "mom6": c.iloc[i] / c.iloc[i - 126] - 1,
        "hi_dist": c.iloc[i] / h.iloc[i - 251:i + 1].max() - 1,
        "close": c.iloc[i],
    }).dropna()
    x["vol_rank"] = x.vol60.rank(pct=True)
    x = x[(x.vol_rank >= 1 - VOL_TOP) & (x.mom6 >= MIN_MOM6) & (x.hi_dist >= -MAX_BELOW_HIGH)]
    return x.sort_values("mom6", ascending=False)


def simulate(C, O, H, L, t, d, trail=TRAIL):
    """Buy at the next open after signal date d; exit at the first of: low <= initial stop (at the
    stop, or the open if it gapped below), close <= best close * (1 - trail) once that level is above
    the initial stop (at the next open), or MAX_DAYS (at the close). None if there is no next bar."""
    j = C.columns.get_loc(t)
    i0 = C.index.get_loc(d) + 1
    if i0 >= len(C) or np.isnan(O.iat[i0, j]):
        return None
    entry = O.iat[i0, j]
    stop = entry * (1 - INITIAL_STOP)
    peak = entry
    last = min(i0 + MAX_DAYS - 1, len(C) - 1)
    base = dict(entry=entry, entry_i=i0)
    for i in range(i0, last + 1):
        o, lo, c = O.iat[i, j], L.iat[i, j], C.iat[i, j]
        if np.isnan(c):
            continue
        if i > i0 and o <= stop:
            return dict(base, exit_i=i, exit=o, why="stop (gap)", peak=peak)
        if lo <= stop:
            return dict(base, exit_i=i, exit=stop, why="stop", peak=peak)
        peak = max(peak, c)
        trail_px = peak * (1 - trail)
        if trail_px > stop and c <= trail_px:
            k = min(i + 1, len(C) - 1)
            px = O.iat[k, j] if not np.isnan(O.iat[k, j]) else c
            return dict(base, exit_i=k, exit=px, why="trailing exit", peak=peak)
    still_open = i0 + MAX_DAYS - 1 > len(C) - 1
    return dict(base, exit_i=last, exit=C.iat[last, j], why="open" if still_open else "12 months", peak=peak,
                exit_level=max(stop, peak * (1 - trail)))


def _simulate_ledger(C, O, H, L, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in ledger.itertuples():
        d = pd.Timestamp(r.signal_date)
        if d not in C.index or r.ticker not in C.columns:
            continue
        s = simulate(C, O, H, L, r.ticker, d)
        if s is None:
            rows.append(dict(ticker=r.ticker, signal_date=d, why="pending entry"))
            continue
        s.update(ticker=r.ticker, signal_date=d, entry_date=C.index[s["entry_i"]],
                 exit_date=C.index[s["exit_i"]], last=C[r.ticker].iloc[s["exit_i"]],
                 ret=s["exit"] / s["entry"] - 1 - (config.COST_ROUND_TRIP / 2 if s["why"] == "open"
                                                   else config.COST_ROUND_TRIP),
                 days=s["exit_i"] - s["entry_i"] + 1)
        rows.append(s)
    return pd.DataFrame(rows)


def run(ind: dict, tickers: list[str]) -> dict:
    """Update the hero paper ledger and return what the report needs."""
    C, O, H, L = ind["C"], ind["O"], ind["H"], ind["L"]
    i = len(C) - 1
    asof = C.index[i]
    cands = screen(C, H, i, tickers)
    ledger = pd.read_csv(LEDGER, parse_dates=["signal_date"]) if LEDGER.exists() else \
        pd.DataFrame(columns=["signal_date", "ticker", "score"])
    trades = _simulate_ledger(C, O, H, L, ledger)
    month = asof.strftime("%Y-%m")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("last_month") != month:  # first run with data in a new month
        held = set(trades.ticker[trades.why.isin(["open", "pending entry"])]) if len(trades) else set()
        free = SLOTS - len(held)
        picks = cands[~cands.index.isin(held)].head(max(free, 0))
        if len(picks):
            new = pd.DataFrame({"signal_date": asof, "ticker": picks.index, "score": picks.mom6.values})
            ledger = pd.concat([ledger, new], ignore_index=True)
            ledger.to_csv(LEDGER, index=False, date_format="%Y-%m-%d")
            trades = _simulate_ledger(C, O, H, L, ledger)
        STATE.write_text(json.dumps({"last_month": month}) + "\n")
    held = set(trades.ticker[trades.why.isin(["open", "pending entry"])]) if len(trades) else set()
    cands = cands.assign(held=cands.index.isin(held))
    return dict(cands=cands, trades=trades, month=month)
