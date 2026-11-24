"""Hero-trade study: which S&P 500 stocks went on to double within 12 months, what was known
beforehand, and how much of the move a simple trading rule would have kept.

    python -m research.hero.hero_study            # uses data/prices.pkl if under 10 hours old
    python -m research.hero.hero_study --refresh  # download prices again

Phase 2 (signals): on the first trading day of each month from Jun 2021, every stock that was in
the S&P 500 on that date is described by what was known that day (volatility, momentum, distance
from the 52-week high, recent gaps). We then check if it touched +100% ("hero") or -50% ("bust")
within the next 252 trading days.

Phase 3 (trading): screens built from Phase 2 are traded with a stop-loss and a trailing exit:
  - every candidate as a separate trade (trade-level results),
  - a 4-slot "hero sleeve" filled monthly (portfolio results),
  - one pick per year (the "once a year hero trade").

KNOWN BIAS: prices come from Yahoo for today's S&P 500 members only. Stocks only count after
their join date (sp500_members.csv, from Wikipedia), which removes stocks that rose INTO the index.
Stocks REMOVED from the index (often after falling) are still missing, so results are optimistic,
most of all for screens that buy beaten-down stocks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from swingbot import config
from swingbot import data as D

HERE = Path(__file__).resolve().parent
START = pd.Timestamp("2021-06-01")
SPLIT = pd.Timestamp("2023-09-01")          # early / late halves of the signal dates
COST = config.COST_ROUND_TRIP
INITIAL_STOP = 0.25                         # exit if the price falls 25% below entry
MAX_DAYS = 252                              # hold at most ~12 months
SLOTS = 4                                   # hero sleeve: 4 positions of equal size


# ---------------------------------------------------------------- data
def load():
    mem = pd.read_csv(HERE / "sp500_members.csv", parse_dates=["added"]).set_index("ticker")
    px = D.prices(mem.index.tolist(), refresh="--refresh" in sys.argv)
    C, O, H, L = (px[k] for k in ["Close", "Open", "High", "Low"])
    tick = [t for t in C.columns if t in mem.index]
    return mem, C[tick], O[tick], H[tick], L[tick], px["Close"]["SPY"]


def features(C, O, H):
    r = C.pct_change()
    return {
        "vol60": r.rolling(60).std() * np.sqrt(252),                 # annualised volatility
        "mom6": C / C.shift(126) - 1,
        "mom12": C / C.shift(252) - 1,
        "hi_dist": C / H.rolling(252).max() - 1,                     # 0 = at 52-week high
        "gap63": (O / C.shift(1) - 1).rolling(63).max(),             # biggest gap up, last 3 months
    }


def month_starts(idx, last_ok):
    days = [idx[idx.searchsorted(d)] for d in pd.date_range(START, idx[-1], freq="MS")]
    return [d for d in days if d <= last_ok]


def snapshot(mem, C, H, L, F):
    """One row per (date, stock in the index): features + what happened in the next 12 months."""
    fwd_hi = H[::-1].rolling(MAX_DAYS, min_periods=200).max()[::-1].shift(-1)
    fwd_lo = L[::-1].rolling(MAX_DAYS, min_periods=200).min()[::-1].shift(-1)
    fwd_c = C.shift(-MAX_DAYS)
    rows = []
    for d in month_starts(C.index, C.index[-1]):
        live = [t for t in C.columns if mem.at[t, "added"] <= d]
        x = pd.DataFrame({k: v.loc[d, live] for k, v in F.items()})
        p = C.loc[d, live]
        x["up"] = fwd_hi.loc[d, live] / p - 1
        x["down"] = fwd_lo.loc[d, live] / p - 1
        x["r12"] = fwd_c.loc[d, live] / p - 1
        x["date"], x["ticker"] = d, live
        rows.append(x)
    X = pd.concat(rows, ignore_index=True).dropna(subset=["vol60", "mom12", "gap63"])
    X["vol_rank"] = X.groupby("date").vol60.rank(pct=True)
    X["half"] = np.where(X.date < SPLIT, "early", "late")
    return X


# ---------------------------------------------------------------- screens
SCREENS = {
    # name: (filter on the snapshot, ranking column, rank descending?)
    "volatile": (lambda x: x.vol_rank >= 0.8, "vol60", True),
    "leader": (lambda x: (x.vol_rank >= 0.8) & (x.mom6 >= 0.30) & (x.hi_dist >= -0.10), "mom6", True),
    "catalyst": (lambda x: (x.vol_rank >= 0.8) & (x.gap63 >= 0.12), "gap63", True),
    "rebound*": (lambda x: (x.vol_rank >= 0.8) & (x.hi_dist <= -0.40), "hi_dist", False),
}


def candidates(X, name):
    f, col, desc = SCREENS[name]
    c = X[f(X)].copy()
    c["rank"] = c.groupby("date")[col].rank(ascending=not desc, method="first")
    return c.sort_values(["date", "rank"])


# ---------------------------------------------------------------- trade simulation
def simulate(C, O, H, L, t, d, trail):
    """Buy at the next open after signal date d. Exit (whichever comes first):
    low <= initial stop (at the stop, or the open if it gapped below), close <= best close * (1 - trail)
    (at the next open), or MAX_DAYS (at the close). Returns None if there is no next day."""
    j = C.columns.get_loc(t)
    i0 = C.index.get_loc(d) + 1
    if i0 >= len(C) or np.isnan(O.iat[i0, j]):
        return None
    entry = O.iat[i0, j]
    stop = entry * (1 - INITIAL_STOP)
    peak = entry
    last = min(i0 + MAX_DAYS - 1, len(C) - 1)
    for i in range(i0, last + 1):
        o, lo, c = O.iat[i, j], L.iat[i, j], C.iat[i, j]
        if np.isnan(c):
            continue
        if i > i0 and o <= stop:
            return dict(exit_i=i, exit=o, why="stop (gap)", entry=entry, entry_i=i0)
        if lo <= stop:
            return dict(exit_i=i, exit=stop, why="stop", entry=entry, entry_i=i0)
        peak = max(peak, c)
        trail_px = peak * (1 - trail)
        if trail_px > stop and c <= trail_px:  # trailing exit only once it is above the initial stop
            k = min(i + 1, len(C) - 1)
            px = O.iat[k, j] if not np.isnan(O.iat[k, j]) else c
            return dict(exit_i=k, exit=px, why="trailing", entry=entry, entry_i=i0)
    open_ = last == len(C) - 1 and i0 + MAX_DAYS - 1 > len(C) - 1
    return dict(exit_i=last, exit=C.iat[last, j], why="open" if open_ else "12 months", entry=entry, entry_i=i0)


def trade_level(C, O, H, L, cand, trail):
    rows = []
    for r in cand.itertuples():
        s = simulate(C, O, H, L, r.ticker, r.date, trail)
        if s:
            s.update(date=r.date, ticker=r.ticker, rank=r.rank, half=r.half,
                     ret=s["exit"] / s["entry"] - 1 - COST, days=s["exit_i"] - s["entry_i"] + 1)
            rows.append(s)
    return pd.DataFrame(rows)


def sleeve(C, trades, slots=SLOTS):
    """Hero sleeve: on each month start, fill empty slots with the best-ranked new candidates
    (a ticker is never held twice). Equal size = sleeve equity / slots. Daily marked equity."""
    by_date = {d: g.sort_values("rank") for d, g in trades.groupby("date")}
    idx = C.index[C.index.searchsorted(START):]
    cash, held, curve = 1.0, {}, []
    for i, d in enumerate(idx):
        gi = C.index.get_loc(d)
        for t, p in list(held.items()):
            if p["exit_i"] == gi:
                cash += p["sh"] * p["exit"] * (1 - COST / 2)
                del held[t]
        # signals are dated the day before entry; enter at today's open
        prev = C.index[gi - 1] if gi > 0 else None
        if prev in by_date:
            eq = cash + sum(p["sh"] * p["mark"] for p in held.values())
            for r in by_date[prev].itertuples():
                if len(held) >= slots or r.ticker in held or r.entry_i != gi:
                    continue
                alloc = min(eq / slots, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                held[r.ticker] = dict(sh=alloc * (1 - COST / 2) / r.entry, exit_i=r.exit_i, exit=r.exit, mark=r.entry)
        for t, p in held.items():
            v = C.at[d, t]
            if not np.isnan(v):
                p["mark"] = v
        curve.append(cash + sum(p["sh"] * p["mark"] for p in held.values()))
    return pd.Series(curve, index=idx)


def curve_stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, max_dd=(eq / eq.cummax() - 1).min())


def yearly(eq):
    y = eq.resample("YE").last()
    y = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), y])
    return y.pct_change().dropna()


# ---------------------------------------------------------------- report
def pct(x, d=0):
    return "—" if pd.isna(x) else f"{x:+.{d}%}"


def main():
    mem, C, O, H, L, spy = load()
    F = features(C, O, H)
    X = snapshot(mem, C, H, L, F)
    out = []
    w = out.append
    w(f"Data through {C.index[-1].date()}; {len(X):,} stock-months on {X.date.nunique()} dates.\n")

    # Phase 2 summary (only rows with a full 12 months ahead)
    full = X.dropna(subset=["up", "down"])
    w("## Phase 2: chance of touching +100% (hero) / -50% (bust) within 12 months\n")
    w("| Screen | Hero early | Hero late | Bust early | Bust late | Median 12m early | Median 12m late |")
    w("|---|---|---|---|---|---|---|")
    rows = [("All S&P 500 stocks", full)] + [(n, full[SCREENS[n][0](full)]) for n in SCREENS]
    for n, g in rows:
        a = g.groupby("half").agg(h=("up", lambda s: (s >= 1).mean()), b=("down", lambda s: (s <= -.5).mean()),
                                  m=("r12", "median"))
        w(f"| {n} | {pct(a.h.get('early'),1)} | {pct(a.h.get('late'),1)} | {pct(a.b.get('early'),1)} | "
          f"{pct(a.b.get('late'),1)} | {pct(a.m.get('early'))} | {pct(a.m.get('late'))} |")

    # Phase 3: trade level
    w("\n## Phase 3a: every candidate as a trade (stop -25%, trailing exit, max 12 months)\n")
    w("| Screen | Trailing exit | Trades | Kept >= +100% | Kept >= +50% | Lost money | Avg | Median | Avg early | Avg late |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    T = {}
    for n in SCREENS:
        cand = candidates(X, n)
        for trail in (0.20, 0.30, 0.40):
            tr = trade_level(C, O, H, L, cand, trail)
            T[(n, trail)] = tr
            c = tr[tr.why != "open"]
            h = c.groupby("half").ret.mean()
            w(f"| {n} | {trail:.0%} | {len(c):,} | {(c.ret >= 1).mean():.1%} | {(c.ret >= .5).mean():.1%} | "
              f"{(c.ret <= 0).mean():.0%} | {pct(c.ret.mean(),1)} | {pct(c.ret.median(),1)} | "
              f"{pct(h.get('early'),1)} | {pct(h.get('late'),1)} |")

    # Phase 3b: sleeve
    w(f"\n## Phase 3b: hero sleeve ({SLOTS} positions, filled monthly), vs S&P 500\n")
    s = spy.loc[spy.index >= START]
    sy = yearly(s)
    years = [d.year for d in sy.index]
    w("| Screen | Trailing | CAGR | Max drawdown | " + " | ".join(str(y) for y in years) + " |")
    w("|---|---|---|---|" + "---|" * len(years))
    ss = curve_stats(s)
    w(f"| S&P 500 | — | {pct(ss['cagr'],1)} | {pct(ss['max_dd'],1)} | " + " | ".join(pct(v) for v in sy) + " |")
    best = {}
    for (n, trail), tr in T.items():
        eq = sleeve(C, tr)
        st, yr = curve_stats(eq), yearly(eq)
        best[(n, trail)] = st["cagr"]
        w(f"| {n} | {trail:.0%} | {pct(st['cagr'],1)} | {pct(st['max_dd'],1)} | " + " | ".join(pct(v) for v in yr) + " |")

    # Phase 3c: one pick per year (first trading day of each year, plus Jun 2021), 30% trail
    w("\n## Phase 3c: the single top-ranked pick on the first trading day of each year (trailing 30%)\n")
    w("| Screen | " + " | ".join(str(y) for y in years) + " |")
    w("|---|" + "---|" * len(years))
    for n in SCREENS:
        tr = T[(n, 0.30)]
        cells = []
        for y in years:
            first = tr[(tr.date.dt.year == y)]
            first = first[first.date == first.date.min()] if len(first) else first
            if first.empty:
                cells.append("—")
                continue
            r = first.sort_values("rank").iloc[0]
            cells.append(f"{r.ticker} {pct(r.ret)}" + (" (open)" if r.why == "open" else ""))
        w(f"| {n} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    (HERE / "results.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
