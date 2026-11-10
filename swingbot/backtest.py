"""Trade simulation shared by the backtest and the daily paper ledger."""
import numpy as np
import pandas as pd

from . import config
from .rules import RULES, Rule


def simulate_trade(ind: dict, t: str, sig_i: int, rule: Rule) -> dict | None:
    """Signal on bar sig_i (close). Enter at open of sig_i+1. Walk forward until an exit.
    Conservative: if stop and target are both touched on the same bar, assume the stop."""
    O, H, L, C = (ind[k][t].values for k in "OHLC")
    sma5 = ind["sma5"][t].values
    atr = ind["atr"][t].values[sig_i]
    n = len(C)
    e_i = sig_i + 1
    if e_i >= n or np.isnan(O[e_i]) or np.isnan(atr):
        return None
    entry = O[e_i]
    stop = entry - rule.stop_atr * atr
    target = entry + rule.target_atr * atr if rule.target_atr else np.inf
    exit_px, exit_i, why = None, None, None
    for i in range(e_i, n):
        if np.isnan(C[i]):
            continue
        if i > e_i and O[i] <= stop:
            exit_px, why = O[i], "stop (gap)"
        elif L[i] <= stop:
            exit_px, why = stop, "stop"
        elif i > e_i and O[i] >= target:
            exit_px, why = O[i], "target (gap)"
        elif H[i] >= target:
            exit_px, why = target, "target"
        elif rule.exit_above_sma5 and C[i] > sma5[i]:
            exit_px, why = C[i], "closed > 5-day avg"
        elif i - e_i + 1 >= rule.max_days:
            exit_px, why = C[i], "time stop"
        if exit_px is not None:
            exit_i = i
            break
    dates = ind["C"].index
    rec = {"ticker": t, "rule": rule.name, "signal_date": dates[sig_i], "entry_date": dates[e_i],
           "entry": entry, "stop": stop, "target": None if np.isinf(target) else target}
    if exit_px is None:  # still open
        last = n - 1
        while np.isnan(C[last]):
            last -= 1
        rec.update(exit_date=None, exit=None, exit_reason="open", last=C[last],
                   ret=C[last] / entry - 1 - config.COST_ROUND_TRIP / 2, days=last - e_i + 1)
    else:
        rec.update(exit_date=dates[exit_i], exit=exit_px, exit_reason=why, last=exit_px,
                   ret=exit_px / entry - 1 - config.COST_ROUND_TRIP, days=exit_i - e_i + 1)
    return rec


def all_trades(ind: dict, sigs: dict, start: str) -> pd.DataFrame:
    rows = []
    idx = ind["C"].index
    s0 = idx.searchsorted(pd.Timestamp(start))
    for name, sig in sigs.items():
        rule = RULES[name]
        sig = sig.iloc[s0:]
        sc = ind["_scores"][name]
        for d, t in sig.stack()[lambda x: x].index:
            i = idx.get_loc(d)
            r = simulate_trade(ind, t, i, rule)
            if r:
                r["score"] = sc.at[d, t]
                r["market_ok"] = bool(ind["market_ok"].iloc[i])
                rows.append(r)
    return pd.DataFrame(rows)


def stats(tr: pd.DataFrame) -> dict:
    tr = tr[tr.exit_reason != "open"]
    if tr.empty:
        return dict(trades=0)
    w, l = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    return dict(trades=len(tr), win_rate=len(w) / len(tr), avg_win=w.mean() if len(w) else 0,
                avg_loss=l.mean() if len(l) else 0, expectancy=tr.ret.mean(),
                profit_factor=(w.sum() / -l.sum()) if l.sum() < 0 else np.inf,
                avg_days=tr.days.mean())


def portfolio(ind: dict, trades: pd.DataFrame, slots: int, start: str) -> pd.Series:
    """Daily equity curve (start = 1.0) holding at most `slots` positions of equal size.
    Signals are taken best-score-first; a ticker is never held twice."""
    C = ind["C"]
    days = C.index[C.index.searchsorted(pd.Timestamp(start)):]
    by_entry = {d: g.sort_values("score", ascending=False) for d, g in trades.groupby("entry_date")}
    cash, held, curve = 1.0, {}, []
    for d in days:
        # exits first (at today's exit price)
        for t, p in list(held.items()):
            if p["exit_date"] is not None and p["exit_date"] == d:
                cash += p["shares"] * p["exit"] * (1 - config.COST_ROUND_TRIP / 2)
                del held[t]
        equity_prev = cash + sum(p["shares"] * p["mark"] for p in held.values())
        for _, r in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(held) >= slots or r.ticker in held:
                continue
            alloc = min(equity_prev / slots, cash)
            if alloc <= 0:
                break
            sh = alloc * (1 - config.COST_ROUND_TRIP / 2) / r.entry
            cash -= alloc
            held[r.ticker] = dict(shares=sh, exit_date=r.exit_date, exit=r.exit, mark=r.entry)
            if r.exit_date == d:  # same-day stop/target
                cash += sh * r.exit * (1 - config.COST_ROUND_TRIP / 2)
                del held[r.ticker]
        for t, p in held.items():
            px = C.at[d, t]
            if not np.isnan(px):
                p["mark"] = px
        curve.append(cash + sum(p["shares"] * p["mark"] for p in held.values()))
    return pd.Series(curve, index=days)


def curve_stats(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    r = eq.pct_change().dropna()
    return dict(total=eq.iloc[-1] / eq.iloc[0] - 1, cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
                max_dd=dd.min(), sharpe=(r.mean() / r.std() * np.sqrt(252)) if r.std() else 0)
