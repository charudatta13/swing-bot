"""Swing-trading rules. Each rule turns end-of-day data into a signal on day t;
the trade is entered at the NEXT day's open (no look-ahead).

Every rule defines its exits: stop-loss and target in ATR multiples, a time stop,
and optionally an exit when price closes back above its 5-day average."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as D


@dataclass
class Rule:
    name: str
    label: str
    description: str
    stop_atr: float
    target_atr: float | None
    max_days: int
    exit_above_sma5: bool = False


RULES = {
    "pullback": Rule("pullback", "Trend pullback",
                     "Stock in an uptrend (price > 50-day > 200-day average) pulls back until "
                     "14-day RSI drops below 40.", stop_atr=2.0, target_atr=3.0, max_days=15),
    "breakout": Rule("breakout", "52-week-high breakout",
                     "Closes at a new 52-week high on at least 2x normal volume, near the day's high, "
                     "above its 200-day average.", stop_atr=2.0, target_atr=4.0, max_days=20),
    "meanrev": Rule("meanrev", "Oversold bounce",
                    "Stock above its 200-day average falls hard for a few days (2-day RSI < 5). "
                    "Exit when it closes back above its 5-day average.",
                    stop_atr=3.0, target_atr=None, max_days=10, exit_above_sma5=True),
    "pead": Rule("pead", "Post-earnings drift",
                 "Beats EPS estimates and jumps at least 4% on the earnings reaction day while "
                 "above its 200-day average. Rides the drift that often follows.",
                 stop_atr=2.5, target_atr=6.0, max_days=20),
}


def _rsi(c: pd.DataFrame, n: int) -> pd.DataFrame:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def indicators(px: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    o, h, l, c, v = (px[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
    tr = tr.reindex(c.index)
    ind = dict(O=o, H=h, L=l, C=c, V=v)
    ind["sma5"] = c.rolling(5).mean()
    ind["sma50"] = c.rolling(50).mean()
    ind["sma200"] = c.rolling(200).mean()
    ind["atr"] = tr.rolling(14).mean()
    ind["rsi2"] = _rsi(c, 2)
    ind["rsi14"] = _rsi(c, 14)
    ind["hi252"] = h.rolling(252).max().shift(1)
    ind["vol50"] = v.rolling(50).mean().shift(1)
    ind["mom126"] = c / c.shift(126) - 1
    spy = c["SPY"]
    ind["market_ok"] = spy > spy.rolling(200).mean()
    return ind


def signals(ind: dict, funds: dict[str, dict] | None = None) -> dict[str, pd.DataFrame]:
    """Boolean DataFrames (dates x tickers) per rule, plus a ranking score."""
    c, h, l, v = ind["C"], ind["H"], ind["L"], ind["V"]
    s200, s50 = ind["sma200"], ind["sma50"]
    liquid = (c > 5) & (ind["vol50"] * c > 20e6)  # > $20M/day traded
    out = {}
    out["pullback"] = liquid & (c > s50) & (s50 > s200) & (ind["rsi14"] < 40) & (ind["rsi14"].shift() >= 40)
    rng = (h - l).replace(0, np.nan)
    out["breakout"] = liquid & (c > ind["hi252"]) & (v > 2 * ind["vol50"]) & (c > s200) & ((c - l) / rng > 0.75)
    out["meanrev"] = liquid & (c > s200) & (ind["rsi2"] < 5)
    pead = pd.DataFrame(False, index=c.index, columns=c.columns)
    if funds:
        days = c.index
        for t, f in funds.items():
            if t not in c.columns:
                continue
            for r in D.earnings_reaction_days(f, days, c[t]):
                d = r["react_day"]
                i = days.get_loc(d)
                if i == 0 or r["surprise"] is None:
                    continue
                ret = c[t].iloc[i] / c[t].iloc[i - 1] - 1
                if r["surprise"] > 0 and ret > 0.04 and c[t].iloc[i] > s200[t].iloc[i]:
                    pead.at[d, t] = True
    out["pead"] = pead & liquid
    if "SPY" in c.columns:
        for k in out:
            out[k]["SPY"] = False
    return out


def score(ind: dict, rule: str) -> pd.DataFrame:
    """Higher = take first when there are more signals than open slots."""
    if rule == "meanrev":
        return -ind["rsi2"]
    if rule == "breakout":
        return ind["V"] / ind["vol50"]
    return ind["mom126"]
