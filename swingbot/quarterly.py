"""Q2 vs Q3 2026: how each stock's price and fundamentals changed, and a first-pass
tag for *why* (earnings reaction vs. sector move vs. valuation change)."""
import numpy as np
import pandas as pd

from . import config
from . import data as D


def _px_at(c: pd.Series, date: str) -> float:
    s = c.loc[:pd.Timestamp(date)].dropna()
    return s.iloc[-1] if len(s) else np.nan


def _quarter_for(period_end: pd.Timestamp) -> str | None:
    """Map a fiscal period end to the calendar quarter it mostly covers."""
    if period_end.year != 2026:
        return None
    return {4: "Q2", 5: "Q2", 6: "Q2", 7: "Q3", 8: "Q3", 9: "Q3"}.get(period_end.month)


def _fund_row(f: dict) -> dict:
    qs = {pd.Timestamp(k): v for k, v in f.get("quarters", {}).items()}
    pick = {}
    for pe in sorted(qs):
        q = _quarter_for(pe)
        if q:
            pick[q] = (pe, qs[pe])  # latest one in that window wins
    out = {}
    if "Q2" in pick and "Q3" in pick:
        (pe2, a), (pe3, b) = pick["Q2"], pick["Q3"]
        rev2, rev3 = a.get("Total Revenue"), b.get("Total Revenue")
        out["q3_period_end"] = pe3.date()
        out["rev_qoq"] = (rev3 / rev2 - 1) if rev2 and rev3 else np.nan
        out["eps_q2"], out["eps_q3"] = a.get("Diluted EPS"), b.get("Diluted EPS")

        def margin(x, k):
            return x[k] / x["Total Revenue"] if x.get(k) is not None and x.get("Total Revenue") else np.nan
        out["gm_chg"] = margin(b, "Gross Profit") - margin(a, "Gross Profit")
        out["om_chg"] = margin(b, "Operating Income") - margin(a, "Operating Income")
        # year-over-year revenue for the Q3 quarter
        yago = [pe for pe in qs if abs((pe3 - pe).days - 365) < 25]
        if yago and qs[yago[0]].get("Total Revenue") and rev3:
            out["rev_yoy"] = rev3 / qs[yago[0]]["Total Revenue"] - 1
    return out


def analyze(univ: pd.DataFrame, px: dict, funds: dict) -> pd.DataFrame:
    C = px["Close"]
    q1, q2, q3 = (config.Q_ENDS[k] for k in ["Q1", "Q2", "Q3"])
    days = C.index
    rows = []
    for _, u in univ.iterrows():
        t = u.ticker
        if t not in C.columns:
            continue
        c = C[t]
        if c.isna().all():  # no prices for this stock (failed download): nothing to compare
            continue
        p1, p2, p3, pl = _px_at(c, q1), _px_at(c, q2), _px_at(c, q3), c.dropna().iloc[-1]
        r = dict(ticker=t, name=u["name"], sector=u.sector, industry=u.industry,
                 ret_q2=p2 / p1 - 1, ret_q3=p3 / p2 - 1, ret_qtd=pl / p3 - 1, price=pl)
        f = funds.get(t, {})
        r.update(_fund_row(f))
        # Earnings reports that landed during Q3 (Jul 1 - Sep 30) and the latest one overall
        reacts = D.earnings_reaction_days(f, days, c) if f else []
        in_q3 = [x for x in reacts if pd.Timestamp(q2) < x["react_day"] <= pd.Timestamp(q3)]
        if in_q3:
            x = in_q3[-1]
            i = days.get_loc(x["react_day"])
            r["q3_earn_date"] = x["react_day"].date()
            r["q3_surprise"] = x["surprise"] / 100 if x["surprise"] is not None else np.nan
            r["q3_earn_move"] = c.iloc[i] / c.iloc[i - 1] - 1
        if reacts:
            x = max(reacts, key=lambda z: z["react_day"])
            i = days.get_loc(x["react_day"])
            r["last_earn_date"] = x["react_day"].date()
            r["last_surprise"] = x["surprise"] / 100 if x["surprise"] is not None else np.nan
            r["last_earn_move"] = c.iloc[i] / c.iloc[i - 1] - 1
        rows.append(r)
    df = pd.DataFrame(rows)
    spy = C["SPY"]
    spy_q3 = _px_at(spy, q3) / _px_at(spy, q2) - 1
    df["sector_q3"] = df.groupby("sector").ret_q3.transform("median")
    df["vs_sector_q3"] = df.ret_q3 - df.sector_q3
    df["spy_q3"] = spy_q3
    df["driver"] = df.apply(_driver, axis=1)
    return df


def _driver(r) -> str:
    """Plain-English first guess at what drove the Q3 move."""
    parts = []
    move = r.ret_q3
    if pd.notna(r.get("q3_earn_move")) and abs(r.q3_earn_move) >= 0.05 and abs(r.q3_earn_move) >= 0.3 * abs(move):
        s, m = r.get("q3_surprise"), r.q3_earn_move
        verb = f"rose {m:.0%}" if m > 0 else f"fell {-m:.0%}"
        if pd.notna(s) and s > 0:
            txt = f"Beat EPS estimate; stock {verb} on earnings day"
            if m < 0:
                txt += " (beat wasn't enough: guidance or expectations likely)"
        elif pd.notna(s) and s < 0:
            txt = f"Missed EPS estimate; stock {verb} on earnings day"
        else:
            txt = f"Stock {verb} on earnings day"
        parts.append(txt)
    if abs(r.sector_q3) > 0.08 and np.sign(r.sector_q3) == np.sign(move) and abs(r.vs_sector_q3) < abs(move) / 2:
        parts.append(f"Sector-wide move ({r.sector} {r.sector_q3:+.0%})")
    eg = r.get("rev_qoq")
    if pd.notna(eg) and abs(eg) > 0.10:
        parts.append(f"Revenue {'up' if eg > 0 else 'down'} {abs(eg):.0%} q/q")
    om = r.get("om_chg")
    if pd.notna(om) and abs(om) > 0.05:
        parts.append(f"Operating margin {'expanded' if om > 0 else 'shrank'} {abs(om)*100:.0f} pts")
    if not parts:
        parts.append("Stock-specific / valuation (no single earnings or sector driver) - check news")
    return "; ".join(parts)


def sector_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sector").agg(stocks=("ticker", "count"), q2=("ret_q2", "median"),
                                 q3=("ret_q3", "median"), qtd=("ret_qtd", "median"),
                                 rev_qoq=("rev_qoq", "median"), beat_rate=("q3_surprise", lambda s: (s > 0).mean()))
    g["shift"] = g.q3 - g.q2
    return g.sort_values("q3", ascending=False)
