"""Run after the market closes:  python -m swingbot.daily
1. refresh data  2. Q2 vs Q3 study  3. backtest each rule (re-run weekly)
4. today's setups  5. update the paper ledger  6. write reports/report_<date>.html"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config
from . import data as D
from . import quarterly as Q
from . import report
from .rules import RULES, indicators, score, signals

LEDGER = D.DATA / "paper_signals.csv"
BT_CACHE = D.DATA / "backtest.pkl"


def rule_status(bt: dict, paper: pd.DataFrame) -> dict:
    """The 'learning' step: a rule stays active only while its evidence is good.
    Backtest sets the baseline; live paper results can switch it to reduced/paused."""
    out = {}
    for name in RULES:
        s = bt[name]["stats"]
        p = paper[(paper.rule == name) & (paper.exit_reason != "open")] if len(paper) else paper
        recent = p.tail(30)
        ps = B.stats(recent) if len(recent) else dict(trades=0)
        if s.get("trades", 0) < 30 or s["expectancy"] <= 0:
            st, why = "Off", "Backtest shows no edge"
        elif ps["trades"] >= 20 and ps["expectancy"] < 0:
            st, why = "Paused", f"Last {ps['trades']} paper trades lost money on average"
        elif ps["trades"] >= 20 and ps["expectancy"] < s["expectancy"] / 2:
            st, why = "Reduced", "Paper results well below backtest - half size"
        else:
            st, why = "Active", ("Backtest edge; waiting for 20+ paper trades" if ps["trades"] < 20
                                 else "Paper results in line with backtest")
        out[name] = dict(status=st, why=why, paper=ps)
    return out


def main(refresh_fund: bool = False, rerun_backtest: bool = False, allow_stale: bool = False):
    t0 = time.time()
    univ = D.universe(refresh=rerun_backtest)
    tickers = univ.ticker.tolist()
    px = D.prices(tickers)
    newest, want = px["Close"].index[-1], D.last_session()
    if newest < want and not allow_stale:
        # Stop before anything is written: a report built on old prices looks normal but is a
        # day behind (setups for a session that already happened, P&L not updated).
        sys.exit(f"STALE DATA: newest prices are for {newest.date()}, expected {want.date()}. "
                 "Yahoo has not published the latest day yet. Nothing was written. Run again later, "
                 "or use --allow-stale to build the report anyway.")
    funds = D.fundamentals(tickers, refresh=refresh_fund)
    ind = indicators(px)
    sigs = signals(ind, funds)
    if config.USE_MARKET_FILTER:
        sigs = {k: v & ind["market_ok"].values[:, None] for k, v in sigs.items()}
    ind["_scores"] = {k: score(ind, k) for k in RULES}
    asof = ind["C"].index[-1]
    print(f"Data through {asof.date()} ({time.time()-t0:.0f}s)")

    # --- Backtest (cached for a week) ---
    bt_age_days = (time.time() - pickle.loads(BT_CACHE.read_bytes()).get("_created", 0)) / 86400         if BT_CACHE.exists() else 1e9
    if rerun_backtest or bt_age_days > 7:
        trades = B.all_trades(ind, sigs, config.BACKTEST_START)
        bt = {}
        mid = pd.Timestamp(config.BACKTEST_START) + (asof - pd.Timestamp(config.BACKTEST_START)) * 0.6
        for name in RULES:
            tr = trades[trades.rule == name]
            eq = B.portfolio(ind, tr, config.MAX_OPEN_POSITIONS, config.BACKTEST_START)
            bt[name] = dict(stats=B.stats(tr), early=B.stats(tr[tr.entry_date < mid]),
                            late=B.stats(tr[tr.entry_date >= mid]), curve=eq, port=B.curve_stats(eq),
                            by_year=tr[tr.exit_reason != "open"].groupby(tr.entry_date.dt.year).ret.agg(["count", "mean"]))
        combo_eq = B.portfolio(ind, trades, config.MAX_OPEN_POSITIONS, config.BACKTEST_START)
        spy = ind["C"]["SPY"].loc[combo_eq.index[0]:]
        spy_eq = spy / spy.iloc[0]
        bt["_combined"] = dict(curve=combo_eq, port=B.curve_stats(combo_eq), stats=B.stats(trades))
        bt["_spy"] = dict(curve=spy_eq, port=B.curve_stats(spy_eq))
        bt["_split_date"] = mid
        bt["_created"] = time.time()
        BT_CACHE.write_bytes(pickle.dumps(bt))
        print(f"Backtest done ({time.time()-t0:.0f}s)")
    bt = pickle.loads(BT_CACHE.read_bytes())

    # --- Paper ledger: log today's top signals, then re-simulate everything logged ---
    ledger = pd.read_csv(LEDGER, parse_dates=["signal_date"]) if LEDGER.exists() else \
        pd.DataFrame(columns=["signal_date", "ticker", "rule", "score"])
    paper_prev = _simulate_ledger(ind, ledger)
    status = rule_status(bt, paper_prev)

    i = len(ind["C"]) - 1
    setups = []
    for name, sig in sigs.items():
        if status[name]["status"] in ("Off", "Paused"):
            continue
        for t in sig.columns[sig.iloc[i].fillna(False).values]:
            setups.append(_setup(ind, t, i, name, status[name]["status"]))
    # Best-evidence rule first, then strongest signal within the rule
    setups = sorted(setups, key=lambda s: (-bt[s["rule"]]["stats"]["expectancy"], -s["score"]))
    todays = pd.DataFrame(setups[:config.MAX_NEW_PAPER_TRADES_PER_DAY])
    if len(todays):
        new = todays[["ticker", "rule", "score"]].assign(signal_date=asof)
        ledger = pd.concat([ledger, new]).drop_duplicates(["signal_date", "ticker", "rule"])
        ledger.to_csv(LEDGER, index=False)
    paper = _simulate_ledger(ind, ledger)

    # --- Q2 vs Q3 ---
    qdf = Q.analyze(univ, px, funds)
    qdf.to_csv(D.DATA / "q2_q3_analysis.csv", index=False)
    notes_path = D.ROOT / "research" / "why_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}

    market = dict(asof=asof.date(), spy=ind["C"]["SPY"].iloc[-1],
                  spy_200=ind["C"]["SPY"].rolling(200).mean().iloc[-1],
                  ok=bool(ind["market_ok"].iloc[-1]),
                  breadth=float((ind["C"].iloc[-1] > ind["sma200"].iloc[-1]).mean()))
    out = report.write(market, setups, paper, status, bt, qdf, Q.sector_table(qdf), notes)
    print(f"Report: {out}  ({time.time()-t0:.0f}s)")
    return out


def _simulate_ledger(ind, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = ind["C"].index
    for _, r in ledger.iterrows():
        d = pd.Timestamp(r.signal_date)
        if d not in idx or r.ticker not in ind["C"].columns:
            continue
        i = idx.get_loc(d)
        if i + 1 >= len(idx):
            rows.append(dict(ticker=r.ticker, rule=r.rule, signal_date=d, exit_reason="pending entry",
                             entry=np.nan, ret=np.nan, days=0))
            continue
        s = B.simulate_trade(ind, r.ticker, i, RULES[r.rule])
        if s:
            rows.append(s)
    return pd.DataFrame(rows)


def _setup(ind, t, i, name, st) -> dict:
    rule = RULES[name]
    c = ind["C"][t].iloc[i]
    atr = ind["atr"][t].iloc[i]
    stop = c - rule.stop_atr * atr
    target = c + rule.target_atr * atr if rule.target_atr else None
    risk_dollars = config.ACCOUNT_SIZE * config.RISK_PER_TRADE_PCT / 100
    if st == "Reduced":
        risk_dollars /= 2
    shares = int(min(risk_dollars / (c - stop), config.ACCOUNT_SIZE * config.MAX_POSITION_PCT / 100 / c))
    return dict(ticker=t, rule=name, rule_label=rule.label, status=st, close=c, stop=stop, target=target,
                max_days=rule.max_days, shares=shares, position=shares * c,
                risk=shares * (c - stop), score=float(ind["_scores"][name][t].iloc[i]),
                rule_rank=list(RULES).index(name),
                rsi14=ind["rsi14"][t].iloc[i], mom126=ind["mom126"][t].iloc[i])


if __name__ == "__main__":
    main(refresh_fund="--refresh" in sys.argv, rerun_backtest="--backtest" in sys.argv,
         allow_stale="--allow-stale" in sys.argv)
