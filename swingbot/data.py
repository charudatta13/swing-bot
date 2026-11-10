"""Free data: S&P 500 list (Wikipedia), daily prices, quarterly financials and
earnings dates (Yahoo Finance via yfinance). Everything is cached under data/."""
import io
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from . import config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FUND = DATA / "fundamentals"
DATA.mkdir(exist_ok=True)
FUND.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 swing-research-tool"}


def _fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def universe(refresh: bool = False) -> pd.DataFrame:
    path = DATA / "sp500.csv"
    if refresh or not _fresh(path, 24 * 7):
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers=UA, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        t = t.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                              "GICS Sub-Industry": "industry"})
        t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
        t[["ticker", "name", "sector", "industry"]].to_csv(path, index=False)
    return pd.read_csv(path)


def prices(tickers: list[str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Returns {'Open','High','Low','Close','Volume'} -> wide DataFrame (dates x tickers).
    Prices are split/dividend adjusted. SPY is always included."""
    path = DATA / "prices.pkl"
    if not refresh and _fresh(path, 10):
        return pickle.loads(path.read_bytes())
    tick = sorted(set(tickers) | {"SPY"})
    raw = yf.download(tick, start=config.HISTORY_START, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    # Keep only days with real closes (drops an unfinished session where Close is still NaN).
    days = raw["Close"].dropna(how="all").index
    days = days[raw["Close"].loc[days].notna().mean(axis=1) > 0.5]
    out = {f: raw[f].loc[days] for f in ["Open", "High", "Low", "Close", "Volume"]}
    path.write_bytes(pickle.dumps(out))
    return out


def _fetch_one(t: str) -> dict:
    for attempt in range(3):
        try:
            tk = yf.Ticker(t)
            rec = {"ticker": t, "quarters": {}, "earnings": [], "fetched": time.time()}
            q = tk.quarterly_income_stmt
            if q is not None and not q.empty:
                rows = ["Total Revenue", "Diluted EPS", "Net Income", "Gross Profit", "Operating Income"]
                for col in q.columns:
                    rec["quarters"][str(pd.Timestamp(col).date())] = {
                        r: (None if r not in q.index or pd.isna(q.at[r, col]) else float(q.at[r, col]))
                        for r in rows}
            e = tk.get_earnings_dates(limit=28)
            if e is not None and not e.empty:
                for ts, row in e.iterrows():
                    rec["earnings"].append({
                        "ts": ts.isoformat(),
                        "est": None if pd.isna(row.get("EPS Estimate")) else float(row["EPS Estimate"]),
                        "eps": None if pd.isna(row.get("Reported EPS")) else float(row["Reported EPS"]),
                        "surprise": None if pd.isna(row.get("Surprise(%)")) else float(row["Surprise(%)"]),
                    })
            return rec
        except Exception as ex:  # rate limits / transient errors
            time.sleep(2 + attempt * 3)
            err = str(ex)
    return {"ticker": t, "quarters": {}, "earnings": [], "error": err}


def fundamentals(tickers: list[str], refresh: bool = False, workers: int = 6) -> dict[str, dict]:
    def stale(t):
        # Uses the fetch time stored inside the file (file dates reset on a git checkout).
        p = FUND / f"{t}.json"
        if refresh or not p.exists():
            return True
        rec = json.loads(p.read_text())
        return "error" in rec or time.time() - rec.get("fetched", 0) > 3 * 86400
    todo = [t for t in tickers if stale(t)]
    if todo:
        print(f"Fetching fundamentals for {len(todo)} tickers...")
        with ThreadPoolExecutor(workers) as ex:
            for i, rec in enumerate(ex.map(_fetch_one, todo), 1):
                (FUND / f"{rec['ticker']}.json").write_text(json.dumps(rec))
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}")
    out = {}
    for t in tickers:
        p = FUND / f"{t}.json"
        if p.exists():
            out[t] = json.loads(p.read_text())
    return out


def earnings_reaction_days(fund: dict, trading_days: pd.DatetimeIndex, close: pd.Series) -> list[dict]:
    """For each past earnings report, find the first trading session that could react to it.
    After-close reports (>=16:00) react next day; pre-open (<9:30) the same day;
    unknown times use whichever of the two days moved more."""
    out = []
    for e in fund.get("earnings", []):
        if e["eps"] is None:
            continue
        ts = pd.Timestamp(e["ts"])
        d = pd.Timestamp(ts.date())
        pos = trading_days.searchsorted(d)
        if pos >= len(trading_days) or pos == 0:
            continue
        hour = ts.hour + ts.minute / 60
        same, nxt = pos, pos + 1 if trading_days[pos] == d else pos
        if hour >= 16:
            idx = nxt
        elif 0 < hour < 9.5:
            idx = same
        else:
            cands = [i for i in (same, nxt) if i < len(trading_days)]
            moves = [abs(close.iloc[i] / close.iloc[i - 1] - 1) if pd.notna(close.iloc[i]) else 0 for i in cands]
            idx = cands[int(pd.Series(moves).idxmax())] if moves else same
        if idx >= len(trading_days):
            continue
        out.append({"report_ts": ts, "react_day": trading_days[idx], "surprise": e["surprise"],
                    "eps": e["eps"], "est": e["est"]})
    return out
