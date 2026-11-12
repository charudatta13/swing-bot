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
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)
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


FIELDS = ["Open", "High", "Low", "Close", "Volume"]
ET = "America/New_York"
DATA_READY_ET = pd.Timedelta(hours=16, minutes=30)  # Yahoo's daily bar is normally final by then


class _NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def last_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent NYSE trading day whose closing prices should be available by `now`."""
    now = (now or pd.Timestamp.now(tz=ET)).tz_convert(ET)
    day = now.tz_localize(None).normalize()
    if now.tz_localize(None) - day < DATA_READY_ET:
        day -= pd.Timedelta(days=1)
    holidays = set(_NYSEHolidays().holidays(day - pd.Timedelta(days=14), day))
    while day.weekday() >= 5 or day in holidays:
        day -= pd.Timedelta(days=1)
    return day


def _by_day(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One row per calendar day. Yahoo can stamp the newest bar with a different time (or time
    zone) than older bars, which splits one trading day into two half-empty rows; merge them."""
    out = {}
    for f in FIELDS:
        df = raw[f].copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(ET).tz_localize(None)
        df.index = idx.normalize()
        out[f] = df.groupby(level=0).last()  # last non-null value per ticker and day
    return out


def _adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Split/dividend-adjust prices ourselves instead of yfinance's auto_adjust.
    Yahoo publishes the newest day's Close hours before its 'Adj Close'; auto_adjust multiplies
    by Adj Close / Close, which blanks the whole day until then. The newest bar's adjustment
    factor is always 1 (adjustments only apply to earlier prices), so a missing Adj Close on a
    day that has a Close is filled with factor 1."""
    close, adj = raw["Close"], raw["Adj Close"]
    ratio = (adj / close).where(adj.notna(), 1.0).where(close.notna())
    out = {f: raw[f] * ratio for f in ["Open", "High", "Low", "Close"]}
    out["Volume"] = raw["Volume"]
    return pd.concat(out, axis=1)


def _download(tick: list[str], start: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(tick, start=start, auto_adjust=False, progress=False,
                      threads=True, group_by="column")
    return _by_day(_adjust(raw))


def prices(tickers: list[str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Returns {'Open','High','Low','Close','Volume'} -> wide DataFrame (dates x tickers).
    Prices are split/dividend adjusted. SPY is always included.
    The newest trading day is checked: tickers missing from it are downloaded again, because
    Yahoo's bulk download sometimes leaves the latest day out for many tickers."""
    path = DATA / "prices.pkl"
    if not refresh and _fresh(path, 10):
        return pickle.loads(path.read_bytes())
    tick = sorted(set(tickers) | {"SPY"})
    px = _download(tick, config.HISTORY_START)
    want = last_session()
    for attempt in range(1, 4):
        c = px["Close"]
        have = c.loc[want].notna() if want in c.index else pd.Series(False, index=c.columns)
        missing = have.index[~have].tolist()
        print(f"Prices for {want.date()}: {have.mean():.0%} of {len(have)} tickers")
        if len(missing) <= 0.02 * len(tick):  # a few delisted/odd tickers are normal
            break
        time.sleep(20 * attempt)
        recent = _download(missing, str((want - pd.Timedelta(days=10)).date()))
        for f in FIELDS:
            px[f] = px[f].combine_first(recent[f].reindex(columns=px[f].columns))
    # Drop a session still in progress (a run during market hours gets a live, partial bar),
    # then keep only days with real closes for most tickers.
    px = {f: px[f].loc[:want] for f in FIELDS}
    c = px["Close"]
    days = c.index[c.notna().mean(axis=1) > 0.5]
    dropped = c.index[c.index > days[-1]]
    if len(dropped):
        print(f"Dropped incomplete day(s): {[str(d.date()) for d in dropped]}")
    out = {f: px[f].loc[days] for f in FIELDS}
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
