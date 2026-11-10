"""Writes the daily HTML report (self-contained, no external files)."""
import html
import json

import numpy as np
import pandas as pd

from . import config
from . import data as D
from .rules import RULES

E = html.escape


def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "<span class=mut>—</span>"
    s = f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f"<span class='{cls}'>{s}</span>"


def money(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.2f}"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class=tw><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def equity_chart(series: dict[str, pd.Series]) -> str:
    """Line chart, one shared y-axis (growth of $1), crosshair tooltip."""
    W, H, pl, pr, pt, pb = 760, 280, 48, 150, 12, 28
    first = next(iter(series.values()))
    idx = first.index
    s = {k: v.reindex(idx).ffill() for k, v in series.items()}
    lo = min(v.min() for v in s.values()) * 0.97
    hi = max(v.max() for v in s.values()) * 1.03
    xs = lambda i: pl + (W - pl - pr) * i / (len(idx) - 1)
    ys = lambda v: pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo))
    out = [f"<svg viewBox='0 0 {W} {H}' class=chart role=img aria-label='Growth of $1: strategy vs S&P 500'>"]
    for tv in np.linspace(lo, hi, 5):
        y = ys(tv)
        out.append(f"<line x1={pl} x2={W-pr} y1={y:.1f} y2={y:.1f} class=grid />"
                   f"<text x={pl-6} y={y+4:.1f} class=ax text-anchor=end>${tv:.2f}</text>")
    for yr in sorted(set(idx.year)):
        j = idx.searchsorted(pd.Timestamp(f"{yr}-01-01"))
        if 0 < j < len(idx):
            out.append(f"<text x={xs(j):.1f} y={H-8} class=ax text-anchor=middle>{yr}</text>")
    label_y = {}
    for k, v in sorted(s.items(), key=lambda kv: -kv[1].iloc[-1]):
        y = ys(v.iloc[-1]) + 4
        if label_y and y - max(label_y.values()) < 14:
            y = max(label_y.values()) + 14
        label_y[k] = y
    for n, (k, v) in enumerate(s.items(), 1):
        step = max(1, len(idx) // 400)
        pts = " ".join(f"{xs(i):.1f},{ys(v.iloc[i]):.1f}" for i in range(0, len(idx), step))
        pts += f" {xs(len(idx)-1):.1f},{ys(v.iloc[-1]):.1f}"
        out.append(f"<polyline points='{pts}' class='ln s{n}' />")
        out.append(f"<text x={W-pr+6} y={label_y[k]:.1f} class=dl>{E(k)} ${v.iloc[-1]:.2f}</text>")
    out.append(f"<line class=xh x1=0 x2=0 y1={pt} y2={H-pb} style='display:none' />")
    out.append(f"<rect class=hit x={pl} y={pt} width={W-pl-pr} height={H-pt-pb} />")
    out.append("</svg>")
    data = {"d": [str(d.date()) for d in idx[::5]],
            "s": {k: [round(float(x), 3) for x in v.values[::5]] for k, v in s.items()},
            "pl": pl, "w": W - pl - pr, "W": W}
    legend = "".join(f"<span><i class='sw s{n}'></i>{E(k)}</span>" for n, k in enumerate(s, 1))
    return (f"<div class=legend>{legend}</div><div class=cw>{''.join(out)}<div class=tip></div></div>"
            f"<script type=application/json class=cdata>{json.dumps(data)}</script>")


def write(market, setups, paper, status, bt, qdf, sectors, notes) -> str:
    asof = market["asof"]
    parts = []

    # ---------- Market ----------
    mk = ("Uptrend: S&P 500 above its 200-day average. Long setups allowed." if market["ok"] else
          "Downtrend: S&P 500 below its 200-day average. The market filter blocks new long setups.")
    parts.append(f"""
<section><h2>Market status</h2>
<div class=tiles>
 <div class=tile><div class=k>S&amp;P 500 (SPY)</div><div class=v>{money(market['spy'])}</div><div class=s>200-day avg {money(market['spy_200'])}</div></div>
 <div class=tile><div class=k>Trend filter</div><div class=v>{'✅ On' if market['ok'] else '⛔ Off'}</div><div class=s>{'Longs allowed' if market['ok'] else 'No new longs'}</div></div>
 <div class=tile><div class=k>Breadth</div><div class=v>{market['breadth']*100:.0f}%</div><div class=s>of S&amp;P 500 above 200-day avg</div></div>
</div><p class=note>{mk}</p></section>""")

    # ---------- Setups ----------
    if setups:
        rows = []
        for s in setups[:15]:
            rows.append([f"<b>{E(s['ticker'])}</b>", E(s["rule_label"]), E(s["status"]), money(s["close"]),
                         money(s["stop"]), money(s["target"]) if s["target"] else "exit on bounce",
                         f"{s['max_days']} days", f"{s['shares']} sh ({money(s['position'])})",
                         money(s["risk"])])
        st = table(["Stock", "Rule", "Rule status", "Last close", "Stop-loss", "Target", "Max hold",
                    "Size", "$ at risk"], rows)
        st += (f"<p class=note>Sizes assume a {money(config.ACCOUNT_SIZE)} account, max {config.MAX_POSITION_PCT}% per stock, "
               f"{config.RISK_PER_TRADE_PCT}% account risk to the stop. Stops/targets are from today's close; if it opens far "
               f"away, move them by the same amount or skip. The top {config.MAX_NEW_PAPER_TRADES_PER_DAY} are logged as paper trades.</p>")
    else:
        st = "<p>No setups today.</p>"
    parts.append(f"<section><h2>New setups for next session</h2>{st}</section>")

    # ---------- Paper trades ----------
    if len(paper):
        paper = paper.reindex(columns=sorted(set(paper.columns) | {"exit_date", "entry_date", "exit", "stop", "target", "last"}))
        op =paper[paper.exit_reason.isin(["open", "pending entry"])]
        cl = paper[~paper.exit_reason.isin(["open", "pending entry"])].sort_values("exit_date", ascending=False)
        rows = [[f"<b>{E(r.ticker)}</b>", E(RULES[r.rule].label), str(pd.Timestamp(r.signal_date).date()),
                 money(r.entry), money(r.stop), money(r.target), money(r.last),
                 pct(r.ret), E(r.exit_reason)] for r in op.itertuples()]
        ptxt = "<h3>Open</h3>" + (table(["Stock", "Rule", "Signal", "Entry", "Stop", "Target", "Last", "P&amp;L", "Status"], rows)
                                  if rows else "<p class=mut>None open.</p>")
        rows = [[f"<b>{E(r.ticker)}</b>", E(RULES[r.rule].label), str(pd.Timestamp(r.entry_date).date()),
                 str(pd.Timestamp(r.exit_date).date()), money(r.entry), money(r.exit), pct(r.ret), E(r.exit_reason)]
                for r in cl.head(30).itertuples()]
        ptxt += "<h3>Closed</h3>" + (table(["Stock", "Rule", "In", "Out", "Entry", "Exit", "Return", "Why"], rows)
                                     if rows else "<p class=mut>No closed paper trades yet.</p>")
    else:
        ptxt = "<p class=mut>The paper ledger starts today. Trades open at the next session's open price.</p>"
    parts.append(f"<section><h2>Paper trades (forward test)</h2>{ptxt}</section>")

    # ---------- Scorecard ----------
    rows = []
    for name, rule in RULES.items():
        s, e, l, p, ps = bt[name]["stats"], bt[name]["early"], bt[name]["late"], bt[name]["port"], status[name]["paper"]
        rows.append([f"<b>{E(rule.label)}</b><div class=mut>{E(rule.description)}</div>",
                     f"<b>{E(status[name]['status'])}</b><div class=mut>{E(status[name]['why'])}</div>",
                     f"{s.get('trades',0):,}", pct(s.get("win_rate"), 0, False),
                     f"{pct(s.get('avg_win'))} / {pct(s.get('avg_loss'))}", pct(s.get("expectancy"), 2),
                     f"{pct(e.get('expectancy'),2)} → {pct(l.get('expectancy'),2)}",
                     f"{s.get('avg_days',0):.1f}", pct(p["cagr"]), pct(p["max_dd"]),
                     f"{ps.get('trades',0)} / {pct(ps.get('expectancy'),2) if ps.get('trades') else '—'}"])
    sc = table(["Rule", "Status", "Trades", "Win rate", "Avg win / loss", "Avg per trade", "Early → recent",
                "Avg days", f"CAGR ({config.MAX_OPEN_POSITIONS} slots)", "Max drawdown", "Paper n / avg"], rows, "score")
    cb, sp = bt["_combined"]["port"], bt["_spy"]["port"]
    split = pd.Timestamp(bt["_split_date"]).date()
    sc += f"""<h3>All four rules together vs. buying and holding the S&amp;P 500</h3>
<div class=tiles>
 <div class=tile><div class=k>Rules combined, CAGR</div><div class=v>{pct(cb['cagr'])}</div><div class=s>max drawdown {pct(cb['max_dd'])}, Sharpe {cb['sharpe']:.2f}</div></div>
 <div class=tile><div class=k>S&amp;P 500, CAGR</div><div class=v>{pct(sp['cagr'])}</div><div class=s>max drawdown {pct(sp['max_dd'])}, Sharpe {sp['sharpe']:.2f}</div></div>
</div>
{equity_chart({'Rules combined': bt['_combined']['curve'], 'S&P 500': bt['_spy']['curve']})}
<p class=note><b>How to read this.</b> Backtest from {config.BACKTEST_START} on today's S&amp;P 500 members, entering at the next open, with
{config.COST_ROUND_TRIP*100:.2f}% round-trip costs, at most {config.MAX_OPEN_POSITIONS} positions. "Early → recent" splits trades at {split}:
a rule whose edge shrinks or flips in the recent period is a warning. <b>Results are optimistic</b>: using today's members leaves out
companies that were dropped from the index (survivorship bias), and Yahoo's free earnings history is incomplete.
Paper results are the real test. A rule is paused automatically when its last 20+ paper trades lose money on average.</p>"""
    parts.append(f"<section><h2>Rule scorecard: which rules have an edge</h2>{sc}</section>")

    # ---------- Q2 vs Q3 ----------
    rows = [[E(sec), f"{int(r.stocks)}", pct(r.q2), pct(r.q3), pct(r["shift"]), pct(r.qtd), pct(r.rev_qoq),
             pct(r.beat_rate, 0, False)] for sec, r in sectors.iterrows()]
    q = table(["Sector", "Stocks", "Q2 median", "Q3 median", "Change Q2→Q3", "Since Sep 30", "Revenue q/q (median)",
               "Beat EPS in Q3"], rows)
    spy_q3 = qdf.spy_q3.iloc[0]

    def mover_rows(df):
        out = []
        for r in df.itertuples():
            note = notes.get(r.ticker)
            why = f"<div class=why>{E(note['why'])}</div>" if note else ""
            src = (" <span class=mut>Sources: " + ", ".join(f"<a href='{E(u)}' target=_blank rel=noopener>{i}</a>"
                   for i, u in enumerate(note.get('sources', []), 1)) + "</span>") if note and note.get("sources") else ""
            out.append([f"<b>{E(r.ticker)}</b><div class=mut>{E(r.name)}</div>", E(r.sector), pct(r.ret_q2), pct(r.ret_q3),
                        pct(r.vs_sector_q3), pct(getattr(r, 'q3_surprise', np.nan)), pct(getattr(r, 'q3_earn_move', np.nan)),
                        pct(getattr(r, 'rev_qoq', np.nan)), f"<div class=mut>{E(r.driver)}</div>{why}{src}"])
        return out
    hdr = ["Stock", "Sector", "Q2", "Q3", "Q3 vs sector", "EPS surprise", "Earnings-day move", "Revenue q/q", "Why it moved"]
    valid = qdf.dropna(subset=["ret_q3"])
    q += "<h3>Biggest Q3 gainers</h3>" + table(hdr, mover_rows(valid.nlargest(15, "ret_q3")), "movers")
    q += "<h3>Biggest Q3 losers</h3>" + table(hdr, mover_rows(valid.nsmallest(15, "ret_q3")), "movers")
    flip_up = valid[(valid.ret_q2 < -0.10) & (valid.ret_q3 > 0.15)].nlargest(10, "ret_q3")
    flip_dn = valid[(valid.ret_q2 > 0.10) & (valid.ret_q3 < -0.15)].nsmallest(10, "ret_q3")
    q += "<h3>Reversals: down in Q2, up in Q3</h3>" + (table(hdr, mover_rows(flip_up), "movers") if len(flip_up) else "<p class=mut>None.</p>")
    q += "<h3>Reversals: up in Q2, down in Q3</h3>" + (table(hdr, mover_rows(flip_dn), "movers") if len(flip_dn) else "<p class=mut>None.</p>")
    q += (f"<p class=note>S&amp;P 500 in Q3: {pct(spy_q3)}. Q2 is Mar 31 → Jun 30, 2026; Q3 is Jun 30 → Sep 30, 2026. Fundamentals "
          "compare the fiscal quarter ending Apr–Jun with the one ending Jul–Sep. The grey 'why' line is computed from the numbers; the "
          "line below it, where present, is a summary of news coverage. All 500 stocks are in <code>data/q2_q3_analysis.csv</code>.</p>")
    parts.append(f"<section><h2>Q2 vs Q3 2026: what changed and why</h2>{q}</section>")

    page = TEMPLATE.replace("{{ASOF}}", str(asof)).replace("{{BODY}}", "\n".join(parts))
    out = D.ROOT / "reports" / f"report_{asof}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    (D.ROOT / "reports" / "latest.html").write_text(page, encoding="utf-8")
    (D.ROOT / "reports" / "index.html").write_text(page, encoding="utf-8")
    return str(out)


TEMPLATE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Swing Trading Report</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--mut:#7a7975;--line:#e6e5e0;
--s1:#2a78d6;--s2:#eb6834;--pos:#00791a;--neg:#c02626}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#1a1a19;--card:#222221;--ink:#f0efec;
--ink2:#c3c2b7;--mut:#94938b;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--pos:#4cc46a;--neg:#f07272}}
:root[data-theme=dark]{color-scheme:dark;--bg:#1a1a19;--card:#222221;--ink:#f0efec;--ink2:#c3c2b7;--mut:#94938b;--line:#3a3a37;
--s1:#3987e5;--s2:#d95926;--pos:#4cc46a;--neg:#f07272}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:20px 0 8px;color:var(--ink2)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.sub,.mut{color:var(--mut)}.mut{font-size:12px}.note{color:var(--ink2);font-size:13px;margin:10px 0 0}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.tiles{display:flex;gap:12px;flex-wrap:wrap}.tile{flex:1 1 180px;border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.k{color:var(--ink2);font-size:12px}.v{font-size:22px;font-weight:600}.s{color:var(--mut);font-size:12px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top;font-variant-numeric:tabular-nums}
table.movers td:last-child{min-width:280px}table.score td:first-child{min-width:220px}
.why{margin-top:4px}a{color:var(--s1)}
.disc{background:transparent;border:1px dashed var(--line);font-size:13px;color:var(--ink2)}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:12px 0 4px}.sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}
.cw{position:relative}.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.ax{fill:var(--mut);font-size:11px}.dl{fill:var(--ink2);font-size:11px}
.ln{fill:none;stroke-width:2;stroke-linejoin:round}.ln.s1{stroke:var(--s1)}.ln.s2{stroke:var(--s2)}
.xh{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3}.hit{fill:transparent;cursor:crosshair}
.tip{position:absolute;display:none;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);white-space:nowrap}
</style></head><body><main>
<h1>Swing trading report</h1><div class=sub>Data through market close {{ASOF}} · S&amp;P 500 universe · free data (Yahoo Finance)</div>
<section class=disc>This is a research and paper-trading tool, not investment advice. Rules are mechanical and tested on past data.
Past results don't guarantee future results. You decide what, if anything, to trade.</section>
{{BODY}}
</main>
<script>
document.querySelectorAll('.cw').forEach(cw=>{
 const svg=cw.querySelector('svg'),tip=cw.querySelector('.tip'),xh=svg.querySelector('.xh'),hit=svg.querySelector('.hit');
 const D=JSON.parse(cw.nextElementSibling.textContent);const n=D.d.length;const names=Object.keys(D.s);
 hit.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const sx=D.W/r.width;
  const x=(e.clientX-r.left)*sx;let i=Math.round((x-D.pl)/D.w*(n-1));i=Math.max(0,Math.min(n-1,i));
  const px=D.pl+D.w*i/(n-1);xh.setAttribute('x1',px);xh.setAttribute('x2',px);xh.style.display='';
  tip.innerHTML='<b>'+D.d[i]+'</b>'+names.map((k,j)=>'<div><i class="sw s'+(j+1)+'"></i>'+k+': $'+D.s[k][i].toFixed(2)+'</div>').join('');
  tip.style.display='block';const left=px/sx;tip.style.left=(left>r.width*0.6?left-tip.offsetWidth-10:left+10)+'px';tip.style.top='10px';});
 hit.addEventListener('mouseleave',()=>{tip.style.display='none';xh.style.display='none';});
});
</script></body></html>"""
