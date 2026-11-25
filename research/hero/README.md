# Hero-trade study (Nov 2026)

Question: can the bot find a "hero trade", a stock that doubles within a year, before it happens?

- Code: `hero_study.py` (run `python -m research.hero.hero_study` from the repo root)
- Full output: `results.md`
- S&P 500 join dates: `sp500_members.csv`, from Wikipedia's "List of S&P 500 companies" (CC BY-SA)

## Method
- **Universe:** on the first trading day of each month from Jun 2021, every stock that was in the S&P 500 *on that date*, using each stock's join date. About 450 stocks per date.
- **Hero:** the price touches +100% within the next 12 months. **Bust:** it touches −50%.
- **Trading test:** buy at the next open. Exit when the first of these happens:
  - a −25% stop-loss
  - a trailing exit: the close falls 20%, 30% or 40% below the best close since entry
  - 12 months have passed
- **Costs:** 0.10% per round trip.

## Known biases (results are optimistic)
1. **Survivorship.** Yahoo only has prices for *today's* members. Stocks counted only after their join date removes the ones that rose *into* the index (CVNA, APP, PLTR, COIN…). Stocks *removed* from the index, often after falling, are still missing. This hurts the "rebound" screen most, since it buys stocks already far below their high.
2. **In-sample screens.** The screens were chosen after looking at Phase 2 on the same 2021–2026 data.
3. **Setting sensitivity.** Portfolio results swing a lot with the trailing-exit setting. Treat single rows as noisy.
4. **The market.** 2023–2025 was a strong bull market; hero rates were about 3× those of 2021–2023.

## Findings
**Phase 2: what came before a double**

| | Hero 2021–23 | Hero 2023–25 | Bust (early / late) |
|---|---|---|---|
| Any S&P 500 stock | 1.4% | 4.4% | 2.5% / 2.7% |
| Most volatile 20% | 5.6% | 13.6% | 7.1% / 6.5% |
| **Leader:** volatile, 6-month gain ≥ 30%, within 10% of 52-week high | 8.1% | 20.3% | 9.2% / 5.1% |
| **Catalyst:** volatile, a ≥ 12% gap up in the last 3 months | 9.7% | 19.6% | 12.6% / 8.7% |
| Rebound*: volatile, ≥ 40% below 52-week high (*most biased) | 10.3% | 21.8% | 8.6% / 9.9% |

- **Volatility is the main ingredient.** No stock in the calmest 20% doubled in either period.
- **Momentum, gaps, volume and sector strength add little** on their own, once volatility is accounted for.

**Phase 3: what a trading rule keeps.** Each trade is a lottery-like bet:
- **Most trades lose:** about 53–64% lose money, and the *median* trade is −4% to −16%.
- **A few big winners carry it:** only 2–7% of trades keep +100% or more, and the *average* is positive (+5% to +18%).

Hero sleeve (4 positions, filled monthly):

| | Yearly growth | Max drop | 2021* | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | +11.9% | −24.5% | +14% | −18% | +26% | +25% | +18% | +7% |
| **Leader, 20% trailing exit** | **+28.9%** | **−29.0%** | −6% | +4% | +20% | +54% | +43% | +55% |
| Catalyst, 20% trailing exit | +35.4% | −46.9% | +23% | −39% | +12% | +114% | +74% | +69% |
| Volatile only, 30% trailing exit | −3.5% | −49.8% | −17% | −30% | +66% | −27% | +21% | −3% |

\*2021 runs from June; 2026 is to Nov 23.

One pick per year (top-ranked stock on the first trading day, 30% trailing exit):
- **Leader:** DVN +163%, TSLA −25%, SLB +4%, ANET +95%, UAL −20%, WDC +152%
- **Catalyst:** IT +12%, ANET −25%, NOW +72%, NFLX +90%, SMCI +39%, AMD +93%

That's six picks per screen, too few to judge.

## Conclusions
1. **A hero can't be picked reliably ahead of time.** Even the best screens double only 1 time in 5–12, and most of their trades lose money. Heroes show up as the few big wins inside a basket of picks.
2. **The most promising design is "Leader" with a 20% trailing exit.**
   - Its portfolio results are the steadiest of the screens, with a max drop similar to the S&P 500.
   - Trades made money on average in both halves.
   - It buys strength, not beaten-down stocks, so it's less exposed to survivorship bias than "rebound".
3. **"Catalyst" makes more but swings harder** (−39% in 2022, −47% max drop). "Rebound" looks good per trade, but its data is the least trustworthy.

## Decisions
- **Hero:** touches +100% within 12 months.
- **Data:** free data for research. Before any real money, confirm on a dataset that includes stocks removed from the index (for example Norgate or Sharadar).
- **Money:** paper only for now. If it holds up, at most 10% of the account, split across 4 picks (about $250 each on $10,000).
- **Next step:** add the "Leader" screen to the bot as a monthly hero watchlist with 20% trailing-exit paper trades. Judge it after 12–24 months of paper results.
