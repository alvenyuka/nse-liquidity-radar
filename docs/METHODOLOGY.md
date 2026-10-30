# Method details

The plan was checked against likely failure modes before any code was written (see
"Decisions" below for what changed as a result), then re-checked against the actual data
once the schema's real quirks turned up.

## What "illiquid" means, and why

A session is **untraded** when `volume IS NULL` (the source's explicit `"-"`, or a session
missing from the row set entirely) `OR volume <= 0`. The source never once writes a literal
`0` for volume, confirmed before relying on this (`SELECT MIN(volume)... ` across the whole
table returns no zero). Both missing-row and explicit-`"-"` mean the same thing for this
project's purpose: nothing changed hands.

This is a fixed, absolute threshold chosen in advance, not a percentile computed over the
full 2013-2024 sample — a percentile threshold would use 2024 data to label a 2014 row,
which is exactly the kind of leakage a backtest needs to avoid.

## The dense per-stock session grid

`sql/02_session_grid.sql` builds one row per (stock, session) for every session between
that stock's own first and last observed trade date, before any rolling window exists.
Two concrete failure modes this removes:

1. **Calendar gaps stretch a window silently.** `ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING`
   on a sparse table "5 sessions" actually spans however many calendar days the next 5
   *present* rows happen to cover — which can be weeks, if the stock has gaps. On the dense
   grid, 5 rows is always 5 trading sessions.
2. **Unbounded grids invent history.** Bounding each stock to its own first/last date
   (rather than the full market calendar) matters just as much: without it, a stock
   delisted in 2019 reads as "illiquid" for every session from 2019 to 2024, which is the
   stock no longer existing, not a liquidity signal.

`checks/02_grid_within_stock_bounds.sql` asserts the second point directly.

## Labels and the leakage guards

- `trailing_illiquid_5`: `MAX(is_untraded)` over the current row and 4 preceding (5
  sessions including today) — `NULL` on a stock's first 4 sessions. This both is the naive
  persistence baseline's prediction and defines `currently_liquid`.
- `is_illiquid_next_5d`: `MAX(is_untraded)` over `1 FOLLOWING` to `5 FOLLOWING` — never `0
  FOLLOWING` (today), always the 5 sessions strictly after. `NULL` on a stock's last 5
  sessions, where there is no full 5-session future to check.
- Both windows are `ROWS BETWEEN`, never `RANGE` or a date self-join, for the same reason as
  the session grid: date arithmetic on a sparse or irregular calendar silently gets "5
  sessions" wrong.
- `tests/test_label_logic.py` builds a 14-row single-stock sequence by hand, works out the
  expected trailing and forward values on paper first, then runs the actual
  `sql/04_labels.sql` text against it and asserts the SQL matches. This is the one test in
  the project that would have caught a leakage bug before it reached a model.

## Train/test split and the embargo

A single time split (train before 2023, test 2023-2024) is the headline, with a 5-session
**embargo**: the last `EMBARGO_DAYS` sessions before the test period are excluded from
training, because their forward labels look 1-5 sessions ahead and would otherwise reach
into the test period. The walk-forward table (`model/train_model.py`,
`run_walkforward`) repeats this with an expanding window, one extra test year at a time,
2016 through 2024 — the single split is one market regime (2023-2024: thin NSE turnover,
KES depreciation); the walk-forward table is the evidence that the result is not one lucky
year.

No stock is held out entirely, and no stock or sector dummy is used as a feature: an
investment desk already knows its own names, and a per-stock fixed effect would mostly
memorise which names are already known to be thin rather than generalise to a name nobody
has flagged yet.

## Features kept, and features cut

Kept: `avg_volume_20d`, `volatility_20d`, `zeros_share_20d`, `volume_zscore_60d`. All four
use `ROWS BETWEEN n PRECEDING AND CURRENT ROW` — today's close and volume are known at the
moment this signal would be used (end of trading on day t, acted on at the open of t+1).

Cut:
- **The Amihud illiquidity ratio** (`|return| / value traded`), despite being the standard
  academic liquidity measure, because it is undefined on exactly the zero-volume days that
  are the whole point of this project, and so goes blind on them. `zeros_share_20d` (the
  share of the last 20 sessions spent untraded, the Lesmond-Ogden-Trzcinka "zeros" measure)
  replaces it: well-defined everywhere, and the quintile chart in the README shows it
  separates forward risk cleanly on its own. The standardised coefficients
  (`outputs/model_results.json`) show `avg_volume_20d` with the larger magnitude of the
  two, but that feature is heavily right-skewed where `zeros_share_20d` is bounded to
  [0, 1] -- which of the two actually drives more of the model is an open question, not
  settled by the coefficients alone (see the README's Results section).
- **Shares-outstanding turnover** — not present in the source data.
- **12-month High/Low** — present in the source but of unstated construction and redundant
  with the rolling windows already built.
- **Sector and stock dummies** — see above.
- **XGBoost, a FastAPI service, Docker, a scheduled-refresh workflow.** None add anything a
  static page and a plain logistic regression do not already show for a frozen historical
  archive: no live traffic to serve, no deployment target to containerise, nothing to
  refresh on a schedule.

## Decisions (changed from the first draft, before any code was written)

The plan was checked against five likely failure modes before any SQL was written, and all
five changed the design:

1. **The persistence trap.** The full-population PR-AUC is inflated by stocks that are
   already illiquid staying that way. Fixed by reporting the "transition" population
   (`currently_liquid = 1` rows only) as the real headline throughout.
2. **Threshold leakage.** An "illiquid" definition based on a percentile over the whole
   sample would use future data to label the past. Fixed with the absolute `volume <= 0 OR
   NULL` threshold above.
3. **The dense grid**, described above — the single biggest change from the first draft,
   which had assumed a plain per-stock `LAG`/`LEAD` on the raw sparse table.
4. **The embargo and the walk-forward table**, described above.
5. **Amihud replaced with `zeros_share_20d`**, described above.

One more decision made once real data was in hand, not anticipated in the design review:
excluding market indices (`^NASI`, `^N20I`, `^FNK25`, …) from the universe, discovered when
the price-code count came to 77 rather than the expected ~60-64 — described in the README's
"Problems I hit".
