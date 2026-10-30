-- Dense per-stock session grid: one row per (stock_code, session) for every session
-- between that stock's own first and last observed trade date. This has to exist
-- before any rolling window, because raw_prices has real calendar gaps -- e.g. MSC is
-- missing 552 of the 2941 sessions inside its own 2013-2024 span (checked by comparing
-- row counts to session counts before writing this). A window function over the sparse
-- table would silently stretch "the next 5 sessions" across however many calendar weeks
-- those gaps happen to cover, which is the leakage risk this grid exists to remove.
--
-- Bounding each stock to its own first/last date (rather than the full 2013-2024
-- calendar) matters just as much: without it, a stock delisted in 2019 would show as
-- "illiquid" for every session from 2019 to 2024, which is not a liquidity signal, it is
-- the stock no longer existing.
CREATE OR REPLACE TABLE stock_bounds AS
SELECT stock_code, MIN(trade_date) AS first_date, MAX(trade_date) AS last_date
FROM raw_prices
GROUP BY stock_code;

CREATE OR REPLACE TABLE session_grid AS
SELECT
    b.stock_code,
    s.trade_date,
    s.session_no,
    p.close_price,
    p.day_low,
    p.day_high,
    p.volume
FROM stock_bounds AS b
JOIN sessions AS s
    ON s.trade_date BETWEEN b.first_date AND b.last_date
LEFT JOIN raw_prices AS p
    ON p.stock_code = b.stock_code AND p.trade_date = s.trade_date
ORDER BY b.stock_code, s.session_no;
