-- Rolling features, computed only from today and earlier (ROWS BETWEEN N PRECEDING AND
-- CURRENT ROW): today's close is known at the moment this signal would be used, at the
-- end of trading on day t, acted on at the open of t+1 (see docs/METHODOLOGY.md).
--
-- A session is "untraded" when volume is missing -- which happens two ways in this
-- source: the row exists with volume recorded as "-" (not traded that day), or the row
-- is absent entirely from session_grid (checked: raw_prices never once writes a literal
-- 0, confirmed before writing this). Both mean the same thing here, so both become
-- volume_or_zero = 0 for every rolling average and the zeros-share measure below.
CREATE OR REPLACE TABLE features_raw AS
WITH flagged AS (
    SELECT
        *,
        CASE WHEN volume IS NULL OR volume <= 0 THEN 1 ELSE 0 END AS is_untraded,
        COALESCE(volume, 0) AS volume_or_zero,
        close_price / LAG(close_price) OVER (PARTITION BY stock_code ORDER BY session_no) - 1 AS daily_return
    FROM session_grid
)
SELECT
    *,
    AVG(volume_or_zero) OVER w20 AS avg_volume_20d,
    STDDEV_SAMP(daily_return) OVER w20 AS volatility_20d,
    AVG(CAST(is_untraded AS DOUBLE)) OVER w20 AS zeros_share_20d,
    -- NULLIF guards a 60-day stretch of identical volume (stddev exactly 0): without it,
    -- 0/0 is a floating-point NaN, not SQL NULL, and NaN slips straight past every later
    -- "IS NOT NULL" filter (found by sklearn refusing to fit on it, not by inspection).
    (volume_or_zero - AVG(volume_or_zero) OVER w60) / NULLIF(STDDEV_SAMP(volume_or_zero) OVER w60, 0) AS volume_zscore_60d,
    COUNT(*) OVER w20 AS trailing_count_20d,
    COUNT(*) OVER w60 AS trailing_count_60d
FROM flagged
WINDOW
    w20 AS (PARTITION BY stock_code ORDER BY session_no ROWS BETWEEN {{SHORT_WINDOW_MINUS_1}} PRECEDING AND CURRENT ROW),
    w60 AS (PARTITION BY stock_code ORDER BY session_no ROWS BETWEEN {{LONG_WINDOW_MINUS_1}} PRECEDING AND CURRENT ROW);

-- A feature computed on fewer than its full window is not the 20-day or 60-day figure
-- it claims to be -- it is set to NULL instead of a half-built number, so the first
-- {{SHORT_WINDOW_MINUS_1}} (or {{LONG_WINDOW_MINUS_1}}) sessions of every stock's life are
-- visibly missing rather than quietly wrong. checks/05 asserts this direction. The window
-- sizes above and the thresholds below both come from settings.py -- see sql/template.py.
CREATE OR REPLACE TABLE features_final AS
SELECT
    stock_code, trade_date, session_no, close_price, day_low, day_high, volume,
    is_untraded, daily_return,
    CASE WHEN trailing_count_20d >= {{SHORT_WINDOW_DAYS}} THEN avg_volume_20d END AS avg_volume_20d,
    CASE WHEN trailing_count_20d >= {{SHORT_WINDOW_DAYS}} THEN volatility_20d END AS volatility_20d,
    CASE WHEN trailing_count_20d >= {{SHORT_WINDOW_DAYS}} THEN zeros_share_20d END AS zeros_share_20d,
    CASE WHEN trailing_count_60d >= {{LONG_WINDOW_DAYS}} THEN volume_zscore_60d END AS volume_zscore_60d,
    trailing_count_20d,
    trailing_count_60d
FROM features_raw;
