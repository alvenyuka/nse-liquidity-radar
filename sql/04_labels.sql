-- The forward label and the trailing (persistence) indicator, on the same dense grid
-- features_final uses. Both use ROWS, not RANGE or a date self-join: on a sparse table a
-- date-arithmetic window silently stretches "5 sessions" across calendar gaps (see
-- sql/02_session_grid.sql); ROWS BETWEEN on this dense grid cannot.
--
-- trailing_illiquid_5: was this stock untraded on any of the last 5 sessions up to and
-- including today? This is both the naive "persistence" baseline's prediction (today's
-- regime is tomorrow's) and the "currently illiquid" state used to split the headline
-- into the full population and the harder transition subset (currently_liquid = 1).
--
-- is_illiquid_next_5d: will it be untraded on any of the NEXT 5 sessions (1 to 5 FOLLOWING,
-- never 0 FOLLOWING -- that would be today, not the forward label)? NULL on a stock's last
-- 5 sessions, where there is no full 5-session future to check -- tests/test_label_logic.py
-- checks this exact window against a 14-row hand-built example with a known answer.
--
-- The window sizes below come from settings.py (FORWARD_WINDOW_DAYS), substituted by
-- sql/template.py -- the trailing window intentionally uses the same size as the forward
-- window, so "currently illiquid" and "will become illiquid" are measured on comparable spans.
CREATE OR REPLACE TABLE labels AS
WITH base AS (
    SELECT
        stock_code,
        session_no,
        is_untraded,
        MAX(is_untraded) OVER trailing5 AS trailing_illiquid_5_raw,
        COUNT(*) OVER trailing5 AS trailing_count_5,
        MAX(is_untraded) OVER forward5 AS forward_illiquid_5_raw,
        COUNT(*) OVER forward5 AS forward_count_5
    FROM features_final
    WINDOW
        trailing5 AS (PARTITION BY stock_code ORDER BY session_no ROWS BETWEEN {{TRAILING_WINDOW_MINUS_1}} PRECEDING AND CURRENT ROW),
        forward5 AS (PARTITION BY stock_code ORDER BY session_no ROWS BETWEEN 1 FOLLOWING AND {{FORWARD_WINDOW_DAYS}} FOLLOWING)
)
SELECT
    stock_code,
    session_no,
    CASE WHEN trailing_count_5 >= {{FORWARD_WINDOW_DAYS}} THEN trailing_illiquid_5_raw END AS trailing_illiquid_5,
    CASE WHEN trailing_count_5 >= {{FORWARD_WINDOW_DAYS}} THEN 1 - trailing_illiquid_5_raw END AS currently_liquid,
    CASE WHEN forward_count_5 >= {{FORWARD_WINDOW_DAYS}} THEN forward_illiquid_5_raw END AS is_illiquid_next_5d
FROM base;
