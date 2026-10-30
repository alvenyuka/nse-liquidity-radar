-- The NSE's own trading calendar: one row per date any stock actually traded or was
-- quoted on, numbered in order. Every stock's rolling windows and forward labels are
-- indexed by this session_no, not by the calendar date, so a public holiday or a
-- weekend never gets counted as "one trading session".
CREATE OR REPLACE TABLE sessions AS
SELECT
    trade_date,
    ROW_NUMBER() OVER (ORDER BY trade_date) AS session_no
FROM (SELECT DISTINCT trade_date FROM raw_prices)
ORDER BY trade_date;
