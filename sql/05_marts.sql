-- The analysis mart: one row per (stock, session) with every feature, both labels, and
-- the stock's sector for grouping only (sector is never a model input -- it is static,
-- so it carries no day-to-day signal, and a sector or stock dummy would just memorise
-- which names are already known to be thin rather than generalise).
CREATE OR REPLACE TABLE mart_features_labels AS
SELECT
    f.stock_code,
    f.trade_date,
    f.session_no,
    EXTRACT(year FROM f.trade_date) AS trade_year,
    sec.sector,
    sec.stock_name,
    f.close_price,
    f.volume,
    f.is_untraded,
    f.avg_volume_20d,
    f.volatility_20d,
    f.zeros_share_20d,
    f.volume_zscore_60d,
    l.trailing_illiquid_5,
    l.currently_liquid,
    l.is_illiquid_next_5d
FROM features_final AS f
LEFT JOIN labels AS l
    ON l.stock_code = f.stock_code AND l.session_no = f.session_no
LEFT JOIN raw_sectors AS sec
    ON sec.stock_code = f.stock_code
ORDER BY f.stock_code, f.session_no;

-- The most recent row actually on record for each stock (not necessarily the exchange's
-- latest session, since some names stopped trading earlier). This is what the site's
-- watchlist shows: the current reading, not the forward label, which by definition is
-- never known yet for the last 5 sessions of the whole archive.
CREATE OR REPLACE TABLE mart_latest_watchlist AS
SELECT m.*
FROM mart_features_labels AS m
JOIN (
    SELECT stock_code, MAX(session_no) AS session_no
    FROM mart_features_labels
    GROUP BY stock_code
) AS last_seen
    ON last_seen.stock_code = m.stock_code AND last_seen.session_no = m.session_no
ORDER BY m.zeros_share_20d DESC NULLS LAST;
