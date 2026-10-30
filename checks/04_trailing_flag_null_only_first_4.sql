-- trailing_illiquid_5 (current row plus 4 preceding = 5 sessions) must be NULL only on a
-- stock's first 4 sessions, where there is no full 5-session trailing window yet.
WITH ranked AS (
    SELECT
        stock_code,
        session_no,
        trailing_illiquid_5,
        ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY session_no ASC) AS rank_from_start
    FROM labels
)
SELECT *
FROM ranked
WHERE (rank_from_start <= 4 AND trailing_illiquid_5 IS NOT NULL)
   OR (rank_from_start > 4 AND trailing_illiquid_5 IS NULL);
