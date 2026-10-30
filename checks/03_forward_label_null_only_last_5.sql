-- is_illiquid_next_5d must be NULL on exactly a stock's last 5 sessions (no full 5-session
-- future to check), and never NULL anywhere else.
WITH ranked AS (
    SELECT
        stock_code,
        session_no,
        is_illiquid_next_5d,
        ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY session_no DESC) AS rank_from_end
    FROM labels
)
SELECT *
FROM ranked
WHERE (rank_from_end <= 5 AND is_illiquid_next_5d IS NOT NULL)
   OR (rank_from_end > 5 AND is_illiquid_next_5d IS NULL);
