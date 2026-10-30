-- The dense grid must never add a session before a stock's first trade or after its last --
-- that would invent trading history for a name that was not yet listed, or long after it
-- stopped trading.
SELECT g.stock_code, g.trade_date
FROM session_grid AS g
JOIN stock_bounds AS b ON b.stock_code = g.stock_code
WHERE g.trade_date < b.first_date OR g.trade_date > b.last_date;
