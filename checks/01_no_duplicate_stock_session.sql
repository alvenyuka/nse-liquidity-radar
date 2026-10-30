-- Every (stock_code, session) should appear exactly once in the dense grid.
SELECT stock_code, session_no, COUNT(*) AS n
FROM session_grid
GROUP BY stock_code, session_no
HAVING COUNT(*) > 1;
