-- Every stock that ever appears in the price table should resolve to a sector row, now
-- that index codes and malformed sector rows have both been removed in clean/.
SELECT DISTINCT p.stock_code
FROM raw_prices AS p
LEFT JOIN raw_sectors AS s ON s.stock_code = p.stock_code
WHERE s.stock_code IS NULL;
