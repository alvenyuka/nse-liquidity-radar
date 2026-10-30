-- Traded volume cannot be negative, and the source counts shares, so it should be a whole
-- number wherever it is present.
SELECT * FROM raw_prices
WHERE volume IS NOT NULL AND (volume < 0 OR volume <> ROUND(volume));
