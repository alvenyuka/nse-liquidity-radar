-- avg_volume_20d and zeros_share_20d are both plain averages: once a stock has 20 or more
-- trailing sessions they must not be NULL, or the warmup logic in sql/03_features.sql is
-- silently dropping rows it should be able to compute. volume_zscore_60d is deliberately
-- not checked here -- it divides by a 60-day standard deviation, which is genuinely zero
-- (and the z-score genuinely undefined) for a stock with no traded volume at all for 60
-- straight sessions, a real state in this thin market, not a bug.
SELECT stock_code, session_no, trailing_count_20d
FROM features_final
WHERE trailing_count_20d >= 20 AND (avg_volume_20d IS NULL OR zeros_share_20d IS NULL);
