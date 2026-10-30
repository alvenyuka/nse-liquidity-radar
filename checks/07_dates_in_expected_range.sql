-- Every trade date must fall inside the source archive's own stated span (settings.py's
-- EARLIEST_VALID_DATE / LATEST_VALID_DATE, substituted by sql/template.py).
SELECT * FROM raw_prices
WHERE trade_date < DATE '{{EARLIEST_VALID_DATE}}' OR trade_date > DATE '{{LATEST_VALID_DATE}}';
