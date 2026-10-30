"""Unit tests for the price-cleaning rules, on small hand-built frames."""
import pandas as pd

from clean_prices import drop_duplicate_stock_dates, drop_index_codes


def test_drop_index_codes_removes_only_caret_prefixed_codes():
    table = pd.DataFrame({
        "stock_code": ["KCB", "^NASI", "EQTY", "^N20I"],
        "trade_date": pd.to_datetime(["2020-01-01"] * 4),
    })
    kept = drop_index_codes(table)
    assert sorted(kept["stock_code"]) == ["EQTY", "KCB"]


def test_drop_duplicate_stock_dates_keeps_the_matching_year_file():
    # Same (stock, date) appears in two files: one claims it is a 2021 file, one a 2022 file.
    # The date itself is 2021-12-31, so the 2021 file's row should be the one kept.
    table = pd.DataFrame({
        "stock_code": ["KCB", "KCB"],
        "trade_date": pd.to_datetime(["2021-12-31", "2021-12-31"]),
        "intended_year": [2021, 2022],
        "close_price": [50.0, 999.0],  # a deliberately wrong value on the mismatched-year row
    })
    deduplicated = drop_duplicate_stock_dates(table)
    assert len(deduplicated) == 1
    assert deduplicated.iloc[0]["close_price"] == 50.0


def test_drop_duplicate_stock_dates_is_a_no_op_when_nothing_overlaps():
    table = pd.DataFrame({
        "stock_code": ["KCB", "KCB"],
        "trade_date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
        "intended_year": [2021, 2021],
        "close_price": [50.0, 51.0],
    })
    deduplicated = drop_duplicate_stock_dates(table)
    assert len(deduplicated) == 2
