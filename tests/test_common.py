"""Unit tests for the small parsing helpers in clean/common.py, each one checked against a
real quirk found in the actual source files (not a hypothetical one)."""
import pandas as pd

from common import normalise_code, parse_money_or_volume, parse_trade_date, rename_columns_case_insensitive, years_in_filename


def test_parse_money_or_volume_handles_commas_dash_and_percent():
    raw = pd.Series(["3,200", "-", "", "6.25%", "118", None])
    parsed = parse_money_or_volume(raw)
    assert parsed.tolist()[0] == 3200.0
    assert pd.isna(parsed.tolist()[1])
    assert pd.isna(parsed.tolist()[2])
    assert parsed.tolist()[3] == 6.25
    assert parsed.tolist()[4] == 118.0
    assert pd.isna(parsed.tolist()[5])


def test_parse_trade_date_handles_every_format_seen_in_the_source():
    raw = pd.Series(["2-Jan-13", "04-Jan-21", "3-Jan-22", "03-Jan-2023", "2-Jan-2024"])
    parsed = parse_trade_date(raw)
    assert parsed.dt.year.tolist() == [2013, 2021, 2022, 2023, 2024]
    assert parsed.dt.month.tolist() == [1, 1, 1, 1, 1]
    assert parsed.dt.day.tolist() == [2, 4, 3, 3, 2]


def test_normalise_code_strips_and_uppercases():
    raw = pd.Series([" kcb ", "Egad", "WTK"])
    assert normalise_code(raw).tolist() == ["KCB", "EGAD", "WTK"]


def test_rename_columns_case_insensitive_matches_both_schema_eras():
    old_era = pd.DataFrame(columns=["DATE", "CODE", "Day Price"])
    new_era = pd.DataFrame(columns=["Date", "Code", "Day Price"])
    mapping = {"date": "trade_date_raw", "code": "stock_code_raw", "day price": "close_price_raw"}
    assert set(rename_columns_case_insensitive(old_era, mapping).columns) == set(mapping.values())
    assert set(rename_columns_case_insensitive(new_era, mapping).columns) == set(mapping.values())


def test_years_in_filename_finds_every_year_even_when_repeated():
    assert years_in_filename("NSE_data_all_stocks_2021_upto_31dec2021.csv") == [2021, 2021]
    assert years_in_filename("NSE_data_stock_market_sectors_2023_2024.csv") == [2023, 2024]
    assert years_in_filename("NSE_data_all_stocks_2013.csv") == [2013]
