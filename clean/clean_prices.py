"""Cleaning rules for the daily price table (one row per stock per trading day)."""
import pandas as pd

from common import (
    find_price_files, normalise_code, parse_money_or_volume, parse_trade_date,
    rename_columns_case_insensitive,
)
from rule_log import log_rule

TABLE = "prices"

# Keys are lower-cased raw headers, so this matches both the 2013-2021 files ('DATE', 'CODE',
# 'Adjust') and the 2022-onward files ('Date', 'Code', 'Adjusted Price').
NORMALISED_COLUMN_TO_INTERNAL = {
    "date": "trade_date_raw",
    "code": "stock_code_raw",
    "day low": "day_low_raw",
    "day high": "day_high_raw",
    "day price": "close_price_raw",
    "previous": "previous_price_raw",
    "change%": "change_pct_raw",
    "volume": "volume_raw",
}
# "NAME", "12m Low", "12m High", "Change" and "Adjust"/"Adjusted Price" are read but not kept:
# NAME is covered by the sector file's own NAME column, "12m Low/High" mix in a year of history
# we are not modelling, "Change" duplicates "Change%" in a different unit, and the adjusted
# price is a backward price adjustment of unstated method -- returns here are computed from
# the raw close instead.


def read_one_file(file_path, intended_year):
    """One yearly CSV, columns renamed, tagged with the file it came from."""
    raw = pd.read_csv(file_path, dtype=str)
    raw = rename_columns_case_insensitive(raw, NORMALISED_COLUMN_TO_INTERNAL)
    table = raw[list(NORMALISED_COLUMN_TO_INTERNAL.values())].copy()
    table["source_file"] = file_path.name
    table["intended_year"] = intended_year
    return table


def concatenate_all_years(price_files):
    tables = []
    for file_path, intended_year in price_files:
        tables.append(read_one_file(file_path, intended_year))
    combined = pd.concat(tables, ignore_index=True)
    log_rule(TABLE, "concatenate_all_years", 0, len(combined), note=f"{len(price_files)} source files")
    return combined


def parse_columns(table):
    """Turn every text column into the type it actually is."""
    rows = len(table)
    table["trade_date"] = parse_trade_date(table["trade_date_raw"])
    table["stock_code"] = normalise_code(table["stock_code_raw"])
    table["close_price"] = parse_money_or_volume(table["close_price_raw"])
    table["day_low"] = parse_money_or_volume(table["day_low_raw"])
    table["day_high"] = parse_money_or_volume(table["day_high_raw"])
    table["previous_price"] = parse_money_or_volume(table["previous_price_raw"])
    table["change_pct_reported"] = parse_money_or_volume(table["change_pct_raw"])
    table["volume"] = parse_money_or_volume(table["volume_raw"])
    log_rule(TABLE, "parse_columns", rows, len(table), note="dates and numbers parsed, no rows dropped yet")
    return table


def drop_unusable_rows(table):
    """A row with no stock code or no date cannot be placed anywhere; drop it and say how many."""
    rows_in = len(table)
    missing_key = table["stock_code"].isna() | table["trade_date"].isna()
    kept = table.loc[~missing_key].copy()
    log_rule(TABLE, "drop_unusable_rows", rows_in, len(kept), note=f"{int(missing_key.sum())} rows had no code or no date")
    return kept


def check_reported_change_matches_computed(table):
    """A sanity print, not a drop: does the file's own Change% agree with (close/previous - 1)?
    Checked by hand first (73rb78pmzw/NSE_data_all_stocks_2013.csv, rows with both columns
    present): the source writes Change% as an unsigned magnitude -- "6.25%" on a day the price
    fell 6.25%, same as a day it rose 6.25% -- while Change and the prices keep their sign. So
    the comparison here is against the computed percentage's absolute value, not its sign."""
    has_both = table["previous_price"].notna() & table["change_pct_reported"].notna() & (table["previous_price"] != 0)
    computed_pct = (table.loc[has_both, "close_price"] / table.loc[has_both, "previous_price"] - 1) * 100
    difference = (computed_pct.abs() - table.loc[has_both, "change_pct_reported"]).abs()
    mismatched = int((difference > 0.5).sum())
    log_rule(TABLE, "check_reported_change_matches_computed", len(table), len(table),
             note=f"{mismatched} of {int(has_both.sum())} comparable rows differ by more than 0.5 points after allowing for Change%'s unsigned magnitude (kept either way)")
    return table


def drop_index_codes(table):
    """Codes starting with '^' are market indices (NSE All-Share, NSE 20/25-Share, FTSE NSE
    Kenya, Zamara Kenya Equity), confirmed against the sector files' own 'Indices' category.
    This project studies individual listed equities' trading liquidity; an index has no
    comparable single-instrument volume, so it is excluded rather than mixed into the universe."""
    rows_in = len(table)
    is_index = table["stock_code"].str.startswith("^")
    index_codes = sorted(table.loc[is_index, "stock_code"].unique())
    kept = table.loc[~is_index].copy()
    log_rule(TABLE, "drop_index_codes", rows_in, len(kept), note=f"removed {index_codes}")
    return kept


def drop_duplicate_stock_dates(table):
    """The four source datasets touch adjacent years; a boundary date can appear in two files.
    Keep the row from the file whose own intended year matches the date, so the row comes from
    the file that was actually built to describe that year."""
    rows_in = len(table)
    table["year_matches_file"] = table["trade_date"].dt.year == table["intended_year"]
    table = table.sort_values(["stock_code", "trade_date", "year_matches_file"], ascending=[True, True, False])
    deduplicated = table.drop_duplicates(subset=["stock_code", "trade_date"], keep="first").copy()
    deduplicated = deduplicated.drop(columns=["year_matches_file"])
    log_rule(TABLE, "drop_duplicate_stock_dates", rows_in, len(deduplicated),
             note=f"{rows_in - len(deduplicated)} duplicate (stock_code, trade_date) rows removed, keeping the matching-year file")
    return deduplicated


def clean_prices():
    price_files = find_price_files()
    table = concatenate_all_years(price_files)
    table = parse_columns(table)
    table = drop_unusable_rows(table)
    table = check_reported_change_matches_computed(table)
    table = drop_index_codes(table)
    table = drop_duplicate_stock_dates(table)
    final_columns = [
        "stock_code", "trade_date", "close_price", "day_low", "day_high",
        "previous_price", "change_pct_reported", "volume", "source_file",
    ]
    return table[final_columns].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
