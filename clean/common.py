"""Parsing and lookup helpers shared by clean_prices.py and clean_sectors.py."""
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "import settings" works when run as clean/run_clean.py

from rule_log import log_rule

RAW_DIR = PROJECT_DIR / "data" / "raw"

YEAR_PATTERN = re.compile(r"(20[0-2]\d)")


def years_in_filename(file_name):
    """Every plausible 4-digit year (2000-2029) found in a file name, in order."""
    return [int(match) for match in YEAR_PATTERN.findall(file_name)]


def find_price_files():
    """Every 'all_stocks' CSV under data/raw/<dataset>/, each tagged with its one intended year."""
    files = sorted(RAW_DIR.glob("*/NSE_data_all_stocks_*.csv"))
    tagged = []
    for file_path in files:
        years = sorted(set(years_in_filename(file_path.name)))
        if len(years) != 1:
            raise ValueError(f"expected exactly one distinct year in {file_path.name}, found {years}")
        tagged.append((file_path, years[0]))
    return tagged


def find_sector_files():
    """Every sector CSV, each tagged with the latest year it claims to describe ('as of')."""
    files = sorted(RAW_DIR.glob("*/NSE_data_stock_market_sectors*.csv"))
    tagged = []
    for file_path in files:
        years = years_in_filename(file_path.name)
        if len(years) == 0:
            raise ValueError(f"no year found in {file_path.name}")
        tagged.append((file_path, max(years)))
    return tagged


def parse_money_or_volume(text_column):
    """'3,200' -> 3200.0, '6.25%' -> 6.25, '-' or blank -> NaN. The source uses '-' for
    not-traded cells and writes Change% with a literal '%' suffix."""
    cleaned = text_column.astype("string").str.strip()
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.where(~cleaned.isin(["-", "", "nan", "NaN"]))
    return pd.to_numeric(cleaned, errors="coerce")


def parse_trade_date(text_column):
    """'4-Jan-13', '04-Jan-21', '3-Jan-22' and '03-Jan-2023' all appear across the 12 source
    files: day not always zero-padded, year sometimes 2-digit and sometimes 4-digit (checked by
    sampling each file's DATE column before writing this). 'mixed' lets pandas infer each
    row's own format rather than forcing one pattern that only some files match."""
    return pd.to_datetime(text_column.astype("string").str.strip(), format="mixed", dayfirst=True)


def normalise_code(text_column):
    """Stock codes compared later must match on case and stray spaces."""
    return text_column.astype("string").str.strip().str.upper()


def rename_columns_case_insensitive(table, normalised_to_internal):
    """Map raw headers to internal names regardless of case or stray spacing.

    The source schema drifted: files from 2013-2021 use 'DATE'/'CODE'/'NAME'/'Adjust', files
    from 2022 onward use 'Date'/'Code'/'Name'/'Adjusted Price' -- same meaning, different case
    (confirmed by comparing one file's columns from each era before writing this)."""
    rename_map = {}
    for raw_name in table.columns:
        key = raw_name.strip().lower()
        if key in normalised_to_internal:
            rename_map[raw_name] = normalised_to_internal[key]
    return table.rename(columns=rename_map)


def drop_and_log(table, drop_mask, table_name, rule, note=""):
    rows_in = len(table)
    kept = table.loc[~drop_mask].copy()
    log_rule(table_name, rule, rows_in, len(kept), note=note)
    return kept
