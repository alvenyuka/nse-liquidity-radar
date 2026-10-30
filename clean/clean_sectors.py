"""Cleaning rules for the sector-lookup table (one row per stock code).

The NSE's own listed universe changes over time (the sector files grow from 64 codes in 2013
to 75 in 2023-2024 as names list, delist or get renamed). This project does not try to
reconstruct that history: it keeps one canonical sector and name per code, taken from the most
recent sector file that mentions it, and only reports how many codes changed sector along the
way. Sector is used for grouping on the report and site, never as a model feature (a sector
is static and carries no day-to-day signal)."""
import pandas as pd

from common import find_sector_files, normalise_code, rename_columns_case_insensitive
from rule_log import log_rule

TABLE = "sectors"

# Same schema drift as the price files: 2013-2021 files use 'SECTOR'/'CODE'/'NAME',
# 2022-onward files use 'Sector'/'Stock_code'/'Stock_name' (confirmed by comparing headers).
NORMALISED_COLUMN_TO_INTERNAL = {
    "sector": "sector",
    "code": "stock_code_raw",
    "stock_code": "stock_code_raw",
    "name": "stock_name",
    "stock_name": "stock_name",
}


def read_one_file(file_path, as_of_year):
    raw = pd.read_csv(file_path, dtype=str)
    table = rename_columns_case_insensitive(raw, NORMALISED_COLUMN_TO_INTERNAL)
    table["stock_code"] = normalise_code(table["stock_code_raw"])
    table["sector"] = table["sector"].astype("string").str.strip()
    table["stock_name"] = table["stock_name"].astype("string").str.strip()
    table["as_of_year"] = as_of_year
    return table[["stock_code", "sector", "stock_name", "as_of_year"]]


def concatenate_all_years(sector_files):
    tables = []
    for file_path, as_of_year in sector_files:
        tables.append(read_one_file(file_path, as_of_year))
    combined = pd.concat(tables, ignore_index=True)
    log_rule(TABLE, "concatenate_all_years", 0, len(combined), note=f"{len(sector_files)} source files")
    return combined


def drop_malformed_rows(combined):
    """Removes two different things, both caught by the same rule (every real holding has a
    company name, so no-name rows are never real holdings):
    - 2 trailing fully-blank rows at the end of the 2013 file, 2 more at the end of the 2020
      file (4 rows total, found by reading the raw files directly).
    - 3 rows in the 2023-2024 file where a new section's label (e.g. 'Energy and Petroleum')
      has been misplaced into the Stock_code column, Stock_name left empty, Sector still
      showing the previous section -- a source authoring error.
    7 rows removed in total (confirmed against data/clean/rule_log.csv after a run)."""
    rows_in = len(combined)
    is_malformed = combined["stock_name"].isna() | (combined["stock_name"].str.strip() == "")
    dropped_codes = sorted(combined.loc[is_malformed, "stock_code"].dropna().unique())
    kept = combined.loc[~is_malformed].copy()
    log_rule(TABLE, "drop_malformed_rows", rows_in, len(kept), note=f"{int(is_malformed.sum())} rows removed; non-blank dropped codes: {dropped_codes}")
    return kept


def drop_index_codes(combined):
    """Same universe decision as clean_prices.py: exclude market indices, keep only individual
    listed equities (and preference-share lines like KPLC-P4/KPLC-P7, which are real instruments)."""
    rows_in = len(combined)
    is_index = combined["stock_code"].str.startswith("^")
    index_codes = sorted(combined.loc[is_index, "stock_code"].unique())
    kept = combined.loc[~is_index].copy()
    log_rule(TABLE, "drop_index_codes", rows_in, len(kept), note=f"removed {index_codes}")
    return kept


def keep_most_recent_per_code(combined):
    """One row per code: the one from the file with the largest as_of_year."""
    rows_in = len(combined)
    ordered = combined.sort_values(["stock_code", "as_of_year"], ascending=[True, False])
    canonical = ordered.drop_duplicates(subset=["stock_code"], keep="first").copy()
    log_rule(TABLE, "keep_most_recent_per_code", rows_in, len(canonical),
              note=f"{canonical['stock_code'].nunique()} distinct stock codes across all years")
    return canonical


def count_sector_changes(combined):
    """How many codes were filed under more than one sector across the years -- informational
    only, the canonical table above already picked the most recent one."""
    sectors_per_code = combined.groupby("stock_code")["sector"].nunique()
    changed = int((sectors_per_code > 1).sum())
    log_rule(TABLE, "count_sector_changes", len(combined), len(combined),
              note=f"{changed} of {len(sectors_per_code)} codes were filed under more than one sector across the years (kept as most recent)")


def clean_sectors():
    sector_files = find_sector_files()
    combined = concatenate_all_years(sector_files)
    combined = drop_malformed_rows(combined)
    combined = drop_index_codes(combined)
    count_sector_changes(combined)
    canonical = keep_most_recent_per_code(combined)
    return canonical[["stock_code", "sector", "stock_name", "as_of_year"]].sort_values("stock_code").reset_index(drop=True)
