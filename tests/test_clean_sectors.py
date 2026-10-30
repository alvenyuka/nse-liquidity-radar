"""Unit tests for the sector-cleaning rules, on small hand-built frames that reproduce the
real malformed rows found in the 2023-2024 source file."""
import pandas as pd

from clean_sectors import drop_index_codes, drop_malformed_rows, keep_most_recent_per_code


def test_drop_malformed_rows_removes_only_rows_with_no_name():
    table = pd.DataFrame({
        "stock_code": ["KCB", "ENERGY AND PETROLEUM"],
        "sector": ["Banking", "Construction and Allied"],
        "stock_name": ["KCB Group Plc", None],
        "as_of_year": [2024, 2024],
    })
    kept = drop_malformed_rows(table)
    assert kept["stock_code"].tolist() == ["KCB"]


def test_drop_index_codes_removes_only_caret_prefixed_codes():
    table = pd.DataFrame({
        "stock_code": ["KCB", "^NASI"],
        "sector": ["Banking", "Indices"],
        "stock_name": ["KCB Group Plc", "NSE All-Share Index"],
        "as_of_year": [2024, 2024],
    })
    kept = drop_index_codes(table)
    assert kept["stock_code"].tolist() == ["KCB"]


def test_keep_most_recent_per_code_prefers_the_latest_year():
    table = pd.DataFrame({
        "stock_code": ["NCBA", "NCBA"],
        "sector": ["Banking", "Banking"],
        "stock_name": ["NIC Bank Ltd", "NCBA Group Plc"],  # renamed after a merger
        "as_of_year": [2013, 2022],
    })
    canonical = keep_most_recent_per_code(table)
    assert len(canonical) == 1
    assert canonical.iloc[0]["stock_name"] == "NCBA Group Plc"
