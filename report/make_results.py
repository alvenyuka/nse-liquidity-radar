"""Stage 8: collect every headline figure into outputs/results.json.

The README and the site quote only numbers from this file, so a number can never drift
from what the pipeline actually produced.
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rounding import round_numbers  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "nse_liquidity.duckdb"
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Illustrative position sizes (KES) for the "days to liquidate" line on the watchlist: how
# long it would take to exit a position of this size, participating at PARTICIPATION_RATE
# of a stock's own trailing 20-day average KES value traded, without being the only buyer
# or seller that day. A standard execution rule of thumb, not fit to this data.
ILLUSTRATIVE_POSITIONS_KES = [1_000_000, 5_000_000, 10_000_000]
PARTICIPATION_RATE = 0.20


def table_to_records(table):
    return json.loads(table.to_json(orient="records", date_format="iso"))


def first_record(connection, sql):
    return table_to_records(connection.execute(sql).df())[0]


def main():
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    results = {"licence": manifest["licence"], "attribution": manifest["attribution"]}

    # 1. What was downloaded and kept
    results["datasets"] = manifest["datasets"]
    results["checks_run"] = len(list((PROJECT_DIR / "checks").glob("*.sql")))
    rule_log = pd.read_csv(CLEAN_DIR / "rule_log.csv", keep_default_na=False)
    results["cleaning_rules"] = table_to_records(rule_log)

    # 2. The universe
    results["universe"] = first_record(connection,
        """
        SELECT
            COUNT(DISTINCT stock_code) AS stock_codes,
            MIN(trade_date) AS first_date,
            MAX(trade_date) AS last_date,
            (SELECT COUNT(*) FROM sessions) AS total_sessions
        FROM raw_prices
        """)
    results["sectors"] = table_to_records(connection.execute(
        "SELECT sector, COUNT(*) AS stock_codes FROM raw_sectors GROUP BY sector ORDER BY stock_codes DESC").df())

    # 3. The coverage finding that drove the dense-grid design: how much of the grid is a
    # real observed row versus a session the stock simply was not recorded for.
    results["coverage"] = first_record(connection,
        """
        SELECT
            COUNT(*) AS grid_rows,
            SUM(CASE WHEN volume IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_a_volume_figure,
            SUM(CASE WHEN close_price IS NULL THEN 1 ELSE 0 END) AS rows_absent_from_source_entirely,
            SUM(CASE WHEN volume IS NULL AND close_price IS NOT NULL THEN 1 ELSE 0 END) AS rows_present_but_marked_not_traded,
            AVG(CASE WHEN is_untraded THEN 1.0 ELSE 0 END) AS overall_untraded_share
        FROM features_final
        """)
    results["most_incomplete_stocks"] = table_to_records(connection.execute(
        """
        WITH counted AS (
            SELECT b.stock_code,
                   b.first_date, b.last_date,
                   (SELECT COUNT(*) FROM session_grid g WHERE g.stock_code = b.stock_code) AS sessions_in_bounds,
                   (SELECT COUNT(*) FROM raw_prices p WHERE p.stock_code = b.stock_code) AS sessions_actually_recorded
            FROM stock_bounds AS b
        )
        SELECT *, sessions_in_bounds - sessions_actually_recorded AS sessions_missing
        FROM counted
        ORDER BY sessions_missing DESC
        LIMIT 5
        """).df())

    # 4. Label distribution: the full population versus the harder transition subset.
    results["label_distribution"] = first_record(connection,
        """
        SELECT
            COUNT(*) AS rows_with_a_label,
            AVG(is_illiquid_next_5d) AS base_rate_full,
            AVG(CASE WHEN currently_liquid = 1 THEN is_illiquid_next_5d END) AS base_rate_transition,
            SUM(CASE WHEN currently_liquid = 1 THEN 1 ELSE 0 END) AS currently_liquid_rows
        FROM mart_features_labels WHERE is_illiquid_next_5d IS NOT NULL
        """)

    # 5. The live watchlist: the most recent reading for every stock, ranked by how much of
    # the last 20 sessions went untraded, with an illustrative days-to-liquidate column.
    watchlist = connection.execute(
        """
        SELECT stock_code, stock_name, sector, trade_date, avg_volume_20d, close_price,
               zeros_share_20d, volume_zscore_60d, trailing_illiquid_5
        FROM mart_latest_watchlist
        """
    ).df()
    watchlist["avg_kes_value_traded_20d"] = watchlist["avg_volume_20d"] * watchlist["close_price"]
    for position_kes in ILLUSTRATIVE_POSITIONS_KES:
        column = f"days_to_liquidate_{position_kes // 1_000_000}m"
        daily_capacity = watchlist["avg_kes_value_traded_20d"] * PARTICIPATION_RATE
        watchlist[column] = (position_kes / daily_capacity).where(daily_capacity > 0)
    results["days_to_liquidate_assumptions"] = {
        "participation_rate": PARTICIPATION_RATE,
        "illustrative_positions_kes": ILLUSTRATIVE_POSITIONS_KES,
        "method": "position_kes / (participation_rate * 20-day average KES value traded)",
    }
    results["watchlist"] = table_to_records(watchlist.sort_values("zeros_share_20d", ascending=False, na_position="last"))

    connection.close()

    # 6. The baseline and the model
    results["baseline"] = json.loads((OUTPUT_DIR / "baseline_results.json").read_text())
    results["model"] = json.loads((OUTPUT_DIR / "model_results.json").read_text())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(round_numbers(results), indent=2))
    print(f"wrote {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
