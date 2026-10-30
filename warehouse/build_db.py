"""Stage 3: load the cleaned parquet files into a DuckDB file."""
from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
DB_PATH = PROJECT_DIR / "data" / "nse_liquidity.duckdb"

TABLE_FILES = {
    "raw_prices": "prices.parquet",
    "raw_sectors": "sectors.parquet",
}


def main():
    # Open in write mode and drop the old tables, so a re-run never mixes old and new rows.
    connection = duckdb.connect(str(DB_PATH))
    for table_name, file_name in TABLE_FILES.items():
        parquet_path = (CLEAN_DIR / file_name).as_posix()
        connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"{table_name}: {row_count} rows")
    connection.close()


if __name__ == "__main__":
    main()
