"""Stage 4: run every numbered .sql file in this folder, in order, against the DuckDB file.

Each file's text is rendered through template.py first, substituting the {{TOKEN}} window
sizes and thresholds with settings.py's actual values -- so changing a window size in
settings.py changes what this stage computes, not just what its comments claim."""
from pathlib import Path

import duckdb

from template import render

SQL_DIR = Path(__file__).resolve().parent
DB_PATH = SQL_DIR.parent / "data" / "nse_liquidity.duckdb"


def main():
    connection = duckdb.connect(str(DB_PATH))
    sql_files = sorted(SQL_DIR.glob("0*.sql"))
    for sql_file in sql_files:
        print(f"running {sql_file.name}")
        connection.execute(render(sql_file.read_text(encoding="utf-8")))
    connection.close()


if __name__ == "__main__":
    main()
