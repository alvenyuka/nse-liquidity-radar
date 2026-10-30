"""Stage 5: run every checks/*.sql file. Each file is a query that must return zero rows.
Exits with status 1 if any check returns rows.

Each file's text is rendered through sql/template.py first (same substitution the SQL-model
stage uses), so a check like "dates in range" reads its bounds from settings.py too."""
import sys
from pathlib import Path

import duckdb

CHECKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHECKS_DIR.parent / "sql"))
from template import render  # noqa: E402

DEFAULT_DB_PATH = CHECKS_DIR.parent / "data" / "nse_liquidity.duckdb"


def run_checks(db_path, checks_dir=CHECKS_DIR):
    """Return a list of (check name, number of offending rows)."""
    connection = duckdb.connect(str(db_path), read_only=True)
    results = []
    for check_file in sorted(Path(checks_dir).glob("*.sql")):
        offending_rows = connection.execute(render(check_file.read_text(encoding="utf-8"))).fetchall()
        results.append((check_file.name, len(offending_rows)))
    connection.close()
    return results


def main():
    db_path = DEFAULT_DB_PATH
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])

    results = run_checks(db_path)
    failed = 0
    for check_name, offending in results:
        if offending == 0:
            print(f"PASS  {check_name}")
        else:
            print(f"FAIL  {check_name}: {offending} offending rows")
            failed = failed + 1

    print(f"{len(results) - failed} of {len(results)} checks passed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
