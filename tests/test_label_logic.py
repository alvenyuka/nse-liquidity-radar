"""The one test that matters most in this project: does sql/04_labels.sql's forward window
actually mean "the next 5 sessions", and the trailing window actually mean "the last 5
sessions including today" -- on a tiny, hand-built sequence where the right answer is worked
out by hand first, not inferred from the SQL itself.

Sequence for one stock, 14 sessions, is_untraded = 0 except sessions 9 (the only "1"):
    session_no:   1  2  3  4  5  6  7  8  9  10 11 12 13 14
    is_untraded:  0  0  0  0  0  0  0  0  1  0  0  0  0  0

Expected trailing_illiquid_5 (today + 4 preceding; NULL until there are 5 rows):
    session 1-4:  NULL (fewer than 5 rows so far)
    session 5-8:  0    (the window [1..5] .. [4..8] never reaches session 9)
    session 9-13: 1    (the window [5..9] .. [9..13] always includes session 9)
    session 14:   0    (the window [10..14] has left session 9 behind)

Expected is_illiquid_next_5d (the next 1..5 sessions; NULL on the last 5, no full future):
    session 1-3:  0    (the window [2..6] .. [4..8] never reaches session 9)
    session 4-8:  1    (the window [5..9] .. [8..12] always includes session 9)
    session 9:    0    (the window [10..14] has left session 9 behind)
    session 10-14: NULL (fewer than 5 sessions remain)
"""
import sys
from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "sql"))
from template import render  # noqa: E402

# Rendered the same way sql/run_sql.py renders it, so this test runs the exact text the
# pipeline actually executes, not a hand-copied approximation of it.
LABELS_SQL = render((PROJECT_DIR / "sql" / "04_labels.sql").read_text(encoding="utf-8"))

EXPECTED_TRAILING = {1: None, 2: None, 3: None, 4: None,
                     5: 0, 6: 0, 7: 0, 8: 0,
                     9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
                     14: 0}
EXPECTED_FORWARD = {1: 0, 2: 0, 3: 0,
                    4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
                    9: 0,
                    10: None, 11: None, 12: None, 13: None, 14: None}


def build_labels_table():
    """A fresh in-memory DuckDB with a 14-row features_final for one stock, then the real
    sql/04_labels.sql run against it unmodified -- the same text the pipeline executes."""
    connection = duckdb.connect(":memory:")
    connection.execute("""
        CREATE TABLE features_final (stock_code VARCHAR, session_no INTEGER, is_untraded INTEGER)
    """)
    untraded_session = 9
    for session_no in range(1, 15):
        is_untraded = 1 if session_no == untraded_session else 0
        connection.execute(
            "INSERT INTO features_final VALUES ('TEST', ?, ?)", [session_no, is_untraded]
        )
    connection.execute(LABELS_SQL)
    return connection


def test_trailing_illiquid_5_matches_hand_worked_example():
    connection = build_labels_table()
    rows = connection.execute(
        "SELECT session_no, trailing_illiquid_5 FROM labels ORDER BY session_no"
    ).fetchall()
    connection.close()
    actual = {session_no: value for session_no, value in rows}
    assert actual == EXPECTED_TRAILING


def test_forward_label_matches_hand_worked_example():
    connection = build_labels_table()
    rows = connection.execute(
        "SELECT session_no, is_illiquid_next_5d FROM labels ORDER BY session_no"
    ).fetchall()
    connection.close()
    actual = {session_no: value for session_no, value in rows}
    assert actual == EXPECTED_FORWARD


def test_currently_liquid_is_the_opposite_of_trailing_illiquid_5():
    connection = build_labels_table()
    rows = connection.execute(
        "SELECT trailing_illiquid_5, currently_liquid FROM labels WHERE trailing_illiquid_5 IS NOT NULL"
    ).fetchall()
    connection.close()
    for trailing_illiquid_5, currently_liquid in rows:
        assert currently_liquid == 1 - trailing_illiquid_5
