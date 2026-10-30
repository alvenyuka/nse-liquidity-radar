"""Substitutes settings.py's constants into a .sql file's text before it runs, so editing a
window size or a valid-date bound in settings.py actually changes what the SQL computes --
found missing in review: the SQL used to hardcode 19/59/4/1..5 and the date bounds as plain
literals, so a reader who changed settings.py and expected different numbers out would have
seen nothing change.

Tokens look like {{TOKEN_NAME}}, chosen because plain SQL never contains curly braces, so a
token can never collide with real SQL text by accident.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from settings import EARLIEST_VALID_DATE, EMBARGO_DAYS, FORWARD_WINDOW_DAYS, LONG_WINDOW_DAYS, LATEST_VALID_DATE, SHORT_WINDOW_DAYS  # noqa: E402


def template_values():
    """One place computing every derived number the .sql files need, from the same settings
    every Python stage reads."""
    return {
        "SHORT_WINDOW_DAYS": SHORT_WINDOW_DAYS,
        "LONG_WINDOW_DAYS": LONG_WINDOW_DAYS,
        "SHORT_WINDOW_MINUS_1": SHORT_WINDOW_DAYS - 1,  # a 20-day window's ROWS BETWEEN n PRECEDING
        "LONG_WINDOW_MINUS_1": LONG_WINDOW_DAYS - 1,
        # The trailing "currently illiquid" window looks back the same number of sessions the
        # forward label looks ahead, by design (sql/04_labels.sql explains why).
        "TRAILING_WINDOW_MINUS_1": FORWARD_WINDOW_DAYS - 1,
        "FORWARD_WINDOW_DAYS": FORWARD_WINDOW_DAYS,
        "EMBARGO_DAYS": EMBARGO_DAYS,
        "EARLIEST_VALID_DATE": EARLIEST_VALID_DATE.isoformat(),
        "LATEST_VALID_DATE": LATEST_VALID_DATE.isoformat(),
    }


def render(sql_text):
    """Replace every {{TOKEN}} in sql_text with its value. Raises if a token in the text has
    no value, so a typo in a .sql file fails loudly instead of running unsubstituted."""
    values = template_values()
    rendered = sql_text
    for token_name, value in values.items():
        rendered = rendered.replace("{{" + token_name + "}}", str(value))
    if "{{" in rendered:
        unresolved_start = rendered.index("{{")
        raise ValueError(f"unresolved template token near: {rendered[unresolved_start:unresolved_start + 40]!r}")
    return rendered
