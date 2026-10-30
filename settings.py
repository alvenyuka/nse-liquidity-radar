"""Constants shared by every stage, so a number is never typed twice and checked against itself.

Decisions recorded here (see docs/METHODOLOGY.md for the reasoning):
- Illiquid day = zero traded volume. A session a stock does not appear in at all is also
  untraded: the dense session grid (sql/02_session_grid.sql) LEFT JOINs onto it, so a missing
  row's volume comes through as SQL NULL, and sql/03_features.sql's is_untraded flag treats
  NULL the same as volume <= 0. How often this actually happens, per stock, is reported in
  outputs/results.json's "coverage" and "most_incomplete_stocks" sections (not a separate
  check file -- it is a fact to report, not a pass/fail assertion).
- The forward label looks at the next FORWARD_WINDOW_DAYS sessions, never the current one.
- Rolling features use the trailing SHORT_WINDOW_DAYS / LONG_WINDOW_DAYS sessions including today,
  because today's close is known at the moment the signal would be used (end of day t, acted on at
  the open of t+1 -- see docs/METHODOLOGY.md).
- A 5-session gap sits between the end of training and the start of testing in every split, so no
  training row's forward label reaches into the test period.
"""
from datetime import date

FIRST_YEAR = 2013
LAST_FILE_MONTH = (2024, 10)  # the 2024 file stops at the end of October

SHORT_WINDOW_DAYS = 20
LONG_WINDOW_DAYS = 60
FORWARD_WINDOW_DAYS = 5
EMBARGO_DAYS = FORWARD_WINDOW_DAYS

ILLIQUID_VOLUME_THRESHOLD = 0  # a session is illiquid if traded volume is this or less

# A plain time split: everything before this year trains, this year and after tests.
SPLIT_TEST_START_YEAR = 2023

# The walk-forward table re-tests one extra year at a time, starting once there is a full
# LONG_WINDOW_DAYS of history behind the first test year.
WALKFORWARD_FIRST_TEST_YEAR = 2016
WALKFORWARD_LAST_TEST_YEAR = 2024

EARLIEST_VALID_DATE = date(FIRST_YEAR, 1, 1)
LATEST_VALID_DATE = date(LAST_FILE_MONTH[0], LAST_FILE_MONTH[1], 31)

FEATURE_COLUMNS = ["avg_volume_20d", "volatility_20d", "zeros_share_20d", "volume_zscore_60d"]

# The rule-of-thumb baseline: flag a stock whose last 20 sessions were untraded at least
# half the time. Fixed in advance as an interpretable round number ("untraded half the
# month"), not fit to the data -- see docs/METHODOLOGY.md.
RULE_ZEROS_SHARE_THRESHOLD = 0.5
