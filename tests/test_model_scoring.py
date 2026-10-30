"""Unit tests for the flag-matching and scoring helpers in model/train_model.py."""
import numpy as np
import pandas as pd

from train_model import flag_top_n, score_flags


def test_flag_top_n_picks_exactly_n_highest_scores():
    probability = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
    flags = flag_top_n(probability, 2)
    assert flags.sum() == 2
    assert flags[1] and flags[3]
    assert not flags[0] and not flags[2] and not flags[4]


def test_flag_top_n_of_zero_flags_nothing():
    probability = np.array([0.1, 0.9, 0.4])
    flags = flag_top_n(probability, 0)
    assert flags.sum() == 0


def test_score_flags_precision_and_recall():
    actual = pd.Series([1, 1, 0, 1, 0])
    flagged = np.array([True, False, True, True, False])
    # true positives: index 0 and 3 (actual=1, flagged=True) = 2
    # predicted positive: index 0, 2, 3 = 3 -> precision 2/3
    # actual positive: index 0,1,3 = 3 -> recall 2/3
    result = score_flags(actual, flagged)
    assert result["flagged"] == 3
    assert abs(result["precision"] - 2 / 3) < 1e-9
    assert abs(result["recall"] - 2 / 3) < 1e-9


def test_score_flags_handles_no_flags_without_division_error():
    actual = pd.Series([1, 0, 1])
    flagged = np.array([False, False, False])
    result = score_flags(actual, flagged)
    assert result["flagged"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
