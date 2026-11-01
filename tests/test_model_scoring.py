"""Unit tests for the flag-matching and scoring helpers in model/train_model.py."""
import numpy as np
import pandas as pd

from settings import FEATURE_COLUMNS
from train_model import build_model, flag_top_n, permutation_importance_by_feature, score_flags


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


def test_permutation_importance_by_feature_ranks_the_real_signal_highest():
    # a label that depends only on avg_volume_20d, with the other three features pure noise --
    # permuting avg_volume_20d should hurt PR-AUC far more than permuting any of the others.
    rng = np.random.default_rng(0)
    rows = 400
    population = pd.DataFrame({
        "avg_volume_20d": rng.normal(size=rows),
        "volatility_20d": rng.normal(size=rows),
        "zeros_share_20d": rng.normal(size=rows),
        "volume_zscore_60d": rng.normal(size=rows),
    })
    population["is_illiquid_next_5d"] = (population["avg_volume_20d"] > 0).astype(int)

    model = build_model()
    model.fit(population[FEATURE_COLUMNS], population["is_illiquid_next_5d"])
    importances = permutation_importance_by_feature(model, population)

    assert set(importances.keys()) == set(FEATURE_COLUMNS)
    for noise_feature in ["volatility_20d", "zeros_share_20d", "volume_zscore_60d"]:
        assert importances["avg_volume_20d"] > importances[noise_feature]
