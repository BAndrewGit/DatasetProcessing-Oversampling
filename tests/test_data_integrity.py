import pytest
import copy
import run_experiment as rexp


def test_preprocess_excludes_target_and_ignored(tiny_adv_df, base_regression_config):
    X, y = rexp.preprocess_data(tiny_adv_df, base_regression_config)
    assert "Risk_Score" not in X.columns
    assert "Behavior_Risk_Level" not in X.columns
    assert len(X) == len(y)
    assert y.name == "Risk_Score"


def test_preprocess_blocks_forbidden_column_in_features(tiny_adv_df, base_regression_config):
    # If user forgets to ignore forbidden column, code must block it
    cfg = copy.deepcopy(base_regression_config)
    cfg["preprocessing"]["ignored_columns"] = []  # remove ignore list

    with pytest.raises(ValueError, match="Forbidden column"):
        rexp.preprocess_data(tiny_adv_df, cfg)


def test_missing_target_raises(tiny_adv_df, base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["data"]["target_column"] = "Nonexistent_Target"
    with pytest.raises(ValueError, match="not found"):
        rexp.preprocess_data(tiny_adv_df, cfg)


def test_savings_mutual_exclusivity(tiny_adv_df):
    # Enforce that Save_Money_Yes and Save_Money_No are mutually exclusive
    both_one = (tiny_adv_df["Save_Money_Yes"] == 1) & (tiny_adv_df["Save_Money_No"] == 1)
    both_zero = (tiny_adv_df["Save_Money_Yes"] == 0) & (tiny_adv_df["Save_Money_No"] == 0)
    assert both_one.sum() == 0
    assert both_zero.sum() == 0

