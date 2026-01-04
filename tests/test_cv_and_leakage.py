import numpy as np
import run_experiment as rexp


def test_repeated_cv_regression_returns_expected_keys(tiny_adv_df, base_regression_config):
    X, y = rexp.preprocess_data(tiny_adv_df, base_regression_config)
    model = rexp.build_model(base_regression_config)
    res = rexp.run_repeated_cv_regression(model, X, y, base_regression_config)

    for k in ["mae", "rmse", "spearman", "r2"]:
        assert k in res
        assert "mean" in res[k] and "std" in res[k] and "all" in res[k]
        assert len(res[k]["all"]) == base_regression_config["cross_validation"]["n_splits"] * base_regression_config["cross_validation"]["n_repeats"]

    assert res["n_folds"] == base_regression_config["cross_validation"]["n_splits"]
    assert res["n_repeats"] == base_regression_config["cross_validation"]["n_repeats"]


def test_repeated_cv_classification_returns_expected_keys(tiny_adv_df, base_classification_config):
    X, y = rexp.preprocess_data(tiny_adv_df, base_classification_config)
    model = rexp.build_model(base_classification_config)
    res = rexp.run_repeated_cv_classification(model, X, y, base_classification_config)

    for k in ["macro_f1", "accuracy", "precision", "recall"]:
        assert k in res
        assert "mean" in res[k] and "std" in res[k] and "all" in res[k]
        assert len(res[k]["all"]) == base_classification_config["cross_validation"]["n_splits"] * base_classification_config["cross_validation"]["n_repeats"]

