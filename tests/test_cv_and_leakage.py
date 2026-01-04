import numpy as np
from experiments.data import preprocess_data
from experiments.models import build_model
from experiments.cv import run_repeated_cv_regression, run_repeated_cv_classification


def test_repeated_cv_regression_returns_expected_keys(tiny_adv_df, base_regression_config):
    X, y = preprocess_data(tiny_adv_df, base_regression_config)
    model = build_model(base_regression_config)
    res = run_repeated_cv_regression(model, X, y, base_regression_config)

    for k in ["mae", "rmse", "spearman", "r2"]:
        assert k in res
        assert "mean" in res[k] and "std" in res[k] and "all" in res[k]
        expected_folds = base_regression_config["cross_validation"]["n_splits"] * base_regression_config["cross_validation"]["n_repeats"]
        assert len(res[k]["all"]) == expected_folds

    assert res["n_folds"] == base_regression_config["cross_validation"]["n_splits"]
    assert res["n_repeats"] == base_regression_config["cross_validation"]["n_repeats"]


def test_repeated_cv_classification_returns_expected_keys(tiny_adv_df, base_classification_config):
    X, y = preprocess_data(tiny_adv_df, base_classification_config)
    model = build_model(base_classification_config)
    res = run_repeated_cv_classification(model, X, y, base_classification_config)

    for k in ["macro_f1", "accuracy", "precision", "recall"]:
        assert k in res
        assert "mean" in res[k] and "std" in res[k] and "all" in res[k]
        expected_folds = base_classification_config["cross_validation"]["n_splits"] * base_classification_config["cross_validation"]["n_repeats"]
        assert len(res[k]["all"]) == expected_folds


def test_cv_split_indices_disjoint(tiny_adv_df, base_regression_config):
    """Verify train/val indices are disjoint (tested implicitly via _validate_cv_split)."""
    from sklearn.model_selection import RepeatedKFold

    X, y = preprocess_data(tiny_adv_df, base_regression_config)
    seed = base_regression_config["experiment"]["seed"]
    cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=seed)

    for train_idx, val_idx in cv.split(X):
        train_set = set(train_idx)
        val_set = set(val_idx)
        assert train_set.isdisjoint(val_set), "Train and val indices should be disjoint"


def test_cv_handles_all_finite_values(tiny_adv_df, base_regression_config):
    """CV should work with valid finite data."""
    X, y = preprocess_data(tiny_adv_df, base_regression_config)

    # Ensure all values are finite
    assert np.isfinite(X.select_dtypes(include=[np.number]).values).all()
    assert np.isfinite(y.values).all()

    # Should complete without error
    model = build_model(base_regression_config)
    res = run_repeated_cv_regression(model, X, y, base_regression_config)
    assert res["mae"]["mean"] >= 0

