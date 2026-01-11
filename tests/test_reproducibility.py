import json
import os
import copy
import numpy as np
from runners import run_baseline


def _load_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        return json.load(f)


def test_same_seed_same_results_regression(
    base_regression_config, write_yaml, patch_dataset_loader
):
    cfg = copy.deepcopy(base_regression_config)
    cfg["model"]["type"] = "ridge"
    cfg_path = write_yaml(cfg, "reg.yaml")

    run1 = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")
    run2 = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")

    m1 = _load_metrics(run1)
    m2 = _load_metrics(run2)

    # With fixed data + fixed seed + deterministic model, these should match
    for metric in ["mae", "rmse", "spearman", "r2"]:
        a = m1["cv_results"][metric]["mean"]
        b = m2["cv_results"][metric]["mean"]
        assert abs(a - b) < 1e-12, f"{metric} differs between runs"


def test_different_seed_changes_cv_split(
    base_regression_config, write_yaml, patch_dataset_loader
):
    cfg1 = copy.deepcopy(base_regression_config)
    cfg2 = copy.deepcopy(base_regression_config)
    cfg2["experiment"]["seed"] = cfg1["experiment"]["seed"] + 1
    cfg2["experiment"]["name"] = "pytest_regression_seed2"

    p1 = write_yaml(cfg1, "s1.yaml")
    p2 = write_yaml(cfg2, "s2.yaml")

    r1 = run_baseline.run_baseline(p1, dataset_path="IGNORED.csv")
    r2 = run_baseline.run_baseline(p2, dataset_path="IGNORED.csv")

    m1 = _load_metrics(r1)
    m2 = _load_metrics(r2)

    # Different seeds should produce different results
    diffs = []
    for metric in ["mae", "rmse"]:
        diffs.append(abs(m1["cv_results"][metric]["mean"] - m2["cv_results"][metric]["mean"]))
    assert any(d > 1e-8 for d in diffs), "Different seeds should produce different CV splits"


def test_fold_scores_saved_in_metrics(
    base_regression_config, write_yaml, patch_dataset_loader
):
    """Ensure fold-level scores are saved in metrics.json for CI/boxplots."""
    cfg = copy.deepcopy(base_regression_config)
    cfg_path = write_yaml(cfg, "fold_scores.yaml")

    run_dir = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")
    metrics = _load_metrics(run_dir)

    for metric in ["mae", "rmse", "spearman", "r2"]:
        assert "all" in metrics["cv_results"][metric], f"Fold scores missing for {metric}"
        expected_folds = cfg["cross_validation"]["n_splits"] * cfg["cross_validation"]["n_repeats"]
        assert len(metrics["cv_results"][metric]["all"]) == expected_folds

