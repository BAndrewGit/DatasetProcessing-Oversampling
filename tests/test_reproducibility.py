import json
import os
import copy
import numpy as np
import run_experiment as rexp


def _load_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        return json.load(f)


def test_same_seed_same_results_regression(
    base_regression_config, write_yaml, patch_dataset_loader, freeze_time
):
    cfg = copy.deepcopy(base_regression_config)
    cfg["model"]["type"] = "ridge"
    cfg_path = write_yaml(cfg, "reg.yaml")

    run1 = rexp.run_experiment(cfg_path, dataset_path="IGNORED.csv")
    run2 = rexp.run_experiment(cfg_path, dataset_path="IGNORED.csv")

    m1 = _load_metrics(run1)
    m2 = _load_metrics(run2)

    # With fixed data + fixed seed + deterministic model, these should match extremely closely.
    for metric in ["mae", "rmse", "spearman", "r2"]:
        a = m1["cv_results"][metric]["mean"]
        b = m2["cv_results"][metric]["mean"]
        assert abs(a - b) < 1e-12


def test_different_seed_changes_cv_split(
    base_regression_config, write_yaml, patch_dataset_loader, freeze_time
):
    cfg1 = copy.deepcopy(base_regression_config)
    cfg2 = copy.deepcopy(base_regression_config)
    cfg2["experiment"]["seed"] = cfg1["experiment"]["seed"] + 1

    p1 = write_yaml(cfg1, "s1.yaml")
    p2 = write_yaml(cfg2, "s2.yaml")

    r1 = rexp.run_experiment(p1, dataset_path="IGNORED.csv")
    r2 = rexp.run_experiment(p2, dataset_path="IGNORED.csv")

    m1 = _load_metrics(r1)
    m2 = _load_metrics(r2)

    # Not guaranteed to always differ, but highly likely. Use a weak assertion:
    # at least one primary metric should differ beyond tiny tolerance.
    diffs = []
    for metric in ["mae", "rmse"]:
        diffs.append(abs(m1["cv_results"][metric]["mean"] - m2["cv_results"][metric]["mean"]))
    assert any(d > 1e-8 for d in diffs)

