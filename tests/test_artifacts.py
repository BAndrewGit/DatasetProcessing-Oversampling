import os
import json
import yaml
import run_baseline


def test_run_creates_required_artifacts(
    base_regression_config, write_yaml, patch_dataset_loader
):
    cfg_path = write_yaml(base_regression_config, "reg.yaml")
    run_dir = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")

    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    assert os.path.isfile(os.path.join(run_dir, "metrics.json"))
    assert os.path.isfile(os.path.join(run_dir, "model.joblib"))
    assert os.path.isfile(os.path.join(run_dir, "data_profile.json"))  # New: dataset fingerprint

    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    # Required fields
    for key in ["experiment_name", "seed", "model_type", "target_column", "target_type", "cv_results"]:
        assert key in metrics, f"Missing key in metrics.json: {key}"

    # Saved config must contain the same target
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        saved_cfg = yaml.safe_load(f)
    assert saved_cfg["data"]["target_column"] == base_regression_config["data"]["target_column"]


def test_data_profile_contains_required_info(
    base_regression_config, write_yaml, patch_dataset_loader
):
    """Test that data_profile.json contains dataset fingerprinting info."""
    cfg_path = write_yaml(base_regression_config, "profile_test.yaml")
    run_dir = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")

    profile_path = os.path.join(run_dir, "data_profile.json")
    assert os.path.isfile(profile_path)

    with open(profile_path, "r") as f:
        profile = json.load(f)

    # Required fields in data profile
    required_keys = [
        "dataset_path",
        "dataset_hash",
        "total_rows",
        "total_columns",
        "feature_count",
        "features_used",
        "target_column",
        "target_stats",
        "timestamp"
    ]
    for key in required_keys:
        assert key in profile, f"Missing key in data_profile.json: {key}"

    assert isinstance(profile["features_used"], list)
    assert profile["feature_count"] == len(profile["features_used"])


def test_metrics_include_fold_level_scores(
    base_regression_config, write_yaml, patch_dataset_loader
):
    """Test that metrics.json includes fold-level scores for CI/boxplots."""
    cfg_path = write_yaml(base_regression_config, "fold_test.yaml")
    run_dir = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")

    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    # Each metric should have 'all' key with fold-level scores
    for metric in ["mae", "rmse", "spearman", "r2"]:
        assert "all" in metrics["cv_results"][metric]
        assert isinstance(metrics["cv_results"][metric]["all"], list)
        assert len(metrics["cv_results"][metric]["all"]) > 0

