import os
import json
import yaml
import run_experiment as rexp


def test_run_creates_required_artifacts(
    base_regression_config, write_yaml, patch_dataset_loader, freeze_time
):
    cfg_path = write_yaml(base_regression_config, "reg.yaml")
    run_dir = rexp.run_experiment(cfg_path, dataset_path="IGNORED.csv")

    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    assert os.path.isfile(os.path.join(run_dir, "metrics.json"))
    assert os.path.isfile(os.path.join(run_dir, "model.joblib"))

    # plot may or may not be saved depending on config; in this suite it's off.
    # Ensure the run doesn't crash when plots disabled.
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    # Required fields
    for key in ["experiment_name", "seed", "model_type", "target_column", "target_type", "cv_results"]:
        assert key in metrics

    # Saved config must contain the same target
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        saved_cfg = yaml.safe_load(f)
    assert saved_cfg["data"]["target_column"] == base_regression_config["data"]["target_column"]

