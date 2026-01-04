import copy
import run_experiment as rexp


def test_baseline_forces_augmentation_off(
    base_regression_config, write_yaml, patch_dataset_loader, freeze_time
):
    """
    Your run_experiment explicitly disables augmentation if enabled.
    This test guarantees you never silently run augmented baselines.
    """
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["enabled"] = True  # should be force-disabled

    cfg_path = write_yaml(cfg, "aug_enabled.yaml")
    run_dir = rexp.run_experiment(cfg_path, dataset_path="IGNORED.csv")

    # Ensure config saved has augmentation disabled
    import yaml, os
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        saved = yaml.safe_load(f)
    assert saved.get("augmentation", {}).get("enabled", False) is False

