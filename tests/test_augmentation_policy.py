import copy
import pytest
from experiments.config_schema import validate_baseline_config, ConfigValidationError


def test_baseline_config_rejects_augmentation_enabled(base_regression_config):
    """
    Baseline config MUST have augmentation disabled.
    If enabled, should raise ConfigValidationError (not silently disable).
    """
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["enabled"] = True

    with pytest.raises(ConfigValidationError, match="Augmentation is enabled"):
        validate_baseline_config(cfg)


def test_baseline_runner_requires_augmentation_disabled(
    base_regression_config, write_yaml, patch_dataset_loader
):
    """
    run_baseline should fail if augmentation is enabled in config.
    """
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["enabled"] = True

    cfg_path = write_yaml(cfg, "aug_enabled.yaml")

    import run_baseline
    with pytest.raises(ConfigValidationError, match="Augmentation is enabled"):
        run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")


def test_baseline_with_disabled_augmentation_succeeds(
    base_regression_config, write_yaml, patch_dataset_loader
):
    """
    Baseline should succeed when augmentation is properly disabled.
    """
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["enabled"] = False

    cfg_path = write_yaml(cfg, "baseline_correct.yaml")

    import run_baseline
    run_dir = run_baseline.run_baseline(cfg_path, dataset_path="IGNORED.csv")

    import os
    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "metrics.json"))

