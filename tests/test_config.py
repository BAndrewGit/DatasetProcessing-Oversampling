import pytest
import copy
from experiments.config_schema import (
    validate_config,
    validate_baseline_config,
    ConfigValidationError,
    FORBIDDEN_TARGETS
)
from experiments.io import config_hash


def test_validate_target_rejects_forbidden(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["data"]["target_column"] = "Behavior_Risk_Level"
    with pytest.raises(ConfigValidationError, match="FORBIDDEN target"):
        validate_config(cfg)


def test_baseline_config_rejects_augmentation_enabled(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["enabled"] = True
    with pytest.raises(ConfigValidationError, match="Augmentation is enabled"):
        validate_baseline_config(cfg)


def test_config_rejects_forbidden_keys(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["augmentation"]["max_size"] = 10000  # Forbidden key
    with pytest.raises(ConfigValidationError, match="FORBIDDEN keys"):
        validate_config(cfg)


def test_config_rejects_invalid_model_type(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["model"]["type"] = "invalid_model"
    with pytest.raises(ConfigValidationError, match="Invalid model type"):
        validate_config(cfg)


def test_config_rejects_invalid_target_type(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["data"]["target_type"] = "invalid_type"
    with pytest.raises(ConfigValidationError, match="Invalid target_type"):
        validate_config(cfg)


def test_config_hash_deterministic(base_regression_config):
    h1 = config_hash(base_regression_config)
    h2 = config_hash(base_regression_config)
    assert h1 == h2

    # Same content but different key insertion order should still match
    cfg2 = {
        "data": base_regression_config["data"],
        "experiment": base_regression_config["experiment"],
        "model": base_regression_config["model"],
        "preprocessing": base_regression_config["preprocessing"],
        "cross_validation": base_regression_config["cross_validation"],
        "augmentation": base_regression_config["augmentation"],
        "metrics": base_regression_config["metrics"],
    }
    assert config_hash(cfg2) == h1


def test_config_validates_required_keys():
    incomplete_config = {
        "experiment": {"name": "test"}
        # Missing seed, data, model, etc.
    }
    with pytest.raises(ConfigValidationError, match="Missing required"):
        validate_config(incomplete_config)

