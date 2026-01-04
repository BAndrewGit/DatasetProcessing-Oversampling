import pytest
import copy
import run_experiment as rexp


def test_validate_target_rejects_forbidden(base_regression_config):
    cfg = copy.deepcopy(base_regression_config)
    cfg["data"]["target_column"] = "Behavior_Risk_Level"
    with pytest.raises(ValueError, match="forbidden target"):
        rexp.validate_target(cfg)


def test_config_hash_deterministic(base_regression_config):
    h1 = rexp.config_hash(base_regression_config)
    h2 = rexp.config_hash(base_regression_config)
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
    assert rexp.config_hash(cfg2) == h1

