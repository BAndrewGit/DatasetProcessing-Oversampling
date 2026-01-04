import os
import json
import copy
import pytest
import pandas as pd
import numpy as np

# Import from modular experiments package
from experiments.config_schema import validate_config, validate_baseline_config, ConfigValidationError
from experiments.io import load_config, config_hash, dataset_hash, create_run_dir, save_results, save_data_profile
from experiments.data import load_dataset, preprocess_data, validate_data_integrity, validate_save_money_consistency
from experiments.models import build_model
from experiments.cv import run_repeated_cv_regression, run_repeated_cv_classification

# Import baseline runner
import run_baseline


@pytest.fixture(scope="session")
def seed():
    return 42


@pytest.fixture
def tiny_adv_df(seed):
    """
    Small deterministic dataframe resembling ADV style.
    Includes:
      - Risk_Score (regression target)
      - Save_Money_Yes / Save_Money_No (classification target)
      - Behavior_Risk_Level (forbidden column that must never enter features)
      - Some numeric and binary features
    """
    rng = np.random.default_rng(seed)
    n = 40

    df = pd.DataFrame({
        "Risk_Score": rng.normal(loc=0.0, scale=1.0, size=n),
        "Save_Money_Yes": rng.integers(0, 2, size=n),
    })
    df["Save_Money_No"] = 1 - df["Save_Money_Yes"]

    # Forbidden label column must never appear in X
    df["Behavior_Risk_Level"] = rng.integers(0, 2, size=n)

    # Features
    df["Debt_Level"] = rng.integers(0, 5, size=n)
    df["Impulse_Buying_Frequency"] = rng.integers(0, 5, size=n)
    df["Budget_Planning_Plan in detail"] = rng.integers(0, 2, size=n)
    df["Save_Money_Yes_feature_duplicate"] = df["Save_Money_Yes"]

    return df


@pytest.fixture
def base_regression_config(tmp_path, seed):
    """
    Minimal config for regression baseline on Risk_Score.
    """
    cfg = {
        "experiment": {
            "name": "pytest_regression",
            "seed": seed,
            "output_dir": str(tmp_path / "runs")
        },
        "data": {
            "dataset_path": "DUMMY.csv",
            "target_column": "Risk_Score",
            "target_type": "regression"
        },
        "model": {
            "type": "ridge",
            "params": {
                "ridge": {}
            }
        },
        "preprocessing": {
            "columns_to_drop": [],
            "ignored_columns": ["Behavior_Risk_Level"]
        },
        "cross_validation": {
            "n_splits": 4,
            "n_repeats": 2
        },
        "augmentation": {
            "enabled": False
        },
        "metrics": {"save_plots": False}
    }
    return cfg


@pytest.fixture
def base_classification_config(tmp_path, seed):
    cfg = {
        "experiment": {
            "name": "pytest_classification",
            "seed": seed,
            "output_dir": str(tmp_path / "runs")
        },
        "data": {
            "dataset_path": "DUMMY.csv",
            "target_column": "Save_Money_Yes",
            "target_type": "classification"
        },
        "model": {
            "type": "logistic_regression",
            "params": {
                "logistic_regression": {
                    "max_iter": 200,
                    "class_weight": "balanced"
                }
            }
        },
        "preprocessing": {
            "columns_to_drop": [],
            "ignored_columns": ["Behavior_Risk_Level", "Save_Money_No"]
        },
        "cross_validation": {
            "n_splits": 4,
            "n_repeats": 2
        },
        "augmentation": {
            "enabled": False
        },
        "metrics": {"save_plots": False}
    }
    return cfg


@pytest.fixture
def patch_dataset_loader(monkeypatch, tiny_adv_df):
    """
    Monkeypatch load_dataset so experiments don't hit disk.
    """
    def _fake_load_dataset(config, dataset_path=None):
        return tiny_adv_df.copy(), "test_dataset.csv"

    monkeypatch.setattr("experiments.data.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("run_baseline.load_dataset", _fake_load_dataset)
    return _fake_load_dataset


@pytest.fixture
def freeze_time(monkeypatch):
    """
    Make run_dir deterministic by freezing datetime.now().
    """
    import datetime as dt

    class _FixedDT:
        @staticmethod
        def now():
            return dt.datetime(2026, 1, 4, 12, 34, 56)

        @staticmethod
        def strftime(fmt):
            return _FixedDT.now().strftime(fmt)

    monkeypatch.setattr("experiments.io.datetime", _FixedDT)
    return _FixedDT


@pytest.fixture
def write_yaml(tmp_path):
    import yaml
    def _write(cfg, name="temp.yaml"):
        p = tmp_path / name
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        return str(p)
    return _write

