import pytest
import numpy as np
import pandas as pd
from experiments.multitask import MultiTaskModel, train_multitask


def test_multitask_model_forward_pass():
    """Test that multi-task model produces outputs with correct shapes."""
    input_dim = 10
    batch_size = 4

    model = MultiTaskModel(input_dim, hidden_dims=[16, 8], dropout=0.3)

    # Create dummy input
    X = np.random.randn(batch_size, input_dim).astype(np.float32)

    import torch
    X_tensor = torch.FloatTensor(X)

    # Forward pass
    risk_pred, savings_logits = model(X_tensor)

    # Check shapes
    assert risk_pred.shape == (batch_size, 1)
    assert savings_logits.shape == (batch_size, 1)


def test_multitask_blocks_forbidden_target():
    """Test that Behavior_Risk_Level is excluded from features."""
    from runners.run_multitask_experiment import preprocess_multitask_data

    # Create dummy dataset with forbidden column
    df = pd.DataFrame({
        'Feature1': [1, 2, 3, 4],
        'Feature2': [5, 6, 7, 8],
        'Behavior_Risk_Level': [0, 1, 0, 1],  # Forbidden
        'Risk_Score': [0.1, 0.5, 0.3, 0.7],
        'Save_Money_Yes': [0, 1, 0, 1]
    })

    config = {
        'preprocessing': {
            'ignored_columns': ['Behavior_Risk_Level'],
            'columns_to_drop': []
        }
    }

    X, y_risk, y_savings = preprocess_multitask_data(df, config)

    # Verify Behavior_Risk_Level not in features
    assert 'Behavior_Risk_Level' not in X.columns
    assert 'Risk_Score' not in X.columns
    assert 'Save_Money_Yes' not in X.columns

    # Verify feature count
    assert len(X.columns) == 2  # Only Feature1 and Feature2


def test_multitask_requires_both_classes():
    """Test that multitask aborts if Save_Money_Yes has only one class."""
    from runners.run_multitask_experiment import preprocess_multitask_data

    # Create dataset with only one class
    df = pd.DataFrame({
        'Feature1': [1, 2, 3, 4],
        'Risk_Score': [0.1, 0.5, 0.3, 0.7],
        'Save_Money_Yes': [1, 1, 1, 1]  # Only one class
    })

    config = {
        'preprocessing': {
            'ignored_columns': ['Behavior_Risk_Level'],
            'columns_to_drop': []
        }
    }

    with pytest.raises(ValueError, match="has only one class"):
        preprocess_multitask_data(df, config)


def test_multitask_mutual_exclusivity():
    """Test that Save_Money_Yes and Save_Money_No are mutually exclusive."""
    from runners.run_multitask_experiment import preprocess_multitask_data

    # Create dataset with both columns = 1 (invalid)
    df = pd.DataFrame({
        'Feature1': [1, 2, 3, 4],
        'Risk_Score': [0.1, 0.5, 0.3, 0.7],
        'Save_Money_Yes': [1, 1, 0, 0],
        'Save_Money_No': [0, 1, 1, 0]  # Row 1 has both = 1
    })

    config = {
        'preprocessing': {
            'ignored_columns': ['Behavior_Risk_Level'],
            'columns_to_drop': []
        }
    }

    with pytest.raises(ValueError, match="both Save_Money_Yes=1 AND Save_Money_No=1"):
        preprocess_multitask_data(df, config)


def test_multitask_config_validation():
    """Test that multitask config is validated correctly."""
    from experiments.config_schema import validate_config, ConfigValidationError

    config = {
        'experiment': {'name': 'test', 'seed': 42},
        'data': {'dataset_path': 'test.csv'},
        'model': {'hidden_dims': [64, 32]},
        'preprocessing': {'ignored_columns': []},
        'cross_validation': {'n_splits': 5}
    }

    # Should not raise for multitask mode
    validate_config(config, mode='multitask')


def test_multitask_training_runs():
    """Test that multitask training completes without errors."""
    # Create small dummy dataset
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_risk_train = np.random.rand(n_samples).astype(np.float32)
    y_savings_train = np.random.randint(0, 2, n_samples).astype(np.float32)

    X_val = np.random.randn(20, n_features).astype(np.float32)
    y_risk_val = np.random.rand(20).astype(np.float32)
    y_savings_val = np.random.randint(0, 2, 20).astype(np.float32)

    config = {
        'hidden_dims': [16, 8],
        'dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 0.01,
        'batch_size': 16,
        'max_epochs': 5,  # Small for testing
        'patience': 3,
        'risk_loss': 'huber',
        'clip_grad': 1.0
    }

    # Should complete without errors
    metrics = train_multitask(
        X_train, y_risk_train, y_savings_train,
        X_val, y_risk_val, y_savings_val,
        config, seed=42
    )

    # Check that metrics are returned
    assert 'risk_mae' in metrics
    assert 'risk_rmse' in metrics
    assert 'risk_spearman' in metrics
    assert 'risk_r2' in metrics
    assert 'savings_macro_f1' in metrics
    assert 'savings_accuracy' in metrics

