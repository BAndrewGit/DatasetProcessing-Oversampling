# Multi-task Learning Experiment Runner
# Ablation: single-task vs multi-task comparison

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json

from experiments.config_schema import validate_config, ConfigValidationError, FORBIDDEN_TARGETS
from experiments.io import load_config, create_run_dir
from experiments.data import load_dataset, validate_save_money_consistency, RISK_SCORE_COMPONENTS
from experiments.multitask import train_single_task_risk, train_single_task_savings, train_multitask
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def preprocess_multitask_data(df, config):
    """
    Preprocess data for multi-task learning.

    CRITICAL: Excludes RISK_SCORE_COMPONENTS to prevent leakage!
    These columns are used to calculate Risk_Score, so including them
    would make the task trivially easy and contaminate results.

    Returns:
        X: Feature matrix
        y_risk: Risk_Score target
        y_savings: Save_Money_Yes target
    """
    # Columns to drop/ignore
    cols_to_drop = config['preprocessing'].get('columns_to_drop', [])
    ignored = config['preprocessing'].get('ignored_columns', [])

    # Hard constraint: Behavior_Risk_Level must be excluded
    if 'Behavior_Risk_Level' not in ignored:
        ignored.append('Behavior_Risk_Level')

    # Targets
    risk_target = 'Risk_Score'
    savings_target = 'Save_Money_Yes'

    # Verify targets exist
    if risk_target not in df.columns:
        raise ValueError(f"Risk target '{risk_target}' not found in dataset")
    if savings_target not in df.columns:
        raise ValueError(f"Savings target '{savings_target}' not found in dataset")

    # =========================================================================
    # FIX 1: EXCLUDE RISK_SCORE_COMPONENTS (LEAKAGE PREVENTION)
    # =========================================================================
    leakage_cols = [c for c in RISK_SCORE_COMPONENTS if c in df.columns]
    if leakage_cols:
        print(f"EXCLUDED (leakage prevention): {leakage_cols}")

    # Build feature set - exclude targets, ignored, dropped, AND leakage cols
    exclude_cols = [risk_target, savings_target] + ignored + cols_to_drop + leakage_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Verify no forbidden columns in features
    for col in feature_cols:
        if col in FORBIDDEN_TARGETS:
            raise ValueError(f"BLOCKED: Forbidden column '{col}' found in features!")

    # DOUBLE-CHECK: Verify no leakage columns slipped through
    bad_cols = [c for c in feature_cols if c in RISK_SCORE_COMPONENTS]
    if bad_cols:
        raise ValueError(f"LEAKAGE BUG: Risk_Score components in multitask features: {bad_cols}")

    X = df[feature_cols].copy()
    y_risk = df[risk_target].copy()
    y_savings = df[savings_target].copy()

    # Hard constraint: Save_Money_Yes and Save_Money_No must be mutually exclusive
    if 'Save_Money_No' in df.columns:
        both_one = (df['Save_Money_Yes'] == 1) & (df['Save_Money_No'] == 1)
        both_zero = (df['Save_Money_Yes'] == 0) & (df['Save_Money_No'] == 0)
        if both_one.sum() > 0:
            raise ValueError(f"Data integrity error: {both_one.sum()} rows have both Save_Money_Yes=1 AND Save_Money_No=1")
        if both_zero.sum() > 0:
            raise ValueError(f"Data integrity error: {both_zero.sum()} rows have both Save_Money_Yes=0 AND Save_Money_No=0")

    # Hard constraint: Save_Money_Yes must have both classes
    if y_savings.nunique() < 2:
        raise ValueError(
            f"ABORT: Save_Money_Yes has only one class ({y_savings.unique()}). "
            f"Cannot train classification model."
        )

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Ignored columns: {ignored}")
    print(f"Risk target distribution: mean={y_risk.mean():.4f}, std={y_risk.std():.4f}")
    print(f"Savings target distribution: {y_savings.value_counts().to_dict()}")

    return X, y_risk, y_savings


def run_multitask_cv(X, y_risk, y_savings, config):
    """
    Run repeated K-fold CV for all three ablation experiments:
    1. Risk-only
    2. Savings-only
    3. Multi-task

    FIX 3: Uses RepeatedStratifiedKFold to handle imbalanced classification.
    Stratifies on y_savings (158/31 class imbalance).

    Returns:
        results: dict with aggregated metrics for each experiment
    """
    seed = config['experiment']['seed']
    n_splits = config['cross_validation'].get('n_splits', 5)
    n_repeats = config['cross_validation'].get('n_repeats', 3)
    total_folds = n_splits * n_repeats

    print(f"\nRunning {n_splits}-fold x {n_repeats} repeats STRATIFIED CV ({total_folds} folds total)...")

    # FIX 3: Use RepeatedStratifiedKFold for proper class balance in each fold
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    risk_only_metrics = []
    savings_only_metrics = []
    multitask_metrics = []
    # Keep last model for each ablation to save
    saved_models = {'risk_only': None, 'savings_only': None, 'multitask': None}

    # FIX 3: Split using y_savings for stratification
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_savings)):
        if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
            print(f"  Fold {fold_idx + 1}/{total_folds}...")

        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_risk_train, y_risk_val = y_risk.iloc[train_idx].values, y_risk.iloc[val_idx].values
        y_savings_train, y_savings_val = y_savings.iloc[train_idx].values, y_savings.iloc[val_idx].values

        # Risk-only
        risk_metrics, risk_model = train_single_task_risk(
            X_train, y_risk_train, X_val, y_risk_val,
            config['model'], seed + fold_idx
        )
        risk_only_metrics.append(risk_metrics)
        saved_models['risk_only'] = risk_model

        # Savings-only
        savings_metrics, savings_model = train_single_task_savings(
            X_train, y_savings_train, X_val, y_savings_val,
            config['model'], seed + fold_idx
        )
        savings_only_metrics.append(savings_metrics)
        saved_models['savings_only'] = savings_model

        # Multi-task
        mt_metrics, mt_model = train_multitask(
            X_train, y_risk_train, y_savings_train,
            X_val, y_risk_val, y_savings_val,
            config['model'], seed + fold_idx
        )
        multitask_metrics.append(mt_metrics)
        saved_models['multitask'] = mt_model

    # Aggregate results
    results = {
        'risk_only': _aggregate_metrics(risk_only_metrics),
        'savings_only': _aggregate_metrics(savings_only_metrics),
        'multitask': _aggregate_metrics(multitask_metrics)
    }

    return results, saved_models


def _aggregate_metrics(metrics_list):
    # Aggregate metrics across folds
    if not metrics_list:
        return {}

    aggregated = {}
    metric_names = list(metrics_list[0].keys())

    for metric_name in metric_names:
        try:
            values = [float(m[metric_name]) for m in metrics_list if m.get(metric_name) is not None]
            if values:
                aggregated[metric_name] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)),
                    'all': values
                }
        except (TypeError, ValueError):
            pass  # Skip non-numeric metrics

    return aggregated


def print_results(results):
    """Print comparison of all three experiments."""
    print("\n" + "=" * 80)
    print("MULTI-TASK ABLATION RESULTS")
    print("=" * 80)

    # Risk metrics
    print("\n--- RISK REGRESSION (Risk_Score) ---")
    print(f"{'Experiment':<20} {'MAE':<15} {'RMSE':<15} {'Spearman rho':<15} {'R2':<15}")
    print("-" * 80)

    for exp_name in ['risk_only', 'multitask']:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]

        mae = exp_results.get('risk_mae', {})
        rmse = exp_results.get('risk_rmse', {})
        spearman = exp_results.get('risk_spearman', {})
        r2 = exp_results.get('risk_r2', {})

        label = "Risk-only (baseline)" if exp_name == 'risk_only' else "Multi-task"

        print(f"{label:<20} "
              f"{mae.get('mean', 0):.4f}±{mae.get('std', 0):.4f}   "
              f"{rmse.get('mean', 0):.4f}±{rmse.get('std', 0):.4f}   "
              f"{spearman.get('mean', 0):.4f}±{spearman.get('std', 0):.4f}   "
              f"{r2.get('mean', 0):.4f}±{r2.get('std', 0):.4f}")

    # Savings metrics
    print("\n--- SAVINGS CLASSIFICATION (Save_Money_Yes) ---")
    print(f"{'Experiment':<20} {'Macro-F1':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 80)

    for exp_name in ['savings_only', 'multitask']:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]

        f1 = exp_results.get('savings_macro_f1', {})
        acc = exp_results.get('savings_accuracy', {})
        prec = exp_results.get('savings_precision', {})
        rec = exp_results.get('savings_recall', {})

        label = "Savings-only (baseline)" if exp_name == 'savings_only' else "Multi-task"

        print(f"{label:<20} "
              f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}   "
              f"{acc.get('mean', 0):.4f}±{acc.get('std', 0):.4f}   "
              f"{prec.get('mean', 0):.4f}±{prec.get('std', 0):.4f}   "
              f"{rec.get('mean', 0):.4f}±{rec.get('std', 0):.4f}")

    print("=" * 80)


def run_multitask_experiment(config_path, dataset_path=None, output_dir=None):
    """
    Run multi-task learning ablation experiment.
    """
    # Load and validate config
    config = load_config(config_path)

    # Override output_dir if provided
    if output_dir:
        config['experiment']['output_dir'] = output_dir

    try:
        validate_config(config, mode='multitask')
    except ConfigValidationError as e:
        print(f"\nCONFIG ERROR:\n{e}")
        raise

    seed = config['experiment']['seed']
    set_seeds(seed)

    print("=" * 80)
    print("MULTI-TASK LEARNING ABLATION EXPERIMENT")
    print("=" * 80)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {seed}")
    print("=" * 80)
    print("Ablation design:")
    print("  1. Risk-only model (baseline)")
    print("  2. Savings-only model (baseline)")
    print("  3. Multi-task model (shared trunk + both heads)")
    print("=" * 80)

    # Load data
    df, actual_path = load_dataset(config, dataset_path)

    # Validate Save_Money consistency
    validate_save_money_consistency(df)

    # Preprocess for multi-task
    X, y_risk, y_savings = preprocess_multitask_data(df, config)

    # Run CV ablation
    results, saved_models = run_multitask_cv(X, y_risk, y_savings, config)

    # Print results
    print_results(results)

    # Save results
    run_dir = create_run_dir(config)

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Save results - simple serialization
    def serialize_value(v):
        if isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [serialize_value(x) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    with open(os.path.join(run_dir, 'ablation_results.json'), 'w') as f:
        json.dump(serialize_value(results), f, indent=2)

    # Save trained models if returned
    try:
        import torch as _torch
        if 'risk_only' in saved_models and saved_models['risk_only'] is not None:
            _torch.save(saved_models['risk_only'].state_dict(), os.path.join(run_dir, 'risk_only_model.pth'))
        if 'savings_only' in saved_models and saved_models['savings_only'] is not None:
            _torch.save(saved_models['savings_only'].state_dict(), os.path.join(run_dir, 'savings_only_model.pth'))
        if 'multitask' in saved_models and saved_models['multitask'] is not None:
            _torch.save(saved_models['multitask'].state_dict(), os.path.join(run_dir, 'multitask_model.pth'))
    except Exception as e:
        print(f"Warning: failed to save multitask models: {e}")

    # Also save a fitted StandardScaler for analysis tooling (joblib) and a small metadata file
    try:
        from sklearn.preprocessing import StandardScaler
        from experiments.save_model import save_sklearn_model, write_model_metadata
        import joblib as _joblib

        scaler = StandardScaler()
        # X may be a DataFrame from preprocess; fit on full features
        try:
            _X_for_scaler = X.values if hasattr(X, 'values') else X
        except Exception:
            _X_for_scaler = X
        scaler.fit(_X_for_scaler)
        scaler_path = os.path.join(run_dir, 'scaler.joblib')
        save_sklearn_model(scaler, scaler_path)

        # Write metadata pointing to saved model files
        metadata = {
            'pytorch_state_dicts': {
                'risk_only': 'risk_only_model.pth' if 'risk_only' in saved_models and saved_models['risk_only'] is not None else None,
                'savings_only': 'savings_only_model.pth' if 'savings_only' in saved_models and saved_models['savings_only'] is not None else None,
                'multitask': 'multitask_model.pth' if 'multitask' in saved_models and saved_models['multitask'] is not None else None
            },
            'sklearn_objects': {
                'scaler': 'scaler.joblib'
            }
        }
        write_model_metadata(run_dir, metadata)
    except Exception as e:
        print(f"Warning: failed to save scaler or metadata for multitask run: {e}")

    print(f"\nResults saved to: {run_dir}")

    return run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-task learning ablation experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset CSV (overrides config)')

    args = parser.parse_args()

    run_multitask_experiment(args.config, args.dataset)
