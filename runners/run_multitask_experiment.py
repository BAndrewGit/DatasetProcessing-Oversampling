# Multi-task Learning Experiment Runner
# Ablation: single-task vs multi-task comparison

import sys
import os

# Add runners dir to path and import startup to silence warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _startup  # noqa: F401
except ImportError:
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json

from experiments.config_schema import validate_config, ConfigValidationError, FORBIDDEN_TARGETS
from experiments.io import load_config, create_run_dir
from experiments.data import load_dataset, validate_save_money_consistency, RISK_SCORE_COMPONENTS
from experiments.model_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_DIM,
    MODEL_SCALER_MODE,
    MODEL_SCALED_FEATURE_COLUMNS,
)
from experiments.multitask import train_single_task_risk, train_single_task_savings, train_multitask
from experiments.multitask_plots import generate_all_multitask_plots
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
    Preprocess data for multi-task learning using the shared model contract.

    Returns:
        X: Feature matrix
        y_risk: Risk_Score target
        y_savings: Save_Money_Yes target
    """
    risk_target = 'Risk_Score'
    savings_target = 'Save_Money_Yes'

    if risk_target not in df.columns:
        raise ValueError(f"Risk target '{risk_target}' not found in dataset")
    if savings_target not in df.columns:
        raise ValueError(f"Savings target '{savings_target}' not found in dataset")

    leakage_cols = [column for column in MODEL_FEATURE_COLUMNS if column in RISK_SCORE_COMPONENTS]
    if leakage_cols:
        raise ValueError(f"Model contract contains leakage columns: {leakage_cols}")

    forbidden_in_contract = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column in set(FORBIDDEN_TARGETS)
        or column in {"Save_Money_No", risk_target, savings_target, "Behavior_Risk_Level"}
    ]
    if forbidden_in_contract:
        raise ValueError(f"Model contract contains forbidden columns: {forbidden_in_contract}")

    missing_features = [column for column in MODEL_FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required model features: {missing_features}")

    feature_cols = list(MODEL_FEATURE_COLUMNS)

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
    print(f"Contract input_dim: {MODEL_INPUT_DIM}")
    print(f"Scaled columns: {MODEL_SCALED_FEATURE_COLUMNS}")
    print(f"Risk target distribution: mean={y_risk.mean():.4f}, std={y_risk.std():.4f}")
    print(f"Savings target distribution: {y_savings.value_counts().to_dict()}")

    return X, y_risk, y_savings


def run_multitask_cv(X, y_risk, y_savings, config, multitask_only=False):
    """
    Run repeated K-fold CV for all three ablation experiments:
    1. Risk-only (optional)
    2. Savings-only (optional)
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

    # SANITY CHECK: Verify no feature is perfectly correlated with target
    try:
        from experiments.sanity_checks import check_multitask_f1_leakage
        feature_names = list(X.columns) if hasattr(X, 'columns') else None
        leakage_check = check_multitask_f1_leakage(X.values if hasattr(X, 'values') else X,
                                                   y_savings.values if hasattr(y_savings, 'values') else y_savings,
                                                   feature_names, seed)
        if leakage_check['likely_leakage']:
            print("\n*** LEAKAGE WARNING ***")
            for check in leakage_check['checks']:
                print(f"  - {check}")
            print(f"  Suspicious features: {leakage_check['suspicious_features']}")
            print("*** END WARNING ***\n")
    except Exception as e:
        print(f"  (leakage check skipped: {e})")

    risk_only_metrics = []
    savings_only_metrics = []
    multitask_metrics = []

    # Collect gradient logs and thresholds from multitask training
    all_gradient_logs = []
    all_thresholds = []
    all_training_logs = []

    # Keep last model for each ablation to save
    saved_models = {'risk_only': None, 'savings_only': None, 'multitask': None}

    # FIX 3: Split using y_savings for stratification
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_savings)):
        if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
            print(f"  Fold {fold_idx + 1}/{total_folds}...")


        # Partial Scaling
        from sklearn.preprocessing import StandardScaler
        scale_cols = list(MODEL_SCALED_FEATURE_COLUMNS)
        scale_idx = [i for i, c in enumerate(X.columns) if c in scale_cols]
        fold_scaler = StandardScaler()
        if scale_idx:
            X_train[:, scale_idx] = fold_scaler.fit_transform(X_train[:, scale_idx])
            X_val[:, scale_idx] = fold_scaler.transform(X_val[:, scale_idx])


        if not multitask_only:
            # Risk-only
            risk_metrics, risk_model = train_single_task_risk(
                X_train, y_risk_train, X_val, y_risk_val,
                config['model'], seed + fold_idx
            )
            # Filter out internal keys starting with '_'
            risk_only_metrics.append({k: v for k, v in risk_metrics.items() if not k.startswith('_')})
            saved_models['risk_only'] = risk_model

            # Savings-only
            savings_metrics, savings_model = train_single_task_savings(
                X_train, y_savings_train, X_val, y_savings_val,
                config['model'], seed + fold_idx
            )
            savings_only_metrics.append({k: v for k, v in savings_metrics.items() if not k.startswith('_')})
            saved_models['savings_only'] = savings_model

        # Multi-task
        mt_metrics, mt_model = train_multitask(
            X_train, y_risk_train, y_savings_train,
            X_val, y_risk_val, y_savings_val,
            config['model'], seed + fold_idx
        )

        # Extract gradient logs and threshold from metrics
        if '_gradient_logs' in mt_metrics:
            all_gradient_logs.extend(mt_metrics['_gradient_logs'])
        if '_optimal_threshold' in mt_metrics:
            all_thresholds.append(mt_metrics['_optimal_threshold'])
        if '_training_log' in mt_metrics:
            all_training_logs.append(mt_metrics['_training_log'])

        # Filter out internal keys for aggregation
        multitask_metrics.append({k: v for k, v in mt_metrics.items() if not k.startswith('_')})
        saved_models['multitask'] = mt_model

    # Aggregate results
    results = {
        'multitask': _aggregate_metrics(multitask_metrics)
    }
    if not multitask_only:
        results['risk_only'] = _aggregate_metrics(risk_only_metrics)
        results['savings_only'] = _aggregate_metrics(savings_only_metrics)

    # Add gradient logs and threshold info for plotting
    if all_gradient_logs:
        results['grad_logs'] = all_gradient_logs
    if all_thresholds:
        results['thresholds'] = {
            'mean': float(np.mean(all_thresholds)),
            'std': float(np.std(all_thresholds)),
            'all': all_thresholds
        }
        print(f"\n  Optimal savings threshold: {np.mean(all_thresholds):.3f} ± {np.std(all_thresholds):.3f}")
    if all_training_logs:
        # Keep only last fold's training log for plotting (to avoid huge data)
        results['epoch_logs'] = all_training_logs[-1]

    return results, saved_models


def _aggregate_metrics(metrics_list):
    """Aggregate metrics across folds with mean, std, median, and IQR."""
    if not metrics_list:
        return {}

    aggregated = {}
    metric_names = list(metrics_list[0].keys())

    for metric_name in metric_names:
        try:
            values = [float(m[metric_name]) for m in metrics_list if m.get(metric_name) is not None]
            if values:
                arr = np.array(values)
                aggregated[metric_name] = {
                    'mean': float(np.nanmean(arr)),
                    'std': float(np.nanstd(arr)),
                    'median': float(np.nanmedian(arr)),
                    'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                    'p25': float(np.percentile(arr, 25)),
                    'p75': float(np.percentile(arr, 75)),
                    'all': values
                }
        except (TypeError, ValueError):
            pass

    return aggregated


def print_results(results):
    """Print comparison of all three experiments with median (IQR) for robustness."""
    print("\n" + "=" * 80)
    print("MULTI-TASK ABLATION RESULTS (median [IQR])")
    print("=" * 80)

    def fmt(m):
        """Format metric as median [p25-p75]."""
        if not m:
            return "N/A"
        return f"{m.get('median', 0):.4f} [{m.get('p25', 0):.3f}-{m.get('p75', 0):.3f}]"

    # Risk metrics - focus on MAE and Spearman (not R²)
    print("\n--- RISK REGRESSION (Risk_Score) ---")
    print(f"{'Experiment':<22} {'MAE (lower=better)':<25} {'Spearman (higher=better)':<25}")
    print("-" * 75)

    for exp_name in ['risk_only', 'multitask']:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]
        mae = exp_results.get('risk_mae', {})
        spearman = exp_results.get('risk_spearman', {})
        label = "Risk-only (baseline)" if exp_name == 'risk_only' else "Multi-task"
        print(f"{label:<22} {fmt(mae):<25} {fmt(spearman):<25}")

    # Savings metrics
    print("\n--- SAVINGS CLASSIFICATION (Save_Money_Yes) ---")
    print(f"{'Experiment':<22} {'Macro-F1 (higher=better)':<25} {'Accuracy':<25}")
    print("-" * 75)

    for exp_name in ['savings_only', 'multitask']:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]
        f1 = exp_results.get('savings_macro_f1', {})
        acc = exp_results.get('savings_accuracy', {})
        label = "Savings-only (baseline)" if exp_name == 'savings_only' else "Multi-task"
        print(f"{label:<22} {fmt(f1):<25} {fmt(acc):<25}")

    print("=" * 75)


def run_multitask_experiment(config_path, dataset_path=None, output_dir=None, multitask_only=False):
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
    if multitask_only:
        print("  1. Multi-task only mode (final production comparison)")
    else:
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

    # Save results
    run_dir = create_run_dir(config)

    with open(os.path.join(run_dir, "feature_columns.json"), "w") as f:
        json.dump(list(MODEL_FEATURE_COLUMNS), f, indent=2)

    # Run CV ablation
    results, saved_models = run_multitask_cv(X, y_risk, y_savings, config, multitask_only=multitask_only)

    # Print results
    print_results(results)


    # ==========================================================================
    # GENERATE MULTITASK PLOTS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GENERATING MULTITASK PLOTS")
    print("=" * 60)

    # Extract epoch logs if available (would need to be collected during training)
    epoch_logs = results.get('epoch_logs', {})
    grad_logs = results.get('grad_logs', [])

    try:
        plots = generate_all_multitask_plots(
            results=results,
            epoch_logs=epoch_logs if epoch_logs else None,
            grad_logs=grad_logs if grad_logs else None,
            output_dir=run_dir
        )
        print(f"Generated {len(plots)} multitask plots")
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

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
        scale_cols = list(MODEL_SCALED_FEATURE_COLUMNS)
        scale_idx = [i for i, c in enumerate(X.columns) if c in scale_cols]
        if scale_idx:
            scaler.fit(X.iloc[:, scale_idx].values)

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
        metadata['scaled_feature_columns'] = list(MODEL_SCALED_FEATURE_COLUMNS)
        metadata['scaler_mode'] = MODEL_SCALER_MODE
        metadata['input_dim'] = MODEL_INPUT_DIM

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
