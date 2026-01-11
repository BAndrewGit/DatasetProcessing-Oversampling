# Domain Transfer Experiment Runner
# Ablation study: ADV-only vs ADV + GMSC auxiliary supervision
# Evaluation on ADV real data only

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from experiments.io import load_config, create_run_dir
from experiments.config_schema import FORBIDDEN_TARGETS
from experiments.domain_transfer import (
    train_adv_only,
    train_with_gmsc_transfer
)


def set_seeds(seed: int):
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


def load_adv_data(config: dict, dataset_path: str = None) -> tuple:
    """
    Load and preprocess ADV dataset.

    Returns:
        X: Feature matrix
        y_risk: Risk_Score target
        y_savings: Save_Money_Yes target
    """
    path = dataset_path or config['data'].get('adv_dataset_path')

    if path is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        print("Select ADV dataset file...")
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        root.destroy()

    print(f"Loading ADV dataset: {path}")
    df = pd.read_csv(path)

    # Targets
    if 'Risk_Score' not in df.columns:
        raise ValueError("ADV dataset missing Risk_Score column")
    if 'Save_Money_Yes' not in df.columns:
        raise ValueError("ADV dataset missing Save_Money_Yes column")

    y_risk = df['Risk_Score'].values.astype(np.float32)
    y_savings = df['Save_Money_Yes'].values.astype(np.float32)

    # Features - exclude targets and forbidden columns
    ignored = config['preprocessing'].get('ignored_columns', [])
    cols_to_drop = config['preprocessing'].get('columns_to_drop', [])

    exclude = ['Risk_Score', 'Save_Money_Yes', 'Save_Money_No'] + ignored + cols_to_drop
    exclude = [c for c in exclude if c in df.columns]

    feature_cols = [c for c in df.columns if c not in exclude]

    # Verify no forbidden columns
    for col in feature_cols:
        if col in FORBIDDEN_TARGETS:
            raise ValueError(f"Forbidden column '{col}' in ADV features")

    X = df[feature_cols].values.astype(np.float32)

    print(f"  ADV shape: {X.shape}")
    print(f"  Risk range: [{y_risk.min():.3f}, {y_risk.max():.3f}]")
    print(f"  Savings distribution: {np.bincount(y_savings.astype(int))}")

    return X, y_risk, y_savings, feature_cols


def load_gmsc_data(config: dict, dataset_path: str = None) -> tuple:
    """
    Load and preprocess GMSC dataset.

    Returns:
        X: Feature matrix
        y: SeriousDlqin2yrs target (binary)
    """
    path = dataset_path or config['data'].get('gmsc_dataset_path')

    if path is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        print("Select GMSC dataset file...")
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        root.destroy()

    print(f"Loading GMSC dataset: {path}")
    df = pd.read_csv(path)

    # Target
    if 'SeriousDlqin2yrs' not in df.columns:
        raise ValueError("GMSC dataset missing SeriousDlqin2yrs column")

    y = df['SeriousDlqin2yrs'].values.astype(np.float32)

    # Features - exclude ID column and target
    exclude = ['Unnamed: 0', 'SeriousDlqin2yrs']
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values.astype(np.float32)

    # Handle missing values (GMSC has NaN in MonthlyIncome and NumberOfDependents)
    # Simple median imputation
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X[nan_mask, i] = median_val

    print(f"  GMSC shape: {X.shape}")
    print(f"  Delinquency rate: {y.mean():.3%}")

    return X, y, feature_cols


def run_domain_transfer_cv(
    adv_X: np.ndarray, adv_y_risk: np.ndarray, adv_y_savings: np.ndarray,
    gmsc_X: np.ndarray, gmsc_y: np.ndarray,
    config: dict, seed: int
) -> dict:
    """
    Run repeated K-fold CV for domain transfer ablation.

    Compares:
    1. ADV-only training
    2. ADV + GMSC transfer (with domain alignment)

    Evaluation on ADV data only.
    """
    n_splits = config['cross_validation'].get('n_splits', 5)
    n_repeats = config['cross_validation'].get('n_repeats', 5)

    print(f"\nRunning {n_splits}-fold x {n_repeats} repeats CV...")

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # Storage for all fold metrics
    adv_only_metrics = []
    transfer_metrics = []

    # Model config
    model_config = config.get('model', {})
    model_config['adv_ratio'] = config['data'].get('adv_ratio', 0.7)

    # Domain alignment config
    alignment_config = config.get('domain_alignment', {})
    model_config['alignment_enabled'] = alignment_config.get('enabled', True)
    model_config['alignment_method'] = alignment_config.get('method', 'coral')
    model_config['alignment_weight'] = alignment_config.get('weight', 0.1)

    total_folds = n_splits * n_repeats

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(adv_X)):
        print(f"\nFold {fold_idx + 1}/{total_folds}")

        # Split ADV data
        adv_X_train, adv_X_val = adv_X[train_idx], adv_X[val_idx]
        adv_y_risk_train, adv_y_risk_val = adv_y_risk[train_idx], adv_y_risk[val_idx]
        adv_y_savings_train, adv_y_savings_val = adv_y_savings[train_idx], adv_y_savings[val_idx]

        # Scale features (fit on train only)
        adv_scaler = StandardScaler()
        adv_X_train_s = adv_scaler.fit_transform(adv_X_train)
        adv_X_val_s = adv_scaler.transform(adv_X_val)

        gmsc_scaler = StandardScaler()
        gmsc_X_s = gmsc_scaler.fit_transform(gmsc_X)

        fold_seed = seed + fold_idx

        # Experiment 1: ADV-only
        print("  Training ADV-only model...")
        adv_metrics = train_adv_only(
            adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
            adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
            model_config, fold_seed
        )
        adv_only_metrics.append(adv_metrics)

        # Experiment 2: ADV + GMSC transfer
        print("  Training ADV + GMSC transfer model...")
        transfer_m = train_with_gmsc_transfer(
            adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
            adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
            gmsc_X_s, gmsc_y,
            model_config, fold_seed
        )
        transfer_metrics.append(transfer_m)

        # Progress report
        print(f"  ADV-only:  Risk MAE={adv_metrics['risk_mae']:.4f}, "
              f"Savings F1={adv_metrics['savings_macro_f1']:.4f}")
        print(f"  Transfer:  Risk MAE={transfer_m['risk_mae']:.4f}, "
              f"Savings F1={transfer_m['savings_macro_f1']:.4f}")

    # Aggregate results
    results = {
        'adv_only': _aggregate_metrics(adv_only_metrics),
        'transfer': _aggregate_metrics(transfer_metrics)
    }

    return results


def _aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics across folds."""
    aggregated = {}

    if not metrics_list:
        return aggregated

    metric_names = metrics_list[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'all': [float(v) for v in values]
            }

    return aggregated


def print_results(results: dict):
    """Print comparison of ADV-only vs Transfer."""
    print("\n" + "=" * 90)
    print("DOMAIN TRANSFER ABLATION RESULTS")
    print("=" * 90)

    # Risk metrics
    print("\n--- RISK REGRESSION (Risk_Score) - ADV Data Only ---")
    print(f"{'Experiment':<25} {'MAE':<18} {'RMSE':<18} {'Spearman ρ':<18} {'R²':<18}")
    print("-" * 90)

    for exp_name, label in [('adv_only', 'ADV-only (baseline)'), ('transfer', 'ADV + GMSC transfer')]:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]

        mae = exp_results.get('risk_mae', {})
        rmse = exp_results.get('risk_rmse', {})
        spearman = exp_results.get('risk_spearman', {})
        r2 = exp_results.get('risk_r2', {})

        print(f"{label:<25} "
              f"{mae.get('mean', 0):.4f}±{mae.get('std', 0):.4f}     "
              f"{rmse.get('mean', 0):.4f}±{rmse.get('std', 0):.4f}     "
              f"{spearman.get('mean', 0):.4f}±{spearman.get('std', 0):.4f}     "
              f"{r2.get('mean', 0):.4f}±{r2.get('std', 0):.4f}")

    # Savings metrics
    print("\n--- SAVINGS CLASSIFICATION (Save_Money_Yes) - ADV Data Only ---")
    print(f"{'Experiment':<25} {'Macro-F1':<18} {'Accuracy':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 90)

    for exp_name, label in [('adv_only', 'ADV-only (baseline)'), ('transfer', 'ADV + GMSC transfer')]:
        if exp_name not in results:
            continue
        exp_results = results[exp_name]

        f1 = exp_results.get('savings_macro_f1', {})
        acc = exp_results.get('savings_accuracy', {})
        prec = exp_results.get('savings_precision', {})
        rec = exp_results.get('savings_recall', {})

        print(f"{label:<25} "
              f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}     "
              f"{acc.get('mean', 0):.4f}±{acc.get('std', 0):.4f}     "
              f"{prec.get('mean', 0):.4f}±{prec.get('std', 0):.4f}     "
              f"{rec.get('mean', 0):.4f}±{rec.get('std', 0):.4f}")

    # Improvement analysis
    print("\n--- TRANSFER IMPROVEMENT ANALYSIS ---")

    adv_only = results.get('adv_only', {})
    transfer = results.get('transfer', {})

    if adv_only and transfer:
        # Risk improvement (lower MAE is better)
        adv_mae = adv_only.get('risk_mae', {}).get('mean', 0)
        tf_mae = transfer.get('risk_mae', {}).get('mean', 0)
        mae_improvement = (adv_mae - tf_mae) / adv_mae * 100 if adv_mae > 0 else 0

        # Savings improvement (higher F1 is better)
        adv_f1 = adv_only.get('savings_macro_f1', {}).get('mean', 0)
        tf_f1 = transfer.get('savings_macro_f1', {}).get('mean', 0)
        f1_improvement = (tf_f1 - adv_f1) / adv_f1 * 100 if adv_f1 > 0 else 0

        print(f"Risk MAE improvement:    {mae_improvement:+.2f}%")
        print(f"Savings F1 improvement:  {f1_improvement:+.2f}%")

        # Stability comparison
        adv_mae_std = adv_only.get('risk_mae', {}).get('std', 0)
        tf_mae_std = transfer.get('risk_mae', {}).get('std', 0)
        stability = (adv_mae_std - tf_mae_std) / adv_mae_std * 100 if adv_mae_std > 0 else 0
        print(f"Risk MAE stability:      {stability:+.2f}% (variance reduction)")

    print("=" * 90)


def analyze_transfer_benefit(results: dict) -> dict:
    """Analyze whether transfer learning provides benefit."""
    adv_only = results.get('adv_only', {})
    transfer = results.get('transfer', {})

    analysis = {
        'risk_improved': False,
        'savings_improved': False,
        'overall_beneficial': False,
        'details': {}
    }

    if not adv_only or not transfer:
        return analysis

    # Risk: transfer is better if MAE is lower
    adv_mae = adv_only.get('risk_mae', {}).get('mean', float('inf'))
    tf_mae = transfer.get('risk_mae', {}).get('mean', float('inf'))
    adv_mae_std = adv_only.get('risk_mae', {}).get('std', 0)
    tf_mae_std = transfer.get('risk_mae', {}).get('std', 0)

    # Statistical significance: non-overlapping confidence intervals (rough check)
    risk_improved = tf_mae < adv_mae and (tf_mae + tf_mae_std) < (adv_mae + adv_mae_std)
    analysis['risk_improved'] = risk_improved
    analysis['details']['risk_mae_diff'] = adv_mae - tf_mae

    # Savings: transfer is better if F1 is higher
    adv_f1 = adv_only.get('savings_macro_f1', {}).get('mean', 0)
    tf_f1 = transfer.get('savings_macro_f1', {}).get('mean', 0)

    savings_improved = tf_f1 > adv_f1
    analysis['savings_improved'] = savings_improved
    analysis['details']['savings_f1_diff'] = tf_f1 - adv_f1

    # Overall: beneficial if either improves without significantly hurting the other
    # Risk shouldn't get >5% worse, savings shouldn't get >5% worse
    risk_ok = tf_mae <= adv_mae * 1.05
    savings_ok = tf_f1 >= adv_f1 * 0.95

    analysis['overall_beneficial'] = (risk_improved or savings_improved) and risk_ok and savings_ok

    return analysis


def run_domain_transfer_experiment(config_path: str, adv_path: str = None, gmsc_path: str = None):
    """
    Run full domain transfer experiment.

    Args:
        config_path: Path to config YAML
        adv_path: Optional path to ADV dataset (overrides config)
        gmsc_path: Optional path to GMSC dataset (overrides config)

    Returns:
        run_dir: Path to experiment output directory
    """
    # Load config
    config = load_config(config_path)

    seed = config['experiment']['seed']
    set_seeds(seed)

    print("=" * 90)
    print("DOMAIN TRANSFER EXPERIMENT")
    print("GMSC as Auxiliary Supervision for ADV Risk Modeling")
    print("=" * 90)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {seed}")
    print("=" * 90)
    print("Architecture:")
    print("  ADV features  -> ADV adapter  -┐")
    print("                                 ├-> Shared trunk -> Risk head (ADV regression)")
    print("  GMSC features -> GMSC adapter -┘              -> Risk head (GMSC classification)")
    print("                                                -> Savings head (ADV only)")
    print("=" * 90)
    print("Ablation design:")
    print("  1. ADV-only (baseline)")
    print("  2. ADV + GMSC transfer (with domain alignment)")
    print("=" * 90)

    # Load data
    adv_X, adv_y_risk, adv_y_savings, adv_features = load_adv_data(config, adv_path)
    gmsc_X, gmsc_y, gmsc_features = load_gmsc_data(config, gmsc_path)

    print(f"\nADV features ({len(adv_features)}): {adv_features[:5]}...")
    print(f"GMSC features ({len(gmsc_features)}): {gmsc_features}")

    # Run CV ablation
    results = run_domain_transfer_cv(
        adv_X, adv_y_risk, adv_y_savings,
        gmsc_X, gmsc_y,
        config, seed
    )

    # Print results
    print_results(results)

    # Analyze benefit
    analysis = analyze_transfer_benefit(results)

    print("\n--- TRANSFER BENEFIT ANALYSIS ---")
    print(f"Risk improved:      {'Yes' if analysis['risk_improved'] else 'No'}")
    print(f"Savings improved:   {'Yes' if analysis['savings_improved'] else 'No'}")
    print(f"Overall beneficial: {'Yes' if analysis['overall_beneficial'] else 'No'}")

    if analysis['overall_beneficial']:
        print("\n✅ GMSC TRANSFER IS BENEFICIAL")
        print("   Recommendation: Include GMSC auxiliary supervision in final model")
        verdict = "beneficial"
    else:
        print("\n❌ GMSC TRANSFER IS NOT BENEFICIAL")
        print("   Recommendation: Use ADV-only model")
        print("   This is valid science - transfer doesn't always help")
        verdict = "not_beneficial"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['experiment'].get('output_dir', 'runs')
    run_dir = os.path.join(output_dir, f"domain_transfer_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save full results
    results_json = {
        'experiment': 'domain_transfer',
        'timestamp': timestamp,
        'seed': seed,
        'adv_features': len(adv_features),
        'gmsc_features': len(gmsc_features),
        'adv_samples': len(adv_X),
        'gmsc_samples': len(gmsc_X),
        'adv_ratio': config['data'].get('adv_ratio', 0.7),
        'alignment_method': config.get('domain_alignment', {}).get('method', 'coral'),
        'results': results,
        'analysis': analysis,
        'verdict': verdict
    }

    with open(os.path.join(run_dir, 'domain_transfer_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save config
    import yaml
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nResults saved to: {run_dir}")

    return run_dir, verdict, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain transfer experiment: ADV + GMSC')
    parser.add_argument('--config', '-c', type=str,
                        default='configs/domain_transfer_experiment.yaml',
                        help='Path to config YAML')
    parser.add_argument('--adv', type=str, default=None,
                        help='Path to ADV dataset (overrides config)')
    parser.add_argument('--gmsc', type=str, default=None,
                        help='Path to GMSC dataset (overrides config)')

    args = parser.parse_args()

    run_domain_transfer_experiment(args.config, args.adv, args.gmsc)

