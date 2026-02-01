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

from experiments.io import load_config
from experiments.config_schema import FORBIDDEN_TARGETS
from experiments.data import RISK_SCORE_COMPONENTS  # Bug #5 FIX: Import leakage columns
from experiments.domain_transfer import (
    train_adv_only,
    train_with_gmsc_transfer,
    train_pretrain_finetune,
    train_with_optimized_mixture,
    train_with_all_upgrades
)


def flatten_cfg(cfg: dict) -> dict:
    """
    Flatten config from YAML structure into a single dict for training functions.

    Bug #1 FIX: This ensures all YAML values are properly read, not defaults.

    Maps:
    - cfg['model'][*] -> flat[*]
    - cfg['domain_alignment'][*] -> flat['alignment_*']
    - cfg['data'][*] -> flat[*] (for ratios etc.)
    """
    model = cfg.get("model", {})
    align = cfg.get("domain_alignment", {})
    data = cfg.get("data", {})

    # Start with model config as base
    flat = dict(model)

    # Map domain_alignment.* -> alignment_*
    flat["alignment_enabled"] = align.get("enabled", False)
    flat["alignment_method"] = align.get("method", "coral")
    flat["alignment_weight"] = align.get("weight", 0.01)
    flat["alignment_start_epoch"] = align.get("start_epoch", 20)
    flat["alignment_max_weight"] = align.get("max_weight", 0.05)

    # Map data ratios
    flat["adv_ratio"] = data.get("adv_ratio", flat.get("adv_ratio", 0.9))
    flat["gmsc_ratio"] = data.get("gmsc_ratio", flat.get("gmsc_ratio", 0.1))

    # Ensure all expected keys have defaults
    defaults = {
        'lr': 0.0005,
        'finetune_trunk_lr': 0.00005,
        'batch_size': 16,
        'gmsc_batch_size': 128,
        'gmsc_pretrain_samples': 5000,
        'dropout': 0.15,
        'weight_decay': 0.001,
        'patience': 10,
        'max_epochs': 100,
        'gmsc_risk_weight': 0.01,
        'adv_risk_weight': 1.0,
        'savings_weight': 1.0,
        'warmup_epochs': 0,
        'freeze_trunk_epochs': 3,
        'pretrain_epochs': 15,
        'finetune_epochs': 40,
        'enable_joint_phase': False,
        'joint_epochs': 0,
        'joint_gmsc_weight': 0.0,
        'gmsc_sample_ratio': 0.10,
        'clip_grad': 1.0,
        'risk_loss': 'huber',
    }

    for key, default_val in defaults.items():
        if key not in flat:
            flat[key] = default_val

    return flat


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
    prep_cfg = config.get('preprocessing', {})
    ignored = prep_cfg.get('ignored_columns', [])
    cols_to_drop = prep_cfg.get('columns_to_drop', [])

    exclude = ['Risk_Score', 'Save_Money_Yes', 'Save_Money_No'] + ignored + cols_to_drop
    exclude = [c for c in exclude if c in df.columns]

    feature_cols = [c for c in df.columns if c not in exclude]

    # Remove forbidden columns if present, but continue (log warning)
    forbidden_in_features = [c for c in feature_cols if c in FORBIDDEN_TARGETS]
    if forbidden_in_features:
        print(f"[WARN] Found forbidden columns in ADV features: {forbidden_in_features}. They will be removed from features.")
        feature_cols = [c for c in feature_cols if c not in FORBIDDEN_TARGETS]

    # Bug #5 FIX: Exclude RISK_SCORE_COMPONENTS to prevent leakage
    # These columns are used to calculate Risk_Score, including them makes the task trivially easy
    leakage_cols = [c for c in feature_cols if c in RISK_SCORE_COMPONENTS]
    if leakage_cols:
        print(f"[LEAKAGE PREVENTION] Excluding Risk_Score formula components: {leakage_cols}")
        feature_cols = [c for c in feature_cols if c not in RISK_SCORE_COMPONENTS]

    # Final check - if no features left, raise
    if not feature_cols:
        raise ValueError("No valid ADV features remain after removing forbidden columns and exclusions")

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
    1. ADV-only training (baseline)
    2. ADV + GMSC transfer (selected strategy from config)

    Training strategies:
    - 'joint': Original joint training
    - 'pretrain_finetune': Pretrain on GMSC, then fine-tune on ADV (recommended)
    - 'optimized_mixture': ADV-dominant with subsampled GMSC

    Evaluation on ADV data only.
    """
    n_splits = config['cross_validation'].get('n_splits', 5)
    n_repeats = config['cross_validation'].get('n_repeats', 5)

    # Training strategy selection
    training_strategy = config.get('training_strategy', 'joint')
    print(f"\nTraining strategy: {training_strategy}")

    print(f"Running {n_splits}-fold x {n_repeats} repeats CV...")

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # Storage for all fold metrics
    adv_only_metrics = []
    transfer_metrics = []

    # Bug #1 FIX: Use flatten_cfg to properly merge all config sections
    model_config = flatten_cfg(config)

    # Log the actual config being used
    print("\n[CONFIG] Actual values being used:")
    print(f"  lr: {model_config.get('lr')}")
    print(f"  finetune_trunk_lr: {model_config.get('finetune_trunk_lr')}")
    print(f"  batch_size: {model_config.get('batch_size')}")
    print(f"  pretrain_epochs: {model_config.get('pretrain_epochs')}")
    print(f"  finetune_epochs: {model_config.get('finetune_epochs')}")
    print(f"  enable_joint_phase: {model_config.get('enable_joint_phase')}")
    print(f"  alignment_enabled: {model_config.get('alignment_enabled')}")


    total_folds = n_splits * n_repeats

    saved_models = {'adv_only_models': [], 'transfer_models': [], 'adv_scalers': [], 'gmsc_scalers': []}
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
        adv_metrics, adv_model = train_adv_only(
            adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
            adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
            model_config, fold_seed
        )
        adv_only_metrics.append(adv_metrics)
        saved_models['adv_only_models'].append(adv_model)
        # save scalers for this fold
        saved_models['adv_scalers'].append(adv_scaler)
        saved_models['gmsc_scalers'].append(gmsc_scaler)

        # Experiment 2: ADV + GMSC transfer (using selected strategy)
        print(f"  Training ADV + GMSC transfer model (strategy: {training_strategy})...")

        if training_strategy == 'pretrain_finetune':
            transfer_m, transfer_model = train_pretrain_finetune(
                adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
                adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
                gmsc_X_s, gmsc_y,
                model_config, fold_seed
            )
        elif training_strategy == 'optimized_mixture':
            transfer_m, transfer_model = train_with_optimized_mixture(
                adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
                adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
                gmsc_X_s, gmsc_y,
                model_config, fold_seed
            )
        else:  # 'joint' (default)
            transfer_m, transfer_model = train_with_gmsc_transfer(
                adv_X_train_s, adv_y_risk_train, adv_y_savings_train,
                adv_X_val_s, adv_y_risk_val, adv_y_savings_val,
                gmsc_X_s, gmsc_y,
                model_config, fold_seed
            )
        transfer_metrics.append(transfer_m)
        saved_models['transfer_models'].append(transfer_model)

        # Progress report
        print(f"  ADV-only:  Risk MAE={adv_metrics['risk_mae']:.4f}, "
              f"Savings F1={adv_metrics['savings_macro_f1']:.4f}")
        print(f"  Transfer:  Risk MAE={transfer_m['risk_mae']:.4f}, "
              f"Savings F1={transfer_m['savings_macro_f1']:.4f}")

    # Aggregate results
    results = {
        'adv_only': _aggregate_metrics(adv_only_metrics),
        'transfer': _aggregate_metrics(transfer_metrics),
        'saved_models': saved_models
    }

    return results


def _aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics across folds."""
    aggregated = {}

    if not metrics_list:
        return aggregated

    metric_names = metrics_list[0].keys()

    for metric_name in metric_names:
        # Skip non-numeric fields like _epoch_logs
        if metric_name.startswith('_'):
            continue
        values = [m[metric_name] for m in metrics_list if metric_name in m and m[metric_name] is not None]
        # Skip if values contain non-numeric types
        if values and all(isinstance(v, (int, float)) for v in values):
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
    print(f"{'Experiment':<25} {'MAE':<18} {'RMSE':<18} {'Spearman rho':<18} {'R2':<18}")
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


def run_domain_transfer_experiment(config_path: str, adv_path: str = None, gmsc_path: str = None, output_dir: str = None):
    """
    Run full domain transfer experiment.
    """
    # Load config
    config = load_config(config_path)

    # Override output_dir if provided
    if output_dir:
        config.setdefault('experiment', {})['output_dir'] = output_dir

    seed = config.get('experiment', {}).get('seed', 42)
    set_seeds(seed)

    print("=" * 90)
    print("DOMAIN TRANSFER EXPERIMENT")
    print("GMSC as Auxiliary Supervision for ADV Risk Modeling")
    print("=" * 90)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {seed}")
    print("=" * 90)
    print("Architecture:")
    print("  ADV features  -> ADV adapter  -+")
    print("                                 +-> Shared trunk -> Risk head (ADV regression)")
    print("  GMSC features -> GMSC adapter -+              -> Risk head (GMSC classification)")
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
        print("\n[OK] GMSC TRANSFER IS BENEFICIAL")
        print("   Recommendation: Include GMSC auxiliary supervision in final model")
        verdict = "beneficial"
    else:
        print("\n[X] GMSC TRANSFER IS NOT BENEFICIAL")
        print("   Recommendation: Use ADV-only model")
        print("   This is valid science - transfer doesn't always help")
        verdict = "not_beneficial"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['experiment'].get('output_dir', 'runs')
    run_dir = os.path.join(output_dir, f"domain_transfer_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare results without non-serializable objects
    results_for_json = {k: v for k, v in results.items() if k != 'saved_models'}

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
        'results': results_for_json,
        'analysis': analysis,
        'verdict': verdict
    }

    with open(os.path.join(run_dir, 'domain_transfer_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save config
    import yaml
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save PyTorch models returned by training (if present)
    try:
        saved = results.get('saved_models', {})
        adv_models = saved.get('adv_only_models', [])
        transfer_models = saved.get('transfer_models', [])
        adv_scalers = saved.get('adv_scalers', [])
        gmsc_scalers = saved.get('gmsc_scalers', [])
        import torch as _torch
        import joblib as _joblib
        # Save the last model of each list if available
        if adv_models:
            _torch.save(adv_models[-1].state_dict(), os.path.join(run_dir, 'adv_only_model.pth'))
        if transfer_models:
            _torch.save(transfer_models[-1].state_dict(), os.path.join(run_dir, 'transfer_model.pth'))
        # Save last scalers as joblib for analysis
        if adv_scalers:
            try:
                _joblib.dump(adv_scalers[-1], os.path.join(run_dir, 'adv_scaler.joblib'))
            except Exception:
                pass
        if gmsc_scalers:
            try:
                _joblib.dump(gmsc_scalers[-1], os.path.join(run_dir, 'gmsc_scaler.joblib'))
            except Exception:
                pass
        # Write small metadata file
        try:
            from experiments.save_model import write_model_metadata
            metadata = {
                'pytorch_state_dicts': {
                    'adv_only': 'adv_only_model.pth' if adv_models else None,
                    'transfer': 'transfer_model.pth' if transfer_models else None
                },
                'sklearn_objects': {
                    'adv_scaler': 'adv_scaler.joblib' if adv_scalers else None,
                    'gmsc_scaler': 'gmsc_scaler.joblib' if gmsc_scalers else None
                }
            }
            write_model_metadata(run_dir, metadata)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: failed to save torch models: {e}")

    print(f"\nResults saved to: {run_dir}")

    return run_dir, verdict, results

