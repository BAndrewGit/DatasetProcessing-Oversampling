import os
import json
import argparse
import random
import hashlib
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)
from scipy.stats import spearmanr

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

FORBIDDEN_TARGETS = ["Behavior_Risk_Level"]


def set_seeds(seed):
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


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def config_hash(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def validate_target(config):
    target = config['data']['target_column']
    if target in FORBIDDEN_TARGETS:
        raise ValueError(
            f"BLOCKED: '{target}' is a forbidden target (circular label). "
            f"Use 'Risk_Score' for regression or 'Save_Money_Yes' for classification."
        )
    return target


def load_dataset(config, dataset_path=None):
    path = dataset_path or config['data'].get('dataset_path')
    if path is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Select the dataset file...")
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            parent=root
        )
        root.destroy()
        if not path:
            raise ValueError("No dataset selected")

    print(f"Loading dataset: {path}")
    return pd.read_csv(path)


def preprocess_data(df, config):
    target = validate_target(config)

    # Drop auxiliary columns
    cols_to_drop = config['preprocessing'].get('columns_to_drop', [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # MANDATORY: Remove ignored columns from features (but keep in df for reference)
    ignored = config['preprocessing'].get('ignored_columns', [])
    ignored = [c for c in ignored if c in df.columns and c != target]

    # Verify target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Build feature set excluding target and ignored columns
    feature_cols = [c for c in df.columns if c != target and c not in ignored]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Print what we're ignoring
    if ignored:
        print(f"IGNORED columns (not used in training): {ignored}")

    # Verify no forbidden columns in features
    for col in X.columns:
        if col in FORBIDDEN_TARGETS:
            raise ValueError(f"BLOCKED: Forbidden column '{col}' found in features!")

    return X, y


def build_model(config):
    model_type = config['model']['type']
    params = config['model']['params'].get(model_type, {})
    seed = config['experiment']['seed']
    target_type = config['data'].get('target_type', 'regression')

    # Regression models
    # Note: Ridge and Lasso don't have random_state in older sklearn versions
    # They are deterministic solvers, so reproducibility comes from data ordering
    if model_type == 'ridge':
        return Ridge(**params)
    elif model_type == 'lasso':
        return Lasso(**params)
    elif model_type == 'xgboost_reg':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBRegressor(random_state=seed, verbosity=0, **params)
    elif model_type == 'lightgbm_reg':
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        return LGBMRegressor(random_state=seed, verbose=-1, **params)
    elif model_type == 'random_forest_reg':
        return RandomForestRegressor(random_state=seed, **params)

    # Classification models
    elif model_type == 'logistic_regression':
        return LogisticRegression(random_state=seed, **params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=seed, **params)
    elif model_type == 'xgboost_clf':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(random_state=seed, verbosity=0, use_label_encoder=False, eval_metric='logloss', **params)
    elif model_type == 'lightgbm_clf':
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        return LGBMClassifier(random_state=seed, verbose=-1, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_repeated_cv_regression(model, X, y, config):
    cv_config = config.get('cross_validation', {})
    seed = config['experiment']['seed']
    n_splits = cv_config.get('n_splits', 5)
    n_repeats = cv_config.get('n_repeats', 10)

    print(f"Running {n_splits}-fold × {n_repeats} repeats CV (regression)...")

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    mae_scores = []
    rmse_scores = []
    spearman_scores = []
    r2_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = build_model(config)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        spearman_scores.append(spearmanr(y_val, y_pred)[0])
        r2_scores.append(r2_score(y_val, y_pred))

    return {
        'mae': {'mean': np.mean(mae_scores), 'std': np.std(mae_scores), 'all': mae_scores},
        'rmse': {'mean': np.mean(rmse_scores), 'std': np.std(rmse_scores), 'all': rmse_scores},
        'spearman': {'mean': np.mean(spearman_scores), 'std': np.std(spearman_scores), 'all': spearman_scores},
        'r2': {'mean': np.mean(r2_scores), 'std': np.std(r2_scores), 'all': r2_scores},
        'n_folds': n_splits,
        'n_repeats': n_repeats
    }


def run_repeated_cv_classification(model, X, y, config):
    cv_config = config.get('cross_validation', {})
    seed = config['experiment']['seed']
    n_splits = cv_config.get('n_splits', 5)
    n_repeats = cv_config.get('n_repeats', 10)

    print(f"Running {n_splits}-fold × {n_repeats} repeats CV (classification)...")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    f1_scores = []
    acc_scores = []
    prec_scores = []
    rec_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = build_model(config)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        f1_scores.append(f1_score(y_val, y_pred, average='macro'))
        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        rec_scores.append(recall_score(y_val, y_pred, average='macro', zero_division=0))

    return {
        'macro_f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'all': f1_scores},
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'all': acc_scores},
        'precision': {'mean': np.mean(prec_scores), 'std': np.std(prec_scores), 'all': prec_scores},
        'recall': {'mean': np.mean(rec_scores), 'std': np.std(rec_scores), 'all': rec_scores},
        'n_folds': n_splits,
        'n_repeats': n_repeats
    }


def save_results(run_dir, config, cv_results, model, X, y, target_type):
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save metrics (convert numpy arrays to lists for JSON)
    results_json = {
        'experiment_name': config['experiment']['name'],
        'seed': config['experiment']['seed'],
        'model_type': config['model']['type'],
        'target_column': config['data']['target_column'],
        'target_type': target_type,
        'cv_results': {}
    }

    for metric, values in cv_results.items():
        if isinstance(values, dict) and 'mean' in values:
            results_json['cv_results'][metric] = {
                'mean': float(values['mean']),
                'std': float(values['std'])
            }
        else:
            results_json['cv_results'][metric] = values

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Train final model on all data and save
    model.fit(X, y)
    model_path = os.path.join(run_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save CV distribution plot
    if config.get('metrics', {}).get('save_plots', True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metric_names = [k for k in cv_results.keys() if isinstance(cv_results[k], dict) and 'all' in cv_results[k]]

        for i, metric in enumerate(metric_names[:4]):
            ax = axes[i]
            scores = cv_results[metric]['all']
            ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(cv_results[metric]['mean'], color='red', linestyle='--',
                      label=f"Mean: {cv_results[metric]['mean']:.4f}")
            ax.set_title(f"{metric.upper()} Distribution")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.suptitle(f"{config['model']['type']} - {config['data']['target_column']}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'cv_distribution.png'), dpi=150)
        plt.close()

    print(f"Results saved to: {run_dir}")


def run_experiment(config_path, dataset_path=None):
    config = load_config(config_path)
    seed = config['experiment']['seed']
    set_seeds(seed)

    # Validate target is not forbidden
    target = validate_target(config)
    target_type = config['data'].get('target_type', 'regression')

    print("=" * 60)
    print(f"CLEAN BASELINE EXPERIMENT")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Target: {target} ({target_type})")
    print(f"Seed: {seed}")
    print("=" * 60)
    print(f"NOTE: Behavior_Risk_Level is FORBIDDEN and will NOT be used.")
    print("=" * 60)

    # Load and preprocess data
    df = load_dataset(config, dataset_path)
    X, y = preprocess_data(df, config)

    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    if target_type == 'regression':
        print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    else:
        print(f"Class distribution: {y.value_counts().to_dict()}")

    # Check augmentation is disabled for baseline
    if config.get('augmentation', {}).get('enabled', False):
        print("\nWARNING: Augmentation is enabled but this is a BASELINE experiment!")
        print("Disabling augmentation for clean baseline...")
        config['augmentation']['enabled'] = False

    # Build model
    model = build_model(config)
    print(f"\nModel: {config['model']['type']}")

    # Run repeated CV
    if target_type == 'regression':
        cv_results = run_repeated_cv_regression(model, X, y, config)
        print("\n" + "=" * 60)
        print("REGRESSION RESULTS (Repeated K-Fold CV)")
        print("=" * 60)
        print(f"MAE:      {cv_results['mae']['mean']:.4f} ± {cv_results['mae']['std']:.4f}")
        print(f"RMSE:     {cv_results['rmse']['mean']:.4f} ± {cv_results['rmse']['std']:.4f}")
        print(f"Spearman: {cv_results['spearman']['mean']:.4f} ± {cv_results['spearman']['std']:.4f}")
        print(f"R²:       {cv_results['r2']['mean']:.4f} ± {cv_results['r2']['std']:.4f}")
    else:
        cv_results = run_repeated_cv_classification(model, X, y, config)
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS (Repeated Stratified K-Fold CV)")
        print("=" * 60)
        print(f"Macro-F1:  {cv_results['macro_f1']['mean']:.4f} ± {cv_results['macro_f1']['std']:.4f}")
        print(f"Accuracy:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        print(f"Recall:    {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['name']}_{timestamp}_{config_hash(config)}"
    output_dir = config['experiment'].get('output_dir', 'runs')
    run_dir = os.path.join(output_dir, run_name)

    # Save results
    save_results(run_dir, config, cv_results, model, X, y, target_type)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    return run_dir


def run_all_baselines(dataset_path=None):
    print("\n" + "=" * 70)
    print("RUNNING ALL BASELINE MODELS")
    print("=" * 70)

    # Regression baselines
    regression_models = ['ridge', 'lasso', 'xgboost_reg']
    classification_models = ['logistic_regression', 'random_forest']

    results_summary = {'regression': {}, 'classification': {}}

    # Run regression baselines
    print("\n>>> REGRESSION BASELINES (Target: Risk_Score)")
    for model_type in regression_models:
        try:
            config = load_config('configs/baseline_regression.yaml')
            config['model']['type'] = model_type
            config['experiment']['name'] = f"baseline_reg_{model_type}"

            # Save temp config
            temp_config = f'configs/_temp_{model_type}.yaml'
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            run_dir = run_experiment(temp_config, dataset_path)

            # Load results
            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)

            results_summary['regression'][model_type] = metrics['cv_results']

            # Cleanup temp config
            os.remove(temp_config)
        except Exception as e:
            print(f"Error running {model_type}: {e}")

    # Run classification baselines
    print("\n>>> CLASSIFICATION BASELINES (Target: Save_Money_Yes)")
    for model_type in classification_models:
        try:
            config = load_config('configs/baseline_classification.yaml')
            config['model']['type'] = model_type
            config['experiment']['name'] = f"baseline_clf_{model_type}"

            temp_config = f'configs/_temp_{model_type}.yaml'
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            run_dir = run_experiment(temp_config, dataset_path)

            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)

            results_summary['classification'][model_type] = metrics['cv_results']

            os.remove(temp_config)
        except Exception as e:
            print(f"Error running {model_type}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print("\nREGRESSION (Risk_Score):")
    print("-" * 50)
    for model, metrics in results_summary['regression'].items():
        print(f"{model:20s} | MAE: {metrics.get('mae', {}).get('mean', 'N/A'):.4f} | "
              f"RMSE: {metrics.get('rmse', {}).get('mean', 'N/A'):.4f} | "
              f"Spearman: {metrics.get('spearman', {}).get('mean', 'N/A'):.4f}")

    print("\nCLASSIFICATION (Save_Money_Yes):")
    print("-" * 50)
    for model, metrics in results_summary['classification'].items():
        print(f"{model:20s} | Macro-F1: {metrics.get('macro_f1', {}).get('mean', 'N/A'):.4f} | "
              f"Accuracy: {metrics.get('accuracy', {}).get('mean', 'N/A'):.4f}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description='Run clean baseline ML experiment (no circular labels)')
    parser.add_argument('--config', '-c', type=str, default='configs/baseline_regression.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to dataset CSV (overrides config)')
    parser.add_argument('--all-baselines', action='store_true',
                       help='Run all baseline models (regression + classification)')
    args = parser.parse_args()

    if args.all_baselines:
        run_all_baselines(args.dataset)
    else:
        run_experiment(args.config, args.dataset)


if __name__ == "__main__":
    main()

