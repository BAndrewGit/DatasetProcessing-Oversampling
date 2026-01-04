# Augmentation Experiment Runner
# Tests whether synthetic data improves real-only performance
# Follows strict quality gates - synthetic rejected if ANY gate fails

# HARD REMOVALS (FOREVER):
# - NO retrain loops on enriched data
# - NO exponential dataset growth
# - NO final 50/50 balancing
# - NO GAN retraining on synthetic output

import os
import json
import argparse
import random
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score
)
from scipy.stats import spearmanr

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    SMOTE = None
    HAS_SMOTE = False

from DataAugmentation.quality_gates import SyntheticQualityGates, validate_synthetic_ratio


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


FORBIDDEN_TARGETS = ["Behavior_Risk_Level"]


def load_dataset(path):
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


def generate_synthetic_smote(X_train, y_train, ratio=0.15, seed=42):
    # SMOTE for classification - generates synthetic minority samples
    # Ratio controls how much synthetic data to add
    if not HAS_SMOTE:
        raise ImportError("imbalanced-learn not installed. Run: pip install imbalanced-learn")

    n_samples = int(len(X_train) * ratio)
    if n_samples < 5:
        return None, None

    # Find minority class
    classes, counts = np.unique(y_train, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    minority_count = np.min(counts)

    # Calculate target count for minority class after SMOTE
    target_minority = minority_count + n_samples

    smote = SMOTE(
        sampling_strategy={minority_class: target_minority},
        random_state=seed,
        k_neighbors=min(5, minority_count - 1)
    )

    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Extract only synthetic samples
        X_syn = X_resampled[len(X_train):]
        y_syn = y_resampled[len(X_train):]

        return X_syn, y_syn
    except Exception as e:
        print(f"SMOTE failed: {e}")
        return None, None


def generate_synthetic_regression(X_train, y_train, ratio=0.15, seed=42):
    # Simple synthetic generation for regression via jittering
    # Adds small noise to existing samples
    np.random.seed(seed)

    n_samples = int(len(X_train) * ratio)
    if n_samples < 5:
        return None, None

    # Randomly select samples to jitter
    indices = np.random.choice(len(X_train), n_samples, replace=True)
    X_selected = X_train[indices].copy()
    y_selected = y_train[indices].copy()

    # Add small noise (scaled by feature std)
    noise_scale = 0.1
    std_X = np.std(X_train, axis=0)
    std_X[std_X == 0] = 1
    noise_X = np.random.normal(0, noise_scale, X_selected.shape) * std_X
    X_syn = X_selected + noise_X

    # Add noise to target too
    std_y = np.std(y_train)
    noise_y = np.random.normal(0, noise_scale * float(std_y), len(y_selected))
    y_syn = y_selected + noise_y

    return X_syn, y_syn


def generate_cluster_synthetic(X_train, y_train, target_type='regression', ratio=0.20, seed=42):
    # Cluster-aware synthetic generation
    # Generates synthetic samples within each cluster
    try:
        from DataAugmentation.cluster_enrichment import generate_cluster_aware_synthetic
        X_syn, y_syn, cluster_info = generate_cluster_aware_synthetic(
            X_train, y_train,
            target_type=target_type,
            max_ratio=ratio,
            seed=seed
        )
        return X_syn, y_syn
    except Exception as e:
        print(f"Cluster-aware generation failed: {e}")
        return None, None


def run_augmentation_experiment(config_path, dataset_path=None):
    config = load_config(config_path)
    seed = config['experiment']['seed']
    set_seeds(seed)

    print("=" * 70)
    print("AUGMENTATION EXPERIMENT")
    print("Testing if synthetic data improves real-only performance")
    print("=" * 70)
    print("RULES:")
    print("  - Synthetic generated INSIDE training folds only")
    print("  - Synthetic ratio: 15-30%")
    print("  - Never reused across runs")
    print("  - ALL quality gates must pass")
    print("=" * 70)

    # Load data
    df = load_dataset(dataset_path or config['data'].get('dataset_path'))

    target = config['data']['target_column']
    target_type = config['data'].get('target_type', 'regression')

    if target in FORBIDDEN_TARGETS:
        raise ValueError(f"BLOCKED: '{target}' is a forbidden target")

    # Build features
    cols_to_drop = config['preprocessing'].get('columns_to_drop', [])
    ignored = config['preprocessing'].get('ignored_columns', [])

    feature_cols = [c for c in df.columns
                   if c != target
                   and c not in cols_to_drop
                   and c not in ignored
                   and c not in FORBIDDEN_TARGETS]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    print(f"\nTarget: {target} ({target_type})")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Get augmentation config
    aug_config = config.get('augmentation', {})
    synthetic_ratio = aug_config.get('synthetic_ratio', 0.15)
    max_ratio = aug_config.get('max_ratio', 0.30)
    synthetic_method = aug_config.get('method', 'jitter')  # jitter, smote, or cluster

    # Validate ratio
    if synthetic_ratio > max_ratio:
        print(f"WARNING: synthetic_ratio {synthetic_ratio} exceeds max {max_ratio}. Capping.")
        synthetic_ratio = max_ratio

    print(f"\nSynthetic ratio: {synthetic_ratio:.0%}")
    print(f"Synthetic method: {synthetic_method}")

    # CV setup
    cv_config = config.get('cross_validation', {})
    n_splits = cv_config.get('n_splits', 5)
    n_repeats = cv_config.get('n_repeats', 5)

    if target_type == 'regression':
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # Results
    results_real_only = {'mae': [], 'rmse': [], 'spearman': []} if target_type == 'regression' else {'f1': [], 'acc': []}
    results_augmented = {'mae': [], 'rmse': [], 'spearman': []} if target_type == 'regression' else {'f1': [], 'acc': []}
    quality_gate_results = []

    print(f"\nRunning {n_splits}-fold x {n_repeats} repeats CV...")
    print("-" * 70)

    fold_idx = 0
    synthetic_accepted = 0
    synthetic_rejected = 0

    for train_idx, test_idx in cv.split(X, y if target_type == 'classification' else None):
        fold_idx += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Real-only baseline
        if target_type == 'regression':
            model_real = Ridge(random_state=seed)
            model_real.fit(X_train_s, y_train)
            pred_real = model_real.predict(X_test_s)

            results_real_only['mae'].append(mean_absolute_error(y_test, pred_real))
            results_real_only['rmse'].append(np.sqrt(mean_squared_error(y_test, pred_real)))
            results_real_only['spearman'].append(spearmanr(y_test, pred_real)[0])
        else:
            model_real = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000)
            model_real.fit(X_train_s, y_train)
            pred_real = model_real.predict(X_test_s)

            results_real_only['f1'].append(f1_score(y_test, pred_real, average='macro'))
            results_real_only['acc'].append(accuracy_score(y_test, pred_real))

        # Generate synthetic data INSIDE this fold
        if synthetic_method == 'cluster':
            X_syn, y_syn = generate_cluster_synthetic(X_train_s, y_train, target_type, ratio=synthetic_ratio, seed=seed + fold_idx)
        elif target_type == 'regression' or synthetic_method == 'jitter':
            X_syn, y_syn = generate_synthetic_regression(X_train_s, y_train, ratio=synthetic_ratio, seed=seed + fold_idx)
        else:
            X_syn, y_syn = generate_synthetic_smote(X_train_s, y_train, ratio=synthetic_ratio, seed=seed + fold_idx)

        if X_syn is None or len(X_syn) == 0:
            # No synthetic data generated - use real-only results
            if target_type == 'regression':
                results_augmented['mae'].append(results_real_only['mae'][-1])
                results_augmented['rmse'].append(results_real_only['rmse'][-1])
                results_augmented['spearman'].append(results_real_only['spearman'][-1])
            else:
                results_augmented['f1'].append(results_real_only['f1'][-1])
                results_augmented['acc'].append(results_real_only['acc'][-1])
            continue

        # Validate synthetic ratio
        try:
            validate_synthetic_ratio(len(X_syn), len(X_train_s), max_ratio=max_ratio)
        except ValueError as e:
            print(f"  Fold {fold_idx}: Synthetic ratio exceeded - {e}")
            continue

        # Run quality gates
        gates = SyntheticQualityGates(seed=seed, verbose=False)
        passed, gate_results = gates.run_all_gates(X_train_s, y_train, X_syn, y_syn, target_type)
        quality_gate_results.append(gate_results)

        if passed:
            synthetic_accepted += 1
            # Use augmented training
            X_train_aug = np.vstack([X_train_s, X_syn])
            y_train_aug = np.concatenate([y_train, y_syn])

            if target_type == 'regression':
                model_aug = Ridge(random_state=seed)
                model_aug.fit(X_train_aug, y_train_aug)
                pred_aug = model_aug.predict(X_test_s)

                results_augmented['mae'].append(mean_absolute_error(y_test, pred_aug))
                results_augmented['rmse'].append(np.sqrt(mean_squared_error(y_test, pred_aug)))
                results_augmented['spearman'].append(spearmanr(y_test, pred_aug)[0])
            else:
                model_aug = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000)
                model_aug.fit(X_train_aug, y_train_aug)
                pred_aug = model_aug.predict(X_test_s)

                results_augmented['f1'].append(f1_score(y_test, pred_aug, average='macro'))
                results_augmented['acc'].append(accuracy_score(y_test, pred_aug))
        else:
            synthetic_rejected += 1
            # Quality gates failed - use real-only
            if target_type == 'regression':
                results_augmented['mae'].append(results_real_only['mae'][-1])
                results_augmented['rmse'].append(results_real_only['rmse'][-1])
                results_augmented['spearman'].append(results_real_only['spearman'][-1])
            else:
                results_augmented['f1'].append(results_real_only['f1'][-1])
                results_augmented['acc'].append(results_real_only['acc'][-1])

        if fold_idx % 5 == 0:
            print(f"  Completed fold {fold_idx}/{n_splits * n_repeats}")

    # Compute summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nQuality Gates: {synthetic_accepted} accepted, {synthetic_rejected} rejected")

    if target_type == 'regression':
        print("\nREGRESSION METRICS:")
        print("-" * 50)
        print(f"{'Metric':<15} {'Real-Only':<20} {'Augmented':<20}")
        print("-" * 50)

        for metric in ['mae', 'rmse', 'spearman']:
            real_mean = np.mean(results_real_only[metric])
            real_std = np.std(results_real_only[metric])
            aug_mean = np.mean(results_augmented[metric])
            aug_std = np.std(results_augmented[metric])

            print(f"{metric.upper():<15} {real_mean:.4f} ± {real_std:.4f}    {aug_mean:.4f} ± {aug_std:.4f}")

        # Decision
        mae_real = np.mean(results_real_only['mae'])
        mae_aug = np.mean(results_augmented['mae'])
        improvement = (mae_real - mae_aug) / mae_real * 100

        std_real = np.std(results_real_only['mae'])
        std_aug = np.std(results_augmented['mae'])
        stability_change = (std_real - std_aug) / std_real * 100 if std_real > 0 else 0
    else:
        print("\nCLASSIFICATION METRICS:")
        print("-" * 50)
        print(f"{'Metric':<15} {'Real-Only':<20} {'Augmented':<20}")
        print("-" * 50)

        for metric in ['f1', 'acc']:
            real_mean = np.mean(results_real_only[metric])
            real_std = np.std(results_real_only[metric])
            aug_mean = np.mean(results_augmented[metric])
            aug_std = np.std(results_augmented[metric])

            print(f"{metric.upper():<15} {real_mean:.4f} ± {real_std:.4f}    {aug_mean:.4f} ± {aug_std:.4f}")

        # Decision
        f1_real = np.mean(results_real_only['f1'])
        f1_aug = np.mean(results_augmented['f1'])
        improvement = (f1_aug - f1_real) / f1_real * 100

        std_real = np.std(results_real_only['f1'])
        std_aug = np.std(results_augmented['f1'])
        stability_change = (std_real - std_aug) / std_real * 100 if std_real > 0 else 0

    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if target_type == 'regression':
        print(f"MAE improvement: {improvement:+.2f}%")
    else:
        print(f"F1 improvement: {improvement:+.2f}%")
    print(f"Stability improvement: {stability_change:+.2f}%")

    # Threshold: synthetic must improve by at least 1% AND not hurt stability
    useful = improvement > 1.0 and stability_change > -20

    if useful:
        print("\n✅ SYNTHETIC DATA IS USEFUL")
        print("   Recommendation: Include in final pipeline")
        verdict = "useful"
    else:
        print("\n❌ SYNTHETIC DATA IS NOT USEFUL")
        print("   Recommendation: Remove from pipeline")
        print("   This is GOOD science - augmentation does not help on small-n")
        verdict = "not_useful"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['experiment'].get('output_dir', 'runs')
    run_dir = os.path.join(output_dir, f"augmentation_experiment_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results_json = {
        'experiment': 'augmentation_test',
        'timestamp': timestamp,
        'target': target,
        'target_type': target_type,
        'synthetic_ratio': synthetic_ratio,
        'n_folds': n_splits * n_repeats,
        'synthetic_accepted': synthetic_accepted,
        'synthetic_rejected': synthetic_rejected,
        'results_real_only': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_real_only.items()},
        'results_augmented': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_augmented.items()},
        'improvement_pct': float(improvement),
        'stability_change_pct': float(stability_change),
        'verdict': verdict
    }

    with open(os.path.join(run_dir, 'augmentation_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nResults saved to: {run_dir}")

    return verdict, results_json


def main():
    parser = argparse.ArgumentParser(description='Test if synthetic augmentation helps')
    parser.add_argument('--config', '-c', type=str, default='configs/augmentation_experiment.yaml',
                       help='Path to config YAML')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to dataset CSV')
    args = parser.parse_args()

    verdict, results = run_augmentation_experiment(args.config, args.dataset)
    return verdict


if __name__ == "__main__":
    main()

