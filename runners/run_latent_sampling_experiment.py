# Import startup module first to silence joblib/loky warnings
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _startup  # noqa: F401 - silences warnings

import argparse
import yaml
import pandas as pd
import numpy as np
import json

# ensure project root is on sys.path so `import experiments` works when running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.latent_experiment import run_latent_fold, run_latent_fold_with_tuning
from experiments.save_model import save_sklearn_model, write_model_metadata
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/latent_sampling/experiment.yaml")
    parser.add_argument("--dataset", required=True, help="data/processed/1_encoded.csv")
    parser.add_argument("--output", required=False, help="runs/latent_sampling_<ts>")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output:
        out_root = args.output
    else:
        import datetime
        out_root = os.path.join("runs", f"latent_sampling_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(out_root, exist_ok=True)
    # save exact config used
    with open(os.path.join(out_root, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    df = pd.read_csv(args.dataset)
    target = cfg.get("target", cfg.get("data", {}).get("target_column", "Risk_Score"))
    task = cfg.get("task", "regression")
    X = df.drop(columns=[cfg.get("forbidden_column", "Behavior_Risk_Level"), target], errors="ignore")
    y = df[target]
    seed = cfg.get("seed", cfg.get("experiment", {}).get("seed", 42))
    n_splits = cfg.get("n_splits", cfg.get("cross_validation", {}).get("n_splits", 5))

    # Normalize config: support both flat keys and nested keys
    # e.g., pca_candidates or pca.ks
    normalized_cfg = dict(cfg)
    if 'pca_candidates' not in normalized_cfg and 'pca' in cfg:
        normalized_cfg['pca_candidates'] = cfg['pca'].get('ks', [30, 25, 20, 15, 10, 8, 5, 3])
    if 'k_candidates' not in normalized_cfg and 'clustering' in cfg:
        normalized_cfg['k_candidates'] = cfg['clustering'].get('ks', [2, 3, 4, 5])
    if 'synth_grid' not in normalized_cfg and 'sampling' in cfg:
        normalized_cfg['synth_grid'] = cfg['sampling'].get('synth_counts', [0, 10, 50, 100])
    if 'global_cap_frac' not in normalized_cfg and 'sampling' in cfg:
        normalized_cfg['global_cap_frac'] = cfg['sampling'].get('global_cap', 0.3)
    if 'per_cluster_cap_frac' not in normalized_cfg and 'sampling' in cfg:
        normalized_cfg['per_cluster_cap_frac'] = cfg['sampling'].get('per_cluster_cap', 0.2)

    rk = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = 0
    all_metrics = []
    use_nested = normalized_cfg.get('use_nested_cv', False)

    # Run leakage checks on first fold (fast sanity check)
    print("\n=== LEAKAGE SANITY CHECKS ===")
    from experiments.sanity_checks import run_all_leakage_checks
    from sklearn.linear_model import HuberRegressor, LogisticRegression
    first_train_idx, first_val_idx = next(rk.split(X))

    # FIX 1: Always run shuffled-y test with actual model_fn
    if task == "regression":
        model_fn = lambda: HuberRegressor(epsilon=1.35, max_iter=500)
    else:
        model_fn = lambda: LogisticRegression(max_iter=500)

    leakage_results = run_all_leakage_checks(
        X.values, y.values, first_train_idx, first_val_idx,
        feature_names=list(X.columns), task=task, model_fn=model_fn
    )

    # Save leakage check results
    with open(os.path.join(out_root, "leakage_check.json"), "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            if isinstance(obj, tuple):
                return [convert(i) for i in obj]
            return obj
        json.dump(convert(leakage_results), f, indent=2)

    if leakage_results['overall']['severity'] == 'CRITICAL':
        print(f"  [CRITICAL] {leakage_results['overall']['message']}")
        for check_name, check_result in leakage_results.items():
            if check_name != 'overall' and check_result.get('severity') == 'CRITICAL':
                msg = check_result.get('message', check_result.get('issues', str(check_result)))
                print(f"    - {check_name}: {msg}")

        # Show specific diagnostics for perfect baseline (most common issue)
        if leakage_results.get('perfect_baseline', {}).get('severity') == 'CRITICAL':
            pb = leakage_results['perfect_baseline']
            print(f"\n  === PERFECT BASELINE DIAGNOSTIC ===")
            print(f"  R² = {pb.get('r2', 'N/A'):.6f} (suspicious if > 0.999)")
            print(f"  MAE = {pb.get('mae', 'N/A'):.2e}")
            print(f"  Relative MAE = {pb.get('relative_mae', 'N/A'):.2e} (suspicious if < 1e-6)")
            print(f"\n  POSSIBLE CAUSES:")
            print(f"    1. Target column (or formula components) is in features")
            print(f"    2. Scaler/PCA fitted on full data before CV split")
            print(f"    3. Duplicate rows between train/val")
            print(f"    4. Target is trivially recoverable from a single feature")
            print(f"\n  TO FIX: Check RISK_SCORE_COMPONENTS exclusion, verify CV split integrity")

        print("\n  [!!!] WARNING: Proceeding with experiment but results are likely INVALID!")
        print("  [!!!] Fix the leakage issue before trusting any metrics.\n")
    elif leakage_results['overall']['severity'] == 'WARNING':
        print(f"  [WARNING] {leakage_results['overall']['message']}")
    else:
        print(f"  [OK] All leakage checks passed")
    print("")

    # Reset KFold for actual experiment
    rk = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in rk.split(X):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        out_dir = os.path.join(out_root, f"fold_{fold}")

        # Use nested CV tuning if enabled
        if use_nested:
            res = run_latent_fold_with_tuning(X_train, y_train, X_val, y_val, normalized_cfg.copy(), out_dir, task=task)
        else:
            res = run_latent_fold(X_train, y_train, X_val, y_val, normalized_cfg, out_dir, task=task)
        all_metrics.append(res)

    # Save summary.json
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Aggregate metrics across folds
    aggregated = {}
    if all_metrics:
        def robust_aggregate(vals):
            """Compute robust statistics: median, IQR, mean, std."""
            arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
            if len(arr) == 0:
                return None
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'n': len(arr)
            }

        # Collect baseline metrics with robust stats
        baseline_keys = all_metrics[0].get('baseline', {}).keys() if all_metrics[0].get('baseline') else []
        for key in baseline_keys:
            if key == 'r2_warning':
                continue
            vals = [m['baseline'].get(key) for m in all_metrics if m.get('baseline') and m['baseline'].get(key) is not None]
            if vals:
                stats = robust_aggregate(vals)
                if stats:
                    aggregated[f'baseline_{key}_mean'] = stats['mean']
                    aggregated[f'baseline_{key}_std'] = stats['std']
                    aggregated[f'baseline_{key}_median'] = stats['median']
                    aggregated[f'baseline_{key}_iqr'] = stats['iqr']
                    aggregated[f'baseline_{key}_p25'] = stats['p25']
                    aggregated[f'baseline_{key}_p75'] = stats['p75']

        # Collect candidate metrics with robust stats
        candidate_metrics = [m.get('candidate') for m in all_metrics if m.get('candidate')]
        if candidate_metrics:
            cand_keys = [k for k in candidate_metrics[0].keys() if k != 'r2_warning']
            for key in cand_keys:
                vals = [m.get(key) for m in candidate_metrics if m.get(key) is not None]
                if vals:
                    stats = robust_aggregate(vals)
                    if stats:
                        aggregated[f'candidate_{key}_mean'] = stats['mean']
                        aggregated[f'candidate_{key}_std'] = stats['std']
                        aggregated[f'candidate_{key}_median'] = stats['median']
                        aggregated[f'candidate_{key}_iqr'] = stats['iqr']

        # use_synth ratio
        use_synth_count = sum(1 for m in all_metrics if m.get('use_synth'))
        aggregated['folds_using_synth'] = use_synth_count
        aggregated['total_folds'] = len(all_metrics)

        # FIX 5: Fold-by-fold comparison - track improved / unchanged / degraded
        fold_comparison = []
        n_improved = 0
        n_unchanged = 0
        n_degraded = 0
        n_catastrophic = 0
        worst_degradation = 0.0
        worst_fold = None

        for i, m in enumerate(all_metrics):
            baseline = m.get('baseline', {})
            candidate = m.get('candidate', {})
            use_synth = m.get('use_synth', False)

            if baseline and candidate and use_synth:
                base_mae = baseline.get('mae', 0)
                cand_mae = candidate.get('mae', 0)

                if base_mae > 0:
                    delta_pct = (cand_mae - base_mae) / base_mae * 100
                else:
                    delta_pct = 0 if cand_mae == 0 else float('inf')

                # Classify fold outcome
                if delta_pct < -2:  # >2% improvement
                    status = 'improved'
                    n_improved += 1
                elif delta_pct > 5:  # >5% degradation
                    status = 'degraded'
                    n_degraded += 1
                    if delta_pct > worst_degradation:
                        worst_degradation = delta_pct
                        worst_fold = f"fold_{i+1}"
                else:
                    status = 'unchanged'
                    n_unchanged += 1

                # Check for catastrophic (>100% degradation)
                if candidate.get('catastrophic', False) or delta_pct > 100:
                    n_catastrophic += 1
                    status = 'catastrophic'

                fold_comparison.append({
                    'fold': f"fold_{i+1}",
                    'baseline_mae': base_mae,
                    'candidate_mae': cand_mae,
                    'delta_pct': delta_pct,
                    'status': status,
                    'use_synth': use_synth
                })
            else:
                fold_comparison.append({
                    'fold': f"fold_{i+1}",
                    'baseline_mae': baseline.get('mae'),
                    'candidate_mae': None,
                    'delta_pct': None,
                    'status': 'skipped' if not use_synth else 'no_candidate',
                    'use_synth': use_synth
                })

        aggregated['fold_comparison'] = fold_comparison
        aggregated['n_improved'] = n_improved
        aggregated['n_unchanged'] = n_unchanged
        aggregated['n_degraded'] = n_degraded
        aggregated['n_catastrophic'] = n_catastrophic
        aggregated['worst_degradation_pct'] = worst_degradation
        aggregated['worst_fold'] = worst_fold

        # Determine verdict based on STABILITY, not means
        # FIX 5: Decision based on % folds improved and worst-case
        synth_ratio = use_synth_count / len(all_metrics) if all_metrics else 0

        # Strict verdict logic:
        # - "useful": majority improved, no catastrophic, worst < 20%
        # - "partial": some improved, no catastrophic
        # - "not_useful": any catastrophic OR majority degraded OR worst > 50%
        if n_catastrophic > 0:
            verdict = "not_useful"
            verdict_detail = f"CATASTROPHIC failure in {n_catastrophic} fold(s) - synthetic data REJECTED"
        elif worst_degradation > 50:
            verdict = "not_useful"
            verdict_detail = f"Worst fold degradation {worst_degradation:.1f}% (>{50}%) - synthetic data REJECTED"
        elif n_degraded > n_improved:
            verdict = "not_useful"
            verdict_detail = f"More folds degraded ({n_degraded}) than improved ({n_improved}) - synthetic data REJECTED"
        elif synth_ratio < 0.5:
            verdict = "not_useful"
            verdict_detail = f"Only {use_synth_count}/{len(all_metrics)} folds passed quality gates"
        elif n_improved > n_degraded and worst_degradation < 20:
            verdict = "useful"
            verdict_detail = f"{n_improved} improved, {n_unchanged} unchanged, {n_degraded} degraded (worst: {worst_degradation:.1f}%)"
        else:
            verdict = "partial"
            verdict_detail = f"{n_improved} improved, {n_unchanged} unchanged, {n_degraded} degraded (worst: {worst_degradation:.1f}%)"

        aggregated['verdict'] = verdict
        aggregated['verdict_detail'] = verdict_detail

        # Check for high variance (instability warning)
        n_folds = len(all_metrics)
        if n_folds < 5:
            print(f"\n  [WARN] Only {n_folds} folds - results may be unstable. Recommend 5+ folds.")

        # Report PRIMARY metrics using MEDIAN [IQR] format (robust to outliers)
        print(f"\n  === REGRESSION METRICS (ROBUST: median [p25-p75]) ===")

        def fmt_robust(key):
            med = aggregated.get(f'baseline_{key}_median')
            p25 = aggregated.get(f'baseline_{key}_p25')
            p75 = aggregated.get(f'baseline_{key}_p75')
            if med is None:
                return "N/A"
            return f"{med:.4f} [{p25:.3f}-{p75:.3f}]"

        if 'baseline_mae_median' in aggregated:
            print(f"  Baseline MAE:      {fmt_robust('mae')}")
        if 'baseline_spearman_median' in aggregated:
            print(f"  Baseline Spearman: {fmt_robust('spearman')}")
        if 'baseline_r2_median' in aggregated:
            r2_med = aggregated.get('baseline_r2_median', 0)
            r2_note = " (WARNING: negative!)" if r2_med < 0 else ""
            print(f"  Baseline R2:       {fmt_robust('r2')}{r2_note}")

        # FIX 5: Report fold-by-fold stability
        print(f"\n  === FOLD-BY-FOLD STABILITY ===")
        print(f"  Improved:    {n_improved}/{len(all_metrics)} folds")
        print(f"  Unchanged:   {n_unchanged}/{len(all_metrics)} folds")
        print(f"  Degraded:    {n_degraded}/{len(all_metrics)} folds")
        if n_catastrophic > 0:
            print(f"  CATASTROPHIC: {n_catastrophic}/{len(all_metrics)} folds [!!!]")
        if worst_fold:
            print(f"  Worst case:  {worst_fold} ({worst_degradation:.1f}% degradation)")

        # Show variance warning if std is high relative to mean
        if 'baseline_mae_mean' in aggregated and 'baseline_mae_std' in aggregated:
            cv = aggregated['baseline_mae_std'] / aggregated['baseline_mae_mean'] if aggregated['baseline_mae_mean'] > 0 else 0
            if cv > 0.3:
                print(f"  [WARN] High variance in MAE (CV={cv:.2f}). Results may be unstable.")

        print(f"\n  [VERDICT] {verdict.upper()}: {verdict_detail}")

    # Save aggregated metrics.json at run root
    with open(os.path.join(out_root, "metrics.json"), "w") as f:
        json.dump(aggregated, f, indent=2)

    # --- Save a representative model.joblib trained on full dataset ---
    try:
        # Train a final model on full data for analysis tooling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        if task == "regression":
            # Use robust model for small data (same logic as latent_experiment)
            n_samples = len(X)
            if n_samples < 200:
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor(epsilon=1.35, max_iter=500)
            elif n_samples < 500:
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
            else:
                model = MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=seed, max_iter=500)
        else:
            model = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, random_state=seed, max_iter=500)

        model.fit(X_scaled_df, y.values)

        # Save model and scaler
        save_sklearn_model(model, os.path.join(out_root, 'model.joblib'))
        save_sklearn_model(scaler, os.path.join(out_root, 'scaler.joblib'))

        # Save data profile
        data_profile = {
            'features_used': list(X.columns),
            'n_features': len(X.columns),
            'n_samples': len(X),
            'target': target,
            'task': task
        }
        with open(os.path.join(out_root, 'data_profile.json'), 'w') as f:
            json.dump(data_profile, f, indent=2)

        # Write metadata
        metadata = {
            'sklearn_objects': {
                'model': 'model.joblib',
                'scaler': 'scaler.joblib'
            },
            'task': task,
            'target': target,
            'n_folds': cfg.get('n_splits', 5),
            'seed': seed
        }
        write_model_metadata(out_root, metadata)

        print(f"Saved representative model to: {os.path.join(out_root, 'model.joblib')}")
    except Exception as e:
        print(f"Warning: failed to save representative model: {e}")

    # --- Generate and save oversampled dataset using latent space ---
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from experiments.latent_sampling import LatentSampler

        print("Generating oversampled dataset...")

        # Fit PCA on full data
        pca_k = normalized_cfg.get('pca_candidates', [10, 8, 5])[0]  # Use first candidate
        pca_k = min(pca_k, X.shape[1], X.shape[0] - 1)

        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X.values)

        pca = PCA(n_components=pca_k, whiten=True, random_state=seed)
        Z = pca.fit_transform(X_scaled_full)

        # Cluster in latent space
        n_clusters = normalized_cfg.get('k_candidates', [2, 3, 4])[0]
        n_clusters = min(n_clusters, len(X) // 5)  # At least 5 samples per cluster
        n_clusters = max(2, n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(Z)

        # Generate synthetic samples
        sampler = LatentSampler(
            per_cluster_cap_frac=normalized_cfg.get('per_cluster_cap_frac', 0.2),
            global_cap_frac=normalized_cfg.get('global_cap_frac', 0.3),
            random_state=seed
        )

        # Request synthetic samples (e.g., 20% of original)
        n_synth_total = int(len(X) * normalized_cfg.get('global_cap_frac', 0.3))

        Z_synth, X_synth_scaled = sampler.sample_with_caps(
            Z, labels, n_synth_total, pca, X_scaled_full,
            method=normalized_cfg.get('sampling_method', 'gaussian')
        )

        if X_synth_scaled.shape[0] > 0:
            # Inverse transform to original scale
            X_synth = scaler_full.inverse_transform(X_synth_scaled)

            # ============================================================
            # POST-PROCESSING: Restore correct column types
            # ============================================================
            X_synth_df = pd.DataFrame(X_synth, columns=X.columns)

            # Detect column types from original data
            original_dtypes = df[X.columns].dtypes

            # Identify binary columns (only 0 and 1 in original)
            binary_cols = []
            int_cols = []
            float_cols = []

            for col in X.columns:
                unique_vals = df[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(col)
                elif original_dtypes[col] in ['int64', 'int32', 'int']:
                    int_cols.append(col)
                else:
                    float_cols.append(col)

            # Identify one-hot groups (columns with same prefix)
            onehot_groups = {}
            onehot_prefixes = [
                'Family_Status_', 'Gender_', 'Financial_Attitude_',
                'Budget_Planning_', 'Save_Money_', 'Impulse_Buying_Category_',
                'Impulse_Buying_Reason_', 'Financial_Investments_',
                'Savings_Goal_', 'Savings_Obstacle_', 'Credit_Usage_',
                'Expense_Distribution_'
            ]

            for prefix in onehot_prefixes:
                group_cols = [c for c in X.columns if c.startswith(prefix)]
                if len(group_cols) > 1:
                    onehot_groups[prefix] = group_cols

            # Process binary columns: round to 0 or 1
            for col in binary_cols:
                X_synth_df[col] = np.clip(np.round(X_synth_df[col]), 0, 1).astype(int)

            # Process integer columns: round to nearest int, clip to original range
            for col in int_cols:
                if col not in binary_cols:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    X_synth_df[col] = np.clip(np.round(X_synth_df[col]), col_min, col_max).astype(int)

            # Process float columns: clip to original range
            for col in float_cols:
                col_min = df[col].min()
                col_max = df[col].max()
                X_synth_df[col] = np.clip(X_synth_df[col], col_min, col_max)

            # Enforce one-hot constraints: exactly one 1 per group
            for prefix, group_cols in onehot_groups.items():
                if all(c in X_synth_df.columns for c in group_cols):
                    # For each row, set the column with highest value to 1, rest to 0
                    group_values = X_synth_df[group_cols].values
                    max_indices = np.argmax(group_values, axis=1)
                    new_values = np.zeros_like(group_values, dtype=int)
                    for i, max_idx in enumerate(max_indices):
                        new_values[i, max_idx] = 1
                    X_synth_df[group_cols] = new_values

            # Convert back to numpy array
            X_synth_processed = X_synth_df.values

            print(f"  Post-processed: {len(binary_cols)} binary cols, {len(int_cols)} int cols, {len(float_cols)} float cols")
            print(f"  One-hot groups enforced: {list(onehot_groups.keys())}")
            # ============================================================

            # Assign labels to synthetic samples (copy from nearest real sample)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1).fit(X_scaled_full)
            _, indices = nn.kneighbors(X_synth_scaled)
            y_synth = y.values[indices.ravel()]

            # Create combined dataset with processed synthetic data
            X_combined = np.vstack([X.values, X_synth_processed])
            y_combined = np.concatenate([y.values, y_synth])
            is_synthetic = np.concatenate([np.zeros(len(X)), np.ones(len(X_synth))])

            # Create DataFrame with correct dtypes
            df_oversampled = pd.DataFrame(X_combined, columns=X.columns)

            # Restore original dtypes
            for col in X.columns:
                if col in binary_cols or col in int_cols:
                    df_oversampled[col] = df_oversampled[col].astype(int)

            df_oversampled[target] = y_combined
            df_oversampled['is_synthetic'] = is_synthetic.astype(int)

            # Save oversampled dataset
            oversampled_path = os.path.join(out_root, 'oversampled_dataset.csv')
            df_oversampled.to_csv(oversampled_path, index=False)

            # Also save a version without the is_synthetic column for direct use
            df_clean = df_oversampled.drop(columns=['is_synthetic'])
            clean_path = os.path.join(out_root, 'augmented_data.csv')
            df_clean.to_csv(clean_path, index=False)

            print(f"Saved oversampled dataset to: {oversampled_path}")
            print(f"  - Original samples: {len(X)}")
            print(f"  - Synthetic samples: {len(X_synth)}")
            print(f"  - Total samples: {len(df_oversampled)}")
            print(f"  - Augmentation ratio: {len(X_synth)/len(X)*100:.1f}%")
            print(f"Saved clean augmented data to: {clean_path}")

            # Update data profile
            data_profile['oversampled_dataset'] = oversampled_path
            data_profile['augmented_data'] = clean_path
            data_profile['n_synthetic'] = int(len(X_synth))
            data_profile['n_total'] = int(len(df_oversampled))
            data_profile['augmentation_ratio'] = float(len(X_synth) / len(X))
            data_profile['postprocessing'] = {
                'binary_cols': len(binary_cols),
                'int_cols': len(int_cols),
                'float_cols': len(float_cols),
                'onehot_groups': list(onehot_groups.keys())
            }
            with open(os.path.join(out_root, 'data_profile.json'), 'w') as f:
                json.dump(data_profile, f, indent=2)
        else:
            print("Warning: No synthetic samples generated")

    except Exception as e:
        import traceback
        print(f"Warning: failed to generate oversampled dataset: {e}")
        traceback.print_exc()

    # --- Generate aggregate plots across all folds ---
    try:
        print("Generating aggregate plots across folds...")
        from experiments.latent_plots import LatentExperimentPlotter
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Collect performance data from all folds
        all_perf_data = []
        fold_dirs = [d for d in os.listdir(out_root) if d.startswith('fold_') and os.path.isdir(os.path.join(out_root, d))]

        for fold_dir in sorted(fold_dirs):
            perf_path = os.path.join(out_root, fold_dir, 'performance_vs_synth_count.csv')
            if os.path.exists(perf_path):
                fold_perf = pd.read_csv(perf_path)
                fold_perf['fold'] = fold_dir
                all_perf_data.append(fold_perf)

        if all_perf_data:
            combined_perf = pd.concat(all_perf_data, ignore_index=True)

            # Save combined performance data
            combined_perf.to_csv(os.path.join(out_root, 'all_folds_performance.csv'), index=False)

            # Create aggregate plots
            agg_dir = os.path.join(out_root, 'aggregate_plots')
            os.makedirs(agg_dir, exist_ok=True)

            # Determine metric based on task
            metric_col = 'mae' if task == 'regression' else 'macro_f1'

            # 1. Boxplot: Performance across folds for each synth_count
            plt.figure(figsize=(12, 6))
            synth_counts = sorted(combined_perf['synth_count'].unique())
            data_for_boxplot = []
            labels = []
            for sc in synth_counts:
                subset = combined_perf[combined_perf['synth_count'] == sc][metric_col].dropna()
                if len(subset) > 0:
                    data_for_boxplot.append(subset.values)
                    labels.append(f'n={sc}' if sc > 0 else 'Baseline')

            if data_for_boxplot:
                bp = plt.boxplot(data_for_boxplot, tick_labels=labels, patch_artist=True)
                colors = ['lightgray'] + ['lightblue'] * (len(data_for_boxplot) - 1)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

            ylabel = 'MAE (lower is better)' if task == 'regression' else 'Macro-F1 (higher is better)'
            plt.xlabel('Synthetic Sample Count', fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            plt.title(f'Performance Across Folds ({n_splits} folds)', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, 'performance_boxplot_all_folds.png'), dpi=150)
            plt.close()

            # 2. Line plot: Mean performance ± std across folds
            plt.figure(figsize=(10, 5))
            means = []
            stds = []
            valid_counts = []

            for sc in synth_counts:
                subset = combined_perf[combined_perf['synth_count'] == sc][metric_col].dropna()
                if len(subset) > 0:
                    means.append(subset.mean())
                    stds.append(subset.std())
                    valid_counts.append(sc)

            if means:
                plt.errorbar(valid_counts, means, yerr=stds, fmt='o-', capsize=5,
                           markersize=8, linewidth=2, color='steelblue', ecolor='gray')

                # Highlight baseline
                if 0 in valid_counts:
                    baseline_idx = valid_counts.index(0)
                    plt.axhline(y=means[baseline_idx], color='red', linestyle='--',
                              alpha=0.7, label='Baseline')

            plt.xlabel('Synthetic Sample Count', fontsize=11)
            plt.ylabel(f'{ylabel} (mean ± std)', fontsize=11)
            plt.title(f'Aggregate Performance ({n_splits} folds)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, 'performance_aggregate_line.png'), dpi=150)
            plt.close()

            # 3. Acceptance rate per synth_count
            # FIX: Use gate_pass column and exclude baselines (is_baseline=True)
            gate_col = 'gate_pass' if 'gate_pass' in combined_perf.columns else 'accepted'
            if gate_col in combined_perf.columns:
                plt.figure(figsize=(8, 5))
                acceptance_rates = []

                # Exclude baselines from acceptance calculation
                if 'is_baseline' in combined_perf.columns:
                    synth_only = combined_perf[~combined_perf['is_baseline'].fillna(False).astype(bool)]
                else:
                    synth_only = combined_perf[combined_perf['synth_count'] > 0]

                for sc in synth_counts:
                    if sc == 0:
                        continue
                    subset = synth_only[synth_only['synth_count'] == sc][gate_col].dropna()
                    if len(subset) > 0:
                        # Handle boolean and numeric values
                        rate = subset.astype(float).mean()
                        acceptance_rates.append((sc, rate))

                if acceptance_rates:
                    scs, rates = zip(*acceptance_rates)
                    colors = ['green' if r > 0.5 else 'orange' if r > 0 else 'red' for r in rates]
                    plt.bar(range(len(scs)), rates, color=colors, alpha=0.7, edgecolor='black')
                    plt.xticks(range(len(scs)), [f'n={s}' for s in scs])
                    plt.xlabel('Synthetic Sample Count', fontsize=11)
                    plt.ylabel('Gate Pass Rate', fontsize=11)
                    plt.title('Quality Gate Pass Rate per Synth Count (excluding baseline)', fontsize=12)
                    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(agg_dir, 'gate_pass_rate.png'), dpi=150)
                    plt.close()
                else:
                    print("  Warning: No synth rows found for acceptance rate plot")

            print(f"  Saved aggregate plots to: {agg_dir}")

        # Create a comprehensive summary report
        summary_report = {
            'experiment_id': os.path.basename(out_root),
            'task': task,
            'target': target,
            'n_folds': n_splits,
            'seed': seed,
            'pca_candidates': normalized_cfg.get('pca_candidates', []),
            'k_candidates': normalized_cfg.get('k_candidates', []),
            'synth_grid': normalized_cfg.get('synth_grid', []),
            'aggregated_metrics': dict(aggregated),
            'plots_generated': {
                'per_fold': ['pca/', 'clustering/', 'synthetic_audit/'],
                'aggregate': ['performance_boxplot_all_folds.png', 'performance_aggregate_line.png', 'acceptance_rate.png']
            }
        }

        with open(os.path.join(out_root, 'experiment_report.json'), 'w') as f:
            json.dump(summary_report, f, indent=2)

        print(f"  Saved experiment report to: {os.path.join(out_root, 'experiment_report.json')}")

    except Exception as e:
        import traceback
        print(f"Warning: failed to generate aggregate plots: {e}")
        traceback.print_exc()

    print(f"Latent sampling experiment complete. Results in: {out_root}")


if __name__ == "__main__":
    main()
