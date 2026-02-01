import argparse
import os
import yaml
import pandas as pd
import numpy as np
import json
# ensure project root is on sys.path so `import experiments` works when running script directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.latent_experiment import run_latent_fold
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
    for train_idx, val_idx in rk.split(X):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        out_dir = os.path.join(out_root, f"fold_{fold}")
        res = run_latent_fold(X_train, y_train, X_val, y_val, normalized_cfg, out_dir, task=task)
        all_metrics.append(res)

    # Save summary.json
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Aggregate metrics across folds
    aggregated = {}
    if all_metrics:
        # Collect baseline metrics
        baseline_keys = all_metrics[0].get('baseline', {}).keys() if all_metrics[0].get('baseline') else []
        for key in baseline_keys:
            vals = [m['baseline'].get(key) for m in all_metrics if m.get('baseline') and m['baseline'].get(key) is not None]
            if vals:
                aggregated[f'baseline_{key}_mean'] = float(np.mean(vals))
                aggregated[f'baseline_{key}_std'] = float(np.std(vals))

        # Collect candidate metrics
        candidate_metrics = [m.get('candidate') for m in all_metrics if m.get('candidate')]
        if candidate_metrics:
            cand_keys = candidate_metrics[0].keys()
            for key in cand_keys:
                vals = [m.get(key) for m in candidate_metrics if m.get(key) is not None]
                if vals:
                    aggregated[f'candidate_{key}_mean'] = float(np.mean(vals))
                    aggregated[f'candidate_{key}_std'] = float(np.std(vals))

        # use_synth ratio
        use_synth_count = sum(1 for m in all_metrics if m.get('use_synth'))
        aggregated['folds_using_synth'] = use_synth_count
        aggregated['total_folds'] = len(all_metrics)

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

    print(f"Latent sampling experiment complete. Results in: {out_root}")


if __name__ == "__main__":
    main()
