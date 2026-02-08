import os
import json
import inspect
from typing import Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import HuberRegressor, Ridge, LogisticRegression
import sklearn.metrics as skmetrics
from scipy.stats import spearmanr
from .latent_space import PCASelector
from .clustering_latent import LatentClusterer
from .latent_sampling import LatentSampler
from .latent_plots import LatentExperimentPlotter
from .save_model import save_sklearn_model, write_model_metadata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _get_model_factory(task: str, n_train: int, model_type: str = "auto", seed: int = 42) -> Callable:
    """
    Return a factory function that creates consistent models.

    This ensures baseline and augmented models use the SAME architecture
    for fair comparison.
    """
    if task == "regression":
        if model_type == "auto":
            if n_train < 200:
                model_type = "huber"
            elif n_train < 500:
                model_type = "ridge"
            else:
                model_type = "mlp"

        if model_type == "huber":
            return lambda: HuberRegressor(epsilon=1.35, max_iter=500)
        elif model_type == "ridge":
            return lambda: Ridge(alpha=1.0)
        elif model_type == "gbm":
            from sklearn.ensemble import GradientBoostingRegressor
            return lambda: GradientBoostingRegressor(loss='huber', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed)
        else:
            return lambda: MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=seed, max_iter=500)
    else:
        return lambda: LogisticRegression(max_iter=500, random_state=seed)


def run_latent_fold(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                    config: Dict[str, Any], out_dir: str, task: str = "regression"):
    os.makedirs(out_dir, exist_ok=True)
    n_train = X_train.shape[0]
    seed = config.get('seed', 42)

    # FIX A: Scale data BEFORE PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)

    # 1) PCA on SCALED data
    psel = PCASelector(candidates=config.get("pca_candidates"))
    psel.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns))
    psel.save_selection(out_dir)
    sel = psel.choose()

    pca = psel.get_model(sel["chosen_k"], True)  #whiten=True for clustering
    if pca is None:
        pca = psel.get_model(sel["chosen_k"], False)

    Z_train = pca.transform(X_train_scaled)
    Z_val = pca.transform(X_val_scaled)

    # 2) Check if clustering is viable (skip if latent is continuous manifold)
    use_continuous_sampler = config.get("use_continuous_sampler", "auto")
    stability_threshold = config.get("clustering_stability_threshold", 0.5)
    clustering_algorithm = config.get("clustering_algorithm", "kmeans")

    cluster_usable = False
    km = None
    k = 2
    fallback_reason = None

    if use_continuous_sampler == "always":
        print(f"  [INFO] Using continuous sampler (forced by config)")
        fallback_reason = "forced_continuous"
    else:
        # Try clustering first
        clusterer = LatentClusterer(
            k_candidates=config.get("k_candidates", (2,3,4)),
            stability_threshold=stability_threshold,
            n_init=config.get("kmeans_n_init", 50),
            algorithm=clustering_algorithm
        )
        best = clusterer.fit_and_choose(pd.DataFrame(Z_train))
        clusterer.save_report(out_dir, Z=pd.DataFrame(Z_train))
        k = best["chosen_k"]
        km = clusterer.get_model(k)
        cluster_usable = clusterer.is_clustering_usable()
        fallback_reason = best.get("fallback_reason")

    # 3) Initialize sampler based on clustering result
    memo_pct = config.get("memorization_percentile", 5)

    if cluster_usable and use_continuous_sampler != "always":
        # Cluster-based sampling
        print(f"  [OK] Using cluster-based sampling (k={k})")
        sampler = LatentSampler(
            per_cluster_cap_frac=config.get("per_cluster_cap_frac", 0.2),
            global_cap_frac=config.get("global_cap_frac", 0.3),
            synth_weight=config.get("synth_weight", 0.3),
            memorization_percentile=memo_pct
        )
        labels_train = km.predict(Z_train) if hasattr(km, 'predict') else np.zeros(n_train, dtype=int)

        # Compute threshold and generate
        memo_threshold = sampler.compute_real_to_real_threshold(X_train_scaled)
        n_bins = config.get("n_coverage_bins", 5)
        max_synth = int(config.get("global_cap_frac", 0.3) * n_train)

        Z_synth, X_synth_scaled_raw, coverage = sampler.sample_with_coverage(
            Z_train, y_train.values, max_synth, pca, X_train_scaled,
            n_bins=n_bins, method=config.get("sampling_method", "gaussian")
        )
    else:
        # CONTINUOUS SAMPLER: for latent spaces without cluster structure
        # Methods: smote_latent (BEST), knn_mixup, gaussian_noise, kde, gaussian
        print(f"  [INFO] Using CONTINUOUS sampler ({fallback_reason}) - no clustering")
        from .latent_sampling import ContinuousLatentSampler

        cont_method = config.get("continuous_method", "smote_latent")
        k_neighbors = config.get("k_neighbors", 5)
        noise_scale = config.get("noise_scale", 0.1)

        cont_sampler = ContinuousLatentSampler(
            method=cont_method,
            random_state=seed,
            memorization_percentile=memo_pct,
            target_aware=True,
            k_neighbors=k_neighbors,
            noise_scale=noise_scale
        )

        n_bins = config.get("n_coverage_bins", 5)
        max_synth = int(config.get("global_cap_frac", 0.3) * n_train)

        Z_synth, X_synth_scaled_raw, y_synth_cont, gen_stats = cont_sampler.generate(
            Z_train, y_train.values, max_synth, pca_decoder=pca, n_bins=n_bins
        )

        coverage = {'n_bins': n_bins, 'method': cont_method, **gen_stats}
        labels_train = np.zeros(n_train, dtype=int)

        # Create a minimal sampler for compatibility
        sampler = LatentSampler(memorization_percentile=memo_pct)

        # FIX BUG #3: Compute threshold in X space, not Z space!
        # cont_sampler._threshold is in Z (latent) space - NOT valid for X rejection
        memo_threshold = sampler.compute_real_to_real_threshold(X_train_scaled)

    print(f"  Memo threshold (p{memo_pct}): {memo_threshold:.4f}")
    print(f"  Generated {X_synth_scaled_raw.shape[0]} samples ({config.get('continuous_method', 'smote_latent')})")

    # Reject near-duplicates
    X_synth_scaled, rej_stats = sampler.reject_near_duplicates(X_synth_scaled_raw, X_train_scaled, memo_threshold)
    if rej_stats['n_rejected'] > 0:
        print(f"  Rejected {rej_stats['n_rejected']} near-duplicates ({rej_stats['rejection_rate']*100:.1f}%)")

    # FIX 3: Recompute Z_synth from kept X (not mask on Z)
    Z_synth = pca.transform(X_synth_scaled) if X_synth_scaled.shape[0] > 0 else np.empty((0, Z_train.shape[1]))

    # FIX D: Proper synthetic labels via kNN in LATENT space (better distances)
    y_synth = _propagate_labels_knn(X_train_scaled, y_train.values, X_synth_scaled,
                                     task=task, k=3, Z_train=Z_train, Z_synth=Z_synth)

    # Inverse transform to original scale
    X_synth = scaler.inverse_transform(X_synth_scaled) if X_synth_scaled.shape[0] > 0 else X_synth_scaled

    # Save audit
    audit = sampler.audit_basic(X_train_scaled, X_synth_scaled)
    audit['rejection_stats'] = rej_stats
    audit['coverage'] = {'n_bins': coverage['n_bins'], 'allocations': coverage.get('allocations', {})}
    audit['memo_threshold'] = memo_threshold
    with open(os.path.join(out_dir, "synthetic_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # FIX D: Quality gate includes REAL utility check on validation
    # Create model factory for consistent comparison
    model_class = _get_model_factory(task, n_train, config.get("model_type", "auto"), seed)
    feature_names = list(X_train.columns)

    gate_pass = _check_quality_gates(X_train_scaled, X_synth_scaled, y_train.values, y_synth,
                                      X_val_scaled, y_val.values, task, memo_threshold, config,
                                      model_class=model_class, feature_names=feature_names)
    use_synth = gate_pass and X_synth_scaled.shape[0] > 0

    # Plotting
    try:
        plotter = LatentExperimentPlotter(out_dir, random_state=seed)

        # PCA plots - always useful
        plotter.plot_evr_per_component(pca, sel['chosen_k'], True)
        feature_names = list(X_train.columns)
        plotter.plot_pca_loadings_heatmap(pca, feature_names, n_components=min(5, sel['chosen_k']), n_features=15)
        plotter.plot_reconstruction_error_distribution(X_train_scaled, X_val_scaled, pca)

        # Clustering plots - ONLY if clustering is usable
        if cluster_usable and km is not None:
            centroids = km.cluster_centers_ if hasattr(km, 'cluster_centers_') else (km.means_ if hasattr(km, 'means_') else None)
            plotter.plot_latent_scatter(Z_train, labels_train, centroids, title=f'Latent Clusters (K={k})')
            plotter.plot_cluster_size_distribution(labels_train, k)

            k_cands = config.get('k_candidates', [2,3,4,5])
            if len(k_cands) > 1:
                plotter.plot_cluster_stability_ari(Z_train, list(k_cands), n_bootstrap=20)
        else:
            # For continuous sampler: just show latent space without cluster coloring
            plotter.plot_latent_scatter(Z_train, None, None, title='Latent Space (Continuous - No Clusters)')

        # Synthetic data plots - always useful when we have synthetic samples
        if X_synth_scaled.shape[0] > 0:
            plotter.plot_memorization_histogram(X_train_scaled, X_synth_scaled, threshold=memo_threshold)
            plotter.plot_two_sample_roc(X_train_scaled, X_synth_scaled)
            if cluster_usable:
                plotter.plot_latent_with_synthetic(Z_train, Z_synth, labels_train)
            else:
                # Show synthetic in latent without cluster colors
                plotter.plot_latent_with_synthetic(Z_train, Z_synth, None)

        plotter.generate_plots_summary()
    except Exception as e:
        print(f"Warning: plotting error: {e}")

    # 5) Train baseline model - FIX 0: Use model_class consistently
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_df = pd.DataFrame(X_val_scaled, columns=X_train.columns)
    y_train_arr = y_train.values

    # FIX 0: Use model_class from _get_model_factory (already computed above)
    # This ensures baseline, candidate, ablation, and utility gate all use SAME model
    base_model = model_class()
    print(f"  Using {base_model.__class__.__name__} (N={n_train})")

    base_model.fit(X_train_df, y_train_arr)
    y_pred_base = base_model.predict(X_val_df)
    baseline_metrics = _compute_metrics(y_val.values, y_pred_base, task)

    # Anchor tracking - use model_class for consistency
    _run_anchor_tracking(X_train_df, y_train_arr, X_val_df, y_val.values, Z_train, Z_synth,
                         X_synth_scaled, y_synth, pca, scaler, model_class, task, config, out_dir, memo_threshold, sampler)

    # Synth count ablation - use SAME model_class for fair comparison
    _run_synth_ablation(X_train_df, y_train_arr, X_val_df, y_val.values, Z_train,
                        pca, scaler, task, config, out_dir, baseline_metrics, sampler, memo_threshold,
                        model_class=model_class)

    # Train candidate with synth - FIX 0: use model_class, FIX 5: use sample_weight
    cand_metrics = None
    if use_synth and X_synth_scaled.shape[0] > 0:
        X_aug = np.vstack([X_train_scaled, X_synth_scaled])
        y_aug = np.concatenate([y_train_arr, y_synth])

        # Create sample weights
        synth_weight = config.get("synth_weight", 0.3)
        weights = np.concatenate([np.ones(len(y_train_arr)), np.full(len(y_synth), synth_weight)])

        # FIX 0: Use model_class for candidate (same as baseline)
        cand_model = model_class()
        X_aug_df = pd.DataFrame(X_aug, columns=X_train.columns)

        # FIX 5: Actually use sample weights if model supports them
        if "sample_weight" in inspect.signature(cand_model.fit).parameters:
            cand_model.fit(X_aug_df, y_aug, sample_weight=weights)
        else:
            cand_model.fit(X_aug_df, y_aug)

        y_pred_cand = cand_model.predict(X_val_df)
        cand_metrics = _compute_metrics(y_val.values, y_pred_cand, task)

        # FIX E: Catastrophic fold kill switch + detailed diagnostics
        if task == "regression":
            catastrophic_threshold = config.get("catastrophic_threshold", 10.0)
            if cand_metrics['mae'] > baseline_metrics['mae'] * catastrophic_threshold:
                print(f"  [CATASTROPHIC] aug_mae={cand_metrics['mae']:.4f} > {catastrophic_threshold}x baseline={baseline_metrics['mae']:.4f}")
                cand_metrics['catastrophic'] = True

                # Detailed diagnostic: feature range violations
                feature_violations = []
                for i, col in enumerate(X_train.columns):
                    train_min, train_max = X_train_scaled[:, i].min(), X_train_scaled[:, i].max()
                    synth_min, synth_max = X_synth_scaled[:, i].min(), X_synth_scaled[:, i].max()
                    if synth_min < train_min - 0.5 or synth_max > train_max + 0.5:
                        feature_violations.append({
                            'feature': col,
                            'train_range': [float(train_min), float(train_max)],
                            'synth_range': [float(synth_min), float(synth_max)]
                        })

                # Outlier detection: kNN distance to nearest real
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1).fit(X_train_scaled)
                dists, _ = nn.kneighbors(X_synth_scaled)
                p95_dist = np.percentile(dists, 95)
                n_outliers = (dists.ravel() > p95_dist * 2).sum()

                diag_path = os.path.join(out_dir, 'catastrophic_diagnostic.json')
                with open(diag_path, 'w') as f:
                    json.dump({
                        'baseline_mae': baseline_metrics['mae'],
                        'candidate_mae': cand_metrics['mae'],
                        'ratio': cand_metrics['mae'] / max(baseline_metrics['mae'], 1e-10),
                        'threshold': catastrophic_threshold,
                        'n_synth': len(y_synth),
                        'synth_y_stats': {
                            'mean': float(np.mean(y_synth)),
                            'std': float(np.std(y_synth)),
                            'min': float(np.min(y_synth)),
                            'max': float(np.max(y_synth))
                        },
                        'feature_range_violations': feature_violations[:10],  # Top 10
                        'n_feature_violations': len(feature_violations),
                        'n_outlier_synth': int(n_outliers),
                        'outlier_threshold_dist': float(p95_dist * 2),
                        'recommendation': 'Check feature violations and outliers above. '
                                         'Possible causes: PCA reconstruction artifacts, '
                                         'one-hot groups not enforced, numerical instability.'
                    }, f, indent=2)
                use_synth = False  # Force reject

    # Save fold artifacts
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"baseline": baseline_metrics, "candidate": cand_metrics, "use_synth": use_synth}, f, indent=2)

    # Save models
    try:
        save_sklearn_model(base_model, os.path.join(out_dir, 'model.joblib'))
        save_sklearn_model(scaler, os.path.join(out_dir, 'scaler.joblib'))
        save_sklearn_model(pca, os.path.join(out_dir, 'pca.joblib'))
        if km:
            save_sklearn_model(km, os.path.join(out_dir, 'kmeans.joblib'))
        write_model_metadata(out_dir, {'pca_k': sel.get('chosen_k'), 'n_clusters': k, 'task': task, 'use_synth': use_synth})
    except Exception as e:
        print(f"Warning: save error: {e}")

    return {"baseline": baseline_metrics, "candidate": cand_metrics, "use_synth": use_synth}


def _propagate_labels_knn(X_train: np.ndarray, y_train: np.ndarray, X_synth: np.ndarray,
                          task: str = "regression", k: int = 3, eps: float = 1e-12,
                          Z_train: np.ndarray = None, Z_synth: np.ndarray = None) -> np.ndarray:
    """
    FIX D: Propagate labels using distance-weighted kNN in LATENT SPACE (Z).
    Distances in X with many one-hots are garbage. Use Z if available.
    """
    if X_synth.shape[0] == 0:
        return np.array([])

    # FIX D: Use latent space if available (better distances for one-hot features)
    if Z_train is not None and Z_synth is not None:
        nn = NearestNeighbors(n_neighbors=min(k, len(Z_train))).fit(Z_train)
        dists, idx = nn.kneighbors(Z_synth)
    else:
        nn = NearestNeighbors(n_neighbors=min(k, len(X_train))).fit(X_train)
        dists, idx = nn.kneighbors(X_synth)

    if task == "regression":
        # Distance-weighted mean
        w = 1.0 / (dists + eps)
        w = w / w.sum(axis=1, keepdims=True)
        return (y_train[idx] * w).sum(axis=1)
    else:
        # Classification: distance-weighted vote
        y_out = []
        for i in range(idx.shape[0]):
            labels = y_train[idx[i]]
            weights = 1.0 / (dists[i] + eps)
            scores = {}
            for lab, ww in zip(labels, weights):
                scores[lab] = scores.get(lab, 0.0) + ww
            y_out.append(max(scores.items(), key=lambda t: t[1])[0])
        return np.array(y_out)


def _check_quality_gates(X_train: np.ndarray, X_synth: np.ndarray, y_train: np.ndarray, y_synth: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, task: str, memo_thr: float, config: dict,
                         model_class=None, feature_names: list = None) -> bool:
    """
    Quality gates with REAL utility check on validation set.

    Gates:
    1. Memorization: synthetic not too close to real
    2. Two-sample: discriminator can't easily distinguish real vs synth (AUC < 0.75)
    3. Utility: augmented model not worse than real-only on VALIDATION set
    4. Domain validity: one-hot/binary features are valid
    """
    from .sanity_checks import run_quality_gates_strict

    if X_synth.shape[0] == 0:
        print("  [QUALITY GATE FAIL] No synthetic samples generated")
        return False

    print("  Quality Gates (STRICT mode - all must pass):")

    # FIX B: Pass synth_weight for consistent utility check
    synth_weight = config.get("synth_weight", 0.3)

    result = run_quality_gates_strict(
        X_train, X_synth, y_train, y_synth, memo_thr,
        config.get('quality_thresholds', {}),
        X_val=X_val, y_val=y_val, task=task, model_class=model_class,
        feature_names=feature_names, synth_weight=synth_weight
    )

    if result['all_passed']:
        print(f"  [QUALITY GATE PASS] {result['passed_count']}/{result['total_gates']} gates passed")
        return True
    else:
        print(f"  [QUALITY GATE FAIL] {result['failed_count']} gate(s) failed - synthetic data NOT used")
        for gate in result['gates']:
            if not gate['passed']:
                print(f"    - FAIL: {gate['reason']}")
        return False


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    """Compute metrics. For regression: MAE and Spearman are PRIMARY (not R²)."""
    if task == "regression":
        mae = float(skmetrics.mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(skmetrics.mean_squared_error(y_true, y_pred)))
        r2 = float(skmetrics.r2_score(y_true, y_pred))

        # Spearman correlation (robust to scale, measures ranking)
        if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
            spearman = float(spearmanr(y_true, y_pred).correlation)
        else:
            spearman = 0.0

        # PRIMARY metrics for small datasets: MAE, Spearman
        # R² can be misleading (negative = worse than mean prediction)
        return {
            "mae": mae,           # PRIMARY: lower is better
            "spearman": spearman, # PRIMARY: higher is better (ranking accuracy)
            "rmse": rmse,
            "r2": r2,             # SECONDARY: can be negative on small datasets
            "r2_warning": "negative" if r2 < 0 else "ok"
        }
    return {
        "macro_f1": float(skmetrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(skmetrics.accuracy_score(y_true, y_pred)),
        "precision": float(skmetrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(skmetrics.recall_score(y_true, y_pred, average="macro", zero_division=0))
    }


def _run_anchor_tracking(X_train_df, y_train, X_val_df, y_val, Z_train, Z_synth, X_synth_scaled, y_synth,
                         pca, scaler, model_class, task, config, out_dir, memo_thr, sampler):
    """Anchors by target quantiles - use model_class for consistency."""
    try:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        anchor_indices = []
        for q in quantiles:
            target_val = np.percentile(y_val, q * 100)
            idx = int(np.argmin(np.abs(y_val - target_val)))
            if idx not in anchor_indices:
                anchor_indices.append(idx)

        # Baseline predictions using model_class
        base_model = model_class()
        base_model.fit(X_train_df, y_train)
        y_pred_base = base_model.predict(X_val_df)

        anchors = [{"val_index": idx, "quantile": q, "true": float(y_val[idx]), "baseline_pred": float(y_pred_base[idx])}
                   for idx, q in zip(anchor_indices, quantiles[:len(anchor_indices)])]

        synth_counts = config.get("synth_grid", [0, 20, 50])
        anchor_results = {"anchors": anchors, "predictions": {0: [{"val_index": a["val_index"], "pred": a["baseline_pred"], "true": a["true"]} for a in anchors]}}

        for s in synth_counts:
            if s == 0:
                continue
            Zs, Xs_raw, _ = sampler.sample_with_coverage(Z_train, y_train, s, pca, X_train_df.values, n_bins=5)
            Xs, _ = sampler.reject_near_duplicates(Xs_raw, X_train_df.values, memo_thr)
            if Xs.shape[0] == 0:
                anchor_results["predictions"][s] = anchor_results["predictions"][0]
                continue

            # FIX D: Use latent space for label propagation
            Zs_kept = pca.transform(Xs) if Xs.shape[0] > 0 else Zs[:Xs.shape[0]]
            ys = _propagate_labels_knn(X_train_df.values, y_train, Xs, task=task, k=3,
                                       Z_train=Z_train, Z_synth=Zs_kept)
            X_aug = np.vstack([X_train_df.values, Xs])
            y_aug = np.concatenate([y_train, ys])

            # FIX B: Use sample weights for consistency
            synth_weight = config.get("synth_weight", 0.3)
            weights = np.concatenate([np.ones(len(y_train)), np.full(len(ys), synth_weight)])

            cand = model_class()
            if "sample_weight" in inspect.signature(cand.fit).parameters:
                cand.fit(pd.DataFrame(X_aug, columns=X_train_df.columns), y_aug, sample_weight=weights)
            else:
                cand.fit(pd.DataFrame(X_aug, columns=X_train_df.columns), y_aug)

            preds = []
            for a in anchors:
                p = cand.predict(X_val_df.iloc[[a["val_index"]]])[0]
                preds.append({"val_index": a["val_index"], "pred": float(p), "true": a["true"]})
            anchor_results["predictions"][s] = preds

        with open(os.path.join(out_dir, "anchor_behavior.json"), "w") as f:
            json.dump(anchor_results, f, indent=2)

        # Generate anchor predictions plot
        try:
            import matplotlib.pyplot as plt
            counts_sorted = sorted([int(c) for c in anchor_results["predictions"].keys()])
            plt.figure(figsize=(8, 5))
            for i, a in enumerate(anchors):
                ys = [next((p["pred"] for p in anchor_results["predictions"][c] if p["val_index"] == a["val_index"]), None) for c in counts_sorted]
                plt.plot(counts_sorted, ys, marker='o', label=f'q{a.get("quantile", i):.1f}')
            plt.xlabel('Synthetic Count')
            plt.ylabel('Prediction')
            plt.title('Anchor Predictions vs Synthetic Count')
            plt.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'anchor_predictions_vs_synth_count.png'), dpi=100)
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: anchor tracking error: {e}")


def _run_synth_ablation(X_train_df, y_train, X_val_df, y_val, Z_train, pca, scaler, task, config, out_dir,
                        baseline_metrics, sampler, memo_thr, model_class=None):
    """
    Run synthetic count ablation with SAME model as baseline for fair comparison.

    FIX: Use same model_class for baseline and augmented to avoid comparing
    "Huber vs MLP" instead of "real vs real+synth".
    """
    from sklearn.linear_model import HuberRegressor, Ridge, LogisticRegression
    from sklearn.neural_network import MLPRegressor, MLPClassifier

    try:
        synth_grid = config.get('synth_grid', [0, 10, 50, 100])
        # FIX: Baseline is always "passed" - it's the reference point
        perf_rows = [{'synth_count': 0, 'gate_pass': True, 'is_baseline': True, **baseline_metrics}]

        # FIX: Determine model class based on same logic as run_latent_fold
        if model_class is None:
            n_train = len(y_train)
            model_type = config.get("model_type", "auto")

            if task == "regression":
                if model_type == "auto":
                    if n_train < 200:
                        model_class = lambda: HuberRegressor(epsilon=1.35, max_iter=500)
                    elif n_train < 500:
                        model_class = lambda: Ridge(alpha=1.0)
                    else:
                        model_class = lambda: MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=42, max_iter=500)
                elif model_type == "huber":
                    model_class = lambda: HuberRegressor(epsilon=1.35, max_iter=500)
                elif model_type == "ridge":
                    model_class = lambda: Ridge(alpha=1.0)
                else:
                    model_class = lambda: MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=42, max_iter=500)
            else:
                model_class = lambda: LogisticRegression(max_iter=500, random_state=42)

        for s in synth_grid:
            if s == 0:
                continue

            Zs, Xs_raw, cov = sampler.sample_with_coverage(Z_train, y_train, s, pca, X_train_df.values, n_bins=5)
            Xs, rej = sampler.reject_near_duplicates(Xs_raw, X_train_df.values, memo_thr)

            row = {'synth_count': s, 'n_generated': int(Xs.shape[0]), 'n_rejected': rej.get('n_rejected', 0)}

            if Xs.shape[0] > 0:
                # FIX D: Use latent space for label propagation (better distances)
                # Recompute Z_synth from kept samples
                Zs_kept = pca.transform(Xs) if Xs.shape[0] > 0 else Zs[:Xs.shape[0]]
                ys = _propagate_labels_knn(X_train_df.values, y_train, Xs, task=task, k=3,
                                           Z_train=Z_train, Z_synth=Zs_kept)

                row['gate_pass'] = True
                row['is_baseline'] = False

                X_aug = np.vstack([X_train_df.values, Xs])
                y_aug = np.concatenate([y_train, ys])

                # FIX B: Use sample weights for consistency
                synth_weight = config.get("synth_weight", 0.3)
                weights = np.concatenate([np.ones(len(y_train)), np.full(len(ys), synth_weight)])

                model = model_class()
                # Apply sample weights if supported
                if "sample_weight" in inspect.signature(model.fit).parameters:
                    model.fit(pd.DataFrame(X_aug, columns=X_train_df.columns), y_aug, sample_weight=weights)
                else:
                    model.fit(pd.DataFrame(X_aug, columns=X_train_df.columns), y_aug)
                y_pred = model.predict(X_val_df)
                row.update(_compute_metrics(y_val, y_pred, task))
            else:
                row['gate_pass'] = False
                row['is_baseline'] = False

            perf_rows.append(row)

        # Save CSV
        import csv
        keys = sorted(set(k for r in perf_rows for k in r.keys()))
        with open(os.path.join(out_dir, 'performance_vs_synth_count.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(perf_rows)

        # Update metrics
        try:
            with open(os.path.join(out_dir, 'metrics.json'), 'r') as f:
                old = json.load(f)
        except:
            old = {}
        old['performance_vs_synth'] = perf_rows
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(old, f, indent=2)

    except Exception as e:
        import traceback
        print(f"Warning: ablation error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")


def tune_sampler_nested_cv(X_train: np.ndarray, y_train: np.ndarray, Z_train: np.ndarray,
                            pca, config: dict, task: str, model_class, seed: int = 42) -> dict:
    """
    Tune sampler hyperparameters using inner CV (nested CV inner loop).

    This prevents selection bias: we tune on inner folds, report on outer fold.

    Grid:
    - method: smote_latent, knn_mixup, gaussian_noise
    - k_neighbors: 3, 5, 7
    - noise_scale: 0.05, 0.1, 0.2
    - memo_pct: 1, 5
    - synth_count: 20, 50

    Objective: MAE + stability_weight * std(MAE)
    """
    from sklearn.model_selection import KFold
    from .latent_sampling import ContinuousLatentSampler, LatentSampler

    tuning_cfg = config.get('tuning', {})
    methods = tuning_cfg.get('methods', ['smote_latent'])
    k_neighbors_list = tuning_cfg.get('k_neighbors', [5])
    noise_scales = tuning_cfg.get('noise_scale', [0.1])
    memo_pcts = tuning_cfg.get('memo_percentile', [5])
    synth_counts = tuning_cfg.get('synth_counts', [50])
    stability_weight = tuning_cfg.get('stability_weight', 0.5)
    inner_splits = config.get('inner_splits', 3)

    best_score = float('inf')
    best_params = {
        'method': 'smote_latent',
        'k_neighbors': 5,
        'noise_scale': 0.1,
        'memo_pct': 5,
        'synth_count': 50
    }

    results = []

    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    # Build grid (minimal)
    grid = []
    for method in methods:
        for k in k_neighbors_list:
            for noise in noise_scales:
                for memo in memo_pcts:
                    for synth_n in synth_counts:
                        # Skip invalid combos
                        if method == 'gaussian_noise' and k != k_neighbors_list[0]:
                            continue  # k irrelevant for gaussian_noise
                        if method != 'gaussian_noise' and noise != noise_scales[0]:
                            continue  # noise irrelevant for smote/mixup

                        grid.append({
                            'method': method,
                            'k_neighbors': k,
                            'noise_scale': noise,
                            'memo_pct': memo,
                            'synth_count': synth_n
                        })

    print(f"  [Tuning] Grid size: {len(grid)} configs, {inner_splits} inner folds")

    for params in grid:
        fold_maes = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_tr, X_vl = X_train[inner_train_idx], X_train[inner_val_idx]
            y_tr, y_vl = y_train[inner_train_idx], y_train[inner_val_idx]
            Z_tr = Z_train[inner_train_idx]

            try:
                # Create sampler with current params
                sampler = ContinuousLatentSampler(
                    method=params['method'],
                    random_state=seed,
                    memorization_percentile=params['memo_pct'],
                    k_neighbors=params['k_neighbors'],
                    noise_scale=params['noise_scale']
                )

                # Generate synthetic
                Z_synth, X_synth, y_synth, _ = sampler.generate(
                    Z_tr, y_tr, params['synth_count'], pca_decoder=pca, n_bins=5
                )

                if len(X_synth) == 0:
                    fold_maes.append(float('inf'))
                    continue

                # Train augmented model
                X_aug = np.vstack([X_tr, X_synth])
                y_aug = np.concatenate([y_tr, y_synth])

                model = model_class()
                model.fit(X_aug, y_aug)
                y_pred = model.predict(X_vl)

                mae = float(np.mean(np.abs(y_vl - y_pred)))
                fold_maes.append(mae)

            except Exception as e:
                fold_maes.append(float('inf'))

        # Compute objective: mean MAE + stability penalty
        if fold_maes and not all(m == float('inf') for m in fold_maes):
            valid_maes = [m for m in fold_maes if m != float('inf')]
            mean_mae = np.mean(valid_maes)
            std_mae = np.std(valid_maes) if len(valid_maes) > 1 else 0
            score = mean_mae + stability_weight * std_mae
        else:
            score = float('inf')

        results.append({**params, 'score': score, 'mean_mae': mean_mae if score != float('inf') else None})

        if score < best_score:
            best_score = score
            best_params = params.copy()

    print(f"  [Tuning] Best: {best_params['method']}, k={best_params['k_neighbors']}, " +
          f"synth={best_params['synth_count']}, score={best_score:.4f}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


def run_latent_fold_with_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 config: dict, out_dir: str, task: str = "regression"):
    """
    Run latent fold with optional nested CV tuning.

    If use_nested_cv=True in config:
    1. Tune hyperparameters on inner CV (train only)
    2. Train final model with best params
    3. Report on outer val (unbiased estimate)
    """
    os.makedirs(out_dir, exist_ok=True)

    use_nested = config.get('use_nested_cv', False)

    if use_nested:
        print("  [Nested CV] Tuning on inner folds...")

        # Prepare data
        n_train = X_train.shape[0]
        seed = config.get('seed', 42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)

        # PCA - use same API as run_latent_fold
        psel = PCASelector(candidates=config.get("pca_candidates", [10, 8, 5]))
        psel.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns))
        sel = psel.choose()
        pca = psel.get_model(sel["chosen_k"], False)
        Z_train = pca.transform(X_train_scaled)

        # Model factory
        model_class = _get_model_factory(task, n_train, config.get("model_type", "auto"), seed)

        # Tune
        tuning_result = tune_sampler_nested_cv(
            X_train_scaled, y_train.values, Z_train, pca, config, task, model_class, seed
        )

        # Save tuning results
        with open(os.path.join(out_dir, 'tuning_results.json'), 'w') as f:
            json.dump({
                'best_params': tuning_result['best_params'],
                'best_score': tuning_result['best_score']
            }, f, indent=2)

        # Update config with best params for final run
        best = tuning_result['best_params']
        config['continuous_method'] = best['method']
        config['k_neighbors'] = best['k_neighbors']
        config['noise_scale'] = best['noise_scale']
        config['memorization_percentile'] = best['memo_pct']
        config['global_cap_frac'] = best['synth_count'] / n_train

    # Run standard fold with (possibly tuned) config
    return run_latent_fold(X_train, y_train, X_val, y_val, config, out_dir, task)


