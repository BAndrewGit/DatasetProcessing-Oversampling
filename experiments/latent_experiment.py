import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skmetrics
from scipy.stats import spearmanr
from .latent_space import PCASelector
from .clustering_latent import LatentClusterer
from .latent_sampling import LatentSampler
from .save_model import save_sklearn_model, write_model_metadata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def run_latent_fold(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, config: Dict[str,Any], out_dir: str, task: str = "regression"):
    os.makedirs(out_dir, exist_ok=True)
    # 1) PCA selection (train-only)
    psel = PCASelector(candidates=config.get("pca_candidates"))
    psel.fit(X_train)
    psel.save_selection(out_dir)
    sel = psel.choose()
    # PCA-A: chosen model with whiten=False (for reconstruction tasks)
    pca_a = psel.get_model(sel["chosen_k"], False)
    # PCA-B for clustering: prefer whiten=True for more spherical latent space; fall back to chosen whiten if not present
    try:
        pca_b = psel.get_model(sel["chosen_k"], True)
        if pca_b is None:
            pca_b = psel.get_model(sel["chosen_k"], sel.get("chosen_whiten", False))
    except Exception:
        pca_b = psel.get_model(sel["chosen_k"], sel.get("chosen_whiten", False))

    Z_train = pca_b.transform(X_train.values)
    Z_val = pca_b.transform(X_val.values)
    # Save the per-component plot from PCASelector (if generated) into out_dir for inspection
    try:
        # pca_selector already saved plots in out_dir via save_selection
        pass
    except Exception:
        pass

    # 2) clustering
    clusterer = LatentClusterer(k_candidates=config.get("k_candidates", (2,3,4,5)), min_fraction=config.get("min_cluster_frac", 0.05))
    best = clusterer.fit_and_choose(pd.DataFrame(Z_train))
    clusterer.save_report(out_dir, Z=pd.DataFrame(Z_train))
    k = best["chosen_k"]
    km = clusterer.get_model(k)
    labels_train = km.predict(Z_train)

    # 3) sampling counts per cluster following caps
    n_train = X_train.shape[0]
    global_cap = int(config.get("global_cap_frac", 0.3) * n_train)
    counts = {}
    for cid in range(k):
        csize = (labels_train == cid).sum()
        counts[cid] = min(int(config.get("per_cluster_cap_frac", 0.2) * csize), global_cap)  # placeholder per choice

    sampler = LatentSampler(per_cluster_cap_frac=config.get("per_cluster_cap_frac", 0.2), global_cap_frac=config.get("global_cap_frac", 0.3))
    Z_synth = sampler.generate_per_cluster(Z_train, labels_train, counts, method=config.get("sampling_method", "gaussian"))
    # cap total to +30%
    max_total = int(config.get("global_cap_frac", 0.3) * n_train)
    if Z_synth.shape[0] > max_total:
        Z_synth = Z_synth[:max_total]

    X_synth = pca_b.inverse_transform(Z_synth)
    X_synth = sampler.postprocess_decoded(X_synth, X_train.values, onehot_groups=config.get("onehot_groups", None))

    # simple audit
    audit = sampler.audit_basic(X_train.values, X_synth)
    with open(os.path.join(out_dir, "synthetic_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # 4) quality gates (placeholders: require min distance > eps)
    memo_min = audit.get("memorization_min", None)
    try:
        thresh = float(config.get("memorization_min_threshold", 1e-6))
    except Exception:
        thresh = 1e-6
    gate_pass = False if memo_min is None else (float(memo_min) > thresh)
    # real gating logic must include two-sample, utility and stability checks
    # If gate fails, use real-only
    use_synth = gate_pass

    # 5) Train models
    if task == "regression":
        base_model = MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=42, max_iter=500)
    else:
        base_model = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, random_state=42, max_iter=500)

    # baseline: fit on DataFrame with column names to ensure model stores feature names
    X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train.values, columns=[f'f{i}' for i in range(X_train.shape[1])])
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    base_model.fit(X_train_df, y_train_arr)
    # Pass DataFrame with column names to sklearn estimators to avoid feature-name warnings
    X_val_df = X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val.values, columns=X_train_df.columns)
    y_pred_base = base_model.predict(X_val_df)

    if task == "regression":
        baseline_metrics = {
            "mae": float(skmetrics.mean_absolute_error(y_val.values, y_pred_base)),
            "rmse": float(np.sqrt(skmetrics.mean_squared_error(y_val.values, y_pred_base))),
            "r2": float(skmetrics.r2_score(y_val.values, y_pred_base))
        }
    else:
        baseline_metrics = {
            "macro_f1": float(skmetrics.f1_score(y_val.values, y_pred_base, average="macro", zero_division=0)),
            "precision": float(skmetrics.precision_score(y_val.values, y_pred_base, average="macro", zero_division=0)),
            "recall": float(skmetrics.recall_score(y_val.values, y_pred_base, average="macro", zero_division=0)),
            "accuracy": float(skmetrics.accuracy_score(y_val.values, y_pred_base))
        }

    # ------------------ Anchor sample tracking (diagnostic only) ------------------
    try:
        anchors = []
        Z_val_arr = Z_val if isinstance(Z_val, np.ndarray) else np.asarray(Z_val)
        Z_train_arr = Z_train if isinstance(Z_train, np.ndarray) else np.asarray(Z_train)
        # cluster centroids from km
        centroids = km.cluster_centers_
        # for each cluster, pick validation sample closest to centroid (if any)
        for cid in range(k):
            if Z_val_arr.size == 0:
                continue
            dists = np.linalg.norm(Z_val_arr - centroids[cid], axis=1)
            idx = int(np.argmin(dists))
            anchors.append({"val_index": int(idx), "cluster": int(cid), "reason": "cluster_representative", "baseline_pred": float(y_pred_base[idx]) if hasattr(y_pred_base, '__len__') else float(y_pred_base), "true": float(y_val.values[idx])})

        # add one difficult anchor: highest baseline absolute error on validation
        if hasattr(y_pred_base, '__len__') and len(y_pred_base) > 0:
            errors = np.abs(y_val.values - y_pred_base)
            hard_idx = int(np.argmax(errors))
            anchors.append({"val_index": hard_idx, "cluster": int(km.predict(Z_val_arr[hard_idx:hard_idx+1])[0]), "reason": "hard_case", "baseline_pred": float(y_pred_base[hard_idx]), "true": float(y_val.values[hard_idx])})

        # ensure unique anchors and limit to 5
        seen = set()
        uniq_anchors = []
        for a in anchors:
            key = a["val_index"]
            if key in seen:
                continue
            seen.add(key)
            uniq_anchors.append(a)
            if len(uniq_anchors) >= 5:
                break

        # synth counts to evaluate (config-driven or default small set)
        synth_counts = config.get("anchor_synth_counts", config.get("synth_counts", [0, 10, 50]))
        # cap list to reasonable size
        if not isinstance(synth_counts, (list, tuple)):
            synth_counts = [int(synth_counts)]

        # helper to allocate per-cluster counts proportional to cluster sizes
        cluster_sizes = np.array([ (labels_train == cid).sum() for cid in range(k) ])
        def allocate_counts(total):
            if total <= 0:
                return {cid: 0 for cid in range(k)}
            prop = cluster_sizes / cluster_sizes.sum() if cluster_sizes.sum() > 0 else np.ones(k) / k
            raw = np.floor(prop * total).astype(int)
            # adjust to match total
            diff = int(total - raw.sum())
            i = 0
            while diff > 0:
                raw[i % k] += 1
                i += 1
                diff -= 1
            # enforce per-cluster cap
            per_cap = int(config.get("per_cluster_cap_frac", 0.2) * n_train)
            for cid in range(k):
                raw[cid] = min(raw[cid], per_cap)
            return {int(cid): int(raw[cid]) for cid in range(k)}

        anchor_results = {"anchors": uniq_anchors, "predictions": {}}

        # baseline already present as baseline_metrics
        anchor_results["predictions"][0] = []
        for a in uniq_anchors:
            anchor_results["predictions"][0].append({"val_index": a["val_index"], "pred": float(a["baseline_pred"]), "true": float(a["true"])})

        for s in synth_counts:
            s = int(s)
            if s == 0:
                continue
            # cap total synth by global cap
            total_cap = int(config.get("global_cap_frac", 0.3) * n_train)
            total = min(s, total_cap)
            if total <= 0:
                # record same as baseline
                anchor_results["predictions"][s] = anchor_results["predictions"][0]
                continue
            counts = allocate_counts(total)
            Zs = sampler.generate_per_cluster(Z_train, labels_train, counts, method=config.get("sampling_method", "gaussian"))
            if Zs.shape[0] > total:
                Zs = Zs[:total]
            Xs = pca_b.inverse_transform(Zs)
            Xs = sampler.postprocess_decoded(Xs, X_train.values, onehot_groups=config.get("onehot_groups", None))

            # Label synthetic rows: placeholder
            if task == "regression":
                y_synth = np.repeat(y_train.mean(), Xs.shape[0])
            else:
                y_synth = np.repeat(pd.Series(y_train).mode().iat[0], Xs.shape[0])

            # train candidate model on augmented data
            X_aug = np.vstack([X_train_df.values, Xs])
            y_aug = np.concatenate([y_train_arr, y_synth])
            cand = base_model.__class__(**base_model.get_params())
            X_aug_df = pd.DataFrame(X_aug, columns=X_train_df.columns)
            cand.fit(X_aug_df, y_aug)

            # get predictions for anchors
            preds = []
            for a in uniq_anchors:
                idx = int(a["val_index"])
                x_anchor = X_val.values[idx:idx+1]
                # Wrap anchor into DataFrame with proper column names to avoid sklearn warnings
                if isinstance(X_train, pd.DataFrame):
                    x_anchor_df = pd.DataFrame(x_anchor, columns=X_train.columns)
                else:
                    x_anchor_df = pd.DataFrame(x_anchor)
                if task == "regression":
                    p = cand.predict(x_anchor_df)[0]
                else:
                    if hasattr(cand, "predict_proba"):
                        p = float(cand.predict_proba(x_anchor_df)[0][1])
                    else:
                        p = float(cand.predict(x_anchor_df)[0])
                preds.append({"val_index": idx, "pred": float(p), "true": float(a["true"])})
            anchor_results["predictions"][s] = preds

        # Save anchor behavior json
        try:
            with open(os.path.join(out_dir, "anchor_behavior.json"), "w") as f:
                json.dump(anchor_results, f, indent=2)
        except Exception:
            pass

        # Plot predictions vs synth count (one line per anchor)
        try:
            counts_sorted = sorted([int(c) for c in anchor_results["predictions"].keys()])
            plt.figure(figsize=(6, 4))
            for i, a in enumerate(uniq_anchors):
                ys = [next((p["pred"] for p in anchor_results["predictions"][c] if p["val_index"] == a["val_index"]), None) for c in counts_sorted]
                plt.plot(counts_sorted, ys, marker='o', label=f'anchor_{i}_idx{a["val_index"]}')
            plt.xlabel('synth_count')
            plt.ylabel('prediction')
            plt.title('Anchor predictions vs synth count (diagnostic)')
            plt.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'anchor_predictions_vs_synth_count.png'))
            plt.close()
        except Exception:
            pass
    except Exception:
        # Anchors are diagnostic only; failures shouldn't break experiment
        pass
    # ------------------ End anchor tracking ------------------

    # ------------------ Story 7.7: MLP ablation curve (performance vs synth count) ------------------
    try:
        # fixed MLP hyperparams
        mlp_hidden = config.get('mlp_hidden', 64)
        mlp_max_iter = config.get('mlp_max_iter', 500)
        mlp_random_state = config.get('seed', 42)
        synth_grid = config.get('synth_grid', [0, 10, 50, 100, 500])

        perf_rows = []

        # Baseline already computed as baseline_metrics
        base_row = {'synth_count': 0}
        if task == 'regression':
            base_row.update({
                'mae': baseline_metrics['mae'],
                'rmse': baseline_metrics['rmse'],
                'r2': baseline_metrics['r2'],
                'spearman': float(spearmanr(y_val.values, y_pred_base).correlation) if len(y_val) > 1 else None,
                'accepted': False
            })
        else:
            # classification baseline metrics are in baseline_metrics
            base_row.update({'macro_f1': baseline_metrics['macro_f1'], 'accuracy': baseline_metrics['accuracy'], 'accepted': False})

        perf_rows.append(base_row)

        # iterate grid
        for s in synth_grid:
            s = int(s)
            if s == 0:
                continue
            # sample with caps and decode preview not needed
            Zs, Xs = sampler.sample_with_caps(Z_train, labels_train, s, pca_b, X_train.values, onehot_groups=config.get('onehot_groups', None), method=config.get('sampling_method', 'gaussian'), per_cluster_cap_frac=config.get('per_cluster_cap_frac', None), global_cap_frac=config.get('global_cap_frac', None), preview_path=None)

            audit = sampler.run_quality_gates(X_train.values, Xs, y_train.values, X_val.values, y_val.values, task=task, thresholds=config.get('quality_thresholds', None), n_stability_repeats=config.get('stability_repeats', 3))

            row = {'synth_count': s, 'n_synth_generated': int(Xs.shape[0]), 'accepted': bool(audit.get('decision', False))}

            if audit.get('decision', False) and Xs.shape[0] > 0:
                # train same MLP architecture
                if task == 'regression':
                    model = MLPRegressor(hidden_layer_sizes=(mlp_hidden,), early_stopping=True, random_state=mlp_random_state, max_iter=mlp_max_iter)
                else:
                    model = MLPClassifier(hidden_layer_sizes=(mlp_hidden,), early_stopping=True, random_state=mlp_random_state, max_iter=mlp_max_iter)

                X_aug = np.vstack([X_train.values, Xs])
                if task == 'regression':
                    y_aug = np.concatenate([y_train.values, np.repeat(y_train.mean(), Xs.shape[0])])
                else:
                    y_aug = np.concatenate([y_train.values, np.repeat(pd.Series(y_train).mode().iat[0], Xs.shape[0])])

                # Train model on DataFrame so it stores feature names
                X_aug_df = pd.DataFrame(X_aug, columns=X_train_df.columns)
                model.fit(X_aug_df, y_aug)
                # Use X_val_df created earlier (DataFrame with proper columns) for predictions
                y_pred = model.predict(X_val_df)

                if task == 'regression':
                    mae = float(skmetrics.mean_absolute_error(y_val.values, y_pred))
                    rmse = float(np.sqrt(skmetrics.mean_squared_error(y_val.values, y_pred)))
                    r2 = float(skmetrics.r2_score(y_val.values, y_pred))
                    spearman = float(spearmanr(y_val.values, y_pred).correlation) if len(y_val) > 1 else None
                    row.update({'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': spearman})
                else:
                    macro_f1 = float(skmetrics.f1_score(y_val.values, y_pred, average='macro', zero_division=0))
                    acc = float(skmetrics.accuracy_score(y_val.values, y_pred))
                    prec = float(skmetrics.precision_score(y_val.values, y_pred, average='macro', zero_division=0))
                    rec = float(skmetrics.recall_score(y_val.values, y_pred, average='macro', zero_division=0))
                    row.update({'macro_f1': macro_f1, 'accuracy': acc, 'precision': prec, 'recall': rec})
            else:
                # record nulls if rejected
                if task == 'regression':
                    row.update({'mae': None, 'rmse': None, 'r2': None, 'spearman': None})
                else:
                    row.update({'macro_f1': None, 'accuracy': None, 'precision': None, 'recall': None})

            # attach audit details
            row['audit'] = audit
            perf_rows.append(row)

        # Save performance_vs_synth_count.csv
        import csv
        csv_path = os.path.join(out_dir, 'performance_vs_synth_count.csv')
        # normalize rows to dicts
        keys = set()
        for r in perf_rows:
            keys.update(r.keys())
        keys = list(sorted(keys))
        with open(csv_path, 'w', newline='') as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=keys)
            writer.writeheader()
            for r in perf_rows:
                writer.writerow({k: (r.get(k) if not isinstance(r.get(k), dict) else json.dumps(r.get(k))) for k in keys})

        # include in fold-level metrics JSON
        try:
            with open(os.path.join(out_dir, 'metrics.json'), 'r') as mf:
                old = json.load(mf)
        except Exception:
            old = {}
        old['performance_vs_synth'] = perf_rows
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as mf:
            json.dump(old, mf, indent=2)
    except Exception:
        # do not fail whole fold on ablation errors
        pass
    # ------------------ End Story 7.7 ablation ------------------

    # candidate with synth if allowed
    if use_synth and X_synth.shape[0] > 0:
        X_aug = np.vstack([X_train_df.values, X_synth])
        if task == "regression":
            y_aug = np.concatenate([y_train_arr, np.repeat(y_train_arr.mean(), X_synth.shape[0])])
        else:
            # label synthetic rows with mode as placeholder
            mode_val = pd.Series(y_train_arr).mode().iat[0] if len(pd.Series(y_train_arr).mode())>0 else y_train_arr.mean()
            y_aug = np.concatenate([y_train_arr, np.repeat(mode_val, X_synth.shape[0])])
        cand_model = base_model.__class__(**base_model.get_params())
        X_aug_df = pd.DataFrame(X_aug, columns=X_train_df.columns)
        cand_model.fit(X_aug_df, y_aug)
        y_pred_cand = cand_model.predict(X_val_df)
        if task == "regression":
            cand_metrics = {
                "mae": float(skmetrics.mean_absolute_error(y_val.values, y_pred_cand)),
                "rmse": float(np.sqrt(skmetrics.mean_squared_error(y_val.values, y_pred_cand))),
                "r2": float(skmetrics.r2_score(y_val.values, y_pred_cand))
            }
        else:
            cand_metrics = {
                "macro_f1": float(skmetrics.f1_score(y_val.values, y_pred_cand, average="macro")),
                "precision": float(skmetrics.precision_score(y_val.values, y_pred_cand, average="macro", zero_division=0)),
                "recall": float(skmetrics.recall_score(y_val.values, y_pred_cand, average="macro", zero_division=0)),
                "accuracy": float(skmetrics.accuracy_score(y_val.values, y_pred_cand))
            }
    else:
        cand_metrics = None

    # Save fold artifacts
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"baseline": baseline_metrics, "candidate": cand_metrics, "use_synth": bool(use_synth)}, f, indent=2)

    # --- Save model.joblib for this fold (baseline MLP + scaler + PCA pipeline) ---
    try:
        # Create a scaler fitted on train data
        scaler = StandardScaler()
        scaler.fit(X_train_df.values)

        # Save the baseline model
        save_sklearn_model(base_model, os.path.join(out_dir, 'model.joblib'))

        # Save scaler separately
        save_sklearn_model(scaler, os.path.join(out_dir, 'scaler.joblib'))

        # Save PCA model (the one used for clustering/latent)
        save_sklearn_model(pca_b, os.path.join(out_dir, 'pca.joblib'))

        # Save clusterer KMeans model
        if km is not None:
            save_sklearn_model(km, os.path.join(out_dir, 'kmeans.joblib'))

        # Write metadata
        metadata = {
            'sklearn_objects': {
                'model': 'model.joblib',
                'scaler': 'scaler.joblib',
                'pca': 'pca.joblib',
                'kmeans': 'kmeans.joblib' if km is not None else None
            },
            'pca_k': sel.get('chosen_k'),
            'pca_whiten': sel.get('chosen_whiten'),
            'n_clusters': k,
            'task': task,
            'use_synth': bool(use_synth)
        }
        write_model_metadata(out_dir, metadata)
    except Exception as e:
        print(f"Warning: failed to save latent fold models: {e}")

    return {"baseline": baseline_metrics, "candidate": cand_metrics, "use_synth": bool(use_synth)}
