"""
Comprehensive plotting module for Latent Space Analysis + Clustering + Oversampling experiments.

Generates all MUST-HAVE plots required for committee-ready analysis:

1) PCA selection / explained variance
   - EVR per component (bar/line)
   - Cumulative EVR with thresholds (0.80, 0.90, 0.95)
   - K chosen marker
   - Reconstruction error vs K

2) Latent space visualization
   - Scatter latent (PC1 vs PC2) colored by cluster
   - Cluster size distribution

3) Cluster selection evidence
   - Silhouette score vs number of clusters (C=2..5)
   - Davies-Bouldin index vs number of clusters
   - Stability plot: ARI (bootstrap) vs number of clusters

4) Synthetic audit (quality gates)
   - Memorization / kNN distance histogram
   - Two-sample test ROC curve
   - Utility plot: real-only vs real+synth performance
   - Anchor prediction stability vs synthetic count

5) NICE-TO-HAVE (bonus)
   - Reconstruction error distribution (train vs val)
   - PCA loadings heatmap (top features per PC)
   - Distance-to-centroid distribution per cluster
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class LatentExperimentPlotter:
    """
    Generates all required plots for latent space experiment.
    Organizes outputs into subfolders: pca/, clustering/, synthetic_audit/
    """

    def __init__(self, out_dir: str, random_state: int = 42):
        self.out_dir = out_dir
        self.random_state = random_state

        # Create subfolder structure
        self.pca_dir = os.path.join(out_dir, 'pca')
        self.clustering_dir = os.path.join(out_dir, 'clustering')
        self.synthetic_dir = os.path.join(out_dir, 'synthetic_audit')

        for d in [self.pca_dir, self.clustering_dir, self.synthetic_dir]:
            os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) PCA PLOTS
    # -------------------------------------------------------------------------

    def plot_evr_per_component(self, pca_model, chosen_k: int, whiten: bool):
        """
        Plot EVR per component as bar chart with cumulative line.
        Shows thresholds at 0.80, 0.90, 0.95.
        """
        evr = pca_model.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        n_comp = len(evr)

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Bar chart for per-component EVR
        x = np.arange(1, n_comp + 1)
        bars = ax1.bar(x, evr, alpha=0.7, color='steelblue', label='Per-component EVR')
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=11, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # Cumulative line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, cum_evr, color='darkorange', marker='o', markersize=4, linewidth=2, label='Cumulative EVR')
        ax2.set_ylabel('Cumulative EVR', fontsize=11, color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')

        # Threshold lines
        for thresh, style, color in [(0.80, '--', 'green'), (0.90, '-.', 'red'), (0.95, ':', 'purple')]:
            ax2.axhline(y=thresh, linestyle=style, color=color, alpha=0.7, label=f'{thresh:.0%} threshold')

        # Mark chosen K
        ax1.axvline(x=chosen_k, color='crimson', linestyle='-', linewidth=2, alpha=0.8)
        ax1.annotate(f'Chosen K={chosen_k}', xy=(chosen_k, max(evr)*0.9),
                    xytext=(chosen_k + 1, max(evr)*0.95),
                    fontsize=10, color='crimson',
                    arrowprops=dict(arrowstyle='->', color='crimson'))

        # Title and legend
        plt.title(f'PCA Explained Variance (K={chosen_k}, whiten={whiten})', fontsize=12)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.pca_dir, 'evr_per_component.png'), dpi=150)
        plt.close()

    def plot_evr_vs_k(self, pca_results: Dict[Tuple[int, bool], Dict], chosen_k: int, chosen_whiten: bool):
        """
        Plot total EVR vs n_components for both whiten=True and whiten=False.
        Shows the chosen K with a marker.
        """
        # Extract data
        data_false = []
        data_true = []

        for (k, whiten), result in pca_results.items():
            if whiten:
                data_true.append((k, result['evr']))
            else:
                data_false.append((k, result['evr']))

        data_false.sort(key=lambda x: x[0])
        data_true.sort(key=lambda x: x[0])

        plt.figure(figsize=(8, 5))

        if data_false:
            ks_f, evrs_f = zip(*data_false)
            plt.plot(ks_f, evrs_f, 'o-', color='blue', label='whiten=False', markersize=6)

        if data_true:
            ks_t, evrs_t = zip(*data_true)
            plt.plot(ks_t, evrs_t, 's-', color='green', label='whiten=True', markersize=6)

        # Mark chosen K
        chosen_evr = pca_results.get((chosen_k, chosen_whiten), {}).get('evr', 0)
        plt.scatter([chosen_k], [chosen_evr], c='red', s=150, zorder=5, marker='*', label=f'Chosen: K={chosen_k}')

        plt.xlabel('Number of Components (K)', fontsize=11)
        plt.ylabel('Total Explained Variance Ratio', fontsize=11)
        plt.title('PCA: EVR vs Number of Components', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add threshold lines
        plt.axhline(y=0.90, linestyle='--', color='orange', alpha=0.5, label='90%')
        plt.axhline(y=0.95, linestyle=':', color='red', alpha=0.5, label='95%')

        plt.tight_layout()
        plt.savefig(os.path.join(self.pca_dir, 'evr_vs_k.png'), dpi=150)
        plt.close()

    def plot_reconstruction_error_vs_k(self, pca_results: Dict[Tuple[int, bool], Dict], chosen_k: int):
        """
        Plot reconstruction error vs K for model selection.
        """
        data_false = []
        data_true = []

        for (k, whiten), result in pca_results.items():
            recon = result.get('recon_error', None)
            if recon is not None:
                if whiten:
                    data_true.append((k, recon))
                else:
                    data_false.append((k, recon))

        data_false.sort(key=lambda x: x[0])
        data_true.sort(key=lambda x: x[0])

        plt.figure(figsize=(8, 5))

        if data_false:
            ks_f, recons_f = zip(*data_false)
            plt.plot(ks_f, recons_f, 'o-', color='blue', label='whiten=False', markersize=6)

        if data_true:
            ks_t, recons_t = zip(*data_true)
            plt.plot(ks_t, recons_t, 's-', color='green', label='whiten=True', markersize=6)

        plt.axvline(x=chosen_k, color='red', linestyle='--', linewidth=2, label=f'Chosen K={chosen_k}')

        plt.xlabel('Number of Components (K)', fontsize=11)
        plt.ylabel('Reconstruction Error (MSE)', fontsize=11)
        plt.title('PCA: Reconstruction Error vs K', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.pca_dir, 'reconstruction_error_vs_k.png'), dpi=150)
        plt.close()

    def plot_pca_loadings_heatmap(self, pca_model, feature_names: List[str], n_components: int = 5, n_features: int = 15):
        """
        Heatmap of top PCA loadings (components vs features).
        """
        loadings = pca_model.components_[:n_components]

        # Get top features by max absolute loading across components
        max_abs_loadings = np.max(np.abs(loadings), axis=0)
        top_feature_idx = np.argsort(max_abs_loadings)[-n_features:][::-1]

        loadings_subset = loadings[:, top_feature_idx]
        feature_subset = [feature_names[i] if i < len(feature_names) else f'f{i}' for i in top_feature_idx]

        plt.figure(figsize=(12, 6))
        sns.heatmap(loadings_subset,
                   xticklabels=feature_subset,
                   yticklabels=[f'PC{i+1}' for i in range(n_components)],
                   cmap='RdBu_r', center=0, annot=True, fmt='.2f')
        plt.title('PCA Loadings Heatmap (Top Features)', fontsize=12)
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        plt.tight_layout()
        plt.savefig(os.path.join(self.pca_dir, 'pca_loadings_heatmap.png'), dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 2) LATENT SPACE VISUALIZATION
    # -------------------------------------------------------------------------

    def plot_latent_scatter(self, Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None,
                           title: str = 'Latent Space Clusters'):
        """
        2D scatter plot of latent space colored by cluster.
        If Z has more than 2 dimensions, uses first 2 PCs.
        """
        if Z.shape[1] > 2:
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            Z_2d = pca_2d.fit_transform(Z)
            xlabel, ylabel = 'PC1 (of latent)', 'PC2 (of latent)'
            centroids_2d = pca_2d.transform(centroids) if centroids is not None else None
        else:
            Z_2d = Z[:, :2]
            xlabel, ylabel = 'Z1', 'Z2'
            centroids_2d = centroids[:, :2] if centroids is not None else None

        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i / max(1, len(unique_labels) - 1)) for i in range(len(unique_labels))]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[colors[i]],
                       label=f'Cluster {label} (n={mask.sum()})', alpha=0.6, s=30)

        if centroids_2d is not None:
            plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black',
                       marker='X', s=200, edgecolors='white', linewidths=2,
                       label='Centroids', zorder=5)

        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(title, fontsize=12)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'latent_scatter.png'), dpi=150)
        plt.close()

    def plot_cluster_size_distribution(self, labels: np.ndarray, k: int):
        """
        Bar chart showing size of each cluster.
        Flags if any cluster is below minimum threshold.
        """
        sizes = np.bincount(labels, minlength=k)
        n_total = len(labels)
        min_threshold = 0.05 * n_total  # 5% threshold

        plt.figure(figsize=(8, 5))

        colors = ['red' if s < min_threshold else 'steelblue' for s in sizes]
        bars = plt.bar(range(k), sizes, color=colors, alpha=0.8, edgecolor='black')

        # Add percentage labels
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            pct = 100 * size / n_total
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', fontsize=10)

        plt.axhline(y=min_threshold, color='red', linestyle='--',
                   label=f'5% threshold ({int(min_threshold)})')

        plt.xlabel('Cluster ID', fontsize=11)
        plt.ylabel('Number of Samples', fontsize=11)
        plt.title(f'Cluster Size Distribution (K={k})', fontsize=12)
        plt.xticks(range(k))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'cluster_sizes.png'), dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 3) CLUSTER SELECTION EVIDENCE
    # -------------------------------------------------------------------------

    def plot_silhouette_vs_k(self, cluster_reports: Dict[int, Dict], chosen_k: int):
        """
        Plot silhouette score vs number of clusters.
        """
        ks = []
        silhouettes = []

        for k, report in sorted(cluster_reports.items()):
            if not report.get('rejected', True):
                sil = report.get('silhouette')
                if sil is not None:
                    ks.append(k)
                    silhouettes.append(sil)

        if not ks:
            return

        plt.figure(figsize=(7, 5))
        plt.plot(ks, silhouettes, 'o-', color='purple', markersize=8, linewidth=2)
        plt.scatter([chosen_k], [silhouettes[ks.index(chosen_k)] if chosen_k in ks else 0],
                   c='red', s=150, zorder=5, marker='*', label=f'Chosen K={chosen_k}')

        # Quality thresholds
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (≥0.5)')
        plt.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.25)')

        plt.xlabel('Number of Clusters', fontsize=11)
        plt.ylabel('Silhouette Score', fontsize=11)
        plt.title('Silhouette Score vs Number of Clusters', fontsize=12)
        plt.xticks(ks)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'silhouette_vs_k.png'), dpi=150)
        plt.close()

    def plot_davies_bouldin_vs_k(self, cluster_reports: Dict[int, Dict], chosen_k: int):
        """
        Plot Davies-Bouldin index vs number of clusters (lower is better).
        """
        ks = []
        dbs = []

        for k, report in sorted(cluster_reports.items()):
            if not report.get('rejected', True):
                db = report.get('davies_bouldin')
                if db is not None:
                    ks.append(k)
                    dbs.append(db)

        if not ks:
            return

        plt.figure(figsize=(7, 5))
        plt.plot(ks, dbs, 'o-', color='teal', markersize=8, linewidth=2)
        plt.scatter([chosen_k], [dbs[ks.index(chosen_k)] if chosen_k in ks else 0],
                   c='red', s=150, zorder=5, marker='*', label=f'Chosen K={chosen_k}')

        plt.xlabel('Number of Clusters', fontsize=11)
        plt.ylabel('Davies-Bouldin Index (lower is better)', fontsize=11)
        plt.title('Davies-Bouldin Index vs Number of Clusters', fontsize=12)
        plt.xticks(ks)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'davies_bouldin_vs_k.png'), dpi=150)
        plt.close()

    def plot_cluster_stability_ari(self, Z: np.ndarray, k_candidates: List[int],
                                    n_bootstrap: int = 10, sample_frac: float = 0.8):
        """
        Bootstrap stability analysis: compute ARI between clusterings on subsamples.
        Higher ARI = more stable clustering.
        """
        from sklearn.metrics import adjusted_rand_score

        n_samples = Z.shape[0]
        subsample_size = int(sample_frac * n_samples)

        stability_results = {}

        for k in k_candidates:
            ari_scores = []

            for i in range(n_bootstrap):
                # Sample two overlapping subsets
                rng = np.random.RandomState(self.random_state + i)
                idx1 = rng.choice(n_samples, subsample_size, replace=False)
                idx2 = rng.choice(n_samples, subsample_size, replace=False)

                # Find common indices
                common = np.intersect1d(idx1, idx2)
                if len(common) < k + 1:
                    continue

                # Cluster both subsets
                km1 = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                km2 = KMeans(n_clusters=k, random_state=self.random_state + 1000, n_init=10)

                labels1_full = km1.fit_predict(Z[idx1])
                labels2_full = km2.fit_predict(Z[idx2])

                # Get labels for common samples
                common_in_idx1 = np.isin(idx1, common)
                common_in_idx2 = np.isin(idx2, common)

                labels1 = labels1_full[common_in_idx1]
                labels2 = labels2_full[common_in_idx2]

                # Compute ARI
                ari = adjusted_rand_score(labels1, labels2)
                ari_scores.append(ari)

            if ari_scores:
                stability_results[k] = {
                    'mean': np.mean(ari_scores),
                    'std': np.std(ari_scores),
                    'scores': ari_scores
                }

        if not stability_results:
            return

        # Plot
        ks = sorted(stability_results.keys())
        means = [stability_results[k]['mean'] for k in ks]
        stds = [stability_results[k]['std'] for k in ks]

        plt.figure(figsize=(7, 5))
        plt.errorbar(ks, means, yerr=stds, fmt='o-', capsize=5, markersize=8,
                    color='darkgreen', linewidth=2, ecolor='gray')

        # Stability threshold
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good stability (≥0.8)')
        plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair stability (≥0.6)')

        plt.xlabel('Number of Clusters', fontsize=11)
        plt.ylabel('ARI (mean ± std)', fontsize=11)
        plt.title(f'Cluster Stability (Bootstrap n={n_bootstrap})', fontsize=12)
        plt.xticks(ks)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'stability_ari_vs_k.png'), dpi=150)
        plt.close()

        # Save stability data
        with open(os.path.join(self.clustering_dir, 'stability_ari.json'), 'w') as f:
            json.dump({str(k): v for k, v in stability_results.items()}, f, indent=2)

    def plot_distance_to_centroid_distribution(self, Z: np.ndarray, labels: np.ndarray,
                                                centroids: np.ndarray):
        """
        Distribution of distances from samples to their cluster centroid.
        """
        unique_labels = np.unique(labels)

        fig, axes = plt.subplots(1, len(unique_labels), figsize=(4 * len(unique_labels), 4))
        if len(unique_labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, unique_labels):
            mask = labels == label
            Z_cluster = Z[mask]
            centroid = centroids[label]

            distances = np.linalg.norm(Z_cluster - centroid, axis=1)

            ax.hist(distances, bins=20, alpha=0.7, color=f'C{label}', edgecolor='black')
            ax.axvline(np.median(distances), color='red', linestyle='--',
                      label=f'Median={np.median(distances):.2f}')
            ax.set_xlabel('Distance to Centroid')
            ax.set_ylabel('Count')
            ax.set_title(f'Cluster {label} (n={mask.sum()})')
            ax.legend(fontsize=8)

        plt.suptitle('Distance to Centroid Distribution per Cluster', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.clustering_dir, 'distance_to_centroid.png'), dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 4) SYNTHETIC AUDIT PLOTS
    # -------------------------------------------------------------------------

    def plot_memorization_histogram(self, X_train: np.ndarray, X_synth: np.ndarray,
                                     threshold: float = None):
        """
        Histogram of kNN distances from synthetic to nearest real sample.
        """
        if X_synth is None or X_synth.shape[0] == 0:
            return

        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        dists, _ = nn.kneighbors(X_synth)
        dists = dists.ravel()

        plt.figure(figsize=(8, 5))
        plt.hist(dists, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

        plt.axvline(np.median(dists), color='green', linestyle='--', linewidth=2,
                   label=f'Median={np.median(dists):.4f}')
        plt.axvline(np.min(dists), color='red', linestyle=':', linewidth=2,
                   label=f'Min={np.min(dists):.4f}')

        if threshold is not None:
            plt.axvline(threshold, color='orange', linestyle='-', linewidth=2,
                       label=f'Threshold={threshold:.4f}')

        plt.xlabel('Distance to Nearest Real Sample', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.title('Synthetic Memorization Check (kNN Distance)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'memorization_knn_histogram.png'), dpi=150)
        plt.close()

        # Save stats
        stats = {
            'n_synth': int(X_synth.shape[0]),
            'min_distance': float(np.min(dists)),
            'median_distance': float(np.median(dists)),
            'mean_distance': float(np.mean(dists)),
            'max_distance': float(np.max(dists))
        }
        with open(os.path.join(self.synthetic_dir, 'memorization_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def plot_two_sample_roc(self, X_train: np.ndarray, X_synth: np.ndarray,
                            auc_threshold: float = 0.75):
        """
        ROC curve for two-sample test (real vs synthetic classifier).
        AUC close to 0.5 = good (can't distinguish), AUC close to 1.0 = bad.
        """
        if X_synth is None or X_synth.shape[0] == 0:
            return

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        # Balance dataset
        n_real = min(X_train.shape[0], 500)
        n_synth = min(X_synth.shape[0], 500)

        np.random.seed(self.random_state)
        idx_real = np.random.choice(X_train.shape[0], n_real, replace=False)
        idx_synth = np.random.choice(X_synth.shape[0], n_synth, replace=False)

        X_combined = np.vstack([X_train[idx_real], X_synth[idx_synth]])
        y_combined = np.concatenate([np.zeros(n_real), np.ones(n_synth)])

        # Split and train discriminator
        X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_combined, test_size=0.3,
                                                   random_state=self.random_state, stratify=y_combined)

        clf = LogisticRegression(max_iter=500, solver='lbfgs')
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')

        # Color background based on quality
        if roc_auc < auc_threshold:
            plt.fill_between([0, 1], [0, 0], [1, 1], alpha=0.1, color='green')
            verdict = 'PASS'
        else:
            plt.fill_between([0, 1], [0, 0], [1, 1], alpha=0.1, color='red')
            verdict = 'FAIL'

        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title(f'Two-Sample Test ROC (AUC<{auc_threshold} required) - {verdict}', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'two_sample_roc.png'), dpi=150)
        plt.close()

        # Save result
        result = {
            'auc': float(roc_auc),
            'threshold': float(auc_threshold),
            'passed': bool(roc_auc < auc_threshold)
        }
        with open(os.path.join(self.synthetic_dir, 'two_sample_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

    def plot_utility_comparison(self, performance_data: List[Dict], task: str = 'regression'):
        """
        Boxplot/violin comparing real-only vs real+synth performance across folds.
        """
        if not performance_data:
            return

        # Extract baseline (synth_count=0) and best augmented
        baseline_scores = []
        augmented_scores = {}

        metric_key = 'mae' if task == 'regression' else 'macro_f1'

        for entry in performance_data:
            synth_count = entry.get('synth_count', 0)
            score = entry.get(metric_key)

            if score is not None:
                if synth_count == 0:
                    baseline_scores.append(score)
                else:
                    if synth_count not in augmented_scores:
                        augmented_scores[synth_count] = []
                    augmented_scores[synth_count].append(score)

        if not baseline_scores and not augmented_scores:
            return

        # Create grouped plot
        plt.figure(figsize=(10, 5))

        all_data = []
        labels = []

        if baseline_scores:
            all_data.append(baseline_scores)
            labels.append('Baseline\n(no synth)')

        for synth_count in sorted(augmented_scores.keys()):
            all_data.append(augmented_scores[synth_count])
            labels.append(f'n={synth_count}')

        if all_data:
            bp = plt.boxplot(all_data, labels=labels, patch_artist=True)
            colors = ['lightgray'] + ['lightblue'] * (len(all_data) - 1)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

        metric_label = 'MAE (lower is better)' if task == 'regression' else 'Macro-F1 (higher is better)'
        plt.xlabel('Configuration', fontsize=11)
        plt.ylabel(metric_label, fontsize=11)
        plt.title(f'Utility: Real-only vs Real+Synthetic ({metric_key})', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'utility_comparison.png'), dpi=150)
        plt.close()

    def plot_anchor_predictions_vs_synth(self, anchor_results: Dict):
        """
        Line plot showing anchor prediction stability vs synthetic count.
        """
        if not anchor_results or 'predictions' not in anchor_results:
            return

        predictions = anchor_results['predictions']
        anchors = anchor_results.get('anchors', [])

        if not predictions or not anchors:
            return

        synth_counts = sorted([int(k) for k in predictions.keys()])

        plt.figure(figsize=(10, 5))

        for i, anchor in enumerate(anchors):
            val_idx = anchor['val_index']
            true_val = anchor.get('true', 0)

            preds = []
            for sc in synth_counts:
                pred_list = predictions.get(sc, predictions.get(str(sc), []))
                pred = next((p['pred'] for p in pred_list if p['val_index'] == val_idx), None)
                preds.append(pred)

            reason = anchor.get('reason', 'unknown')
            label = f'Anchor {i} (idx={val_idx}, {reason})'
            plt.plot(synth_counts, preds, 'o-', label=label, markersize=6)

            # True value line
            plt.axhline(y=true_val, linestyle=':', alpha=0.3, color=f'C{i}')

        plt.xlabel('Synthetic Count', fontsize=11)
        plt.ylabel('Prediction', fontsize=11)
        plt.title('Anchor Predictions vs Synthetic Count (Stability Diagnostic)', fontsize=12)
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'anchor_stability.png'), dpi=150)
        plt.close()

    def plot_performance_vs_synth_count(self, performance_data: List[Dict], task: str = 'regression'):
        """
        Line plot of model performance vs synthetic sample count.
        Shows both accepted and rejected configurations.
        """
        if not performance_data:
            return

        metric_key = 'mae' if task == 'regression' else 'macro_f1'

        synth_counts = []
        scores = []
        accepted = []

        for entry in performance_data:
            sc = entry.get('synth_count', 0)
            score = entry.get(metric_key)
            acc = entry.get('accepted', False)

            synth_counts.append(sc)
            scores.append(score)
            accepted.append(acc)

        plt.figure(figsize=(9, 5))

        # Plot all points
        for sc, score, acc in zip(synth_counts, scores, accepted):
            if score is not None:
                color = 'green' if acc else 'red'
                marker = 'o' if acc else 'x'
                plt.scatter(sc, score, c=color, marker=marker, s=100,
                           label='Accepted' if acc else 'Rejected', zorder=3)

        # Connect accepted points with line
        acc_points = [(sc, s) for sc, s, a in zip(synth_counts, scores, accepted) if a and s is not None]
        if acc_points:
            acc_points.sort(key=lambda x: x[0])
            plt.plot([p[0] for p in acc_points], [p[1] for p in acc_points],
                    'g--', alpha=0.5, linewidth=1)

        # Baseline marker
        baseline = next((entry for entry in performance_data if entry.get('synth_count', 0) == 0), None)
        if baseline and baseline.get(metric_key) is not None:
            plt.axhline(y=baseline[metric_key], color='blue', linestyle='--',
                       alpha=0.7, label='Baseline')

        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        metric_label = 'MAE (lower is better)' if task == 'regression' else 'Macro-F1 (higher is better)'
        plt.xlabel('Synthetic Sample Count', fontsize=11)
        plt.ylabel(metric_label, fontsize=11)
        plt.title(f'Model Performance vs Synthetic Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'performance_vs_synth_count.png'), dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 5) NICE-TO-HAVE BONUS PLOTS
    # -------------------------------------------------------------------------

    def plot_reconstruction_error_distribution(self, X_train: np.ndarray, X_val: np.ndarray,
                                                pca_model):
        """
        Distribution of reconstruction error for train vs validation.
        """
        # Reconstruct
        Z_train = pca_model.transform(X_train)
        Z_val = pca_model.transform(X_val)
        X_train_recon = pca_model.inverse_transform(Z_train)
        X_val_recon = pca_model.inverse_transform(Z_val)

        # Per-sample reconstruction error (MSE)
        train_errors = np.mean((X_train - X_train_recon) ** 2, axis=1)
        val_errors = np.mean((X_val - X_val_recon) ** 2, axis=1)

        plt.figure(figsize=(8, 5))
        plt.hist(train_errors, bins=30, alpha=0.6, label=f'Train (n={len(train_errors)})', color='blue')
        plt.hist(val_errors, bins=30, alpha=0.6, label=f'Val (n={len(val_errors)})', color='orange')

        plt.axvline(np.median(train_errors), color='blue', linestyle='--',
                   label=f'Train median={np.median(train_errors):.4f}')
        plt.axvline(np.median(val_errors), color='orange', linestyle='--',
                   label=f'Val median={np.median(val_errors):.4f}')

        plt.xlabel('Reconstruction Error (MSE per sample)', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.title('Reconstruction Error Distribution', fontsize=12)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.pca_dir, 'reconstruction_error_distribution.png'), dpi=150)
        plt.close()

    def plot_latent_with_synthetic(self, Z_real: np.ndarray, Z_synth: np.ndarray,
                                    labels_real: np.ndarray):
        """
        Scatter plot showing real vs synthetic points in latent space.
        """
        if Z_synth is None or Z_synth.shape[0] == 0:
            return

        # Project to 2D if needed
        if Z_real.shape[1] > 2:
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            Z_real_2d = pca_2d.fit_transform(Z_real)
            Z_synth_2d = pca_2d.transform(Z_synth)
        else:
            Z_real_2d = Z_real[:, :2]
            Z_synth_2d = Z_synth[:, :2]

        plt.figure(figsize=(8, 6))

        # Plot real by cluster
        unique_labels = np.unique(labels_real)
        for label in unique_labels:
            mask = labels_real == label
            plt.scatter(Z_real_2d[mask, 0], Z_real_2d[mask, 1],
                       c=f'C{label}', alpha=0.4, s=20, label=f'Real cluster {label}')

        # Plot synthetic
        plt.scatter(Z_synth_2d[:, 0], Z_synth_2d[:, 1],
                   c='red', marker='x', s=30, alpha=0.8, label='Synthetic')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Latent Space: Real vs Synthetic', fontsize=12)
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.synthetic_dir, 'latent_real_vs_synth.png'), dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # SUMMARY GENERATION
    # -------------------------------------------------------------------------

    def generate_plots_summary(self) -> Dict[str, List[str]]:
        """
        Generate a summary of all plots created.
        """
        summary = {'pca': [], 'clustering': [], 'synthetic_audit': []}

        for subdir, key in [(self.pca_dir, 'pca'),
                           (self.clustering_dir, 'clustering'),
                           (self.synthetic_dir, 'synthetic_audit')]:
            if os.path.exists(subdir):
                files = [f for f in os.listdir(subdir) if f.endswith('.png')]
                summary[key] = files

        with open(os.path.join(self.out_dir, 'plots_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        return summary
