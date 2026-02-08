import json
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class LatentClusterer:
    """
    Clustering in latent space with multiple algorithms and stability validation.

    Algorithms supported:
    - kmeans: K-Means (baseline)
    - gmm: Gaussian Mixture Model with BIC selection (RECOMMENDED for elliptical clusters)
    - hdbscan: Density-based clustering (good for irregular shapes)

    Stability is measured via pairwise ARI between bootstrap runs.
    """

    def __init__(self, k_candidates=(2,3,4,5), min_fraction=0.05, random_state=42,
                 n_init=100, stability_threshold=0.6, algorithm='auto'):
        self.k_candidates = k_candidates
        self.min_fraction = min_fraction
        self.random_state = int(random_state) if random_state is not None else None
        self.n_init = n_init
        self.stability_threshold = stability_threshold
        self.algorithm = algorithm  # 'kmeans', 'gmm', 'hdbscan', or 'auto'
        self.best = None
        self.models = {}
        self.stability_results = {}
        self.latent_diagnostics = {}

    def _check_latent_health(self, Z_arr: np.ndarray) -> dict:
        """Check if latent space is healthy (not collapsed/degenerate)."""
        n, d = Z_arr.shape
        stds = np.std(Z_arr, axis=0)
        n_dead_dims = int(np.sum(stds < 1e-6))

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(Z_arr)
        dists, _ = nn.kneighbors(Z_arr)
        min_dists = dists[:, 1]
        n_near_duplicates = int(np.sum(min_dists < 1e-4))
        n_approx_unique = n - n_near_duplicates
        is_healthy = (n_dead_dims < d * 0.5) and (n_near_duplicates < n * 0.3)

        diagnostics = {
            'n_samples': n, 'n_dims': d, 'n_dead_dims': n_dead_dims,
            'n_near_duplicates': n_near_duplicates, 'n_approx_unique': n_approx_unique,
            'std_min': float(stds.min()), 'std_max': float(stds.max()),
            'std_mean': float(stds.mean()), 'is_healthy': is_healthy
        }

        if not is_healthy:
            print(f"  [WARN] Latent unhealthy: {n_dead_dims}/{d} dead dims, {n_near_duplicates}/{n} near-duplicates")

        return diagnostics

    def _fit_gmm(self, Z_arr: np.ndarray, k: int) -> Tuple[Optional[GaussianMixture], np.ndarray, float]:
        """Fit GMM and return model, labels, and BIC."""
        best_bic = float('inf')
        best_model = None
        best_labels = None

        for cov_type in ['full', 'diag', 'spherical']:
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type=cov_type,
                    n_init=min(10, self.n_init), random_state=self.random_state, max_iter=200
                )
                gmm.fit(Z_arr)
                bic = gmm.bic(Z_arr)
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_labels = gmm.predict(Z_arr)
            except Exception:
                continue

        return best_model, best_labels if best_labels is not None else np.zeros(len(Z_arr)), best_bic

    def _fit_hdbscan(self, Z_arr: np.ndarray) -> Tuple[Optional[Any], np.ndarray, int]:
        """Fit HDBSCAN and return model, labels, and number of clusters found."""
        try:
            from hdbscan import HDBSCAN
            # For small datasets (N<300), use smaller min_cluster_size
            n = len(Z_arr)
            if n < 100:
                min_cluster_size = 3
            elif n < 300:
                min_cluster_size = max(3, int(n * 0.02))  # 2% of data
            else:
                min_cluster_size = max(5, int(n * 0.05))

            hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
            labels = hdb.fit_predict(Z_arr)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            return hdb, labels, n_clusters
        except ImportError:
            print("  [WARN] HDBSCAN not installed, skipping")
            return None, np.zeros(len(Z_arr)), 0

    def compute_stability_pairwise(self, Z_arr: np.ndarray, k: int, n_bootstrap: int = 30,
                                    subsample_ratio: float = 0.8, algorithm: str = 'kmeans') -> dict:
        """Compute clustering stability via PAIRWISE ARI between bootstrap runs."""
        rng = np.random.RandomState(self.random_state)
        n = len(Z_arr)
        subsample_size = int(n * subsample_ratio)

        all_labels = []
        for i in range(n_bootstrap):
            idx = rng.choice(n, subsample_size, replace=False)
            Z_sub = Z_arr[idx]

            if algorithm == 'gmm':
                gmm = GaussianMixture(n_components=k, n_init=5, random_state=self.random_state + i)
                gmm.fit(Z_sub)
                labels_full = gmm.predict(Z_arr)
            else:
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state + i)
                km.fit(Z_sub)
                labels_full = km.predict(Z_arr)

            all_labels.append(labels_full)

        # Pairwise ARI between all bootstrap runs
        ari_scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)

        ari_mean = float(np.mean(ari_scores)) if ari_scores else 0.0
        ari_med = float(np.median(ari_scores)) if ari_scores else 0.0
        ari_p10 = float(np.percentile(ari_scores, 10)) if ari_scores else 0.0

        is_stable = (ari_med >= self.stability_threshold) and (ari_p10 >= self.stability_threshold - 0.15)

        return {
            'ari_mean': ari_mean, 'ari_median': ari_med, 'ari_p10': ari_p10,
            'ari_std': float(np.std(ari_scores)) if ari_scores else 0.0,
            'n_pairs': len(ari_scores), 'is_stable': is_stable
        }

    def fit_and_choose(self, Z: pd.DataFrame):
        Z_arr = Z.values if isinstance(Z, pd.DataFrame) else np.asarray(Z)
        n = Z_arr.shape[0]
        reports: Dict[int, Dict[str, Any]] = {}

        self.latent_diagnostics = self._check_latent_health(Z_arr)
        if not self.latent_diagnostics['is_healthy']:
            print(f"  [WARN] Latent space is degenerate. Clustering may fail.")

        algo = self.algorithm
        if algo == 'auto':
            algo = 'kmeans'  # KMeans is most stable on small data

        print(f"  Clustering algorithm: {algo.upper()}")

        # HDBSCAN: density-based, auto-determines k
        hdbscan_result = None
        if algo == 'hdbscan':
            hdb_model, hdb_labels, hdb_k = self._fit_hdbscan(Z_arr)
            if hdb_k >= 2:
                # Compute stability for HDBSCAN
                hdb_stability = self._compute_hdbscan_stability(Z_arr, n_bootstrap=20)

                try:
                    hdb_sil = float(silhouette_score(Z_arr[hdb_labels >= 0], hdb_labels[hdb_labels >= 0])) if (hdb_labels >= 0).sum() > hdb_k else None
                except:
                    hdb_sil = None

                n_noise = int((hdb_labels == -1).sum())
                noise_ratio = n_noise / n

                hdbscan_result = {
                    'model': hdb_model, 'labels': hdb_labels, 'k': hdb_k,
                    'silhouette': hdb_sil, 'n_noise': n_noise, 'noise_ratio': noise_ratio,
                    'stability': hdb_stability
                }

                # HDBSCAN is usable if: stable, not too much noise, k >= 2
                is_usable = hdb_stability.get('is_stable', False) and noise_ratio < 0.3

                print(f"  HDBSCAN: k={hdb_k}, noise={noise_ratio*100:.1f}%, " +
                      f"ARI={hdb_stability.get('ari_median', 0):.3f}, usable={is_usable}")

                if is_usable:
                    self.models['hdbscan'] = {"model": hdb_model, "labels": hdb_labels}
                    self.best = {
                        "chosen_k": hdb_k,
                        "reports": {'hdbscan': hdbscan_result},
                        "cluster_usable": True,
                        "fallback_reason": None,
                        "algorithm": 'hdbscan',
                        "latent_diagnostics": self.latent_diagnostics,
                        "hdbscan_result": {'k': hdb_k, 'noise_ratio': noise_ratio}
                    }
                    return self.best
                else:
                    print(f"  [WARN] HDBSCAN unstable or too noisy. Trying KMeans fallback.")
                    algo = 'kmeans'  # Fallback
            else:
                print(f"  [WARN] HDBSCAN found <2 clusters. Trying KMeans fallback.")
                algo = 'kmeans'

        # Evaluate k candidates with GMM or KMeans
        for k in self.k_candidates:
            if algo == 'gmm':
                model, labels, bic = self._fit_gmm(Z_arr, k)
                if model is None:
                    reports[k] = {"k": k, "rejected": True, "reason": "gmm_failed"}
                    continue
                extra_metrics = {'bic': bic, 'algorithm': 'gmm'}
            else:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
                labels = km.fit_predict(Z_arr)
                model = km
                extra_metrics = {'algorithm': 'kmeans'}

            sizes = np.bincount(labels, minlength=k)
            n_eff = len(np.unique(labels))

            if n_eff < k:
                reports[k] = {"k": k, "rejected": True, "reason": "degenerate", "n_eff": int(n_eff)}
                continue

            max_ratio = sizes.max() / n
            if max_ratio > 0.95:
                reports[k] = {"k": k, "rejected": True, "reason": "collapsed", "max_ratio": float(max_ratio)}
                continue

            if (sizes < max(1, int(self.min_fraction * n))).any():
                reports[k] = {"k": k, "rejected": True, "reason": "small_cluster", "sizes": sizes.tolist()}
                continue

            try:
                sil = float(silhouette_score(Z_arr, labels))
            except:
                sil = None

            db = float(davies_bouldin_score(Z_arr, labels))

            stability = self.compute_stability_pairwise(Z_arr, k, n_bootstrap=30, algorithm=algo)
            self.stability_results[k] = stability

            reports[k] = {
                "k": k, "rejected": False, "silhouette": sil, "davies_bouldin": db,
                "sizes": sizes.tolist(),
                "stability_ari_mean": stability['ari_mean'],
                "stability_ari_median": stability['ari_median'],
                "stability_ari_p10": stability['ari_p10'],
                "is_stable": bool(stability['is_stable']),
                **extra_metrics
            }

            if algo == 'gmm':
                reports[k]["centroids"] = model.means_.tolist()
            else:
                reports[k]["centroids"] = model.cluster_centers_.tolist()

            self.models[k] = {"model": model, "labels": labels}

        # Choose best k
        candidates = [r for r in reports.values() if not r.get("rejected", True)]

        if not candidates:
            chosen_k = int(min(self.k_candidates))
            cluster_usable = False
            fallback_reason = "no_structure"
            print(f"  [WARN] No valid clustering. Fallback to target-aware oversampling.")
        else:
            stable_cands = [c for c in candidates if c.get("is_stable", False)]
            if stable_cands:
                if algo == 'gmm':
                    chosen_k = int(min(stable_cands, key=lambda x: x.get("bic", float('inf')))["k"])
                else:
                    chosen_k = int(max(stable_cands, key=lambda x: x.get("silhouette", 0) or 0)["k"])
                cluster_usable = True
                fallback_reason = None
                print(f"  [OK] Stable clustering: k={chosen_k}")
            else:
                if algo == 'gmm':
                    chosen_k = int(min(candidates, key=lambda x: x.get("bic", float('inf')))["k"])
                else:
                    cand_with_sil = [c for c in candidates if c.get("silhouette") is not None]
                    chosen_k = int(max(cand_with_sil, key=lambda x: x["silhouette"])["k"]) if cand_with_sil else int(candidates[0]["k"])
                cluster_usable = False
                fallback_reason = "unstable"
                print(f"  [WARN] Unstable clustering (ARI < {self.stability_threshold}). k={chosen_k} for visualization only.")

        self.best = {
            "chosen_k": chosen_k,
            "reports": reports,
            "cluster_usable": bool(cluster_usable),
            "fallback_reason": fallback_reason,
            "algorithm": algo,
            "latent_diagnostics": self.latent_diagnostics,
            "hdbscan_result": {'k': hdbscan_result['k']} if hdbscan_result else None
        }
        return self.best

    def _compute_hdbscan_stability(self, Z_arr: np.ndarray, n_bootstrap: int = 20) -> dict:
        """Compute HDBSCAN stability via pairwise ARI."""
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            return {'ari_median': 0.0, 'is_stable': False}

        rng = np.random.RandomState(self.random_state)
        n = len(Z_arr)
        subsample_size = int(n * 0.8)
        min_cluster_size = max(5, int(n * 0.05))

        all_labels = []
        for i in range(n_bootstrap):
            idx = rng.choice(n, subsample_size, replace=False)
            hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
            hdb.fit(Z_arr[idx])
            # Approximate predict for full data
            labels_full = hdb.fit_predict(Z_arr)
            all_labels.append(labels_full)

        ari_scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)

        ari_med = float(np.median(ari_scores)) if ari_scores else 0.0
        is_stable = ari_med >= self.stability_threshold - 0.1

        return {'ari_median': ari_med, 'ari_std': float(np.std(ari_scores)) if ari_scores else 0.0, 'is_stable': is_stable}

    def get_model(self, k: int):
        return self.models.get(k, {}).get("model", None)

    def is_clustering_usable(self) -> bool:
        return self.best.get("cluster_usable", False) if self.best else False

    def save_report(self, out_dir: str, Z: Any = None):
        os.makedirs(out_dir, exist_ok=True)
        report = self.best.copy() if self.best is not None else {"chosen_k": None, "reports": {}}
        with open(os.path.join(out_dir, "cluster_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        try:
            if Z is not None and self.best and self.best.get("chosen_k") is not None:
                Z_arr = Z.values if isinstance(Z, pd.DataFrame) else np.asarray(Z)
                k = int(self.best["chosen_k"])
                model_entry = self.models.get(k)
                if model_entry is not None:
                    labels = model_entry.get("labels")
                    if Z_arr.shape[1] > 2:
                        reducer = PCA(n_components=2, random_state=self.random_state)
                        Z2 = reducer.fit_transform(Z_arr)
                    else:
                        Z2 = Z_arr[:, :2]

                    plt.figure(figsize=(6, 5))
                    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
                    algo = self.best.get('algorithm', 'kmeans')
                    plt.title(f'Latent clusters ({algo.upper()}, k={k})')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'latent_cluster_scatter.png'))
                    plt.close()
        except Exception:
            pass
