import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class LatentClusterer:
    """
    Fit KMeans on latent train-space, choose k by silhouette / DB while enforcing min cluster size.
    """
    def __init__(self, k_candidates=(2,3,4,5), min_fraction=0.05, random_state=42):
        self.k_candidates = k_candidates
        self.min_fraction = min_fraction
        self.random_state = int(random_state) if random_state is not None else None
        self.best = None
        self.models = {}

    def fit_and_choose(self, Z: pd.DataFrame):
        # Accept Z as DataFrame or ndarray
        if isinstance(Z, pd.DataFrame):
            Z_arr = Z.values
        else:
            Z_arr = np.asarray(Z)
        n = Z_arr.shape[0]
        reports: Dict[int, Dict[str, Any]] = {}
        for k in self.k_candidates:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(Z_arr)
            sizes = np.bincount(labels, minlength=k)
            if (sizes < max(1, int(self.min_fraction * n))).any():
                reports[k] = {"k": k, "rejected": True, "reason": "small_cluster", "sizes": sizes.tolist()}
                continue
            sil = None
            try:
                sil = float(silhouette_score(Z_arr, labels))
            except Exception:
                sil = None
            db = float(davies_bouldin_score(Z_arr, labels))
            centroids = km.cluster_centers_.tolist()
            reports[k] = {"k": k, "rejected": False, "silhouette": sil, "davies_bouldin": db, "sizes": sizes.tolist(), "centroids": centroids}
            self.models[k] = {"model": km, "labels": labels}
        # choose k: prefer non-rejected with highest silhouette (if available) else lowest DB
        candidates = [r for r in reports.values() if not r.get("rejected", True)]
        if not candidates:
            # all rejected -> fallback to smallest k in candidate list
            chosen_k = int(min(self.k_candidates))
        else:
            # prefer silhouette if available
            cand_with_sil = [c for c in candidates if c.get("silhouette") is not None]
            if cand_with_sil:
                chosen_k = int(max(cand_with_sil, key=lambda x: x["silhouette"])["k"])
            else:
                chosen_k = int(min(candidates, key=lambda x: x["davies_bouldin"])["k"])
        self.best = {"chosen_k": chosen_k, "reports": reports}
        return self.best

    def get_model(self, k:int):
        return self.models.get(k, {}).get("model", None)

    def save_report(self, out_dir: str, Z: Any = None):
        """
        Save cluster_report.json. If Z (latent array) is provided, also create a 2D scatter plot of clusters.
        """
        os.makedirs(out_dir, exist_ok=True)
        report = self.best.copy() if self.best is not None else {"chosen_k": None, "reports": {}}
        # Flatten reports to serializable
        with open(os.path.join(out_dir, "cluster_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # If Z provided, create a 2D projection and scatter plot colored by chosen labels
        try:
            if Z is not None and self.best and self.best.get("chosen_k") is not None:
                if isinstance(Z, pd.DataFrame):
                    Z_arr = Z.values
                else:
                    Z_arr = np.asarray(Z)
                k = int(self.best["chosen_k"])
                model_entry = self.models.get(k)
                if model_entry is not None:
                    labels = model_entry.get("labels")
                    # 2D projection: if Z_arr has >=2 dims, use PCA to reduce to 2 for plotting
                    if Z_arr.shape[1] > 2:
                        reducer = PCA(n_components=2, random_state=self.random_state)
                        Z2 = reducer.fit_transform(Z_arr)
                    else:
                        Z2 = Z_arr[:, :2]

                    plt.figure(figsize=(6, 5))
                    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
                    plt.title(f'Latent clusters (k={k})')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.grid(False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'latent_cluster_scatter.png'))
                    plt.close()
        except Exception:
            # plotting must not raise
            pass
