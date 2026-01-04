# Cluster-Aware Synthetic Enrichment (Optional Variant)
#
# Rules:
# - Cluster on REAL data only
# - PCA-reduced space (5-10 PCs)
# - Synthetic generated WITHIN cluster only
# - Max 20% per cluster
# - NO reclustering after enrichment

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusterAwareEnrichment:
    # Optional: cluster-based synthetic generation
    # Synthetic samples stay within their cluster's distribution

    def __init__(self, n_components=5, n_clusters=5, max_synthetic_ratio=0.20, seed=42):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.max_synthetic_ratio = max_synthetic_ratio
        self.seed = seed

        self.pca = None
        self.kmeans = None
        self.scaler = None
        self.cluster_stats = {}

    def fit(self, X):
        # Fit clustering on REAL data only
        # This is done ONCE and never updated

        np.random.seed(self.seed)

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # PCA reduction (5-10 components)
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pca = PCA(n_components=n_comp, random_state=self.seed)
        X_pca = self.pca.fit_transform(X_scaled)

        # KMeans clustering
        n_clust = min(self.n_clusters, len(X) // 5)  # At least 5 samples per cluster
        n_clust = max(2, n_clust)
        self.kmeans = KMeans(n_clusters=n_clust, random_state=self.seed, n_init=10)
        clusters = self.kmeans.fit_predict(X_pca)

        # Store cluster statistics for synthetic generation
        for cluster_id in range(n_clust):
            mask = clusters == cluster_id
            cluster_data = X[mask]

            if len(cluster_data) > 1:
                self.cluster_stats[cluster_id] = {
                    'mean': np.mean(cluster_data, axis=0),
                    'std': np.std(cluster_data, axis=0),
                    'count': len(cluster_data),
                    'indices': np.where(mask)[0]
                }

        return self

    def generate_synthetic(self, X, y, target_type='regression'):
        # Generate synthetic samples within each cluster
        # Respects max 20% synthetic ratio per cluster
        # NO reclustering after this

        if self.kmeans is None:
            raise ValueError("Must call fit() first")

        np.random.seed(self.seed)

        X_syn_all = []
        y_syn_all = []

        for cluster_id, stats in self.cluster_stats.items():
            n_real = stats['count']
            n_syn = int(n_real * self.max_synthetic_ratio)

            if n_syn < 2:
                continue

            # Generate synthetic samples within cluster distribution
            cluster_mean = stats['mean']
            cluster_std = stats['std']
            cluster_std[cluster_std == 0] = 1e-6  # Avoid zero std

            # Gaussian jitter around cluster mean
            noise = np.random.normal(0, 0.3, (n_syn, len(cluster_mean)))
            X_syn = cluster_mean + noise * cluster_std

            # Generate targets based on cluster's target distribution
            cluster_indices = stats['indices']
            y_cluster = y[cluster_indices]

            if target_type == 'regression':
                # Sample targets with noise
                y_sampled = np.random.choice(y_cluster, size=n_syn, replace=True)
                y_noise = np.random.normal(0, np.std(y_cluster) * 0.1, n_syn)
                y_syn = y_sampled + y_noise
            else:
                # Sample class labels proportionally
                y_syn = np.random.choice(y_cluster, size=n_syn, replace=True)

            X_syn_all.append(X_syn)
            y_syn_all.append(y_syn)

        if len(X_syn_all) == 0:
            return None, None

        X_syn = np.vstack(X_syn_all)
        y_syn = np.concatenate(y_syn_all)

        return X_syn.astype(np.float32), y_syn.astype(np.float32)

    def get_cluster_info(self):
        # Return cluster information for analysis
        info = {}
        for cluster_id, stats in self.cluster_stats.items():
            info[f"cluster_{cluster_id}"] = {
                'n_samples': stats['count'],
                'max_synthetic': int(stats['count'] * self.max_synthetic_ratio)
            }
        return info


def generate_cluster_aware_synthetic(X_train, y_train, target_type='regression',
                                      n_components=5, n_clusters=5,
                                      max_ratio=0.20, seed=42):
    # Convenience function for cluster-aware synthetic generation

    enricher = ClusterAwareEnrichment(
        n_components=n_components,
        n_clusters=n_clusters,
        max_synthetic_ratio=max_ratio,
        seed=seed
    )

    enricher.fit(X_train)
    X_syn, y_syn = enricher.generate_synthetic(X_train, y_train, target_type)

    return X_syn, y_syn, enricher.get_cluster_info()

