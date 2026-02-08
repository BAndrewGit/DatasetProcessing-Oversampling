import json
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class LatentSampler:
    """
    Cluster-conditioned sampler in latent space.
    Two methods: gaussian per-cluster or kNN jitter.
    Decoding via PCA.inverse_transform is performed outside and passed as decoder.

    FIXES IMPLEMENTED:
    - Real memorization threshold based on real-to-real kNN distance distribution
    - Coverage constraint using quantile bins (not KMeans)
    - Sample weighting for training
    - Proper rejection of near-duplicates
    """
    def __init__(self, per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=42,
                 synth_weight=0.3, memorization_percentile=5):
        self.per_cluster_cap_frac = per_cluster_cap_frac
        self.global_cap_frac = global_cap_frac
        self.rs = np.random.RandomState(int(random_state) if random_state is not None else None)

        # NEW: Sample weighting for training (0.2-0.5 recommended)
        self.synth_weight = synth_weight

        # NEW: Memorization threshold percentile (p1 or p5)
        self.memorization_percentile = memorization_percentile

        # Will be computed from data
        self._real_to_real_threshold = None

    def compute_real_to_real_threshold(self, X_train: np.ndarray, k: int = 1) -> float:
        """
        Compute memorization threshold based on real-to-real kNN distances.

        Uses p1 or p5 percentile of the distribution of distances from each
        real sample to its k-th nearest neighbor (excluding itself).

        Any synthetic sample with distance < threshold is considered a near-duplicate.
        """
        if X_train.shape[0] < 3:
            return 0.0

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_train)  # +1 to exclude self
        distances, _ = nn.kneighbors(X_train)

        # Take distances to k-th neighbor (index k because 0 is self with dist=0)
        kth_distances = distances[:, k]

        # Compute percentile threshold
        threshold = float(np.percentile(kth_distances, self.memorization_percentile))
        self._real_to_real_threshold = threshold

        return threshold

    def reject_near_duplicates(self, X_synth: np.ndarray, X_train: np.ndarray,
                                threshold: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Reject synthetic samples that are too close to real samples.

        Args:
            X_synth: Synthetic samples
            X_train: Real training samples
            threshold: Distance threshold (if None, uses computed real-to-real threshold)

        Returns:
            X_filtered: Synthetic samples that passed the filter
            stats: Dict with rejection statistics
        """
        if X_synth.shape[0] == 0:
            return X_synth, {'n_original': 0, 'n_rejected': 0, 'n_kept': 0}

        if threshold is None:
            if self._real_to_real_threshold is None:
                threshold = self.compute_real_to_real_threshold(X_train)
            else:
                threshold = self._real_to_real_threshold

        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        distances, _ = nn.kneighbors(X_synth)
        distances = distances.ravel()

        # Keep only samples with distance > threshold
        mask = distances > threshold
        X_filtered = X_synth[mask]

        stats = {
            'n_original': int(X_synth.shape[0]),
            'n_rejected': int((~mask).sum()),
            'n_kept': int(mask.sum()),
            'threshold': float(threshold),
            'rejection_rate': float((~mask).sum() / X_synth.shape[0]) if X_synth.shape[0] > 0 else 0.0,
            'distances_rejected': distances[~mask].tolist() if (~mask).sum() > 0 else [],
            'distances_kept_min': float(distances[mask].min()) if mask.sum() > 0 else None,
            'distances_kept_median': float(np.median(distances[mask])) if mask.sum() > 0 else None,
        }

        return X_filtered, stats

    def create_coverage_bins(self, y_train: np.ndarray, X_train: np.ndarray = None,
                              n_bins: int = 5, feature_idx: int = None) -> Dict:
        """
        Create coverage bins using quantiles (NOT KMeans).

        Bins are based on target distribution to ensure rare regions get coverage.
        Optionally also considers a feature axis.

        Args:
            y_train: Target values
            X_train: Features (optional, for 2D binning)
            n_bins: Number of bins
            feature_idx: Feature index for 2D binning (optional)

        Returns:
            Dict with bin edges, counts, and sample indices per bin
        """
        y_train = np.asarray(y_train)

        # Create quantile bins on target
        bin_edges = np.percentile(y_train, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        n_bins = len(bin_edges) - 1

        # Assign samples to bins
        bin_indices = np.digitize(y_train, bin_edges[1:-1])  # 0 to n_bins-1

        bins = {}
        for i in range(n_bins):
            mask = bin_indices == i
            bins[i] = {
                'count': int(mask.sum()),
                'indices': np.where(mask)[0].tolist(),
                'y_range': (float(bin_edges[i]), float(bin_edges[i + 1])),
                'target_mean': float(y_train[mask].mean()) if mask.sum() > 0 else 0.0,
            }

        return {
            'bin_edges': bin_edges.tolist(),
            'n_bins': n_bins,
            'bins': bins,
            'total_samples': len(y_train)
        }

    def sample_with_coverage(self, Z: np.ndarray, y_train: np.ndarray,
                              n_samples: int, pca_decoder, X_train: np.ndarray,
                              n_bins: int = 5, method: str = "gaussian") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic samples with coverage constraint.

        Ensures rare bins (by target quantile) get proportional coverage.
        Uses quantile bins instead of KMeans clusters.

        Args:
            Z: Latent representations
            y_train: Target values
            n_samples: Total samples to generate
            pca_decoder: PCA for inverse transform
            X_train: Original features (for postprocessing)
            n_bins: Number of quantile bins
            method: "gaussian" or "knn_jitter"

        Returns:
            Z_synth: Synthetic latent vectors
            X_synth: Decoded synthetic features
            coverage_stats: Dict with coverage information
        """
        # Create coverage bins
        coverage = self.create_coverage_bins(y_train, n_bins=n_bins)
        bins = coverage['bins']

        # Calculate samples per bin (proportional to size, but with minimum guarantee)
        total_real = coverage['total_samples']
        min_per_bin = max(1, n_samples // (len(bins) * 3))  # At least some from each bin

        # Allocate proportionally with minimum guarantee
        allocations = {}
        remaining = n_samples

        for bin_id, bin_info in bins.items():
            if bin_info['count'] < 2:  # Need at least 2 for covariance
                allocations[bin_id] = 0
                continue

            # Proportional allocation
            prop = bin_info['count'] / total_real
            alloc = max(min_per_bin, int(n_samples * prop))

            # Cap to avoid over-allocating
            alloc = min(alloc, int(bin_info['count'] * self.per_cluster_cap_frac))
            allocations[bin_id] = alloc
            remaining -= alloc

        # Distribute remaining proportionally
        if remaining > 0:
            for bin_id in sorted(allocations.keys()):
                if bins[bin_id]['count'] >= 2:
                    add = min(remaining, 5)
                    allocations[bin_id] += add
                    remaining -= add
                    if remaining <= 0:
                        break

        # Generate samples per bin
        synths_z = []
        synths_bins = []

        for bin_id, n_alloc in allocations.items():
            if n_alloc <= 0:
                continue

            bin_indices = bins[bin_id]['indices']
            Z_bin = Z[bin_indices]

            if Z_bin.shape[0] < 2:
                continue

            if method == "gaussian":
                z_new = self.gaussian_sample(Z_bin, n_alloc)
            else:
                z_new = self.knn_jitter(Z_bin, n_alloc)

            synths_z.append(z_new)
            synths_bins.extend([bin_id] * len(z_new))

        if not synths_z:
            return np.zeros((0, Z.shape[1])), np.zeros((0, X_train.shape[1])), coverage

        Z_synth = np.vstack(synths_z)

        # Decode
        X_synth = pca_decoder.inverse_transform(Z_synth)
        X_synth = self.postprocess_decoded(X_synth, X_train)

        coverage['allocations'] = allocations
        coverage['synth_bin_labels'] = synths_bins

        return Z_synth, X_synth, coverage

    def create_sample_weights(self, n_real: int, n_synth: int,
                               synth_weight: float = None) -> np.ndarray:
        """
        Create sample weights for training with synthetic data.

        Real samples get weight 1.0, synthetic samples get synth_weight.

        Args:
            n_real: Number of real samples
            n_synth: Number of synthetic samples
            synth_weight: Weight for synthetic samples (default: self.synth_weight)

        Returns:
            weights: Array of sample weights
        """
        if synth_weight is None:
            synth_weight = self.synth_weight

        real_weights = np.ones(n_real)
        synth_weights = np.full(n_synth, synth_weight)

        return np.concatenate([real_weights, synth_weights])

    def gaussian_sample(self, Z_cluster: np.ndarray, n_samples: int) -> np.ndarray:
        mu = Z_cluster.mean(axis=0)
        cov = np.cov(Z_cluster, rowvar=False)
        # regularize cov
        cov += 1e-6 * np.eye(cov.shape[0])
        return self.rs.multivariate_normal(mu, cov, size=n_samples)

    def knn_jitter(self, Z_cluster: np.ndarray, n_samples: int, k=5, scale=0.05) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=min(k, Z_cluster.shape[0])).fit(Z_cluster)
        inds = self.rs.randint(0, Z_cluster.shape[0], size=n_samples)
        base = Z_cluster[inds]
        nbrs = nn.kneighbors(base, return_distance=False)[:, 0]
        jitter = (Z_cluster[nbrs] - base) * (self.rs.randn(n_samples, Z_cluster.shape[1]) * scale)
        return base + jitter

    def generate_per_cluster(self, Z: np.ndarray, labels: np.ndarray, counts: Dict[int,int], method="gaussian") -> np.ndarray:
        synths = []
        for cluster_id, n in counts.items():
            mask = labels == cluster_id
            Zc = Z[mask]
            if Zc.shape[0] < 2 or n <= 0:
                continue
            if method == "gaussian":
                zs = self.gaussian_sample(Zc, n)
            else:
                zs = self.knn_jitter(Zc, n)
            synths.append(zs)
        if not synths:
            return np.zeros((0, Z.shape[1]))
        return np.vstack(synths)

    def postprocess_decoded(self, X_synth: np.ndarray, X_train: np.ndarray, onehot_groups: Dict[str, Tuple[int,int]] = None) -> np.ndarray:
        # clip numeric to min/max of train per column
        mins = X_train.min(axis=0)
        maxs = X_train.max(axis=0)
        X_clipped = np.clip(X_synth, mins, maxs)
        if onehot_groups:
            Xc = X_clipped.copy()
            for col_start, col_end in onehot_groups.values():
                block = Xc[:, col_start:col_end]
                arg = block.argmax(axis=1)
                new_block = np.zeros_like(block)
                new_block[np.arange(block.shape[0]), arg] = 1
                Xc[:, col_start:col_end] = new_block
            return Xc
        return X_clipped

    def audit_basic(self, X_train: np.ndarray, X_synth: np.ndarray) -> Dict[str, Any]:
        # simple memorization check: nearest neighbor distance statistics
        if X_synth.shape[0] == 0:
            return {"n_synth": 0, "memorization_min": None, "memorization_median": None}
        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        dists, _ = nn.kneighbors(X_synth)
        dists = dists.ravel()
        return {"n_synth": int(X_synth.shape[0]), "memorization_min": float(dists.min()), "memorization_median": float(np.median(dists))}

    def compute_counts_with_caps(self, labels: np.ndarray, requested_total: int, per_cluster_cap_frac: float = None, global_cap_frac: float = None) -> Dict[int, int]:
        """
        Compute per-cluster sample counts respecting per-cluster and global caps.
        labels: array-like cluster labels for training set
        requested_total: desired total synthetic samples
        returns dict cluster_id -> n_samples
        """
        if per_cluster_cap_frac is None:
            per_cluster_cap_frac = self.per_cluster_cap_frac
        if global_cap_frac is None:
            global_cap_frac = self.global_cap_frac

        labels = np.asarray(labels)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        total_train = labels.shape[0]

        # enforce global cap
        global_cap = int(np.floor(global_cap_frac * total_train)) if total_train > 0 else 0
        total_allowed = min(int(requested_total), global_cap)

        # per-cluster caps
        per_caps = {cid: int(np.floor(per_cluster_cap_frac * sz)) for cid, sz in cluster_sizes.items()}

        # proportional allocation
        if total_allowed <= 0:
            return {cid: 0 for cid in cluster_sizes.keys()}
        prop = {cid: sz / sum(cluster_sizes.values()) for cid, sz in cluster_sizes.items()}
        raw_alloc = {cid: int(np.floor(prop[cid] * total_allowed)) for cid in cluster_sizes.keys()}
        # adjust remainder
        remainder = total_allowed - sum(raw_alloc.values())
        cids = list(cluster_sizes.keys())
        i = 0
        while remainder > 0 and len(cids) > 0:
            raw_alloc[cids[i % len(cids)]] += 1
            i += 1
            remainder -= 1

        # enforce per-cluster cap
        final_alloc = {cid: min(raw_alloc.get(cid, 0), per_caps.get(cid, 0)) for cid in cluster_sizes.keys()}

        # if sum < total_allowed because of caps, distribute remaining to clusters with slack
        remaining = total_allowed - sum(final_alloc.values())
        if remaining > 0:
            # clusters with slack
            slack = {cid: per_caps[cid] - final_alloc[cid] for cid in cluster_sizes.keys()}
            slack_cids = [cid for cid, s in slack.items() if s > 0]
            idx = 0
            while remaining > 0 and slack_cids:
                cid = slack_cids[idx % len(slack_cids)]
                add = min(1, slack[cid])
                final_alloc[cid] += add
                slack[cid] -= add
                if slack[cid] == 0:
                    slack_cids.remove(cid)
                remaining -= add
                idx += 1

        return {int(cid): int(n) for cid, n in final_alloc.items()}

    def sample_with_caps(self, Z: np.ndarray, labels: np.ndarray, requested_total: int, pca_decoder, X_train: np.ndarray, onehot_groups: Dict[str, Tuple[int,int]] = None, method: str = "gaussian", per_cluster_cap_frac: float = None, global_cap_frac: float = None, preview_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-level sampling: compute counts with caps, generate Z_synth within clusters, decode with pca_decoder.inverse_transform,
        postprocess one-hot groups, save a small CSV preview if preview_path provided. Returns (Z_synth, X_synth).
        """
        labels = np.asarray(labels)
        Z = np.asarray(Z)
        counts = self.compute_counts_with_caps(labels, requested_total, per_cluster_cap_frac=per_cluster_cap_frac, global_cap_frac=global_cap_frac)
        total = sum(counts.values())
        if total == 0:
            return np.zeros((0, Z.shape[1])), np.zeros((0, X_train.shape[1]))

        Z_synth = self.generate_per_cluster(Z, labels, counts, method=method)
        # cap to total just in case
        if Z_synth.shape[0] > total:
            Z_synth = Z_synth[:total]

        # decode
        X_synth = pca_decoder.inverse_transform(Z_synth)
        X_synth = self.postprocess_decoded(X_synth, X_train, onehot_groups=onehot_groups)

        # preview save
        try:
            if preview_path and X_synth.shape[0] > 0:
                preview_n = min(100, X_synth.shape[0])
                dfp = pd.DataFrame(X_synth[:preview_n], columns=[f"f{i}" for i in range(X_synth.shape[1])])
                dfp.to_csv(preview_path, index=False)
        except Exception:
            pass

        return Z_synth, X_synth

    def run_quality_gates(self, X_train: np.ndarray, X_synth: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task: str = "regression", model_factory=None, thresholds: Dict[str, Any] = None, n_stability_repeats: int = 5) -> Dict[str, Any]:
        """
        Run fold-level synthetic quality gates and return audit dict with gate results and metrics.

        gates:
          - memorization: min/median NN distance above thresholds
          - two_sample: discriminator AUC < threshold (can't separate real vs synth)
          - utility: augmented performance not worse than threshold (%)
          - stability: variance of metric not worse than threshold (% increase)

        model_factory: callable that returns a fresh estimator supporting fit/predict (for utility/stability)
        thresholds: dict with keys 'memorization_min', 'memorization_median', 'two_sample_auc', 'utility_delta', 'stability_delta'
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
        import math

        if thresholds is None:
            thresholds = {}
        mem_min_thr = float(thresholds.get('memorization_min', 1e-6))
        mem_med_thr = float(thresholds.get('memorization_median', 1e-6))
        two_sample_auc_thr = float(thresholds.get('two_sample_auc', 0.75))
        utility_delta_thr = float(thresholds.get('utility_delta', 0.02))
        stability_delta_thr = float(thresholds.get('stability_delta', 0.2))

        audit = {'gates': {}, 'details': {}}

        # memorization
        if X_synth is None or X_synth.shape[0] == 0:
            audit['gates']['memorization'] = False
            audit['details']['memorization'] = {'n_synth': 0}
            # if no synth, other gates are irrelevant
            audit['decision'] = False
            return audit

        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        dists, _ = nn.kneighbors(X_synth)
        dists = dists.ravel()
        mem_min = float(dists.min())
        mem_med = float(np.median(dists))
        audit['details']['memorization'] = {'min': mem_min, 'median': mem_med}
        audit['gates']['memorization'] = (mem_min > mem_min_thr and mem_med > mem_med_thr)

        # two-sample discriminator
        try:
            # build dataset: sample up to n for balance
            n_real = min(X_train.shape[0], 1000)
            n_synth = min(X_synth.shape[0], 1000)
            idx_real = np.random.choice(np.arange(X_train.shape[0]), n_real, replace=False)
            idx_synth = np.random.choice(np.arange(X_synth.shape[0]), n_synth, replace=False)
            X_ds = np.vstack([X_train[idx_real], X_synth[idx_synth]])
            y_ds = np.concatenate([np.zeros(n_real), np.ones(n_synth)])
            clf = LogisticRegression(max_iter=200, solver='liblinear')
            # train-test split for AUC
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X_ds, y_ds, test_size=0.3, random_state=42, stratify=y_ds)
            clf.fit(X_tr, y_tr)
            if hasattr(clf, 'predict_proba'):
                probs = clf.predict_proba(X_te)[:, 1]
            else:
                probs = clf.decision_function(X_te)
            auc = float(roc_auc_score(y_te, probs))
        except Exception as e:
            auc = 1.0  # worst-case: discriminator succeeded
        audit['details']['two_sample_auc'] = auc
        audit['gates']['two_sample'] = (auc < two_sample_auc_thr)

        # utility: compare real-only vs augmented on val
        if model_factory is None:
            # default model: small MLP or linear depending on task
            if task == 'regression':
                from sklearn.linear_model import Ridge
                def model_factory():
                    return Ridge()
            else:
                from sklearn.linear_model import LogisticRegression
                def model_factory():
                    return LogisticRegression(max_iter=200)

        # Train real-only
        try:
            model_real = model_factory()
            model_real.fit(X_train, y_train)
            y_pred_real = model_real.predict(X_val)
            if task == 'regression':
                real_score = float(mean_absolute_error(y_val, y_pred_real))
            else:
                real_score = float(f1_score(y_val, y_pred_real, average='macro', zero_division=0))
        except Exception as e:
            audit['gates']['utility'] = False
            audit['details']['utility'] = {'error': str(e)}
            # fail safe
            audit['decision'] = False
            return audit

        # Train augmented
        try:
            X_aug = np.vstack([X_train, X_synth])
            if np.isscalar(y_train[0]) or hasattr(y_train, 'shape'):
                y_aug = np.concatenate([y_train, np.repeat(np.mean(y_train), X_synth.shape[0])]) if task == 'regression' else np.concatenate([y_train, np.repeat(pd.Series(y_train).mode().iat[0], X_synth.shape[0])])
            else:
                y_aug = np.concatenate([y_train, np.repeat(np.mean(y_train), X_synth.shape[0])])
            model_aug = model_factory()
            model_aug.fit(X_aug, y_aug)
            y_pred_aug = model_aug.predict(X_val)
            if task == 'regression':
                aug_score = float(mean_absolute_error(y_val, y_pred_aug))
            else:
                aug_score = float(f1_score(y_val, y_pred_aug, average='macro', zero_division=0))
        except Exception as e:
            audit['gates']['utility'] = False
            audit['details']['utility'] = {'error': str(e)}
            audit['decision'] = False
            return audit

        audit['details']['utility'] = {'real_score': real_score, 'aug_score': aug_score}
        # For regression lower-is-better: allow augmented <= real*(1+delta)
        if task == 'regression':
            audit['gates']['utility'] = (aug_score <= real_score * (1.0 + utility_delta_thr))
        else:
            # classification higher-is-better: allow augmented >= real*(1-delta)
            audit['gates']['utility'] = (aug_score >= real_score * (1.0 - utility_delta_thr))

        # stability: measure variance across multiple re-trains
        try:
            real_scores = []
            aug_scores = []
            for i in range(max(1, min(n_stability_repeats, 10))):
                m1 = model_factory()
                m1.fit(X_train, y_train)
                p1 = m1.predict(X_val)
                if task == 'regression':
                    real_scores.append(float(mean_absolute_error(y_val, p1)))
                else:
                    real_scores.append(float(f1_score(y_val, p1, average='macro', zero_division=0)))

                m2 = model_factory()
                m2.fit(np.vstack([X_train, X_synth]), np.concatenate([y_train, np.repeat(np.mean(y_train), X_synth.shape[0])]) if task=='regression' else np.concatenate([y_train, np.repeat(pd.Series(y_train).mode().iat[0], X_synth.shape[0])]))
                p2 = m2.predict(X_val)
                if task == 'regression':
                    aug_scores.append(float(mean_absolute_error(y_val, p2)))
                else:
                    aug_scores.append(float(f1_score(y_val, p2, average='macro', zero_division=0)))

            var_real = float(np.var(real_scores, ddof=1)) if len(real_scores) > 1 else 0.0
            var_aug = float(np.var(aug_scores, ddof=1)) if len(aug_scores) > 1 else 0.0
            audit['details']['stability'] = {'var_real': var_real, 'var_aug': var_aug}
            # stability passes if var_aug <= var_real*(1+stability_delta_thr)
            audit['gates']['stability'] = (var_aug <= var_real * (1.0 + stability_delta_thr))
        except Exception as e:
            audit['gates']['stability'] = False
            audit['details']['stability'] = {'error': str(e)}

        # Final decision: all gates must pass
        all_pass = all(bool(v) for v in audit['gates'].values())
        audit['decision'] = bool(all_pass)
        return audit


class ContinuousLatentSampler:
    """
    Continuous latent space sampler - NO CLUSTERING.

    Best approach when latent space is a continuous manifold (ARI≈0).

    Methods (in order of recommendation):
    - smote_latent: SMOTE-style interpolation between kNN in latent (BEST)
    - knn_mixup: Random convex combination of k nearest neighbors
    - gaussian_noise: Add scaled Gaussian noise around each point
    - kde: Kernel Density Estimation sampling
    - gaussian: Fit global Gaussian and sample

    All methods include:
    - Target-aware sampling (ensures coverage of rare target values)
    - Memorization rejection (removes samples too close to real)
    - Label propagation via kNN weighted average
    """

    def __init__(self, method='smote_latent', random_state=42,
                 memorization_percentile=5, target_aware=True,
                 k_neighbors=5, noise_scale=0.1):
        self.method = method
        self.rs = np.random.RandomState(int(random_state) if random_state is not None else None)
        self.memorization_percentile = memorization_percentile
        self.target_aware = target_aware
        self.k_neighbors = k_neighbors
        self.noise_scale = noise_scale
        self._threshold = None
        self._Z_train = None
        self._y_train = None

    def fit(self, Z: np.ndarray, y: np.ndarray = None):
        """Store training data and compute memorization threshold."""
        self._Z_train = np.asarray(Z)
        if y is not None:
            self._y_train = np.asarray(y)

        # Compute memorization threshold from real-to-real distances
        nn = NearestNeighbors(n_neighbors=2).fit(self._Z_train)
        dists, _ = nn.kneighbors(self._Z_train)
        self._threshold = float(np.percentile(dists[:, 1], self.memorization_percentile))

        return self

    def _sample_gaussian_noise(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add scaled Gaussian noise around randomly selected real points."""
        n = len(self._Z_train)
        indices = self.rs.choice(n, n_samples, replace=True)

        # Compute per-dimension std for scaling
        std_per_dim = np.std(self._Z_train, axis=0) * self.noise_scale

        # Add noise
        noise = self.rs.randn(n_samples, self._Z_train.shape[1]) * std_per_dim
        Z_synth = self._Z_train[indices] + noise

        # Propagate labels
        y_synth = self._y_train[indices] if self._y_train is not None else None

        return Z_synth, y_synth

    def _sample_knn_mixup(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random convex combination of point and one of its k neighbors."""
        n = len(self._Z_train)
        k = min(self.k_neighbors, n - 1)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(self._Z_train)
        _, indices = nn.kneighbors(self._Z_train)

        Z_synth = []
        y_synth = []

        for _ in range(n_samples):
            # Pick random anchor
            anchor_idx = self.rs.randint(n)
            anchor = self._Z_train[anchor_idx]

            # Pick random neighbor (skip self at index 0)
            neighbor_idx = indices[anchor_idx, self.rs.randint(1, k + 1)]
            neighbor = self._Z_train[neighbor_idx]

            # Random interpolation factor
            alpha = self.rs.uniform(0.1, 0.9)
            z_new = anchor * alpha + neighbor * (1 - alpha)
            Z_synth.append(z_new)

            # Interpolate labels
            if self._y_train is not None:
                y_new = self._y_train[anchor_idx] * alpha + self._y_train[neighbor_idx] * (1 - alpha)
                y_synth.append(y_new)

        return np.array(Z_synth), np.array(y_synth) if y_synth else None

    def _sample_smote_latent(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE-style: interpolate between point and random kNN neighbor."""
        n = len(self._Z_train)
        k = min(self.k_neighbors, n - 1)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(self._Z_train)
        _, indices = nn.kneighbors(self._Z_train)

        Z_synth = []
        y_synth = []

        for _ in range(n_samples):
            # Pick random anchor
            anchor_idx = self.rs.randint(n)
            anchor = self._Z_train[anchor_idx]

            # Pick random neighbor
            neighbor_idx = indices[anchor_idx, self.rs.randint(1, k + 1)]
            neighbor = self._Z_train[neighbor_idx]

            # SMOTE: interpolate along the line between anchor and neighbor
            diff = neighbor - anchor
            gap = self.rs.uniform(0, 1)
            z_new = anchor + gap * diff
            Z_synth.append(z_new)

            # Interpolate labels (weighted by distance)
            if self._y_train is not None:
                y_anchor = self._y_train[anchor_idx]
                y_neighbor = self._y_train[neighbor_idx]
                y_new = y_anchor + gap * (y_neighbor - y_anchor)
                y_synth.append(y_new)

        return np.array(Z_synth), np.array(y_synth) if y_synth else None

    def _sample_kde(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """KDE sampling."""
        from sklearn.neighbors import KernelDensity

        bw = self._Z_train.shape[0] ** (-1.0 / (self._Z_train.shape[1] + 4))
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(self._Z_train)

        Z_synth = kde.sample(n_samples, random_state=self.rs)

        # Assign labels via kNN
        y_synth = self._propagate_labels(Z_synth) if self._y_train is not None else None

        return Z_synth, y_synth

    def _sample_gaussian(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Global Gaussian sampling."""
        mean = np.mean(self._Z_train, axis=0)
        cov = np.cov(self._Z_train, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6

        Z_synth = self.rs.multivariate_normal(mean, cov, size=n_samples)
        y_synth = self._propagate_labels(Z_synth) if self._y_train is not None else None

        return Z_synth, y_synth

    def _propagate_labels(self, Z_synth: np.ndarray, k: int = 3) -> np.ndarray:
        """Propagate labels via distance-weighted kNN average."""
        k = min(k, len(self._Z_train))
        nn = NearestNeighbors(n_neighbors=k).fit(self._Z_train)
        dists, indices = nn.kneighbors(Z_synth)

        # Distance-weighted average
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        y_synth = np.sum(self._y_train[indices] * weights, axis=1)
        return y_synth

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample using configured method."""
        if self.method == 'smote_latent':
            return self._sample_smote_latent(n_samples)
        elif self.method == 'knn_mixup':
            return self._sample_knn_mixup(n_samples)
        elif self.method == 'gaussian_noise':
            return self._sample_gaussian_noise(n_samples)
        elif self.method == 'kde':
            return self._sample_kde(n_samples)
        elif self.method == 'gaussian':
            return self._sample_gaussian(n_samples)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def sample_target_aware(self, n_samples: int, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Sample with target coverage: ensure rare target values get representation."""
        if self._y_train is None:
            raise ValueError("Fit with y to use target-aware sampling")

        # Create target quantile bins
        bin_edges = np.percentile(self._y_train, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        n_bins = len(bin_edges) - 1
        bin_indices = np.digitize(self._y_train, bin_edges[1:-1])

        samples_per_bin = max(1, n_samples // n_bins)
        Z_synth_list, y_synth_list = [], []

        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() < 2:
                continue

            # Create mini-sampler for this bin
            Z_bin = self._Z_train[mask]
            y_bin = self._y_train[mask]

            mini_sampler = ContinuousLatentSampler(
                method=self.method, random_state=self.rs.randint(10000),
                k_neighbors=min(self.k_neighbors, len(Z_bin) - 1),
                noise_scale=self.noise_scale
            )
            mini_sampler.fit(Z_bin, y_bin)

            Z_new, y_new = mini_sampler.sample(samples_per_bin)
            Z_synth_list.append(Z_new)
            y_synth_list.append(y_new)

        if not Z_synth_list:
            return np.array([]).reshape(0, self._Z_train.shape[1]), np.array([])

        return np.vstack(Z_synth_list), np.concatenate(y_synth_list)

    def reject_memorized(self, Z_synth: np.ndarray, Z_train: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """Reject samples too close to real data."""
        if Z_synth.shape[0] == 0:
            return Z_synth, {'n_rejected': 0, 'n_kept': 0}

        Z_ref = Z_train if Z_train is not None else self._Z_train
        nn = NearestNeighbors(n_neighbors=1).fit(Z_ref)
        dists, _ = nn.kneighbors(Z_synth)
        dists = dists.ravel()

        mask = dists > self._threshold

        return Z_synth[mask], {
            'n_original': len(Z_synth),
            'n_rejected': int((~mask).sum()),
            'n_kept': int(mask.sum()),
            'threshold': self._threshold,
            'rejection_rate': float((~mask).mean())
        }

    def generate(self, Z_train: np.ndarray, y_train: np.ndarray, n_samples: int,
                 pca_decoder=None, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Full pipeline: fit, sample, reject, decode."""
        self.fit(Z_train, y_train)

        # Oversample to account for rejection
        n_oversample = int(n_samples * 1.5)

        if self.target_aware:
            Z_synth, y_synth = self.sample_target_aware(n_oversample, n_bins)
        else:
            Z_synth, y_synth = self.sample(n_oversample)

        # Reject memorized
        Z_synth, reject_stats = self.reject_memorized(Z_synth)

        # Match y_synth to kept samples
        if y_synth is not None and len(y_synth) > len(Z_synth):
            # Re-propagate labels for kept samples
            y_synth = self._propagate_labels(Z_synth)

        # Trim to requested size
        if len(Z_synth) > n_samples:
            idx = self.rs.choice(len(Z_synth), n_samples, replace=False)
            Z_synth = Z_synth[idx]
            if y_synth is not None:
                y_synth = y_synth[idx]

        # Decode
        X_synth = pca_decoder.inverse_transform(Z_synth) if pca_decoder else Z_synth

        stats = {
            'method': self.method,
            'n_requested': n_samples,
            'n_generated': len(Z_synth),
            'target_aware': self.target_aware,
            **reject_stats
        }

        return Z_synth, X_synth, y_synth if y_synth is not None else np.array([]), stats

