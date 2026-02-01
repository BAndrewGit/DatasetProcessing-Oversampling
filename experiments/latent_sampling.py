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
    """
    def __init__(self, per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=42):
        self.per_cluster_cap_frac = per_cluster_cap_frac
        self.global_cap_frac = global_cap_frac
        self.rs = np.random.RandomState(int(random_state) if random_state is not None else None)

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

