import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from experiments.latent_sampling import LatentSampler


def test_compute_counts_with_caps_behavior():
    rs = np.random.RandomState(1)
    # create labels of 3 clusters with sizes 50, 30, 20
    labels = np.array([0]*50 + [1]*30 + [2]*20)
    sampler = LatentSampler(per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=0)

    requested_total = 1000
    counts = sampler.compute_counts_with_caps(labels, requested_total)

    # check per-cluster cap: each cluster cap = floor(0.2 * cluster_size)
    unique, counts_actual = np.unique(labels, return_counts=True)
    per_caps = {int(u): int(np.floor(0.2 * c)) for u, c in zip(unique, counts_actual)}
    for cid, alloc in counts.items():
        assert alloc <= per_caps[cid]

    # check global cap
    total_alloc = sum(counts.values())
    assert total_alloc <= int(np.floor(0.3 * labels.shape[0]))


def test_quality_gates_reject_obvious_memorization():
    rs = np.random.RandomState(2)
    # X_train: 100 samples random
    X_train = rs.randn(100, 5)
    # X_synth: make identical to some training rows to force memorization
    X_synth = X_train[:10].copy()
    y_train = np.zeros(100)
    X_val = rs.randn(20, 5)
    y_val = np.zeros(20)

    sampler = LatentSampler()
    audit = sampler.run_quality_gates(X_train, X_synth, y_train, X_val, y_val, task='regression', thresholds={'memorization_min': 1e-6, 'memorization_median': 1e-6, 'two_sample_auc': 0.75, 'utility_delta': 0.02, 'stability_delta': 0.2})

    assert isinstance(audit, dict)
    # Since synths duplicate train rows, memorization min should be 0 and gate should fail
    assert audit['details']['memorization']['min'] == 0.0 or audit['gates']['memorization'] is False
    assert audit['decision'] is False


