import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from experiments.latent_sampling import LatentSampler


def test_sample_with_caps_respects_caps_and_saves_preview(tmp_path):
    rs = np.random.RandomState(0)
    n_train = 100
    n_features = 8
    X_train = rs.randn(n_train, n_features)
    # fit PCA to have a decoder
    pca = PCA(n_components=5, random_state=0)
    Z = pca.fit_transform(X_train)

    # cluster Z into 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(Z)

    sampler = LatentSampler(per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=0)

    # Request a very large total to trigger caps
    requested_total = 1000
    counts = sampler.compute_counts_with_caps(labels, requested_total)

    # check per-cluster caps
    cluster_sizes = {int(u): int(c) for u, c in zip(*np.unique(labels, return_counts=True))}
    for cid, alloc in counts.items():
        per_cap = int(np.floor(0.2 * cluster_sizes[cid]))
        assert alloc <= per_cap

    # check global cap
    total_alloc = sum(counts.values())
    assert total_alloc <= int(np.floor(0.3 * n_train))

    # Now actually sample and decode, and save preview
    preview_path = os.path.join(str(tmp_path), "preview.csv")
    Z_synth, X_synth = sampler.sample_with_caps(Z, labels, requested_total, pca, X_train, onehot_groups=None, method='gaussian', preview_path=preview_path)

    # number of generated should match total_alloc (or <= if generation couldn't fill)
    assert Z_synth.shape[0] <= total_alloc
    assert X_synth.shape[0] == Z_synth.shape[0]

    # preview file should exist if any samples generated
    if X_synth.shape[0] > 0:
        assert os.path.exists(preview_path)
        dfp = pd.read_csv(preview_path)
        assert dfp.shape[0] <= 100
        assert dfp.shape[1] == X_synth.shape[1]
    else:
        # if none generated, preview shouldn't exist
        assert not os.path.exists(preview_path)

