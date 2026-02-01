import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from experiments.latent_experiment import run_latent_fold
from experiments.latent_space import PCASelector
from experiments.clustering_latent import LatentClusterer
from experiments.latent_sampling import LatentSampler


def test_run_latent_fold_uses_train_only(monkeypatch, tmp_path):
    # Create synthetic small dataset
    rs = np.random.RandomState(0)
    n_train = 50
    n_val = 20
    n_features = 10
    X = rs.randn(n_train + n_val, n_features)
    y = rs.randn(n_train + n_val)

    X_train = pd.DataFrame(X[:n_train], columns=[f"f{i}" for i in range(n_features)])
    y_train = pd.Series(y[:n_train])
    X_val = pd.DataFrame(X[n_train:], columns=[f"f{i}" for i in range(n_features)])
    y_val = pd.Series(y[n_train:])

    # Flags to ensure our patched methods are called with train-only data
    flags = {"pca_fit_called": False, "cluster_fit_called": False, "sampler_called": False}

    # capture original
    orig_pca_fit = PCASelector.fit

    def fake_pca_fit(self, X_arg):
        # Should be called with X_train only
        assert X_arg.shape[0] == n_train
        flags['pca_fit_called'] = True
        # call original implementation for safety
        return orig_pca_fit(self, X_arg)

    # patch PCASelector.fit
    monkeypatch.setattr(PCASelector, 'fit', fake_pca_fit, raising=True)

    orig_cluster_fit = LatentClusterer.fit_and_choose

    def fake_cluster_fit(self, Z_arg):
        # Z_arg corresponds to Z_train; rows should match n_train
        assert Z_arg.shape[0] == n_train
        flags['cluster_fit_called'] = True
        return orig_cluster_fit(self, Z_arg)

    monkeypatch.setattr(LatentClusterer, 'fit_and_choose', fake_cluster_fit, raising=True)

    orig_sample_with_caps = LatentSampler.sample_with_caps

    def fake_sample_with_caps(self, Z, labels, requested_total, pca, X_train_arg, **kwargs):
        # Ensure Z corresponds to Z_train and X_train_arg corresponds to X_train values
        assert Z.shape[0] == n_train
        assert X_train_arg.shape[0] == n_train
        flags['sampler_called'] = True
        # call original sample_with_caps implementation
        return orig_sample_with_caps(self, Z, labels, requested_total, pca, X_train_arg, **kwargs)

    monkeypatch.setattr(LatentSampler, 'sample_with_caps', fake_sample_with_caps, raising=True)

    # config minimal
    config = {
        'pca_candidates': [5, 10],
        'k_candidates': (2, 3),
        'per_cluster_cap_frac': 0.2,
        'global_cap_frac': 0.3,
        'synth_grid': [0],
    }

    out = tmp_path / "out"
    out_dir = str(out)

    # run fold (should use patched methods and assert their checks)
    res = run_latent_fold(X_train, y_train, X_val, y_val, config, out_dir, task='regression')

    assert flags['pca_fit_called'] is True
    assert flags['cluster_fit_called'] is True
    # sampler may or may not be called depending on synth_grid; ensure patch didn't raise
    assert 'baseline' in res
