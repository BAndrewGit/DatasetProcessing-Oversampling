import numpy as np
from sklearn.decomposition import PCA
from experiments.latent_sampling import LatentSampler


def test_sampler_is_deterministic_given_seed():
    rs = np.random.RandomState(3)
    X_train = rs.randn(80, 6)
    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(X_train)
    # create cluster labels
    labels = np.random.RandomState(0).randint(0, 3, size=Z.shape[0])

    sampler1 = LatentSampler(per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=42)
    sampler2 = LatentSampler(per_cluster_cap_frac=0.2, global_cap_frac=0.3, random_state=42)

    Zs1, Xs1 = sampler1.sample_with_caps(Z, labels, 10, pca, X_train, method='gaussian')
    Zs2, Xs2 = sampler2.sample_with_caps(Z, labels, 10, pca, X_train, method='gaussian')

    # determinism: outputs should be equal
    assert Zs1.shape == Zs2.shape
    assert Xs1.shape == Xs2.shape
    if Zs1.shape[0] > 0:
        assert np.allclose(Zs1, Zs2)
        assert np.allclose(Xs1, Xs2)


