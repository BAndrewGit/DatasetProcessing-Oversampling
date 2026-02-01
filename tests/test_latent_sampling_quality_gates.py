import numpy as np
import pandas as pd
from experiments.latent_sampling import LatentSampler


def test_audit_basic_reports_distances():
    rng = np.random.RandomState(0)
    X_train = rng.randn(50, 5)
    X_synth = X_train[:10] + 1e-3  # near duplicates
    sampler = LatentSampler()
    audit = sampler.audit_basic(X_train, X_synth)
    assert audit["n_synth"] == 10
    assert audit["memorization_min"] >= 0.0

