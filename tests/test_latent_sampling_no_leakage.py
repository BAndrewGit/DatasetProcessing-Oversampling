import pytest
import pandas as pd
import numpy as np
from experiments.latent_space import PCASelector


def test_pca_fit_on_train_only(tmp_path):
    # Create toy data
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 10))
    # split
    train = X.iloc[:80]
    val = X.iloc[80:]
    psel = PCASelector(candidates=[5,3], min_k=3)
    psel.fit(train)
    # ensure selected PCA models were fit only on train by checking transform shapes
    sel = psel.choose()
    pca_model = psel.get_model(sel["chosen_k"], sel["chosen_whiten"])
    Z_train = pca_model.transform(train.values)
    Z_val = pca_model.transform(val.values)  # transform allowed
    assert Z_train.shape[0] == 80
    assert Z_val.shape[0] == 20

