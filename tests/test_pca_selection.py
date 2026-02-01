import json
import os
import numpy as np
import pandas as pd
from experiments.latent_space import PCASelector


def test_pca_selection_writes_json_and_plot(tmp_path):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    X_train = X.iloc[:80]
    X_val = X.iloc[80:]

    out_dir = tmp_path / "pca_run"
    out_dir = str(out_dir)

    psel = PCASelector(candidates=[8,5,3], min_k=3, random_state=0)
    psel.fit(X_train)
    # save outputs
    psel.save_selection(out_dir)

    sel_path = os.path.join(out_dir, "pca_selection.json")
    plot_path = os.path.join(out_dir, "pca_evr.png")

    assert os.path.exists(sel_path), "pca_selection.json not written"
    assert os.path.exists(plot_path), "pca_evr.png not written"

    with open(sel_path) as f:
        sel = json.load(f)

    assert "chosen_k" in sel
    assert "chosen_whiten" in sel
    assert "candidates" in sel
    # chosen_k must be one of tried candidates
    tried = [int(k.split("_")[0]) for k in sel["candidates"].keys()]
    assert int(sel["chosen_k"]) in tried

    # Ensure candidate entries have evr and recon_error
    for v in sel["candidates"].values():
        assert "evr" in v
        assert "recon_error" in v

