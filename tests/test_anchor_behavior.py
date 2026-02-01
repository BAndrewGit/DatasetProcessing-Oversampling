import os
import json
import numpy as np
import pandas as pd
from experiments.latent_experiment import run_latent_fold


def test_anchor_behavior_artifact_written(tmp_path):
    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame({
        f"f{i}": rng.randn(n) for i in range(8)
    })
    # synthetic target (regression)
    y = pd.Series(rng.normal(size=n), name="Risk_Score")

    # split into train/val
    train_idx = np.arange(0, 30)
    val_idx = np.arange(30, 40)
    X_train = df.iloc[train_idx].reset_index(drop=True)
    X_val = df.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    out_dir = str(tmp_path / "fold_test")

    cfg = {
        "pca_candidates": [6,4,3],
        "k_candidates": [2,3],
        "min_cluster_frac": 0.01,
        "per_cluster_cap_frac": 0.2,
        "global_cap_frac": 0.3,
        "sampling_method": "gaussian",
        "memorization_min_threshold": 1e-6,
        "onehot_groups": None,
        "anchor_synth_counts": [0, 5, 10]
    }

    res = run_latent_fold(X_train, y_train, X_val, y_val, cfg, out_dir, task="regression")

    anchor_json = os.path.join(out_dir, "anchor_behavior.json")
    anchor_plot = os.path.join(out_dir, "anchor_predictions_vs_synth_count.png")

    assert os.path.exists(anchor_json), "anchor_behavior.json not created"
    assert os.path.exists(anchor_plot), "anchor_predictions_vs_synth_count.png not created"

    with open(anchor_json) as f:
        data = json.load(f)

    assert "anchors" in data
    assert "predictions" in data
    # predictions keys are stringified in JSON
    keys = set(int(k) for k in data["predictions"].keys())
    assert 0 in keys
    # at least baseline plus one synth count should be present
    assert any(k > 0 for k in keys)

