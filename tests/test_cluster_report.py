import os
import json
import numpy as np
import pandas as pd
from experiments.latent_space import PCASelector
from experiments.clustering_latent import LatentClusterer


def test_cluster_report_and_scatter_written(tmp_path):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(120, 12), columns=[f"f{i}" for i in range(12)])
    X_train = X.iloc[:100]

    out_dir = tmp_path / "cluster_run"
    out_dir = str(out_dir)

    # Fit PCA on train, get whitened latent for clustering
    psel = PCASelector(candidates=[10,8,6,4], min_k=3, random_state=0)
    psel.fit(X_train)
    sel = psel.choose()
    # use whiten=True model for clustering
    pca_b = psel.get_model(int(sel["chosen_k"]), True)
    Z_train = pca_b.transform(X_train.values)

    clusterer = LatentClusterer(k_candidates=(2,3,4,5), min_fraction=0.01, random_state=0)
    best = clusterer.fit_and_choose(pd.DataFrame(Z_train))
    # Save report and plot
    clusterer.save_report(out_dir, Z=pd.DataFrame(Z_train))

    report_path = os.path.join(out_dir, "cluster_report.json")
    plot_path = os.path.join(out_dir, "latent_cluster_scatter.png")

    assert os.path.exists(report_path), "cluster_report.json not written"
    assert os.path.exists(plot_path), "latent_cluster_scatter.png not written"

    with open(report_path) as f:
        rep = json.load(f)

    assert "chosen_k" in rep
    assert "reports" in rep
    # chosen_k should be in k_candidates or fallback
    assert int(rep["chosen_k"]) in [2,3,4,5]
    # check that reports contain silhouette or davies_bouldin keys for at least one k
    any_metrics = any((('silhouette' in v and v['silhouette'] is not None) or ('davies_bouldin' in v)) for v in rep['reports'].values())
    assert any_metrics

