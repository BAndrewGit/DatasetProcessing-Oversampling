import json
import os

from runners.run_release_promote import promote_release_bundle


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_promote_release_bundle_creates_bundle_and_plots(tmp_path):
    decision_dir = tmp_path / "final_decision_20260101_000000"
    bundle_dir = decision_dir / "inference_bundle"
    bundle_dir.mkdir(parents=True)

    # Minimal bundle files.
    for name in [
        "model.pt",
        "scaler.pkl",
        "feature_columns.json",
        "bank_mapping_rules.yaml",
        "thresholds.json",
        "model_metadata.json",
    ]:
        (bundle_dir / name).write_text("x", encoding="utf-8")

    final_comparison = {
        "final_model_family": "multitask",
        "selection_report": {
            "spearman_ok": True,
            "mae_ok": True,
            "stability_ok": True,
            "savings_ok": False,
        },
        "metrics": {
            "risk_mae": {
                "multitask": {"mean": 0.18, "all": [0.17, 0.18, 0.19]},
                "hybrid": {"mean": 0.21, "all": [0.20, 0.22, 0.21]},
            },
            "risk_spearman": {
                "multitask": {"mean": 0.47, "all": [0.40, 0.50, 0.51]},
                "hybrid": {"mean": 0.24, "all": [0.20, 0.25, 0.27]},
            },
            "savings_macro_f1": {
                "multitask": {"mean": 0.93, "all": [0.90, 0.95, 0.94]},
                "hybrid": {"mean": 0.44, "all": [0.40, 0.45, 0.47]},
            },
        },
    }

    _write_json(decision_dir / "final_comparison.json", final_comparison)
    _write_json(decision_dir / "comparison_report.json", {"bundle": {"bundle_dir": str(bundle_dir)}})
    _write_json(decision_dir / "final_multitask_results.json", {"ok": True})
    _write_json(decision_dir / "final_hybrid_results.json", {"ok": True})

    releases_dir = tmp_path / "deployment" / "releases"
    manifest = promote_release_bundle(
        decision_dir=str(decision_dir),
        releases_dir=str(releases_dir),
        release_name="release_test",
        update_current=True,
    )

    release_dir = releases_dir / "release_test"
    assert release_dir.is_dir()
    assert (release_dir / "bundle" / "model.pt").is_file()
    assert (release_dir / "plots" / "metric_bars.png").is_file()
    assert (release_dir / "plots" / "fold_boxplots.png").is_file()
    assert (release_dir / "plots" / "selection_checks.png").is_file()
    assert (release_dir / "promotion_manifest.json").is_file()

    expected_current_dir = os.path.abspath(os.path.join("deployment", "current"))
    assert manifest.get("current_dir") == expected_current_dir


