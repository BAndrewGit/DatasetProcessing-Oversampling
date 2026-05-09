import json
import subprocess
import sys
from pathlib import Path

from experiments.model_contract import MODEL_FEATURE_COLUMNS, MODEL_SCALED_FEATURE_COLUMNS, MODEL_SCALER_MODE


REQUIRED_BUNDLE = [
    "model.pt",
    "scaler.pkl",
    "feature_columns.json",
    "bank_mapping_rules.yaml",
    "thresholds.json",
    "model_metadata.json",
]

REQUIRED_SNAPSHOT = [
    "final_multitask_results.json",
    "final_hybrid_results.json",
    "final_comparison.json",
    "final_multitask_grad_logs.json",
    "final_hybrid_grad_logs.json",
    "comparison_report.json",
]

REQUIRED_PLOTS = [
    "metric_bars.png",
    "fold_boxplots.png",
    "selection_checks.png",
]


def _touch(path: Path, payload: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
        return
    path.write_text(payload, encoding="utf-8")


def _build_release_tree(release_dir: Path) -> None:
    bundle_dir = release_dir / "bundle"
    snapshot_dir = release_dir / "decision_snapshot"
    plots_dir = release_dir / "plots"

    feature_columns = list(MODEL_FEATURE_COLUMNS)
    bank_rules_yaml = (
        "risk_score_high_threshold: 0.7\n"
        "alerts:\n"
        "  - metric: saving_probability\n"
        "    operator: '<'\n"
        "    value: 0.5\n"
        "    message: low_saving_probability\n"
    )

    for name in REQUIRED_BUNDLE:
        if name == "feature_columns.json":
            _touch(bundle_dir / name, payload=json.dumps(feature_columns))
        elif name == "thresholds.json":
            _touch(
                bundle_dir / name,
                payload=json.dumps({"saving_probability_threshold": 0.5, "top_k_factors": 5}),
            )
        elif name == "model_metadata.json":
            _touch(
                bundle_dir / name,
                payload=json.dumps(
                    {
                        "model_type": "multitask_net",
                        "multitask": True,
                        "model_config": {"input_dim": len(feature_columns)},
                        "input_dim": len(feature_columns),
                        "scaled_feature_columns": list(MODEL_SCALED_FEATURE_COLUMNS),
                        "scaler_mode": MODEL_SCALER_MODE,
                    }
                ),
            )
        elif name == "bank_mapping_rules.yaml":
            _touch(bundle_dir / name, payload=bank_rules_yaml)
        else:
            _touch(bundle_dir / name, payload=b"0" * 64)
    for name in REQUIRED_SNAPSHOT:
        _touch(snapshot_dir / name, payload="{}")
    for name in REQUIRED_PLOTS:
        _touch(plots_dir / name, payload=b"plot")

    manifest = {
        "release_dir": str(release_dir),
        "bundle_dir": str(bundle_dir),
        "plots_dir": str(plots_dir),
        "final_model_family": "multitask",
        "source_decision_dir": str(release_dir / "_source_decision"),
        "source_bundle_dir": str(release_dir / "_source_bundle"),
    }
    _touch(release_dir / "promotion_manifest.json", payload=json.dumps(manifest))


def test_verify_release_script_passes_for_complete_release(tmp_path):
    release_dir = tmp_path / "deployment" / "releases" / "release_ok"
    _build_release_tree(release_dir)

    script = Path(__file__).resolve().parents[1] / "runners" / "verify_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_verify_release_script_fails_when_bundle_file_missing(tmp_path):
    release_dir = tmp_path / "deployment" / "releases" / "release_broken"
    _build_release_tree(release_dir)
    (release_dir / "bundle" / "model.pt").unlink()

    script = Path(__file__).resolve().parents[1] / "runners" / "verify_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("model.pt" in msg for msg in payload["errors"])


def test_verify_release_script_fails_when_feature_contract_mismatches(tmp_path):
    release_dir = tmp_path / "deployment" / "releases" / "release_bad_contract"
    _build_release_tree(release_dir)

    _touch(
        release_dir / "bundle" / "model_metadata.json",
        payload=json.dumps(
            {
                "model_type": "multitask_net",
                "multitask": True,
                "model_config": {"input_dim": len(MODEL_FEATURE_COLUMNS)},
                "input_dim": len(MODEL_FEATURE_COLUMNS),
                "scaled_feature_columns": ["Age", "Income_Category", "Essential_Needs_Percentage"],
                "scaler_mode": MODEL_SCALER_MODE,
            }
        ),
    )

    script = Path(__file__).resolve().parents[1] / "runners" / "verify_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("Feature contract mismatch" in msg for msg in payload["errors"])

