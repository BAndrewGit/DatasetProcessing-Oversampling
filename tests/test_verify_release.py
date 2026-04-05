import json
import subprocess
import sys
from pathlib import Path


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
    path.write_text(payload, encoding="utf-8")


def _build_release_tree(release_dir: Path) -> None:
    bundle_dir = release_dir / "bundle"
    snapshot_dir = release_dir / "decision_snapshot"
    plots_dir = release_dir / "plots"

    for name in REQUIRED_BUNDLE:
        _touch(bundle_dir / name)
    for name in REQUIRED_SNAPSHOT:
        _touch(snapshot_dir / name, payload="{}")
    for name in REQUIRED_PLOTS:
        _touch(plots_dir / name)

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

