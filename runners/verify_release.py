# Validate promoted release artifacts for deployment readiness.

import argparse
import json
import os
import sys
from datetime import datetime

import yaml

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNNERS_DIR)
sys.path.insert(0, RUNNERS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from experiments.model_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_DIM,
    MODEL_SCALER_MODE,
    MODEL_SCALED_FEATURE_COLUMNS,
)


REQUIRED_BUNDLE_FILES = [
    "model.pt",
    "scaler.pkl",
    "feature_columns.json",
    "bank_mapping_rules.yaml",
    "thresholds.json",
    "model_metadata.json",
]

REQUIRED_SNAPSHOT_FILES = [
    "final_multitask_results.json",
    "final_hybrid_results.json",
    "final_comparison.json",
    "final_multitask_grad_logs.json",
    "final_hybrid_grad_logs.json",
    "comparison_report.json",
]

REQUIRED_PLOT_FILES = [
    "metric_bars.png",
    "fold_boxplots.png",
    "selection_checks.png",
]


def _latest_release(releases_dir: str) -> str:
    if not os.path.isdir(releases_dir):
        raise FileNotFoundError(f"Releases directory not found: {releases_dir}")

    candidates = []
    for name in os.listdir(releases_dir):
        path = os.path.join(releases_dir, name)
        if os.path.isdir(path):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No release directories found in: {releases_dir}")

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _file_size(path: str) -> int:
    return os.path.getsize(path)


def _normalize(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _validate_feature_contract(report: dict, feature_columns: list, model_metadata: dict) -> None:
    if not feature_columns:
        report["ok"] = False
        report["errors"].append("feature_columns.json is empty")
        return

    if any(not isinstance(column, str) or not column.strip() for column in feature_columns):
        report["ok"] = False
        report["errors"].append("feature_columns.json contains invalid feature names")
        return

    if len(set(feature_columns)) != len(feature_columns):
        report["ok"] = False
        report["errors"].append("feature_columns.json contains duplicate feature names")
        return

    expected_feature_columns = list(MODEL_FEATURE_COLUMNS)
    if feature_columns != expected_feature_columns:
        report["ok"] = False
        report["errors"].append(
            "Feature contract mismatch: feature_columns.json does not match the shared canonical contract"
        )

    model_config = model_metadata.get("model_config")
    input_dim = None
    if isinstance(model_config, dict) and isinstance(model_config.get("input_dim"), int):
        input_dim = int(model_config["input_dim"])
    elif isinstance(model_metadata.get("input_dim"), int):
        input_dim = int(model_metadata["input_dim"])
    else:
        report["ok"] = False
        report["errors"].append("model_metadata.json is missing input_dim")

    if input_dim is not None and input_dim != MODEL_INPUT_DIM:
        report["ok"] = False
        report["errors"].append(
            "Feature contract mismatch: model_metadata input_dim does not match the shared canonical contract"
        )

    if input_dim is not None and len(feature_columns) != input_dim:
        report["ok"] = False
        report["errors"].append(
            "Feature contract mismatch: "
            f"feature_columns.json has {len(feature_columns)} entries, but model_metadata "
            f"declares input_dim={input_dim}"
        )

    scaled_feature_columns = model_metadata.get("scaled_feature_columns")
    if scaled_feature_columns is None:
        report["ok"] = False
        report["errors"].append("model_metadata.json is missing scaled_feature_columns")
        return

    if not isinstance(scaled_feature_columns, list) or any(
        not isinstance(column, str) or not column.strip() for column in scaled_feature_columns
    ):
        report["ok"] = False
        report["errors"].append("model_metadata.json has invalid scaled_feature_columns")
        return

    expected_scaled_feature_columns = list(MODEL_SCALED_FEATURE_COLUMNS)
    if scaled_feature_columns != expected_scaled_feature_columns:
        report["ok"] = False
        report["errors"].append(
            "Feature contract mismatch: model_metadata scaled_feature_columns do not match the shared canonical contract"
        )

    if model_metadata.get("scaler_mode") != MODEL_SCALER_MODE:
        report["ok"] = False
        report["errors"].append(
            "Feature contract mismatch: model_metadata scaler_mode does not match the shared canonical contract"
        )


def verify_release_artifacts(
    release_dir: str = None,
    releases_dir: str = "deployment/releases",
    strict_sources: bool = False,
) -> dict:
    release_dir = release_dir or _latest_release(releases_dir)
    release_dir = os.path.abspath(release_dir)

    manifest_path = os.path.join(release_dir, "promotion_manifest.json")
    bundle_dir = os.path.join(release_dir, "bundle")
    snapshot_dir = os.path.join(release_dir, "decision_snapshot")
    plots_dir = os.path.join(release_dir, "plots")

    report = {
        "checked_at": datetime.now().isoformat(),
        "release_dir": release_dir,
        "ok": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    def fail(message: str) -> None:
        report["ok"] = False
        report["errors"].append(message)

    # Check top-level structure.
    for folder_key, folder_path in {
        "bundle_dir": bundle_dir,
        "decision_snapshot_dir": snapshot_dir,
        "plots_dir": plots_dir,
    }.items():
        exists = os.path.isdir(folder_path)
        report["checks"][folder_key] = exists
        if not exists:
            fail(f"Missing required directory: {folder_path}")

    if not os.path.isfile(manifest_path):
        fail(f"Missing promotion manifest: {manifest_path}")
        manifest = {}
    else:
        report["checks"]["promotion_manifest"] = True
        manifest = _load_json(manifest_path)

    # Validate required artifact files.
    for name in REQUIRED_BUNDLE_FILES:
        path = os.path.join(bundle_dir, name)
        exists = os.path.isfile(path)
        report["checks"][f"bundle:{name}"] = exists
        if not exists:
            fail(f"Missing bundle file: {path}")
            continue
        if _file_size(path) <= 32:
            fail(f"Bundle file too small to be valid: {path}")

    for name in REQUIRED_SNAPSHOT_FILES:
        path = os.path.join(snapshot_dir, name)
        exists = os.path.isfile(path)
        report["checks"][f"snapshot:{name}"] = exists
        if not exists:
            fail(f"Missing decision snapshot file: {path}")
            continue
        if _file_size(path) <= 0:
            fail(f"Decision snapshot file is empty: {path}")
            continue
        try:
            _load_json(path)
        except Exception as exc:
            fail(f"Invalid JSON in decision snapshot file {path}: {exc}")

    for name in REQUIRED_PLOT_FILES:
        path = os.path.join(plots_dir, name)
        exists = os.path.isfile(path)
        report["checks"][f"plot:{name}"] = exists
        if not exists:
            fail(f"Missing plot file: {path}")
            continue
        if _file_size(path) <= 0:
            fail(f"Plot file is empty: {path}")

    # Validate manifest consistency.
    if manifest:
        expected_release = _normalize(release_dir)
        manifest_release = manifest.get("release_dir")
        if manifest_release:
            same_release = _normalize(manifest_release) == expected_release
            report["checks"]["manifest_release_dir_match"] = same_release
            if not same_release:
                fail(
                    "Manifest release_dir mismatch: "
                    f"expected {release_dir}, found {manifest_release}"
                )
        else:
            fail("Manifest missing required field: release_dir")

        manifest_bundle = manifest.get("bundle_dir")
        if manifest_bundle:
            same_bundle = _normalize(manifest_bundle) == _normalize(bundle_dir)
            report["checks"]["manifest_bundle_dir_match"] = same_bundle
            if not same_bundle:
                fail(
                    "Manifest bundle_dir mismatch: "
                    f"expected {bundle_dir}, found {manifest_bundle}"
                )
        else:
            fail("Manifest missing required field: bundle_dir")

        manifest_plots = manifest.get("plots_dir")
        if manifest_plots:
            same_plots = _normalize(manifest_plots) == _normalize(plots_dir)
            report["checks"]["manifest_plots_dir_match"] = same_plots
            if not same_plots:
                fail(
                    "Manifest plots_dir mismatch: "
                    f"expected {plots_dir}, found {manifest_plots}"
                )
        else:
            fail("Manifest missing required field: plots_dir")

        final_model_family = manifest.get("final_model_family")
        family_ok = final_model_family in {"multitask", "hybrid_transfer"}
        report["checks"]["manifest_final_model_family"] = family_ok
        if not family_ok:
            fail(
                "Manifest final_model_family must be one of: "
                f"multitask, hybrid_transfer. Found: {final_model_family}"
            )

        for source_field in ["source_decision_dir", "source_bundle_dir"]:
            source_path = manifest.get(source_field)
            if source_path:
                exists = os.path.exists(source_path)
                report["checks"][f"manifest_{source_field}_exists"] = exists
                if strict_sources and not exists:
                    fail(f"Manifest {source_field} does not exist: {source_path}")
                elif not exists:
                    report["warnings"].append(
                        f"Manifest {source_field} not found on this machine: {source_path}"
                    )
            else:
                if strict_sources:
                    fail(f"Manifest missing required field in strict mode: {source_field}")
                else:
                    report["warnings"].append(
                        f"Manifest does not include optional field: {source_field}"
                    )

    # Validate bundle content consistency after the file existence checks.
    if report["ok"]:
        try:
            feature_columns = _load_json(os.path.join(bundle_dir, "feature_columns.json"))
            thresholds = _load_json(os.path.join(bundle_dir, "thresholds.json"))
            model_metadata = _load_json(os.path.join(bundle_dir, "model_metadata.json"))
            bank_mapping_rules = _load_yaml(os.path.join(bundle_dir, "bank_mapping_rules.yaml"))
        except Exception as exc:
            fail(f"Failed to parse bundle artifacts: {exc}")
        else:
            if not isinstance(feature_columns, list):
                fail("feature_columns.json must contain a JSON array")
            if not isinstance(thresholds, dict):
                fail("thresholds.json must contain a JSON object")
            if not isinstance(model_metadata, dict):
                fail("model_metadata.json must contain a JSON object")
            if not isinstance(bank_mapping_rules, dict):
                fail("bank_mapping_rules.yaml must contain a YAML object")
            if isinstance(feature_columns, list) and isinstance(model_metadata, dict):
                _validate_feature_contract(report, feature_columns, model_metadata)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify promoted release artifacts")
    parser.add_argument("--release-dir", default=None, help="Path to deployment release directory")
    parser.add_argument(
        "--releases-dir",
        default="deployment/releases",
        help="Releases root used when --release-dir is omitted",
    )
    parser.add_argument(
        "--strict-sources",
        action="store_true",
        help="Fail if source paths from manifest are missing",
    )
    args = parser.parse_args()

    report = verify_release_artifacts(
        release_dir=args.release_dir,
        releases_dir=args.releases_dir,
        strict_sources=args.strict_sources,
    )

    print(json.dumps(report, indent=2))
    sys.exit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()

