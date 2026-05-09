import json

from runners.run_hybrid_production import export_inference_bundle
from experiments.model_contract import MODEL_FEATURE_COLUMNS, MODEL_SCALED_FEATURE_COLUMNS, MODEL_SCALER_MODE


def test_export_inference_bundle_creates_expected_files(tmp_path):
    model_src = tmp_path / "winner_model.pth"
    scaler_src = tmp_path / "winner_scaler.joblib"
    model_src.write_bytes(b"model")
    scaler_src.write_bytes(b"scaler")

    winner_checkpoint = {
        "checkpoint_path": str(model_src),
        "scaler_path": str(scaler_src),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "thresholds": {"saving_probability_threshold": 0.5},
        "bank_mapping_rules": {"thresholds": {"discretionary_spending_high": 3.0}},
        "metadata": {
            "family": "multitask",
            "run_id": "pipeline_20260101_010101",
            "features_used": list(MODEL_FEATURE_COLUMNS),
            "metrics_summary": {"multitask": {"risk_mae": 0.18}},
            "source_pipeline_run": "runs/pipeline_20260101_010101",
        },
    }

    bundle_dir = tmp_path / "inference_bundle"
    paths = export_inference_bundle(winner_checkpoint, str(bundle_dir))

    assert (bundle_dir / "model.pt").is_file()
    assert (bundle_dir / "scaler.pkl").is_file()
    assert (bundle_dir / "feature_columns.json").is_file()
    assert (bundle_dir / "bank_mapping_rules.yaml").is_file()
    assert (bundle_dir / "thresholds.json").is_file()
    assert (bundle_dir / "model_metadata.json").is_file()

    with open(paths["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["family"] == "multitask"
    assert metadata["run_id"] == "pipeline_20260101_010101"
    assert metadata["input_dim"] == len(MODEL_FEATURE_COLUMNS)
    assert metadata["scaled_feature_columns"] == list(MODEL_SCALED_FEATURE_COLUMNS)
    assert metadata["scaler_mode"] == MODEL_SCALER_MODE
    with open(paths["feature_columns"], "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    assert feature_columns == list(MODEL_FEATURE_COLUMNS)
    assert "exported_at" in metadata

