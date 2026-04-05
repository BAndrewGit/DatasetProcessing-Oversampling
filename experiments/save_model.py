"""Model saving helpers: joblib for sklearn/scalers, torch for PyTorch state_dicts,
plus minimal metadata writer.
"""

import os
import json
import shutil
from datetime import datetime

import yaml

def save_sklearn_model(obj, path):
    """Save an sklearn-compatible object (pipeline, scaler, estimator) using joblib.
    Returns the path on success.
    """
    try:
        import joblib
    except Exception as e:
        raise

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path


def save_torch_model(model, path):
    """Save a PyTorch model. If model has state_dict, save that, else save model directly.
    Returns the path on success.
    """
    try:
        import torch
    except ImportError:
        raise

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)
    except Exception:
        # Fall back to saving the model object directly
        torch.save(model, path)
    return path


def write_model_metadata(run_dir, metadata: dict, filename: str = 'model_metadata.json'):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return path


def export_inference_bundle(
    bundle_dir: str,
    model_src_path: str,
    scaler_src_path: str,
    feature_columns: list,
    thresholds: dict,
    bank_mapping_rules: dict,
    metadata: dict,
) -> dict:
    """Export a frozen inference bundle for external applications."""
    os.makedirs(bundle_dir, exist_ok=True)

    model_dst = os.path.join(bundle_dir, 'model.pt')
    scaler_dst = os.path.join(bundle_dir, 'scaler.pkl')
    features_dst = os.path.join(bundle_dir, 'feature_columns.json')
    rules_dst = os.path.join(bundle_dir, 'bank_mapping_rules.yaml')
    thresholds_dst = os.path.join(bundle_dir, 'thresholds.json')
    metadata_dst = os.path.join(bundle_dir, 'model_metadata.json')

    shutil.copyfile(model_src_path, model_dst)
    shutil.copyfile(scaler_src_path, scaler_dst)

    with open(features_dst, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, indent=2)
    with open(rules_dst, 'w', encoding='utf-8') as f:
        yaml.safe_dump(bank_mapping_rules, f, sort_keys=False)
    with open(thresholds_dst, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)

    metadata_out = dict(metadata)
    metadata_out['exported_at'] = datetime.now().isoformat()
    with open(metadata_dst, 'w', encoding='utf-8') as f:
        json.dump(metadata_out, f, indent=2)

    return {
        'bundle_dir': bundle_dir,
        'model': model_dst,
        'scaler': scaler_dst,
        'feature_columns': features_dst,
        'bank_mapping_rules': rules_dst,
        'thresholds': thresholds_dst,
        'metadata': metadata_dst,
    }

