"""Model saving helpers: joblib for sklearn/scalers, torch for PyTorch state_dicts,
plus minimal metadata writer.
"""

import os
import json

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
