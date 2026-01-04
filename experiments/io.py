# I/O utilities for experiment pipeline
# Config loading, result saving, run directory management

import os
import json
import hashlib
from datetime import datetime

import yaml
import pandas as pd


def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config


def config_hash(config):
    """Generate deterministic hash of config for run naming."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def dataset_hash(df):
    """Generate hash of dataset content for fingerprinting."""
    # Hash based on shape and sample of data
    content = f"{df.shape}_{df.columns.tolist()}_{df.head(10).to_json()}_{df.tail(10).to_json()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def create_run_dir(config, output_dir=None):
    """Create unique run directory for experiment outputs."""
    output_dir = output_dir or config['experiment'].get('output_dir', 'runs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = config_hash(config)
    run_name = f"{config['experiment']['name']}_{timestamp}_{cfg_hash}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_results(run_dir, config, cv_results, model, X, y, target_type):
    """Save all experiment artifacts to run directory."""
    import joblib
    import matplotlib.pyplot as plt

    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save metrics (include fold-level scores for CI/boxplots)
    results_json = {
        'experiment_name': config['experiment']['name'],
        'seed': config['experiment']['seed'],
        'model_type': config['model']['type'],
        'target_column': config['data']['target_column'],
        'target_type': target_type,
        'cv_results': {}
    }

    for metric, values in cv_results.items():
        if isinstance(values, dict) and 'mean' in values:
            results_json['cv_results'][metric] = {
                'mean': float(values['mean']),
                'std': float(values['std']),
                'all': [float(v) for v in values.get('all', [])]  # Include fold-level scores
            }
        else:
            results_json['cv_results'][metric] = values

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Train final model on all data and save
    model.fit(X, y)
    model_path = os.path.join(run_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save CV distribution plot
    if config.get('metrics', {}).get('save_plots', True):
        _save_cv_plot(run_dir, config, cv_results)

    print(f"Results saved to: {run_dir}")
    return run_dir


def _save_cv_plot(run_dir, config, cv_results):
    """Save CV score distribution plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metric_names = [k for k in cv_results.keys() if isinstance(cv_results[k], dict) and 'all' in cv_results[k]]

    for i, metric in enumerate(metric_names[:4]):
        ax = axes[i]
        scores = cv_results[metric]['all']
        ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(cv_results[metric]['mean'], color='red', linestyle='--',
                  label=f"Mean: {cv_results[metric]['mean']:.4f}")
        ax.set_title(f"{metric.upper()} Distribution")
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.suptitle(f"{config['model']['type']} - {config['data']['target_column']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'cv_distribution.png'), dpi=150)
    plt.close()


def save_data_profile(run_dir, df, X, y, dataset_path):
    """Save dataset fingerprint/profile for reproducibility tracking."""
    profile = {
        'dataset_path': str(dataset_path) if dataset_path else 'interactive_selection',
        'dataset_hash': dataset_hash(df),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'feature_count': len(X.columns),
        'features_used': list(X.columns),
        'target_column': y.name,
        'target_stats': {
            'mean': float(y.mean()) if y.dtype in ['float64', 'int64'] else None,
            'std': float(y.std()) if y.dtype in ['float64', 'int64'] else None,
            'min': float(y.min()) if y.dtype in ['float64', 'int64'] else None,
            'max': float(y.max()) if y.dtype in ['float64', 'int64'] else None,
            'unique_values': int(y.nunique()),
            'value_counts': y.value_counts().to_dict() if y.nunique() <= 10 else None
        },
        'missing_values': int(X.isnull().sum().sum()),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(run_dir, 'data_profile.json'), 'w') as f:
        json.dump(profile, f, indent=2)

    return profile

