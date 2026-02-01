# Baseline experiment runner
# Augmentation is HARD-DISABLED - use run_augmentation_experiment.py for augmentation

import argparse
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.config_schema import validate_baseline_config, ConfigValidationError
from experiments.io import load_config, save_results, create_run_dir, save_data_profile
from experiments.data import load_dataset, preprocess_data, validate_data_integrity, validate_save_money_consistency
from experiments.models import build_model
from experiments.cv import run_repeated_cv_regression, run_repeated_cv_classification


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def run_baseline(config_path, dataset_path=None, output_dir=None):
    """
    Run a clean baseline experiment (NO augmentation).

    Args:
        config_path: Path to YAML config file
        dataset_path: Optional path to dataset CSV (overrides config)
        output_dir: Optional output directory (overrides config)

    Returns:
        run_dir: Path to experiment output directory
    """
    # Load and validate config
    config = load_config(config_path)

    # Override output_dir if provided
    if output_dir:
        config['experiment']['output_dir'] = output_dir

    try:
        validate_baseline_config(config)
    except ConfigValidationError as e:
        print(f"\nCONFIG ERROR:\n{e}")
        raise

    seed = config['experiment']['seed']
    set_seeds(seed)

    target = config['data']['target_column']
    target_type = config['data'].get('target_type', 'regression')

    print("=" * 60)
    print("CLEAN BASELINE EXPERIMENT")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Target: {target} ({target_type})")
    print(f"Seed: {seed}")
    print("=" * 60)
    print("Augmentation: DISABLED (baseline mode)")
    print("=" * 60)

    # Load data
    df, actual_path = load_dataset(config, dataset_path)

    # Validate Save_Money consistency if both columns exist
    validate_save_money_consistency(df)

    # Preprocess
    X, y = preprocess_data(df, config)

    # Validate data integrity
    validate_data_integrity(X, y, config)

    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {len(X.columns)}")

    if target_type == 'regression':
        print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    else:
        print(f"Class distribution: {y.value_counts().to_dict()}")

    # Build model
    model = build_model(config)
    print(f"\nModel: {config['model']['type']}")

    # Run CV
    if target_type == 'regression':
        cv_results = run_repeated_cv_regression(model, X, y, config)
        print("\n" + "=" * 60)
        print("REGRESSION RESULTS (Repeated K-Fold CV)")
        print("=" * 60)
        print(f"MAE:      {cv_results['mae']['mean']:.4f} ± {cv_results['mae']['std']:.4f}")
        print(f"RMSE:     {cv_results['rmse']['mean']:.4f} ± {cv_results['rmse']['std']:.4f}")
        print(f"Spearman: {cv_results['spearman']['mean']:.4f} ± {cv_results['spearman']['std']:.4f}")
        print(f"R2:       {cv_results['r2']['mean']:.4f} ± {cv_results['r2']['std']:.4f}")
    else:
        cv_results = run_repeated_cv_classification(model, X, y, config)
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS (Repeated Stratified K-Fold CV)")
        print("=" * 60)
        print(f"Macro-F1:  {cv_results['macro_f1']['mean']:.4f} ± {cv_results['macro_f1']['std']:.4f}")
        print(f"Accuracy:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        print(f"Recall:    {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")

    # Create run directory
    run_dir = create_run_dir(config)

    # Save data profile (dataset fingerprint)
    save_data_profile(run_dir, df, X, y, actual_path)

    # Save results
    save_results(run_dir, config, cv_results, model, X, y, target_type)

    print("\n" + "=" * 60)
    print("Baseline experiment complete!")
    print("=" * 60)

    return run_dir


def run_all_baselines(dataset_path=None):
    """Run all baseline models for both regression and classification."""
    import yaml
    import json
    import os

    print("\n" + "=" * 70)
    print("RUNNING ALL BASELINE MODELS")
    print("=" * 70)

    regression_models = ['ridge', 'lasso', 'xgboost_reg']
    classification_models = ['logistic_regression', 'random_forest']

    results_summary = {'regression': {}, 'classification': {}}

    # Run regression baselines
    print("\n>>> REGRESSION BASELINES (Target: Risk_Score)")
    for model_type in regression_models:
        try:
            config = load_config('configs/baseline_regression.yaml')
            config['model']['type'] = model_type
            config['experiment']['name'] = f"baseline_reg_{model_type}"

            # Force augmentation off
            config['augmentation'] = {'enabled': False}

            # Save temp config
            temp_config = f'configs/_temp_{model_type}.yaml'
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            run_dir = run_baseline(temp_config, dataset_path)

            # Load results
            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)

            results_summary['regression'][model_type] = metrics['cv_results']

            os.remove(temp_config)
        except Exception as e:
            print(f"Error running {model_type}: {e}")

    # Run classification baselines
    print("\n>>> CLASSIFICATION BASELINES (Target: Save_Money_Yes)")
    for model_type in classification_models:
        try:
            config = load_config('configs/baseline_classification.yaml')
            config['model']['type'] = model_type
            config['experiment']['name'] = f"baseline_clf_{model_type}"

            config['augmentation'] = {'enabled': False}

            temp_config = f'configs/_temp_{model_type}.yaml'
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            run_dir = run_baseline(temp_config, dataset_path)

            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)

            results_summary['classification'][model_type] = metrics['cv_results']

            os.remove(temp_config)
        except Exception as e:
            print(f"Error running {model_type}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print("\nREGRESSION (Risk_Score):")
    print("-" * 50)
    for model, metrics in results_summary['regression'].items():
        mae = metrics.get('mae', {}).get('mean', 'N/A')
        rmse = metrics.get('rmse', {}).get('mean', 'N/A')
        spearman = metrics.get('spearman', {}).get('mean', 'N/A')
        print(f"{model:20s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Spearman: {spearman:.4f}")

    print("\nCLASSIFICATION (Save_Money_Yes):")
    print("-" * 50)
    for model, metrics in results_summary['classification'].items():
        f1 = metrics.get('macro_f1', {}).get('mean', 'N/A')
        acc = metrics.get('accuracy', {}).get('mean', 'N/A')
        print(f"{model:20s} | Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run clean baseline experiment (NO augmentation)',
        epilog='For augmentation experiments, use: python run_augmentation_experiment.py'
    )
    parser.add_argument('--config', '-c', type=str, default='configs/baseline_regression.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to dataset CSV (overrides config)')
    parser.add_argument('--all-baselines', action='store_true',
                       help='Run all baseline models (regression + classification)')
    args = parser.parse_args()

    if args.all_baselines:
        run_all_baselines(args.dataset)
    else:
        run_baseline(args.config, args.dataset)


if __name__ == "__main__":
    main()

