# Analysis Runner
# Orchestrates all interpretability, stability, and paper artifact generation
# Produces publication-ready results from trained models

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from experiments.io import load_config
from experiments.config_schema import FORBIDDEN_TARGETS

from analysis.interpretability import (
    compute_permutation_importance,
    compute_permutation_importance_torch,
    plot_feature_importance,
    identify_actionable_features
)
from analysis.stability import (
    analyze_cv_stability,
    compare_model_stability,
    plot_stability_analysis,
    plot_fold_trajectories,
    compute_stability_summary
)
from analysis.error_analysis import (
    analyze_errors_by_segment,
    identify_failure_cases,
    plot_error_distribution
)
from analysis.what_if import (
    compute_partial_dependence,
    sensitivity_analysis,
    verify_economic_plausibility,
    plot_what_if_curves,
    plot_sensitivity_tornado,
    ECONOMIC_EXPECTATIONS
)
from analysis.paper_artifacts import (
    generate_ablation_table,
    create_methodology_diagram,
    create_results_summary,
    package_latent_sprint_artifacts
)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_experiment_results(run_dir: str) -> Dict:
    """
    Load results from an experiment run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Dict with experiment results
    """
    results = {}

    # Load metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)

    # record run dir for downstream packaging
    results['_run_dir'] = run_dir

    # Load config
    config_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(config_path):
        results['config'] = load_config(config_path)

    # Try config.json too
    config_json = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_json):
        with open(config_json, 'r') as f:
            results['config'] = json.load(f)

    # Load model
    model_path = os.path.join(run_dir, 'model.joblib')
    if os.path.exists(model_path):
        results['model'] = joblib.load(model_path)

    # Also support PyTorch model files (model.pth / model.pt)
    if 'model' not in results:
        for torch_name in ['model.pth', 'model.pt']:
            tpath = os.path.join(run_dir, torch_name)
            if os.path.exists(tpath):
                try:
                    import torch
                    results['model'] = torch.load(tpath)
                    results['_torch_model_path'] = tpath
                    break
                except Exception:
                    # fallback: skip torch model if loading fails
                    pass

    # Load data profile
    profile_path = os.path.join(run_dir, 'data_profile.json')
    if os.path.exists(profile_path):
        with open(profile_path, 'r') as f:
            results['data_profile'] = json.load(f)

    # Load ablation results (for multi-task/domain transfer)
    ablation_path = os.path.join(run_dir, 'ablation_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path, 'r') as f:
            results['ablation'] = json.load(f)

    # Load domain transfer results
    transfer_path = os.path.join(run_dir, 'domain_transfer_results.json')
    if os.path.exists(transfer_path):
        with open(transfer_path, 'r') as f:
            results['transfer'] = json.load(f)

    # Sprint 7 artifacts (latent sampling)
    # Search run_dir and nested fold_* directories for latent artifacts
    # NEW structure: fold_*/pca/, fold_*/clustering/, fold_*/synthetic_audit/
    latent_summary = {
        'pca_selection': None,
        'performance_csv': None,
        'cluster_scatter': None,
        'experiment_report': None,
        'aggregate_plots': [],
        'folds': []
    }

    # Check root-level artifacts
    root_pca = os.path.join(run_dir, 'pca_selection.json')
    root_perf = os.path.join(run_dir, 'performance_vs_synth_count.csv')
    root_scatter = os.path.join(run_dir, 'latent_cluster_scatter.png')
    root_exp_report = os.path.join(run_dir, 'experiment_report.json')
    all_folds_perf = os.path.join(run_dir, 'all_folds_performance.csv')

    if os.path.exists(root_pca):
        latent_summary['pca_selection'] = root_pca
    if os.path.exists(root_perf):
        latent_summary['performance_csv'] = root_perf
    if os.path.exists(root_scatter):
        latent_summary['cluster_scatter'] = root_scatter
    if os.path.exists(root_exp_report):
        latent_summary['experiment_report'] = root_exp_report
    if os.path.exists(all_folds_perf):
        latent_summary['all_folds_performance'] = all_folds_perf

    # Check aggregate_plots directory
    agg_plots_dir = os.path.join(run_dir, 'aggregate_plots')
    if os.path.exists(agg_plots_dir):
        for f in os.listdir(agg_plots_dir):
            if f.endswith('.png'):
                latent_summary['aggregate_plots'].append(os.path.join(agg_plots_dir, f))

    # Walk subdirectories (fold_*) and collect per-fold artifacts
    for root, dirs, files in os.walk(run_dir):
        # limit to a depth (fold folders typically directly under run_dir)
        rel = os.path.relpath(root, run_dir)
        parts = rel.split(os.sep)
        if parts[0].startswith('fold') or 'fold_' in root.lower() or (len(parts) > 1 and parts[1].startswith('fold')):
            # Old structure (direct files in fold_*)
            f_pca = os.path.join(root, 'pca_selection.json')
            f_perf = os.path.join(root, 'performance_vs_synth_count.csv')
            f_scatter = os.path.join(root, 'latent_cluster_scatter.png')

            # New structure (subfolders)
            f_pca_new = os.path.join(root, 'pca', 'evr_per_component.png')
            f_clustering_new = os.path.join(root, 'clustering', 'latent_scatter.png')
            f_synth_new = os.path.join(root, 'synthetic_audit', 'memorization_knn_histogram.png')

            entry = {'fold_dir': root}

            # Collect old structure files
            if os.path.exists(f_pca):
                entry['pca_selection'] = f_pca
                if latent_summary['pca_selection'] is None:
                    latent_summary['pca_selection'] = f_pca
            if os.path.exists(f_perf):
                entry['performance_csv'] = f_perf
                if latent_summary['performance_csv'] is None:
                    latent_summary['performance_csv'] = f_perf
            if os.path.exists(f_scatter):
                entry['cluster_scatter'] = f_scatter
                if latent_summary['cluster_scatter'] is None:
                    latent_summary['cluster_scatter'] = f_scatter

            # Collect new structure subfolders
            pca_dir = os.path.join(root, 'pca')
            clustering_dir = os.path.join(root, 'clustering')
            synth_dir = os.path.join(root, 'synthetic_audit')

            if os.path.exists(pca_dir):
                entry['pca_plots'] = [os.path.join(pca_dir, f) for f in os.listdir(pca_dir) if f.endswith('.png')]
            if os.path.exists(clustering_dir):
                entry['clustering_plots'] = [os.path.join(clustering_dir, f) for f in os.listdir(clustering_dir) if f.endswith('.png')]
                # Use new scatter if old not found
                new_scatter = os.path.join(clustering_dir, 'latent_scatter.png')
                if os.path.exists(new_scatter) and 'cluster_scatter' not in entry:
                    entry['cluster_scatter'] = new_scatter
                    if latent_summary['cluster_scatter'] is None:
                        latent_summary['cluster_scatter'] = new_scatter
            if os.path.exists(synth_dir):
                entry['synthetic_audit_plots'] = [os.path.join(synth_dir, f) for f in os.listdir(synth_dir) if f.endswith('.png')]

            # Check for plots_summary.json
            plots_summary = os.path.join(root, 'plots_summary.json')
            if os.path.exists(plots_summary):
                entry['plots_summary'] = plots_summary

            if len(entry) > 1:  # More than just fold_dir
                latent_summary['folds'].append(entry)

    # If any latent artifacts were found, attach to results
    if latent_summary['pca_selection'] or latent_summary['performance_csv'] or latent_summary['cluster_scatter'] or latent_summary['folds'] or latent_summary['aggregate_plots']:
        results['latent_sprint'] = latent_summary

    return results


def run_interpretability_analysis(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_names: List[str],
    task_type: str,
    output_dir: str,
    is_torch: bool = False,
    X_np: np.ndarray = None
) -> Dict:
    """
    Run interpretability analysis on a model.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        feature_names: Feature names
        task_type: 'regression' or 'classification'
        output_dir: Output directory
        is_torch: Whether model is PyTorch

    Returns:
        Dict with interpretability results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n=== INTERPRETABILITY ANALYSIS ===")

    # Permutation importance
    print("Computing permutation importance...")

    # Use numpy array for internal numeric ops but pass a DataFrame with feature names to sklearn models
    X_arr = X_np if X_np is not None else (X.values if isinstance(X, pd.DataFrame) else np.array(X))
    X_for_model = pd.DataFrame(X_arr, columns=feature_names)

    if is_torch:
        importance_df = compute_permutation_importance_torch(
            model, X_arr, y, feature_names, task=task_type
        )
    else:
        scoring = 'neg_mean_absolute_error' if task_type == 'regression' else 'f1_macro'
        # Pass a DataFrame with column names so sklearn estimators expecting feature_names_in_ won't warn
        importance_df = compute_permutation_importance(
            model, X_for_model, y, feature_names, scoring=scoring
        )

    results['permutation_importance'] = importance_df.to_dict('records')

    # Save importance plot
    fig = plot_feature_importance(
        importance_df,
        top_n=20,
        title=f"Feature Importance ({task_type.title()})",
        save_path=os.path.join(output_dir, 'feature_importance.pdf')
    )
    plt.close(fig)

    # Identify actionable features
    actionability_df = identify_actionable_features(importance_df)
    results['actionability'] = actionability_df.to_dict('records')

    # Save actionability summary
    actionable_count = (actionability_df['actionability'] == 'actionable').sum()
    non_actionable_count = (actionability_df['actionability'] == 'non-actionable').sum()

    results['actionability_summary'] = {
        'actionable': actionable_count,
        'non_actionable': non_actionable_count,
        'top_actionable': actionability_df[
            actionability_df['actionability'] == 'actionable'
        ].head(10)['feature'].tolist()
    }

    print(f"  Found {actionable_count} actionable features")
    print(f"  Found {non_actionable_count} non-actionable features")

    # Save raw importance data
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    return results

def package_latent_runs_into_analysis(experiment_results: Dict[str, Dict], analysis_out_dir: str, task: str = 'regression') -> None:
    """
    For any experiment with latent_sprint present, call packaging helper to move artifacts into analysis output.
    """
    from analysis.paper_artifacts import package_latent_sprint_artifacts
    for exp_name, res in experiment_results.items():
        run_dir = res.get('_run_dir') or res.get('config', {}).get('_run_dir')
        if not run_dir:
            continue
        latent = res.get('latent_sprint')
        if not latent:
            continue
        out_dir = os.path.join(analysis_out_dir, exp_name + '_latent')
        try:
            package_latent_sprint_artifacts(res, run_dir, out_dir, task=task)
            print(f"Packaged latent sprint artifacts for {exp_name} -> {out_dir}")
        except Exception as e:
            print(f"Failed to package latent artifacts for {exp_name}: {e}")

def run_stability_analysis(
    experiment_results: Dict[str, Dict],
    output_dir: str
) -> Dict:
    """
    Run stability analysis across experiments.

    Args:
        experiment_results: Dict mapping experiment name to results
        output_dir: Output directory

    Returns:
        Dict with stability results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n=== STABILITY ANALYSIS ===")

    # Extract CV results for comparison
    model_cv_results = {}

    for exp_name, exp_results in experiment_results.items():
        cv_data = exp_results.get('metrics', {}).get('cv_results', {})
        if cv_data:
            model_cv_results[exp_name] = cv_data

    if not model_cv_results:
        print("  No CV results found for stability analysis")
        return results

    # Analyze stability per experiment
    for exp_name, cv_results in model_cv_results.items():
        stability_df = analyze_cv_stability(cv_results)
        results[f'{exp_name}_stability'] = stability_df.to_dict('records')

        # Check stability criterion (std < 20% of mean)
        # Be robust to empty DataFrame or missing 'stable' column
        if stability_df is None or stability_df.empty:
            stable_metrics = []
            unstable_metrics = []
        else:
            if 'stable' in stability_df.columns:
                stable_metrics = stability_df[stability_df['stable']]['metric'].tolist()
                unstable_metrics = stability_df[~stability_df['stable']]['metric'].tolist()
            else:
                # If 'stable' not present, treat all as stable for reporting
                stable_metrics = stability_df['metric'].tolist()
                unstable_metrics = []

        print(f"  {exp_name}: {len(stable_metrics)} stable, {len(unstable_metrics)} unstable metrics")

    # Compare models
    if len(model_cv_results) >= 2:
        comparison_df = compare_model_stability(model_cv_results)
        results['model_comparison'] = comparison_df.to_dict('records')

        comparison_df.to_csv(os.path.join(output_dir, 'model_stability_comparison.csv'), index=False)

    # Compute overall stability summary
    stability_summary = compute_stability_summary(model_cv_results)
    results['summary'] = stability_summary

    # Create stability plots
    # Find a common metric to visualize
    common_metrics = ['mae', 'risk_mae', 'rmse', 'risk_rmse', 'macro_f1', 'savings_macro_f1']

    for metric in common_metrics:
        has_metric = all(metric in cv for cv in model_cv_results.values())
        if has_metric:
            fig = plot_stability_analysis(
                model_cv_results,
                metric=metric,
                title=f"Stability Analysis: {metric}",
                save_path=os.path.join(output_dir, f'stability_{metric}.pdf')
            )
            plt.close(fig)

            fig = plot_fold_trajectories(
                model_cv_results,
                metric=metric,
                title=f"CV Fold Trajectories: {metric}",
                save_path=os.path.join(output_dir, f'trajectories_{metric}.pdf')
            )
            plt.close(fig)
            break

    print(f"  Best risk model: {stability_summary.get('best_risk_model', 'N/A')}")
    print(f"  Most stable model: {stability_summary.get('most_stable_overall', 'N/A')}")

    return results


def run_error_analysis(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_names: List[str],
    task_type: str,
    output_dir: str,
    cluster_labels: np.ndarray = None,
    is_torch: bool = False,
    X_np: np.ndarray = None
) -> Dict:
    """
    Run error analysis.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        feature_names: Feature names
        task_type: 'regression' or 'classification'
        output_dir: Output directory
        cluster_labels: Optional cluster assignments
        is_torch: Whether model is PyTorch

    Returns:
        Dict with error analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n=== ERROR ANALYSIS ===")

    # Get predictions
    if is_torch:
        import torch
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            if hasattr(model, 'forward_adv'):
                output = model.forward(X_tensor, domain=0)
                y_pred = output['risk'].cpu().numpy()
            elif hasattr(model, 'risk_head'):
                risk_out, _ = model(X_tensor)
                y_pred = risk_out.cpu().numpy()
            else:
                y_pred = model(X_tensor).squeeze().cpu().numpy()
    else:
        # Ensure model.predict receives a DataFrame with proper column names to avoid sklearn UserWarning
        X_pred_input = X_np if X_np is not None else (X.values if isinstance(X, pd.DataFrame) else np.array(X))
        X_pred_df = pd.DataFrame(X_pred_input, columns=feature_names)
        y_pred = model.predict(X_pred_df)

    # Basic error statistics
    # Ensure abs_errors is defined for later checks
    abs_errors = np.array([])
    if task_type == 'regression':
        errors = y - y_pred
        abs_errors = np.abs(errors)

        results['error_stats'] = {
            'mae': float(np.mean(abs_errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mean_bias': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'max_error': float(np.max(abs_errors))
        }
    else:
        y_pred_class = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else y_pred.astype(int)
        results['error_stats'] = {
            'accuracy': float(np.mean(y == y_pred_class)),
            'error_rate': float(np.mean(y != y_pred_class)),
            'n_errors': int(np.sum(y != y_pred_class))
        }

    # Error by segment
    if cluster_labels is not None:
        segment_analysis = analyze_errors_by_segment(
            y, y_pred, cluster_labels, task_type
        )
        results['segment_analysis'] = segment_analysis.to_dict('records')

        # Check cluster balance
        if task_type == 'regression':
            cluster_maes = segment_analysis['mae'].values
            overall_mae = np.mean(abs_errors) if abs_errors.size else np.nan

            # Flag if any cluster contributes >50% to overall error
            dominant = (cluster_maes > overall_mae * 1.5).any()
            results['cluster_balanced'] = not dominant
            print(f"  Cluster balance check: {'PASS' if not dominant else 'WARNING - unbalanced'}")

    # Identify failure cases (use numpy input for consistent behavior)
    X_for_fail = X_np if X_np is not None else (X.values if isinstance(X, pd.DataFrame) else np.array(X))
    failure_cases, failure_patterns = identify_failure_cases(
        y, y_pred, X_for_fail, feature_names, task_type
    )

    results['failure_cases'] = failure_cases.to_dict('records')
    results['n_failures'] = failure_patterns['n_failures']
    results['failure_rate'] = failure_patterns['failure_rate']

    # Top features distinguishing failures
    sig_features = failure_patterns.get('significant_features', [])
    results['failure_discriminating_features'] = [
        {'feature': f, 'p_value': d['p_value'], 'difference': d['difference']}
        for f, d in sig_features[:5]
    ]

    print(f"  Failure rate: {failure_patterns['failure_rate']:.2%}")
    print(f"  Significant discriminating features: {len(sig_features)}")

    # Error distribution plot
    fig = plot_error_distribution(
        y, y_pred, cluster_labels, task_type,
        title="Error Analysis",
        save_path=os.path.join(output_dir, 'error_distribution.pdf')
    )
    plt.close(fig)

    # Save failure cases
    failure_cases.to_csv(os.path.join(output_dir, 'failure_cases.csv'), index=False)

    return results


def run_what_if_analysis(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
    output_dir: str,
    top_n_features: int = 10,
    is_torch: bool = False,
    X_np: np.ndarray = None
) -> Dict:
    """
    Run what-if / sensitivity analysis.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Feature names
        output_dir: Output directory
        top_n_features: Number of top features to analyze
        is_torch: Whether model is PyTorch

    Returns:
        Dict with what-if results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n=== WHAT-IF ANALYSIS ===")

    # Select top features by variance
    feature_vars = np.var(X, axis=0)
    top_indices = np.argsort(feature_vars)[-top_n_features:]

    # Compute partial dependence
    print("Computing partial dependence...")

    X_arr = X_np if X_np is not None else (X.values if isinstance(X, pd.DataFrame) else np.array(X))

    if is_torch:
        from analysis.what_if import compute_partial_dependence_torch
        pd_results = compute_partial_dependence_torch(
            model, X_arr, list(top_indices), feature_names
        )
    else:
        pd_results = compute_partial_dependence(
            model, X_arr, list(top_indices), feature_names
        )

    results['partial_dependence'] = {
        feat: {
            'grid': data['grid'].tolist(),
            'values': data['pd_values'].tolist()
        }
        for feat, data in pd_results.items()
    }

    # Verify economic plausibility
    plausibility_df = verify_economic_plausibility(pd_results, ECONOMIC_EXPECTATIONS)
    results['plausibility'] = plausibility_df.to_dict('records')

    plausible_count = plausibility_df['is_plausible'].sum()
    print(f"  Economically plausible: {plausible_count}/{len(plausibility_df)}")

    # Flag implausible relationships
    implausible = plausibility_df[~plausibility_df['is_plausible']]
    if len(implausible) > 0:
        print("  WARNING: Implausible relationships detected:")
        for _, row in implausible.iterrows():
            print(f"    - {row['feature']}: expected {row['expected_direction']}, "
                  f"got {row['actual_direction']}")

    results['n_plausible'] = int(plausible_count)
    results['n_implausible'] = len(plausibility_df) - int(plausible_count)

    # Sensitivity analysis
    print("Running sensitivity analysis...")

    if is_torch:
        import torch
        model.eval()
        device = next(model.parameters()).device

        def predict_fn(x):
            with torch.no_grad():
                x_t = torch.FloatTensor(x).to(device)
                if hasattr(model, 'forward_adv'):
                    return model.forward(x_t, domain=0)['risk'].cpu().numpy()
                elif hasattr(model, 'risk_head'):
                    return model(x_t)[0].cpu().numpy()
                return model(x_t).squeeze().cpu().numpy()

        sens_df = sensitivity_analysis(None, X_arr[:100], feature_names, predict_fn=predict_fn)
    else:
        sens_df = sensitivity_analysis(model, X_arr[:100], feature_names)

    results['sensitivity'] = sens_df.to_dict('records')

    # Check monotonicity
    monotonic_count = (sens_df['is_monotonic'] != 'non-monotonic').sum()
    print(f"  Monotonic features: {monotonic_count}/{len(sens_df)}")

    # Plot partial dependence
    fig = plot_what_if_curves(
        pd_results,
        title="What-If Analysis: Partial Dependence",
        save_path=os.path.join(output_dir, 'partial_dependence.pdf')
    )
    plt.close(fig)

    # Plot sensitivity tornado
    fig = plot_sensitivity_tornado(
        sens_df,
        title="Sensitivity Analysis",
        save_path=os.path.join(output_dir, 'sensitivity_tornado.pdf')
    )
    plt.close(fig)

    # Save data
    sens_df.to_csv(os.path.join(output_dir, 'sensitivity_analysis.csv'), index=False)
    plausibility_df.to_csv(os.path.join(output_dir, 'plausibility_check.csv'), index=False)

    return results


def generate_paper_artifacts(
    all_results: Dict,
    output_dir: str
) -> Dict:
    """
    Generate all paper-ready artifacts.

    Args:
        all_results: All analysis results
        output_dir: Output directory

    Returns:
        Dict with artifact paths
    """
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {}

    print("\n=== GENERATING PAPER ARTIFACTS ===")

    # Methodology diagrams
    for arch in ['multitask', 'domain_transfer', 'pipeline']:
        path = os.path.join(output_dir, f'diagram_{arch}.pdf')
        fig = create_methodology_diagram(arch, save_path=path)
        plt.close(fig)
        artifacts[f'diagram_{arch}'] = path

    print("  Created methodology diagrams")

    # Generate tables
    experiment_results = all_results.get('experiments', {})

    if experiment_results:
        # Ablation table
        ablation_table = generate_ablation_table(
            experiment_results,
            metrics=['risk_mae', 'risk_rmse', 'risk_spearman', 'savings_macro_f1'],
            format_type='latex'
        )

        with open(os.path.join(output_dir, 'table_ablation.tex'), 'w') as f:
            f.write(ablation_table)
        artifacts['table_ablation'] = os.path.join(output_dir, 'table_ablation.tex')

        # Markdown version
        ablation_md = generate_ablation_table(
            experiment_results,
            metrics=['risk_mae', 'risk_rmse', 'risk_spearman', 'savings_macro_f1'],
            format_type='markdown'
        )

        with open(os.path.join(output_dir, 'table_ablation.md'), 'w') as f:
            f.write(ablation_md)

        print("  Created ablation tables")

    # Results summary
    summary_path = os.path.join(output_dir, 'results_summary.json')
    create_results_summary(all_results, summary_path)
    artifacts['summary'] = summary_path

    print("  Created results summary")

    return artifacts


def run_full_analysis(
    run_dirs: List[str],
    dataset_path: str,
    output_dir: str,
    config_path: str = None,
    seed: int = 42
):
    """
    Run complete analysis pipeline.

    Args:
        run_dirs: List of experiment run directories
        dataset_path: Path to dataset
        output_dir: Output directory for analysis
        config_path: Optional config file path
        seed: Random seed
    """
    set_seeds(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(output_dir, f'analysis_{timestamp}')
    os.makedirs(analysis_dir, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {analysis_dir}")
    print(f"Analyzing {len(run_dirs)} experiment(s)")
    print("=" * 70)

    # Load all experiment results
    all_experiments = {}
    for run_dir in run_dirs:
        exp_name = os.path.basename(run_dir)
        results = load_experiment_results(run_dir)
        if results:
            all_experiments[exp_name] = results
            print(f"Loaded: {exp_name}")

    if not all_experiments:
        print("ERROR: No experiment results found")
        return

    # If some experiments don't have models (e.g., augmentation audit), try to attach baseline models
    # Find baseline regression/classification models in the loaded experiments
    baseline_reg_model = None
    baseline_clf_model = None
    for name, res in all_experiments.items():
        if name.startswith('clean_baseline_regression') and res.get('model') is not None:
            baseline_reg_model = res['model']
        if name.startswith('clean_baseline_classification') and res.get('model') is not None:
            baseline_clf_model = res['model']

    # Attach appropriate baseline model to experiments missing a model so analysis can run interpretability
    for name, res in all_experiments.items():
        if res.get('model') is None:
            # determine target type from config/metrics; default to regression
            cfg = res.get('config', {})
            target_type = cfg.get('data', {}).get('target_type') if cfg else None
            if not target_type:
                # infer from metrics presence
                metrics = res.get('metrics', {})
                cv = metrics.get('cv_results', {})
                if 'mae' in cv:
                    target_type = 'regression'
                elif 'f1' in cv or 'accuracy' in cv:
                    target_type = 'classification'
            if target_type == 'classification' and baseline_clf_model is not None:
                res['model'] = baseline_clf_model
                print(f"[INFO] Attached baseline classification model to {name} for analysis")
            elif (target_type == 'regression' or target_type is None) and baseline_reg_model is not None:
                res['model'] = baseline_reg_model
                print(f"[INFO] Attached baseline regression model to {name} for analysis")
            else:
                # no baseline available; analysis will skip interpretability for this exp
                pass

    # Load dataset for analysis
    if config_path:
        config = load_config(config_path)
    else:
        # Use config from first experiment
        first_exp = list(all_experiments.values())[0]
        config = first_exp.get('config', {})

    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Preprocess
    target_col = config.get('data', {}).get('target_column', 'Risk_Score')
    target_type = config.get('data', {}).get('target_type', 'regression')

    # Package any latent-sprint artifacts (Sprint 7) into analysis dir now that we know target_type
    try:
        package_latent_runs_into_analysis(all_experiments, os.path.join(analysis_dir, 'latent_sprint'), task=target_type)
    except Exception as e:
        print(f"[WARN] Packaging latent sprint artifacts failed: {e}")

    # Try to get features from first experiment's data_profile
    first_exp = list(all_experiments.values())[0]
    data_profile = first_exp.get('data_profile', {})

    if data_profile.get('features_used'):
        # Use exact features that were used in training
        feature_cols = [c for c in data_profile['features_used'] if c in df.columns]
        print(f"Using features from data_profile: {len(feature_cols)}")
    else:
        # Fallback: calculate features
        ignored = config.get('preprocessing', {}).get('ignored_columns', ['Behavior_Risk_Level'])
        cols_to_drop = config.get('preprocessing', {}).get('columns_to_drop', [])
        exclude = [target_col] + ignored + cols_to_drop
        if target_col == 'Risk_Score' and 'Save_Money_Yes' in df.columns:
            exclude.extend(['Save_Money_Yes', 'Save_Money_No'])
        feature_cols = [c for c in df.columns if c not in exclude and c not in FORBIDDEN_TARGETS]

    X = df[feature_cols].astype(np.float32)
    y = df[target_col].astype(np.float32)

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target: {target_col} ({target_type})")

    # Verify no forbidden columns
    for col in feature_cols:
        if col in FORBIDDEN_TARGETS:
            raise ValueError(f"BLOCKED: Forbidden column '{col}' in features!")

    print("\nForbidden column check: PASS")

    # Initialize results container
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'dataset': dataset_path,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'target': target_col,
            'experiments': list(all_experiments.keys())
        },
        'experiments': {},
        'interpretability': {},
        'stability': {},
        'errors': {},
        'what_if': {},
        'artifacts': {}
    }

    # Run analysis for each experiment
    for exp_name, exp_results in all_experiments.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {exp_name}")
        print(f"{'='*60}")

        exp_dir = os.path.join(analysis_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        model = exp_results.get('model')
        is_torch = model is not None and hasattr(model, 'parameters')

        # Get experiment-specific features from data_profile
        exp_profile = exp_results.get('data_profile', {})
        exp_features = exp_profile.get('features_used', feature_cols)
        exp_features = [c for c in exp_features if c in df.columns]

        # If a model exists and exposes feature names, prefer those (sklearn sets feature_names_in_)
        if model is not None:
            try:
                model_feature_names = getattr(model, 'feature_names_in_', None)
                if model_feature_names is None and hasattr(model, 'named_steps'):
                    # Pipeline: the pipeline itself may have feature_names_in_
                    model_feature_names = getattr(model, 'feature_names_in_', None)
                if model_feature_names is not None:
                    # Use intersection to avoid missing columns
                    model_cols = [c for c in list(model_feature_names) if c in df.columns]
                    if model_cols:
                        exp_features = model_cols
            except Exception:
                pass

        # Prepare data with experiment-specific features (keep as DataFrame for sklearn)
        # Use DataFrame to preserve column names when used with sklearn models
        X_exp = df[exp_features].copy()
        X_exp = X_exp.astype(np.float32)
        # Prepare a numpy copy for analysis internals
        X_exp_np = X_exp.values if isinstance(X_exp, pd.DataFrame) else np.array(X_exp)

        # Get target from experiment config if available
        exp_config = exp_results.get('config', {})
        exp_target = exp_config.get('data', {}).get('target_column', target_col)
        exp_target_type = exp_config.get('data', {}).get('target_type', target_type)
        y_exp = df[exp_target].astype(np.float32) if exp_target in df.columns else y

        # Store experiment metrics
        all_results['experiments'][exp_name] = exp_results.get('metrics', {}).get('cv_results', {})

        if model is not None:
            try:
                # Interpretability
                interp_results = run_interpretability_analysis(
                    model, X_exp, y_exp, exp_features, exp_target_type,
                    os.path.join(exp_dir, 'interpretability'),
                    is_torch=is_torch,
                    X_np=X_exp_np
                )
                all_results['interpretability'][exp_name] = interp_results
            except Exception as e:
                print(f"[WARN] Interpretability failed: {e}")

            try:
                # Error analysis
                error_results = run_error_analysis(
                    model, X_exp, y_exp, exp_features, exp_target_type,
                    os.path.join(exp_dir, 'errors'),
                    is_torch=is_torch,
                    X_np=X_exp_np
                )
                all_results['errors'][exp_name] = error_results
            except Exception as e:
                print(f"[WARN] Error analysis failed: {e}")

            try:
                # What-if analysis
                whatif_results = run_what_if_analysis(
                    model, X_exp, exp_features,
                    os.path.join(exp_dir, 'what_if'),
                    is_torch=is_torch,
                    X_np=X_exp_np
                )
                all_results['what_if'][exp_name] = whatif_results
            except Exception as e:
                print(f"[WARN] What-if analysis failed: {e}")

    # Stability analysis (across all experiments)
    stability_results = run_stability_analysis(
        all_experiments,
        os.path.join(analysis_dir, 'stability')
    )
    all_results['stability'] = stability_results

    # Generate paper artifacts
    artifacts = generate_paper_artifacts(
        all_results,
        os.path.join(analysis_dir, 'paper_artifacts')
    )
    all_results['artifacts'] = artifacts

    # Save complete analysis results
    results_path = os.path.join(analysis_dir, 'analysis_results.json')

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {analysis_dir}")
    print(f"Main results file: {results_path}")
    print("=" * 70)

    return analysis_dir, all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive model analysis for publication'
    )
    parser.add_argument(
        '--runs', '-r',
        type=str,
        nargs='+',
        required=True,
        help='Experiment run directories to analyze'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='analysis_output',
        help='Output directory'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Optional config file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    run_full_analysis(
        args.runs,
        args.dataset,
        args.output,
        args.config,
        args.seed
    )


if __name__ == '__main__':
    main()

