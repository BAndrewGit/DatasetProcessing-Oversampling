# Stability Analysis Module
# Analyzes variance and stability across cross-validation folds
# Compares stability between model variants

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy import stats


def analyze_cv_stability(
    cv_results: Dict[str, Dict],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Analyze stability of cross-validation results.

    Args:
        cv_results: Dict with metrics containing 'mean', 'std', 'all'
        metrics: List of metrics to analyze (default: all)

    Returns:
        DataFrame with stability statistics
    """
    if metrics is None:
        metrics = [k for k in cv_results.keys()
                  if isinstance(cv_results[k], dict) and 'all' in cv_results[k]]

    stability_data = []

    for metric in metrics:
        if metric not in cv_results:
            continue

        values = cv_results[metric].get('all', [])
        if not values:
            continue

        values = np.array(values)

        stability_data.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'n_folds': len(values),
            'stable': np.std(values) / np.abs(np.mean(values)) < 0.20 if np.mean(values) != 0 else False
        })

    return pd.DataFrame(stability_data)


def compare_model_stability(
    model_results: Dict[str, Dict[str, Dict]],
    metrics: List[str] = None,
    test_significance: bool = True
) -> pd.DataFrame:
    """
    Compare stability across different models.

    Args:
        model_results: Dict mapping model name to cv_results
        metrics: List of metrics to compare
        test_significance: Whether to run statistical tests

    Returns:
        DataFrame with comparative stability statistics
    """
    if metrics is None:
        all_metrics = set()
        for results in model_results.values():
            for k, v in results.items():
                if isinstance(v, dict) and 'all' in v:
                    all_metrics.add(k)
        metrics = list(all_metrics)

    comparison_data = []

    for metric in metrics:
        row = {'metric': metric}

        model_values = {}
        for model_name, results in model_results.items():
            if metric in results and 'all' in results[metric]:
                values = np.array(results[metric]['all'])
                model_values[model_name] = values

                row[f'{model_name}_mean'] = np.mean(values)
                row[f'{model_name}_std'] = np.std(values)
                row[f'{model_name}_cv'] = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else np.inf

        if test_significance and len(model_values) >= 2:
            model_names = list(model_values.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name_i, name_j = model_names[i], model_names[j]
                    vals_i = model_values[name_i]
                    vals_j = model_values[name_j]

                    if len(vals_i) == len(vals_j):
                        t_stat, p_value = stats.ttest_rel(vals_i, vals_j)
                    else:
                        t_stat, p_value = stats.ttest_ind(vals_i, vals_j)

                    row[f'pvalue_{name_i}_vs_{name_j}'] = p_value
                    row[f'significant_{name_i}_vs_{name_j}'] = p_value < 0.05

        cv_values = {k: np.std(v) / np.abs(np.mean(v)) if np.mean(v) != 0 else np.inf
                     for k, v in model_values.items()}
        if cv_values:
            row['most_stable'] = min(cv_values, key=cv_values.get)

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def plot_stability_analysis(
    model_results: Dict[str, Dict[str, Dict]],
    metric: str,
    title: str = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create stability comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    box_data = []
    labels = []

    for model_name, results in model_results.items():
        if metric in results and 'all' in results[metric]:
            box_data.append(results[metric]['all'])
            labels.append(model_name)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    if box_data:
        bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        ax1.set_ylabel(metric)
        ax1.set_title(f'{metric} Distribution Across Folds')
        ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    cv_values = []

    for model_name, results in model_results.items():
        if metric in results and 'all' in results[metric]:
            values = np.array(results[metric]['all'])
            cv = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else 0
            cv_values.append(cv * 100)

    if cv_values:
        bars = ax2.bar(labels, cv_values, color=colors[:len(labels)], alpha=0.7)
        ax2.axhline(y=20, color='red', linestyle='--', label='20% threshold')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.set_title('Stability Comparison (lower is better)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, cv_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    if title:
        plt.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_fold_trajectories(
    model_results: Dict[str, Dict[str, Dict]],
    metric: str,
    title: str = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot metric values across CV folds as trajectories."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'd']

    for i, (model_name, results) in enumerate(model_results.items()):
        if metric in results and 'all' in results[metric]:
            values = results[metric]['all']
            folds = np.arange(1, len(values) + 1)

            ax.plot(folds, values,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   markersize=4,
                   alpha=0.7,
                   label=f'{model_name} (μ={np.mean(values):.4f})')

            ax.axhline(y=np.mean(values),
                      color=colors[i % len(colors)],
                      linestyle='--',
                      alpha=0.5)

    ax.set_xlabel('Fold')
    ax.set_ylabel(metric)
    ax.set_title(title or f'{metric} Across CV Folds')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compute_stability_summary(
    model_results: Dict[str, Dict[str, Dict]],
    risk_metrics: List[str] = None,
    savings_metrics: List[str] = None
) -> Dict:
    """Compute comprehensive stability summary for all models."""
    if risk_metrics is None:
        risk_metrics = ['risk_mae', 'risk_rmse', 'risk_spearman', 'risk_r2', 'mae', 'rmse', 'spearman', 'r2']

    if savings_metrics is None:
        savings_metrics = ['savings_macro_f1', 'savings_accuracy', 'macro_f1', 'accuracy']

    summary = {
        'models': {},
        'best_risk_model': None,
        'best_savings_model': None,
        'most_stable_overall': None
    }

    risk_scores = {}
    savings_scores = {}
    stability_scores = {}

    for model_name, results in model_results.items():
        model_summary = {
            'risk': {},
            'savings': {},
            'overall_stability': []
        }

        for metric in risk_metrics:
            if metric in results and 'all' in results[metric]:
                values = np.array(results[metric]['all'])
                mean_val = np.mean(values)
                cv = np.std(values) / np.abs(mean_val) if mean_val != 0 else np.inf

                model_summary['risk'][metric] = {
                    'mean': mean_val,
                    'std': np.std(values),
                    'cv': cv
                }
                model_summary['overall_stability'].append(cv)

                if metric in ['risk_mae', 'mae']:
                    risk_scores[model_name] = mean_val

        for metric in savings_metrics:
            if metric in results and 'all' in results[metric]:
                values = np.array(results[metric]['all'])
                mean_val = np.mean(values)
                cv = np.std(values) / np.abs(mean_val) if mean_val != 0 else np.inf

                model_summary['savings'][metric] = {
                    'mean': mean_val,
                    'std': np.std(values),
                    'cv': cv
                }
                model_summary['overall_stability'].append(cv)

                if metric in ['savings_macro_f1', 'macro_f1']:
                    savings_scores[model_name] = mean_val

        if model_summary['overall_stability']:
            model_summary['mean_cv'] = np.mean(model_summary['overall_stability'])
            stability_scores[model_name] = model_summary['mean_cv']

        summary['models'][model_name] = model_summary

    if risk_scores:
        summary['best_risk_model'] = min(risk_scores, key=risk_scores.get)
    if savings_scores:
        summary['best_savings_model'] = max(savings_scores, key=savings_scores.get)
    if stability_scores:
        summary['most_stable_overall'] = min(stability_scores, key=stability_scores.get)

    return summary


def identify_unstable_features(
    cv_importance_results: List[pd.DataFrame],
    feature_names: List[str],
    instability_threshold: float = 0.5
) -> pd.DataFrame:
    """Identify features with unstable importance across CV folds."""
    feature_importance = {feat: [] for feat in feature_names}

    for fold_df in cv_importance_results:
        for _, row in fold_df.iterrows():
            if row['feature'] in feature_importance:
                feature_importance[row['feature']].append(row['importance_mean'])

    stability_data = []

    for feat, values in feature_importance.items():
        if not values:
            continue

        values = np.array(values)
        mean_imp = np.mean(values)
        std_imp = np.std(values)
        cv_imp = std_imp / np.abs(mean_imp) if mean_imp != 0 else np.inf

        stability_data.append({
            'feature': feat,
            'mean_importance': mean_imp,
            'std_importance': std_imp,
            'cv_importance': cv_imp,
            'min_importance': np.min(values),
            'max_importance': np.max(values),
            'is_unstable': cv_imp > instability_threshold
        })

    df = pd.DataFrame(stability_data)
    df = df.sort_values('cv_importance', ascending=False)

    return df

