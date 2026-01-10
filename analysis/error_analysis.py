# Error Analysis Module
# Analyzes model errors by behavioral segments and clusters
# Identifies systematic failure cases

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats


def analyze_errors_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_labels: np.ndarray,
    task_type: str = 'regression',
    segment_names: Dict[int, str] = None
) -> pd.DataFrame:
    """
    Analyze prediction errors by behavioral segments/clusters.

    Args:
        y_true: True target values
        y_pred: Predicted values
        segment_labels: Cluster/segment assignments
        task_type: 'regression' or 'classification'
        segment_names: Optional mapping of segment IDs to names

    Returns:
        DataFrame with per-segment error analysis
    """
    unique_segments = np.unique(segment_labels)

    if segment_names is None:
        segment_names = {s: f'Segment {s}' for s in unique_segments}

    analysis_data = []

    for segment in unique_segments:
        mask = segment_labels == segment
        y_true_seg = y_true[mask]
        y_pred_seg = y_pred[mask]
        n_samples = mask.sum()

        if task_type == 'regression':
            errors = y_true_seg - y_pred_seg
            abs_errors = np.abs(errors)

            analysis_data.append({
                'segment': segment,
                'segment_name': segment_names.get(segment, f'Segment {segment}'),
                'n_samples': n_samples,
                'proportion': n_samples / len(y_true),
                'mae': np.mean(abs_errors),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mean_error': np.mean(errors),  # Bias
                'std_error': np.std(errors),
                'median_error': np.median(errors),
                'max_error': np.max(abs_errors),
                'error_skew': stats.skew(errors) if len(errors) > 2 else 0,
                'y_true_mean': np.mean(y_true_seg),
                'y_pred_mean': np.mean(y_pred_seg)
            })
        else:
            # Classification
            correct = y_true_seg == y_pred_seg

            analysis_data.append({
                'segment': segment,
                'segment_name': segment_names.get(segment, f'Segment {segment}'),
                'n_samples': n_samples,
                'proportion': n_samples / len(y_true),
                'accuracy': np.mean(correct),
                'error_rate': 1 - np.mean(correct),
                'n_errors': np.sum(~correct),
                'class_distribution': dict(zip(*np.unique(y_true_seg, return_counts=True))),
                'pred_distribution': dict(zip(*np.unique(y_pred_seg, return_counts=True)))
            })

    df = pd.DataFrame(analysis_data)

    # Add relative performance column
    if task_type == 'regression':
        overall_mae = np.mean(np.abs(y_true - y_pred))
        df['relative_mae'] = df['mae'] / overall_mae
        df['performance'] = df['relative_mae'].apply(
            lambda x: 'better' if x < 0.9 else ('worse' if x > 1.1 else 'average')
        )
    else:
        overall_acc = np.mean(y_true == y_pred)
        df['relative_accuracy'] = df['accuracy'] / overall_acc
        df['performance'] = df['relative_accuracy'].apply(
            lambda x: 'better' if x > 1.1 else ('worse' if x < 0.9 else 'average')
        )

    return df


def identify_failure_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    task_type: str = 'regression',
    threshold_percentile: float = 95,
    n_cases: int = 20
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify and characterize systematic failure cases.

    Args:
        y_true: True target values
        y_pred: Predicted values
        X: Feature matrix
        feature_names: Feature names
        task_type: 'regression' or 'classification'
        threshold_percentile: Percentile for defining failures
        n_cases: Number of failure cases to return

    Returns:
        failure_cases: DataFrame with failure case details
        failure_patterns: Dict with identified patterns
    """
    if task_type == 'regression':
        errors = np.abs(y_true - y_pred)
        threshold = np.percentile(errors, threshold_percentile)
        failure_mask = errors >= threshold
    else:
        failure_mask = y_true != y_pred

    failure_indices = np.where(failure_mask)[0]

    # Build failure case DataFrame
    cases_data = []
    for idx in failure_indices[:n_cases]:
        case = {
            'index': idx,
            'y_true': y_true[idx],
            'y_pred': y_pred[idx]
        }

        if task_type == 'regression':
            case['error'] = errors[idx]
            case['direction'] = 'over' if y_pred[idx] > y_true[idx] else 'under'

        # Add top features
        for i, feat in enumerate(feature_names[:10]):
            case[feat] = X[idx, i]

        cases_data.append(case)

    failure_cases = pd.DataFrame(cases_data)

    # Identify patterns in failure cases
    X_failures = X[failure_mask]
    X_successes = X[~failure_mask]

    patterns = {
        'n_failures': failure_mask.sum(),
        'failure_rate': failure_mask.mean(),
        'feature_differences': {}
    }

    # Compare feature distributions
    for i, feat in enumerate(feature_names):
        fail_mean = np.mean(X_failures[:, i])
        succ_mean = np.mean(X_successes[:, i])

        # T-test for difference
        if len(X_failures) > 1 and len(X_successes) > 1:
            t_stat, p_value = stats.ttest_ind(X_failures[:, i], X_successes[:, i])
        else:
            t_stat, p_value = 0, 1

        patterns['feature_differences'][feat] = {
            'failure_mean': fail_mean,
            'success_mean': succ_mean,
            'difference': fail_mean - succ_mean,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    # Sort by significance
    significant_features = [
        (feat, data) for feat, data in patterns['feature_differences'].items()
        if data['significant']
    ]
    significant_features.sort(key=lambda x: x[1]['p_value'])
    patterns['significant_features'] = significant_features[:10]

    return failure_cases, patterns


def verify_cluster_balance(
    cv_results_by_cluster: Dict[int, Dict],
    overall_results: Dict
) -> pd.DataFrame:
    """
    Verify that no single cluster dominates performance gains.

    Args:
        cv_results_by_cluster: Dict mapping cluster ID to results
        overall_results: Overall model results

    Returns:
        DataFrame with cluster contribution analysis
    """
    balance_data = []

    for cluster_id, results in cv_results_by_cluster.items():
        cluster_metrics = {}

        for metric in ['mae', 'rmse', 'risk_mae', 'risk_rmse']:
            if metric in results:
                cluster_val = results[metric]
                overall_val = overall_results.get(metric, {}).get('mean', cluster_val)

                cluster_metrics[f'{metric}_cluster'] = cluster_val
                cluster_metrics[f'{metric}_contribution'] = (
                    (overall_val - cluster_val) / overall_val * 100
                    if overall_val != 0 else 0
                )

        balance_data.append({
            'cluster': cluster_id,
            **cluster_metrics
        })

    df = pd.DataFrame(balance_data)

    # Add dominance flag
    for metric in ['mae', 'rmse', 'risk_mae', 'risk_rmse']:
        contrib_col = f'{metric}_contribution'
        if contrib_col in df.columns:
            max_contrib = df[contrib_col].max()
            df[f'{metric}_dominates'] = df[contrib_col] > max_contrib * 0.5

    return df


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_labels: Optional[np.ndarray] = None,
    task_type: str = 'regression',
    title: str = "Error Distribution",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot error distribution analysis.

    Args:
        y_true: True target values
        y_pred: Predicted values
        segment_labels: Optional segment labels
        task_type: 'regression' or 'classification'
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if task_type == 'regression':
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        # Plot 1: Error histogram
        ax1 = axes[0]
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax1.axvline(x=0, color='red', linestyle='--', label='Zero error')
        ax1.axvline(x=np.mean(errors), color='orange', linestyle='--',
                   label=f'Mean bias: {np.mean(errors):.3f}')
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()

        # Plot 2: True vs Predicted scatter
        ax2 = axes[1]
        ax2.scatter(y_true, y_pred, alpha=0.5, s=20, c='#2E86AB')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

        ax2.set_xlabel('True Value')
        ax2.set_ylabel('Predicted Value')
        ax2.set_title('True vs Predicted')
        ax2.legend()

        # Plot 3: Error by segment or quantile
        ax3 = axes[2]

        if segment_labels is not None:
            unique_segments = np.unique(segment_labels)
            segment_errors = [abs_errors[segment_labels == s] for s in unique_segments]
            ax3.boxplot(segment_errors, labels=[f'S{s}' for s in unique_segments])
            ax3.set_xlabel('Segment')
        else:
            # Error by true value quantile
            quantiles = pd.qcut(y_true, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            quantile_errors = [abs_errors[quantiles == q] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
            ax3.boxplot(quantile_errors, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax3.set_xlabel('True Value Quantile')

        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Error by Segment/Quantile')

    else:
        # Classification error analysis
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        # Plot 1: Confusion matrix
        ax1 = axes[0]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax1, cmap='Blues')
        ax1.set_title('Confusion Matrix')

        # Plot 2: Error rate by segment
        ax2 = axes[1]
        if segment_labels is not None:
            unique_segments = np.unique(segment_labels)
            error_rates = [(y_true[segment_labels == s] != y_pred[segment_labels == s]).mean()
                          for s in unique_segments]
            ax2.bar([f'S{s}' for s in unique_segments], error_rates, color='#A23B72', alpha=0.7)
            ax2.set_xlabel('Segment')
            ax2.set_ylabel('Error Rate')
            ax2.set_title('Error Rate by Segment')
        else:
            # Overall metrics
            acc = np.mean(y_true == y_pred)
            ax2.bar(['Accuracy', 'Error Rate'], [acc, 1-acc], color=['#2E86AB', '#A23B72'], alpha=0.7)
            ax2.set_ylabel('Rate')
            ax2.set_title('Classification Performance')

        # Plot 3: Class distribution
        ax3 = axes[2]
        classes, true_counts = np.unique(y_true, return_counts=True)
        _, pred_counts = np.unique(y_pred, return_counts=True)

        x = np.arange(len(classes))
        width = 0.35
        ax3.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
        ax3.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Class {c}' for c in classes])
        ax3.set_ylabel('Count')
        ax3.set_title('Class Distribution')
        ax3.legend()

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_error_summary_table(
    model_errors: Dict[str, Dict],
    task_type: str = 'regression'
) -> pd.DataFrame:
    """
    Create summary table of errors across models.

    Args:
        model_errors: Dict mapping model name to error statistics
        task_type: 'regression' or 'classification'

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for model_name, errors in model_errors.items():
        row = {'model': model_name}

        if task_type == 'regression':
            row['mae'] = errors.get('mae', np.nan)
            row['rmse'] = errors.get('rmse', np.nan)
            row['mean_bias'] = errors.get('mean_error', np.nan)
            row['std_error'] = errors.get('std_error', np.nan)
            row['max_error'] = errors.get('max_error', np.nan)
        else:
            row['accuracy'] = errors.get('accuracy', np.nan)
            row['error_rate'] = errors.get('error_rate', np.nan)
            row['n_errors'] = errors.get('n_errors', np.nan)

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def analyze_bias_by_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    n_bins: int = 5
) -> pd.DataFrame:
    """
    Analyze prediction bias across feature value ranges.

    Args:
        y_true: True target values
        y_pred: Predicted values
        X: Feature matrix
        feature_names: Feature names
        n_bins: Number of bins per feature

    Returns:
        DataFrame with bias analysis per feature
    """
    bias_data = []
    errors = y_true - y_pred

    for i, feat in enumerate(feature_names):
        feat_values = X[:, i]

        # Skip if constant
        if np.std(feat_values) < 1e-10:
            continue

        # Bin the feature
        try:
            bins = pd.qcut(feat_values, q=n_bins, duplicates='drop')
            bin_labels = bins.unique()
        except ValueError:
            continue

        for bin_label in bin_labels:
            mask = bins == bin_label
            if mask.sum() < 5:
                continue

            bias_data.append({
                'feature': feat,
                'bin': str(bin_label),
                'n_samples': mask.sum(),
                'mean_error': np.mean(errors[mask]),
                'std_error': np.std(errors[mask]),
                'mae': np.mean(np.abs(errors[mask])),
                'y_true_mean': np.mean(y_true[mask]),
                'y_pred_mean': np.mean(y_pred[mask])
            })

    return pd.DataFrame(bias_data)

