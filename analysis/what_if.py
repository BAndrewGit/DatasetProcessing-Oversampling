# What-If Analysis Module
# Controlled perturbation analysis and sensitivity testing
# Verifies model responses are monotonic and economically plausible

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable, Union
from sklearn.inspection import partial_dependence
import warnings


def compute_partial_dependence(
    model,
    X: np.ndarray,
    feature_indices: List[int],
    feature_names: List[str],
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict:
    """
    Compute partial dependence for selected features.

    Args:
        model: Trained model with predict method
        X: Feature matrix
        feature_indices: Indices of features to analyze
        feature_names: Feature names
        grid_resolution: Number of grid points
        percentiles: Percentile range for feature values

    Returns:
        Dict with partial dependence results
    """
    results = {}

    for feat_idx in feature_indices:
        feat_name = feature_names[feat_idx]

        # Get feature range
        feat_values = X[:, feat_idx]
        low = np.percentile(feat_values, percentiles[0] * 100)
        high = np.percentile(feat_values, percentiles[1] * 100)

        grid = np.linspace(low, high, grid_resolution)

        # Compute average predictions for each grid value
        pd_values = []

        for grid_val in grid:
            X_modified = X.copy()
            X_modified[:, feat_idx] = grid_val

            preds = model.predict(X_modified)
            if hasattr(preds, 'mean'):
                pd_values.append(preds.mean())
            else:
                pd_values.append(np.mean(preds))

        results[feat_name] = {
            'grid': grid,
            'pd_values': np.array(pd_values),
            'feature_index': feat_idx,
            'feature_range': (low, high),
            'original_mean': np.mean(feat_values),
            'original_std': np.std(feat_values)
        }

    return results


def compute_partial_dependence_torch(
    model,
    X: np.ndarray,
    feature_indices: List[int],
    feature_names: List[str],
    task: str = 'regression',
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict:
    """
    Compute partial dependence for PyTorch models.

    Args:
        model: PyTorch model
        X: Feature matrix
        feature_indices: Indices of features to analyze
        feature_names: Feature names
        task: 'regression' or 'classification'
        grid_resolution: Number of grid points
        percentiles: Percentile range

    Returns:
        Dict with partial dependence results
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    def predict_fn(X_in):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_in).to(device)
            if hasattr(model, 'forward_adv'):
                output = model.forward(X_tensor, domain=0)
                return output['risk'].cpu().numpy()
            elif hasattr(model, 'risk_head'):
                risk_out, _ = model(X_tensor)
                return risk_out.cpu().numpy()
            else:
                return model(X_tensor).squeeze().cpu().numpy()

    results = {}

    for feat_idx in feature_indices:
        feat_name = feature_names[feat_idx]

        feat_values = X[:, feat_idx]
        low = np.percentile(feat_values, percentiles[0] * 100)
        high = np.percentile(feat_values, percentiles[1] * 100)

        grid = np.linspace(low, high, grid_resolution)

        pd_values = []
        for grid_val in grid:
            X_modified = X.copy()
            X_modified[:, feat_idx] = grid_val
            preds = predict_fn(X_modified)
            pd_values.append(np.mean(preds))

        results[feat_name] = {
            'grid': grid,
            'pd_values': np.array(pd_values),
            'feature_index': feat_idx,
            'feature_range': (low, high),
            'original_mean': np.mean(feat_values),
            'original_std': np.std(feat_values)
        }

    return results


def sensitivity_analysis(
    model,
    X_baseline: np.ndarray,
    feature_names: List[str],
    perturbation_range: Tuple[float, float] = (-0.5, 0.5),
    n_steps: int = 21,
    predict_fn: Callable = None
) -> pd.DataFrame:
    """
    Perform sensitivity analysis by perturbing each feature.

    Args:
        model: Trained model or None if predict_fn provided
        X_baseline: Baseline feature vector(s)
        feature_names: Feature names
        perturbation_range: Range of perturbation as fraction of std
        n_steps: Number of perturbation steps
        predict_fn: Optional custom prediction function

    Returns:
        DataFrame with sensitivity results
    """
    if X_baseline.ndim == 1:
        X_baseline = X_baseline.reshape(1, -1)

    if predict_fn is None:
        predict_fn = lambda x: model.predict(x)

    baseline_pred = predict_fn(X_baseline).mean()

    # Compute feature statistics
    feature_stds = np.std(X_baseline, axis=0)
    feature_stds[feature_stds == 0] = 1  # Avoid division by zero

    perturbations = np.linspace(perturbation_range[0], perturbation_range[1], n_steps)

    sensitivity_data = []

    for feat_idx, feat_name in enumerate(feature_names):
        responses = []

        for pert in perturbations:
            X_perturbed = X_baseline.copy()
            X_perturbed[:, feat_idx] += pert * feature_stds[feat_idx]

            pred = predict_fn(X_perturbed).mean()
            responses.append(pred)

        responses = np.array(responses)

        # Calculate sensitivity metrics
        sensitivity_data.append({
            'feature': feat_name,
            'feature_index': feat_idx,
            'baseline_pred': baseline_pred,
            'min_pred': responses.min(),
            'max_pred': responses.max(),
            'pred_range': responses.max() - responses.min(),
            'sensitivity': (responses.max() - responses.min()) / (2 * feature_stds[feat_idx]),
            'is_monotonic': check_monotonicity(responses),
            'direction': 'positive' if responses[-1] > responses[0] else 'negative'
        })

    df = pd.DataFrame(sensitivity_data)
    df = df.sort_values('pred_range', ascending=False).reset_index(drop=True)

    return df


def check_monotonicity(values: np.ndarray, tolerance: float = 0.01) -> str:
    """
    Check if a sequence is monotonic.

    Args:
        values: Sequence of values
        tolerance: Tolerance for small violations

    Returns:
        'increasing', 'decreasing', 'non-monotonic', or 'constant'
    """
    if len(values) < 2:
        return 'constant'

    diffs = np.diff(values)

    if np.all(np.abs(diffs) < tolerance):
        return 'constant'

    positive = np.sum(diffs > tolerance)
    negative = np.sum(diffs < -tolerance)

    if positive > 0 and negative == 0:
        return 'increasing'
    elif negative > 0 and positive == 0:
        return 'decreasing'
    else:
        return 'non-monotonic'


def verify_economic_plausibility(
    pd_results: Dict,
    expected_directions: Dict[str, str]
) -> pd.DataFrame:
    """
    Verify that partial dependence relationships are economically plausible.

    Args:
        pd_results: Partial dependence results
        expected_directions: Dict mapping feature name to expected direction
                           ('positive', 'negative', 'any')

    Returns:
        DataFrame with plausibility verification
    """
    verification_data = []

    for feat_name, pd_data in pd_results.items():
        pd_values = pd_data['pd_values']

        # Determine actual direction
        actual_direction = check_monotonicity(pd_values)

        expected = expected_directions.get(feat_name, 'any')

        if expected == 'any':
            is_plausible = True
        elif expected == 'positive':
            is_plausible = actual_direction in ['increasing', 'constant']
        elif expected == 'negative':
            is_plausible = actual_direction in ['decreasing', 'constant']
        else:
            is_plausible = True

        verification_data.append({
            'feature': feat_name,
            'expected_direction': expected,
            'actual_direction': actual_direction,
            'is_plausible': is_plausible,
            'pd_min': pd_values.min(),
            'pd_max': pd_values.max(),
            'pd_range': pd_values.max() - pd_values.min()
        })

    return pd.DataFrame(verification_data)


# Economic plausibility expectations for financial behavior features
ECONOMIC_EXPECTATIONS = {
    # Higher debt should increase risk
    'Debt_Level': 'positive',
    'DebtRatio': 'positive',

    # More impulse buying should increase risk
    'Impulse_Buying_Frequency': 'positive',

    # Better budget planning should decrease risk
    'Budget_Planning_Plan in detail': 'negative',
    'Budget_Planning_Plan budget in detail': 'negative',

    # Saving money should decrease risk
    'Save_Money_Yes': 'negative',

    # Higher essential needs percentage might indicate financial stress
    'Essential_Needs_Percentage': 'positive',

    # Age effects are complex (could go either way)
    'Age': 'any',
    'age': 'any',

    # Credit utilization increases risk
    'RevolvingUtilizationOfUnsecuredLines': 'positive',

    # Past due indicators increase risk
    'NumberOfTime30-59DaysPastDueNotWorse': 'positive',
    'NumberOfTime60-89DaysPastDueNotWorse': 'positive',
    'NumberOfTimes90DaysLate': 'positive',

    # Income typically decreases risk
    'MonthlyIncome': 'negative',
    'Income_Category': 'negative'
}


def plot_what_if_curves(
    pd_results: Dict,
    n_cols: int = 3,
    figsize: Tuple[int, int] = None,
    title: str = "What-If Analysis: Partial Dependence",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot partial dependence curves.

    Args:
        pd_results: Partial dependence results
        n_cols: Number of columns in subplot grid
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        Figure object
    """
    n_features = len(pd_results)
    n_rows = int(np.ceil(n_features / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, (feat_name, pd_data) in enumerate(pd_results.items()):
        ax = axes[i]

        grid = pd_data['grid']
        pd_values = pd_data['pd_values']

        ax.plot(grid, pd_values, color=colors[i % len(colors)], linewidth=2)

        # Add original mean marker
        orig_mean = pd_data['original_mean']
        idx_closest = np.argmin(np.abs(grid - orig_mean))
        ax.scatter([orig_mean], [pd_values[idx_closest]], color='red',
                  s=100, zorder=5, marker='*', label='Data mean')

        ax.set_xlabel(feat_name[:30] + '...' if len(feat_name) > 30 else feat_name, fontsize=9)
        ax.set_ylabel('Predicted Risk')
        ax.grid(alpha=0.3)

        # Add monotonicity indicator
        monotonicity = check_monotonicity(pd_values)
        ax.set_title(f'{monotonicity}', fontsize=9, style='italic')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sensitivity_tornado(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Sensitivity Analysis (Tornado Chart)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create tornado chart for sensitivity analysis.

    Args:
        sensitivity_df: Sensitivity analysis DataFrame
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    df = sensitivity_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    baseline = df['baseline_pred'].iloc[0]

    # Calculate deviations from baseline
    low_dev = df['min_pred'] - baseline
    high_dev = df['max_pred'] - baseline

    y_pos = np.arange(len(df))

    # Plot bars
    ax.barh(y_pos, low_dev, height=0.4, align='center', color='#2E86AB',
           alpha=0.7, label='Low perturbation')
    ax.barh(y_pos, high_dev, height=0.4, align='center', color='#A23B72',
           alpha=0.7, label='High perturbation')

    # Add baseline line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Deviation from Baseline Prediction')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def individual_what_if(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    feature_to_vary: str,
    value_range: Tuple[float, float],
    n_points: int = 50,
    predict_fn: Callable = None
) -> Dict:
    """
    Perform what-if analysis for a single sample and feature.

    Args:
        model: Trained model
        X_sample: Single sample feature vector
        feature_names: Feature names
        feature_to_vary: Name of feature to vary
        value_range: Range of values to test
        n_points: Number of points
        predict_fn: Optional custom prediction function

    Returns:
        Dict with what-if results
    """
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    if predict_fn is None:
        predict_fn = lambda x: model.predict(x)

    feat_idx = feature_names.index(feature_to_vary)
    original_value = X_sample[0, feat_idx]

    test_values = np.linspace(value_range[0], value_range[1], n_points)
    predictions = []

    for val in test_values:
        X_modified = X_sample.copy()
        X_modified[0, feat_idx] = val
        pred = predict_fn(X_modified)
        predictions.append(pred[0] if hasattr(pred, '__len__') else pred)

    return {
        'feature': feature_to_vary,
        'original_value': original_value,
        'original_prediction': predict_fn(X_sample)[0],
        'test_values': test_values,
        'predictions': np.array(predictions),
        'sensitivity': (max(predictions) - min(predictions)) / (value_range[1] - value_range[0])
    }

