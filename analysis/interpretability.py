# Interpretability Module
# Global and local explanations for trained models
# Uses SHAP and permutation importance for post-hoc interpretability

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import warnings

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Run: pip install shap")


# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

def compute_permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    scoring: str = 'neg_mean_absolute_error',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for any sklearn-compatible model.

    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        scoring: Scoring metric
        random_state: Random seed

    Returns:
        DataFrame with feature importance statistics
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
        'importance_min': result.importances.min(axis=1),
        'importance_max': result.importances.max(axis=1)
    })

    # Sort by mean importance (descending for negative metrics)
    importance_df = importance_df.sort_values(
        'importance_mean',
        ascending=False
    ).reset_index(drop=True)

    return importance_df


def compute_permutation_importance_torch(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task: str = 'regression',
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for PyTorch models.

    Args:
        model: PyTorch model with forward method
        X: Feature matrix (numpy)
        y: Target values
        feature_names: List of feature names
        task: 'regression' or 'classification'
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with feature importance statistics
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    def predict_fn(X_in):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_in).to(device)
            if hasattr(model, 'forward_adv'):
                # Domain transfer model
                output = model.forward(X_tensor, domain=0)
                return output['risk'].cpu().numpy()
            elif hasattr(model, 'risk_head'):
                # Multi-task model
                risk_out, _ = model(X_tensor)
                return risk_out.cpu().numpy()
            else:
                return model(X_tensor).cpu().numpy()

    # Baseline predictions
    y_pred_base = predict_fn(X)

    if task == 'regression':
        base_score = -np.mean(np.abs(y - y_pred_base))
    else:
        y_pred_class = (y_pred_base > 0.5).astype(int)
        base_score = np.mean(y_pred_class == y)

    rng = np.random.RandomState(random_state)
    importances = np.zeros((len(feature_names), n_repeats))

    for feat_idx in range(len(feature_names)):
        for rep in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])

            y_pred_perm = predict_fn(X_permuted)

            if task == 'regression':
                perm_score = -np.mean(np.abs(y - y_pred_perm))
            else:
                y_pred_class = (y_pred_perm > 0.5).astype(int)
                perm_score = np.mean(y_pred_class == y)

            importances[feat_idx, rep] = base_score - perm_score

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances.mean(axis=1),
        'importance_std': importances.std(axis=1),
        'importance_min': importances.min(axis=1),
        'importance_max': importances.max(axis=1)
    })

    importance_df = importance_df.sort_values(
        'importance_mean',
        ascending=False
    ).reset_index(drop=True)

    return importance_df


# =============================================================================
# SHAP VALUES
# =============================================================================

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    model_type: str = 'tree',
    background_samples: int = 100,
    max_samples: int = 500,
    random_state: int = 42
) -> Tuple[np.ndarray, 'shap.Explainer']:
    """
    Compute SHAP values for model interpretability.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        model_type: 'tree' for tree-based, 'kernel' for others
        background_samples: Number of background samples for KernelSHAP
        max_samples: Max samples to explain
        random_state: Random seed

    Returns:
        shap_values: SHAP value matrix
        explainer: SHAP explainer object
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed. Run: pip install shap")

    np.random.seed(random_state)

    # Subsample if needed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    if model_type == 'tree':
        # TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    elif model_type == 'linear':
        # LinearExplainer for linear models
        explainer = shap.LinearExplainer(model, X[:background_samples])
        shap_values = explainer.shap_values(X_explain)
    else:
        # KernelExplainer for any model
        if len(X) > background_samples:
            background_idx = np.random.choice(len(X), background_samples, replace=False)
            background = X[background_idx]
        else:
            background = X

        def predict_fn(x):
            return model.predict(x)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    return shap_values, explainer


def compute_shap_values_torch(
    model,
    X: np.ndarray,
    feature_names: List[str],
    task: str = 'regression',
    background_samples: int = 50,
    max_samples: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, object]:
    """
    Compute SHAP values for PyTorch models using DeepExplainer or KernelSHAP.

    Args:
        model: PyTorch model
        X: Feature matrix
        feature_names: List of feature names
        task: 'regression' or 'classification'
        background_samples: Number of background samples
        max_samples: Max samples to explain
        random_state: Random seed

    Returns:
        shap_values: SHAP value matrix
        explainer: SHAP explainer object
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed. Run: pip install shap")

    import torch

    np.random.seed(random_state)
    model.eval()
    device = next(model.parameters()).device

    # Create prediction function
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

    # Background data
    if len(X) > background_samples:
        bg_idx = np.random.choice(len(X), background_samples, replace=False)
        background = X[bg_idx]
    else:
        background = X

    # Samples to explain
    if len(X) > max_samples:
        explain_idx = np.random.choice(len(X), max_samples, replace=False)
        X_explain = X[explain_idx]
    else:
        X_explain = X

    # Use KernelExplainer for PyTorch
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples=50)

    return shap_values, explainer


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance (Permutation)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    color: str = '#2E86AB'
) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        color: Bar color

    Returns:
        Figure object
    """
    df_plot = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(df_plot))

    ax.barh(
        y_pos,
        df_plot['importance_mean'],
        xerr=df_plot['importance_std'],
        color=color,
        alpha=0.8,
        capsize=3
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (mean decrease in performance)')
    ax.set_title(title)

    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    title: str = "SHAP Summary Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create SHAP summary plot (beeswarm).

    Args:
        shap_values: SHAP values matrix
        X: Feature matrix
        feature_names: Feature names
        max_display: Max features to display
        title: Plot title
        save_path: Path to save figure

    Returns:
        Figure object
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed")

    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def get_local_explanations(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    sample_indices: List[int],
    base_value: float = 0.0
) -> List[Dict]:
    """
    Get local explanations for specific samples.

    Args:
        shap_values: SHAP values matrix
        X: Feature matrix
        feature_names: Feature names
        sample_indices: Indices of samples to explain
        base_value: Expected/base value

    Returns:
        List of explanation dictionaries
    """
    explanations = []

    for idx in sample_indices:
        sample_shap = shap_values[idx]
        sample_features = X[idx]

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]

        explanation = {
            'sample_index': idx,
            'prediction': base_value + np.sum(sample_shap),
            'base_value': base_value,
            'top_contributors': []
        }

        for i in sorted_idx[:10]:  # Top 10 contributors
            explanation['top_contributors'].append({
                'feature': feature_names[i],
                'value': float(sample_features[i]),
                'shap_value': float(sample_shap[i]),
                'direction': 'positive' if sample_shap[i] > 0 else 'negative'
            })

        explanations.append(explanation)

    return explanations


def identify_actionable_features(
    importance_df: pd.DataFrame,
    actionable_keywords: List[str] = None
) -> pd.DataFrame:
    """
    Identify actionable vs non-actionable features.

    Actionable features are those that a person could potentially change
    (e.g., spending habits, savings behavior).
    Non-actionable are demographic or fixed (e.g., age, gender).

    Args:
        importance_df: Feature importance DataFrame
        actionable_keywords: Keywords indicating actionable features

    Returns:
        DataFrame with actionability classification
    """
    if actionable_keywords is None:
        actionable_keywords = [
            'spending', 'saving', 'budget', 'impulse', 'debt',
            'expense', 'credit', 'investment', 'plan', 'attitude'
        ]

    non_actionable_keywords = [
        'age', 'gender', 'family', 'status', 'income_category'
    ]

    def classify_actionability(feature_name):
        feature_lower = feature_name.lower()

        # Check non-actionable first
        for keyword in non_actionable_keywords:
            if keyword in feature_lower:
                return 'non-actionable'

        # Check actionable
        for keyword in actionable_keywords:
            if keyword in feature_lower:
                return 'actionable'

        return 'uncertain'

    df = importance_df.copy()
    df['actionability'] = df['feature'].apply(classify_actionability)

    return df


def create_importance_comparison_plot(
    importance_dfs: Dict[str, pd.DataFrame],
    top_n: int = 15,
    title: str = "Feature Importance Comparison",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison of feature importance across models.

    Args:
        importance_dfs: Dict mapping model name to importance DataFrame
        top_n: Number of top features per model
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    n_models = len(importance_dfs)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)

    if n_models == 1:
        axes = [axes]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, (model_name, df) in enumerate(importance_dfs.items()):
        ax = axes[i]
        df_plot = df.head(top_n)

        y_pos = np.arange(len(df_plot))

        ax.barh(
            y_pos,
            df_plot['importance_mean'],
            xerr=df_plot['importance_std'],
            color=colors[i % len(colors)],
            alpha=0.8,
            capsize=2
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(model_name)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

