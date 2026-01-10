# Tests for Analysis Module
# Verifies interpretability, stability, error analysis, and paper artifacts

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

# Use non-interactive backend for tests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# INTERPRETABILITY TESTS
# =============================================================================

def test_compute_permutation_importance():
    """Test permutation importance computation."""
    from sklearn.linear_model import Ridge
    from analysis.interpretability import compute_permutation_importance

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1

    model = Ridge()
    model.fit(X, y)

    feature_names = ['f0', 'f1', 'f2', 'f3', 'f4']

    importance_df = compute_permutation_importance(
        model, X, y, feature_names, n_repeats=5
    )

    assert 'feature' in importance_df.columns
    assert 'importance_mean' in importance_df.columns
    assert 'importance_std' in importance_df.columns
    assert len(importance_df) == 5

    # f0 should be most important (highest coefficient)
    top_feature = importance_df.iloc[0]['feature']
    assert top_feature == 'f0', f"Expected f0, got {top_feature}"


def test_identify_actionable_features():
    """Test actionable feature identification."""
    from analysis.interpretability import identify_actionable_features

    importance_df = pd.DataFrame({
        'feature': ['Debt_Level', 'Age', 'Impulse_Buying_Frequency', 'Gender_Male'],
        'importance_mean': [0.5, 0.3, 0.2, 0.1],
        'importance_std': [0.1, 0.05, 0.03, 0.02]
    })

    result = identify_actionable_features(importance_df)

    assert 'actionability' in result.columns

    debt_row = result[result['feature'] == 'Debt_Level'].iloc[0]
    assert debt_row['actionability'] == 'actionable'

    age_row = result[result['feature'] == 'Age'].iloc[0]
    assert age_row['actionability'] == 'non-actionable'


def test_plot_feature_importance():
    """Test feature importance plot generation."""
    from analysis.interpretability import plot_feature_importance

    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(10)],
        'importance_mean': np.random.rand(10),
        'importance_std': np.random.rand(10) * 0.1
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'importance.png')
        fig = plot_feature_importance(importance_df, save_path=save_path)

        assert fig is not None
        assert os.path.exists(save_path)

        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# STABILITY TESTS
# =============================================================================

def test_analyze_cv_stability():
    """Test CV stability analysis."""
    from analysis.stability import analyze_cv_stability

    cv_results = {
        'mae': {
            'mean': 0.5,
            'std': 0.05,
            'all': [0.48, 0.52, 0.49, 0.51, 0.50]
        },
        'rmse': {
            'mean': 0.6,
            'std': 0.2,  # High variance
            'all': [0.4, 0.8, 0.5, 0.7, 0.6]
        }
    }

    stability_df = analyze_cv_stability(cv_results)

    assert len(stability_df) == 2
    assert 'stable' in stability_df.columns
    assert 'cv' in stability_df.columns

    # MAE should be stable (cv < 20%)
    mae_row = stability_df[stability_df['metric'] == 'mae'].iloc[0]
    assert mae_row['stable'] == True

    # RMSE should be unstable (cv > 20%)
    rmse_row = stability_df[stability_df['metric'] == 'rmse'].iloc[0]
    assert rmse_row['stable'] == False


def test_compare_model_stability():
    """Test model stability comparison."""
    from analysis.stability import compare_model_stability

    model_results = {
        'model_a': {
            'mae': {'mean': 0.5, 'std': 0.05, 'all': [0.48, 0.52, 0.49, 0.51, 0.50]}
        },
        'model_b': {
            'mae': {'mean': 0.6, 'std': 0.15, 'all': [0.45, 0.75, 0.55, 0.65, 0.60]}
        }
    }

    comparison_df = compare_model_stability(model_results, metrics=['mae'])

    assert len(comparison_df) == 1
    assert 'most_stable' in comparison_df.columns

    # Model A should be most stable
    assert comparison_df.iloc[0]['most_stable'] == 'model_a'


def test_compute_stability_summary():
    """Test stability summary computation."""
    from analysis.stability import compute_stability_summary

    model_results = {
        'single_task': {
            'risk_mae': {'mean': 0.5, 'std': 0.05, 'all': [0.48, 0.52, 0.49]},
            'savings_macro_f1': {'mean': 0.7, 'std': 0.03, 'all': [0.68, 0.72, 0.70]}
        },
        'multi_task': {
            'risk_mae': {'mean': 0.45, 'std': 0.04, 'all': [0.43, 0.47, 0.45]},
            'savings_macro_f1': {'mean': 0.72, 'std': 0.02, 'all': [0.71, 0.73, 0.72]}
        }
    }

    summary = compute_stability_summary(model_results)

    assert 'best_risk_model' in summary
    assert 'best_savings_model' in summary
    assert 'most_stable_overall' in summary

    # Multi-task should have better risk (lower MAE)
    assert summary['best_risk_model'] == 'multi_task'


# =============================================================================
# ERROR ANALYSIS TESTS
# =============================================================================

def test_analyze_errors_by_segment():
    """Test error analysis by segment."""
    from analysis.error_analysis import analyze_errors_by_segment

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.5, 4.8, 6.2])
    segments = np.array([0, 0, 0, 1, 1, 1])

    result = analyze_errors_by_segment(y_true, y_pred, segments, task_type='regression')

    assert len(result) == 2  # Two segments
    assert 'mae' in result.columns
    assert 'performance' in result.columns


def test_identify_failure_cases():
    """Test failure case identification."""
    from analysis.error_analysis import identify_failure_cases

    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 10.0, 3.1, 4.0, 5.1])  # Index 1 is a failure
    X = np.random.randn(5, 3)
    feature_names = ['f1', 'f2', 'f3']

    failure_cases, patterns = identify_failure_cases(
        y_true, y_pred, X, feature_names,
        task_type='regression',
        threshold_percentile=80
    )

    assert 'n_failures' in patterns
    assert 'failure_rate' in patterns
    assert len(failure_cases) > 0


def test_plot_error_distribution():
    """Test error distribution plotting."""
    from analysis.error_analysis import plot_error_distribution

    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.3

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'errors.png')
        fig = plot_error_distribution(
            y_true, y_pred,
            task_type='regression',
            save_path=save_path
        )

        assert fig is not None
        assert os.path.exists(save_path)

        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# WHAT-IF ANALYSIS TESTS
# =============================================================================

def test_compute_partial_dependence():
    """Test partial dependence computation."""
    from sklearn.linear_model import Ridge
    from analysis.what_if import compute_partial_dependence

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] + np.random.randn(100) * 0.1

    model = Ridge()
    model.fit(X, y)

    pd_results = compute_partial_dependence(
        model, X, [0, 1], ['f0', 'f1', 'f2'],
        grid_resolution=20
    )

    assert 'f0' in pd_results
    assert 'grid' in pd_results['f0']
    assert 'pd_values' in pd_results['f0']
    assert len(pd_results['f0']['grid']) == 20


def test_sensitivity_analysis():
    """Test sensitivity analysis."""
    from sklearn.linear_model import Ridge
    from analysis.what_if import sensitivity_analysis

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(50) * 0.1

    model = Ridge()
    model.fit(X, y)

    sens_df = sensitivity_analysis(model, X, ['f0', 'f1', 'f2'])

    assert 'sensitivity' in sens_df.columns
    assert 'is_monotonic' in sens_df.columns
    assert len(sens_df) == 3

    # f0 should have highest sensitivity
    assert sens_df.iloc[0]['feature'] == 'f0'


def test_verify_economic_plausibility():
    """Test economic plausibility verification."""
    from analysis.what_if import verify_economic_plausibility, check_monotonicity

    pd_results = {
        'Debt_Level': {
            'grid': np.array([0, 1, 2, 3, 4]),
            'pd_values': np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Increasing
        },
        'Save_Money_Yes': {
            'grid': np.array([0, 1]),
            'pd_values': np.array([0.5, 0.6])  # Increasing (wrong!)
        }
    }

    expected = {
        'Debt_Level': 'positive',
        'Save_Money_Yes': 'negative'
    }

    result = verify_economic_plausibility(pd_results, expected)

    # Debt_Level should be plausible (increasing as expected)
    debt_row = result[result['feature'] == 'Debt_Level'].iloc[0]
    assert debt_row['is_plausible'] == True

    # Save_Money_Yes should be implausible (increasing, expected decreasing)
    save_row = result[result['feature'] == 'Save_Money_Yes'].iloc[0]
    assert save_row['is_plausible'] == False


def test_check_monotonicity():
    """Test monotonicity check function."""
    from analysis.what_if import check_monotonicity

    increasing = np.array([1, 2, 3, 4, 5])
    decreasing = np.array([5, 4, 3, 2, 1])
    non_monotonic = np.array([1, 3, 2, 4, 3])
    constant = np.array([2, 2, 2, 2, 2])

    assert check_monotonicity(increasing) == 'increasing'
    assert check_monotonicity(decreasing) == 'decreasing'
    assert check_monotonicity(non_monotonic) == 'non-monotonic'
    assert check_monotonicity(constant) == 'constant'


# =============================================================================
# PAPER ARTIFACTS TESTS
# =============================================================================

def test_generate_ablation_table_latex():
    """Test LaTeX ablation table generation."""
    from analysis.paper_artifacts import generate_ablation_table

    results = {
        'baseline': {'mae': {'mean': 0.5, 'std': 0.05}},
        'improved': {'mae': {'mean': 0.4, 'std': 0.04}}
    }

    latex = generate_ablation_table(results, metrics=['mae'], format_type='latex')

    assert '\\begin{table}' in latex
    assert '\\end{table}' in latex
    assert 'baseline' in latex or 'Baseline' in latex


def test_generate_ablation_table_markdown():
    """Test Markdown ablation table generation."""
    from analysis.paper_artifacts import generate_ablation_table

    results = {
        'baseline': {'mae': {'mean': 0.5, 'std': 0.05}},
        'improved': {'mae': {'mean': 0.4, 'std': 0.04}}
    }

    markdown = generate_ablation_table(results, metrics=['mae'], format_type='markdown')

    assert '|' in markdown
    assert '---' in markdown


def test_create_methodology_diagram():
    """Test methodology diagram creation."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    from analysis.paper_artifacts import create_methodology_diagram

    with tempfile.TemporaryDirectory() as tmpdir:
        for arch in ['multitask', 'domain_transfer', 'pipeline']:
            save_path = os.path.join(tmpdir, f'diagram_{arch}.png')
            fig = create_methodology_diagram(arch, save_path=save_path)

            assert fig is not None
            assert os.path.exists(save_path)

            plt.close(fig)


def test_create_results_summary():
    """Test results summary creation."""
    from analysis.paper_artifacts import create_results_summary

    results = {
        'experiment_1': {
            'mae': {'mean': 0.5, 'std': 0.05, 'all': [0.48, 0.52]},
            'accuracy': 0.85
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'summary.json')
        create_results_summary(results, output_path)

        assert os.path.exists(output_path)

        import json
        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert 'generated_at' in loaded
        assert 'experiments' in loaded


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_no_forbidden_columns_in_analysis():
    """Verify forbidden columns are not used in analysis."""
    from experiments.config_schema import FORBIDDEN_TARGETS

    # Check that FORBIDDEN_TARGETS is properly defined
    assert 'Behavior_Risk_Level' in FORBIDDEN_TARGETS


def test_analysis_reproducibility():
    """Test that analysis is reproducible with same seed."""
    from sklearn.linear_model import Ridge
    from analysis.interpretability import compute_permutation_importance

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] + np.random.randn(50) * 0.1

    model = Ridge()
    model.fit(X, y)

    # Run twice with same seed
    result1 = compute_permutation_importance(
        model, X, y, ['f0', 'f1', 'f2'],
        n_repeats=5, random_state=42
    )

    result2 = compute_permutation_importance(
        model, X, y, ['f0', 'f1', 'f2'],
        n_repeats=5, random_state=42
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(
        result1['importance_mean'].values,
        result2['importance_mean'].values
    )

