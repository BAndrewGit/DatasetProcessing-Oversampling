# Analysis and Interpretability Module
# Transforms trained models into publishable scientific results

from .interpretability import (
    compute_permutation_importance,
    compute_shap_values,
    plot_feature_importance,
    plot_shap_summary,
    get_local_explanations
)

from .stability import (
    analyze_cv_stability,
    compare_model_stability,
    plot_stability_analysis
)

from .error_analysis import (
    analyze_errors_by_segment,
    identify_failure_cases,
    plot_error_distribution
)

from .what_if import (
    compute_partial_dependence,
    sensitivity_analysis,
    plot_what_if_curves
)

from .paper_artifacts import (
    generate_ablation_table,
    generate_model_comparison_table,
    create_methodology_diagram,
    export_final_figures
)

__all__ = [
    'compute_permutation_importance',
    'compute_shap_values',
    'plot_feature_importance',
    'plot_shap_summary',
    'get_local_explanations',
    'analyze_cv_stability',
    'compare_model_stability',
    'plot_stability_analysis',
    'analyze_errors_by_segment',
    'identify_failure_cases',
    'plot_error_distribution',
    'compute_partial_dependence',
    'sensitivity_analysis',
    'plot_what_if_curves',
    'generate_ablation_table',
    'generate_model_comparison_table',
    'create_methodology_diagram',
    'export_final_figures'
]

