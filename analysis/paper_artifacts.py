# Paper Artifacts Module
# Generates publication-ready tables, figures, and diagrams
# Follows academic standards for reproducibility

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime


# =============================================================================
# TABLE GENERATORS
# =============================================================================

def generate_ablation_table(
    experiment_results: Dict[str, Dict],
    metrics: List[str],
    experiment_names: Dict[str, str] = None,
    metric_names: Dict[str, str] = None,
    format_type: str = 'latex',
    highlight_best: bool = True
) -> str:
    """
    Generate ablation study table for paper.

    Args:
        experiment_results: Dict mapping experiment ID to results
        metrics: List of metrics to include
        experiment_names: Optional friendly names for experiments
        metric_names: Optional friendly names for metrics
        format_type: 'latex', 'markdown', or 'html'
        highlight_best: Whether to highlight best values

    Returns:
        Formatted table string
    """
    if experiment_names is None:
        experiment_names = {k: k for k in experiment_results.keys()}

    if metric_names is None:
        metric_names = {m: m.replace('_', ' ').title() for m in metrics}

    # Build data matrix
    rows = []
    for exp_id, results in experiment_results.items():
        row = {'Experiment': experiment_names.get(exp_id, exp_id)}

        for metric in metrics:
            if metric in results:
                if isinstance(results[metric], dict):
                    mean = results[metric].get('mean', results[metric].get('value', np.nan))
                    std = results[metric].get('std', 0)
                    row[metric_names.get(metric, metric)] = f'{mean:.4f} ± {std:.4f}'
                else:
                    row[metric_names.get(metric, metric)] = f'{results[metric]:.4f}'
            else:
                row[metric_names.get(metric, metric)] = '-'

        rows.append(row)

    df = pd.DataFrame(rows)

    # Find best values
    if highlight_best:
        best_values = {}
        for col in df.columns[1:]:
            values = []
            for val in df[col]:
                if val != '-':
                    try:
                        # Extract mean from "mean ± std" format
                        numeric = float(val.split('±')[0].strip())
                        values.append(numeric)
                    except:
                        values.append(np.nan)
                else:
                    values.append(np.nan)

            if values:
                # Determine if higher or lower is better
                metric_key = [k for k, v in metric_names.items() if v == col]
                if metric_key:
                    metric_key = metric_key[0]
                    try:
                        if any(x in metric_key.lower() for x in ['mae', 'rmse', 'error', 'loss']):
                            best_idx = np.nanargmin(values)
                        else:
                            best_idx = np.nanargmax(values)
                        best_values[col] = best_idx
                    except ValueError:
                        # All-NaN slice - skip highlighting
                        pass

    # Format output
    if format_type == 'latex':
        return _format_latex_table(df, best_values if highlight_best else {})
    elif format_type == 'markdown':
        return _format_markdown_table(df, best_values if highlight_best else {})
    else:
        return df.to_html()


def _format_latex_table(df: pd.DataFrame, best_values: Dict) -> str:
    """Format DataFrame as LaTeX table."""
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Ablation Study Results}')
    lines.append('\\label{tab:ablation}')

    col_format = 'l' + 'c' * (len(df.columns) - 1)
    lines.append(f'\\begin{{tabular}}{{{col_format}}}')
    lines.append('\\toprule')

    # Header
    header = ' & '.join([f'\\textbf{{{col}}}' for col in df.columns])
    lines.append(header + ' \\\\')
    lines.append('\\midrule')

    # Data rows
    for idx, row in df.iterrows():
        row_vals = []
        for col_idx, (col, val) in enumerate(row.items()):
            if col in best_values and best_values[col] == idx:
                row_vals.append(f'\\textbf{{{val}}}')
            else:
                row_vals.append(str(val))
        lines.append(' & '.join(row_vals) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


def _format_markdown_table(df: pd.DataFrame, best_values: Dict) -> str:
    """Format DataFrame as Markdown table."""
    lines = []

    # Header
    lines.append('| ' + ' | '.join(df.columns) + ' |')
    lines.append('|' + '|'.join(['---'] * len(df.columns)) + '|')

    # Data rows
    for idx, row in df.iterrows():
        row_vals = []
        for col_idx, (col, val) in enumerate(row.items()):
            if col in best_values and best_values[col] == idx:
                row_vals.append(f'**{val}**')
            else:
                row_vals.append(str(val))
        lines.append('| ' + ' | '.join(row_vals) + ' |')

    return '\n'.join(lines)


def generate_model_comparison_table(
    model_results: Dict[str, Dict],
    risk_metrics: List[str] = None,
    savings_metrics: List[str] = None,
    format_type: str = 'latex'
) -> str:
    """
    Generate model comparison table.

    Args:
        model_results: Dict mapping model name to results
        risk_metrics: Risk-related metrics
        savings_metrics: Savings-related metrics
        format_type: 'latex', 'markdown', or 'html'

    Returns:
        Formatted table string
    """
    if risk_metrics is None:
        risk_metrics = ['mae', 'rmse', 'spearman', 'r2', 'risk_mae', 'risk_rmse', 'risk_spearman', 'risk_r2']

    if savings_metrics is None:
        savings_metrics = ['macro_f1', 'accuracy', 'savings_macro_f1', 'savings_accuracy']

    rows = []

    for model_name, results in model_results.items():
        row = {'Model': model_name}

        # Risk metrics
        for metric in risk_metrics:
            if metric in results:
                val = results[metric]
                if isinstance(val, dict):
                    row[f'Risk {metric}'] = f"{val.get('mean', 0):.4f} ± {val.get('std', 0):.4f}"
                else:
                    row[f'Risk {metric}'] = f"{val:.4f}"

        # Savings metrics
        for metric in savings_metrics:
            if metric in results:
                val = results[metric]
                if isinstance(val, dict):
                    row[f'Savings {metric}'] = f"{val.get('mean', 0):.4f} ± {val.get('std', 0):.4f}"
                else:
                    row[f'Savings {metric}'] = f"{val:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    if format_type == 'latex':
        return _format_latex_table(df, {})
    elif format_type == 'markdown':
        return _format_markdown_table(df, {})
    else:
        return df.to_html()


def generate_cv_statistics_table(
    cv_results: Dict[str, Dict],
    metrics: List[str],
    include_ci: bool = True,
    format_type: str = 'latex'
) -> str:
    """
    Generate cross-validation statistics table with confidence intervals.

    Args:
        cv_results: Cross-validation results
        metrics: Metrics to include
        include_ci: Whether to include 95% CI
        format_type: Output format

    Returns:
        Formatted table string
    """
    rows = []

    for metric in metrics:
        if metric not in cv_results:
            continue

        data = cv_results[metric]
        if not isinstance(data, dict) or 'all' not in data:
            continue

        values = np.array(data['all'])
        mean = np.mean(values)
        std = np.std(values)

        row = {
            'Metric': metric.replace('_', ' ').title(),
            'Mean': f'{mean:.4f}',
            'Std': f'{std:.4f}',
            'Min': f'{np.min(values):.4f}',
            'Max': f'{np.max(values):.4f}'
        }

        if include_ci:
            from scipy import stats
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
            row['95% CI'] = f'[{ci[0]:.4f}, {ci[1]:.4f}]'

        rows.append(row)

    df = pd.DataFrame(rows)

    if format_type == 'latex':
        return _format_latex_table(df, {})
    elif format_type == 'markdown':
        return _format_markdown_table(df, {})
    else:
        return df.to_html()


# =============================================================================
# FIGURE GENERATORS
# =============================================================================

def create_methodology_diagram(
    architecture: str = 'multitask',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create methodology/architecture diagram.

    Args:
        architecture: 'multitask', 'domain_transfer', or 'pipeline'
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    if architecture == 'multitask':
        _draw_multitask_diagram(ax)
    elif architecture == 'domain_transfer':
        _draw_domain_transfer_diagram(ax)
    else:
        _draw_pipeline_diagram(ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def _draw_multitask_diagram(ax):
    """Draw multi-task learning architecture diagram."""
    # Input layer
    input_box = mpatches.FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.1",
                                        facecolor='#E8F4FD', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 4, 'Input\nFeatures\n(X)', ha='center', va='center', fontsize=10, weight='bold')

    # Shared trunk
    trunk_box = mpatches.FancyBboxPatch((3.5, 2.5), 2.5, 3, boxstyle="round,pad=0.1",
                                        facecolor='#FFF3CD', edgecolor='#F18F01', linewidth=2)
    ax.add_patch(trunk_box)
    ax.text(4.75, 4, 'Shared\nTrunk', ha='center', va='center', fontsize=10, weight='bold')

    # Risk head
    risk_box = mpatches.FancyBboxPatch((7, 4.5), 2, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax.add_patch(risk_box)
    ax.text(8, 5.25, 'Risk Head\n(Regression)', ha='center', va='center', fontsize=9, weight='bold')

    # Savings head
    savings_box = mpatches.FancyBboxPatch((7, 2), 2, 1.5, boxstyle="round,pad=0.1",
                                          facecolor='#F8D7DA', edgecolor='#DC3545', linewidth=2)
    ax.add_patch(savings_box)
    ax.text(8, 2.75, 'Savings Head\n(Classification)', ha='center', va='center', fontsize=9, weight='bold')

    # Arrows
    ax.annotate('', xy=(3.5, 4), xytext=(2.5, 4),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(7, 5.25), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(7, 2.75), xytext=(6, 3.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    ax.set_title('Multi-Task Learning Architecture', fontsize=14, weight='bold', pad=20)


def _draw_domain_transfer_diagram(ax):
    """Draw domain transfer architecture diagram."""
    # ADV input
    adv_box = mpatches.FancyBboxPatch((0.5, 5), 1.8, 1.5, boxstyle="round,pad=0.1",
                                      facecolor='#E8F4FD', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(adv_box)
    ax.text(1.4, 5.75, 'ADV\nFeatures', ha='center', va='center', fontsize=9, weight='bold')

    # GMSC input
    gmsc_box = mpatches.FancyBboxPatch((0.5, 1.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#E2E3E5', edgecolor='#6C757D', linewidth=2)
    ax.add_patch(gmsc_box)
    ax.text(1.4, 2.25, 'GMSC\nFeatures', ha='center', va='center', fontsize=9, weight='bold')

    # ADV adapter
    adv_adapt = mpatches.FancyBboxPatch((2.8, 5), 1.5, 1.5, boxstyle="round,pad=0.1",
                                        facecolor='#CCE5FF', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(adv_adapt)
    ax.text(3.55, 5.75, 'ADV\nAdapter', ha='center', va='center', fontsize=9)

    # GMSC adapter
    gmsc_adapt = mpatches.FancyBboxPatch((2.8, 1.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                                         facecolor='#D6D8DB', edgecolor='#6C757D', linewidth=2)
    ax.add_patch(gmsc_adapt)
    ax.text(3.55, 2.25, 'GMSC\nAdapter', ha='center', va='center', fontsize=9)

    # Shared trunk
    trunk_box = mpatches.FancyBboxPatch((4.8, 2.5), 1.8, 3, boxstyle="round,pad=0.1",
                                        facecolor='#FFF3CD', edgecolor='#F18F01', linewidth=2)
    ax.add_patch(trunk_box)
    ax.text(5.7, 4, 'Shared\nTrunk', ha='center', va='center', fontsize=10, weight='bold')

    # Task heads
    risk_box = mpatches.FancyBboxPatch((7.2, 5), 2, 1.2, boxstyle="round,pad=0.1",
                                       facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax.add_patch(risk_box)
    ax.text(8.2, 5.6, 'Risk\n(ADV)', ha='center', va='center', fontsize=9)

    gmsc_risk_box = mpatches.FancyBboxPatch((7.2, 3.4), 2, 1.2, boxstyle="round,pad=0.1",
                                            facecolor='#E2E3E5', edgecolor='#6C757D', linewidth=2)
    ax.add_patch(gmsc_risk_box)
    ax.text(8.2, 4, 'Risk\n(GMSC)', ha='center', va='center', fontsize=9)

    savings_box = mpatches.FancyBboxPatch((7.2, 1.8), 2, 1.2, boxstyle="round,pad=0.1",
                                          facecolor='#F8D7DA', edgecolor='#DC3545', linewidth=2)
    ax.add_patch(savings_box)
    ax.text(8.2, 2.4, 'Savings\n(ADV only)', ha='center', va='center', fontsize=9)

    # Arrows
    ax.annotate('', xy=(2.8, 5.75), xytext=(2.3, 5.75), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(2.8, 2.25), xytext=(2.3, 2.25), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(4.8, 4.5), xytext=(4.3, 5.5), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(4.8, 3.5), xytext=(4.3, 2.5), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(7.2, 5.6), xytext=(6.6, 4.5), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(7.2, 4), xytext=(6.6, 4), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(7.2, 2.4), xytext=(6.6, 3.5), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    ax.set_title('Domain Transfer Architecture (ADV + GMSC)', fontsize=14, weight='bold', pad=20)


def _draw_pipeline_diagram(ax):
    """Draw experimental pipeline diagram."""
    stages = [
        ('Data\nLoading', '#E8F4FD', 0.5),
        ('Preprocessing', '#FFF3CD', 2.3),
        ('Cross-\nValidation', '#D4EDDA', 4.1),
        ('Model\nTraining', '#CCE5FF', 5.9),
        ('Evaluation', '#F8D7DA', 7.7)
    ]

    for label, color, x in stages:
        box = mpatches.FancyBboxPatch((x, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                      facecolor=color, edgecolor='#333', linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.75, 4, label, ha='center', va='center', fontsize=9, weight='bold')

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1][2], 4), xytext=(stages[i][2] + 1.5, 4),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    ax.set_title('Experimental Pipeline', fontsize=14, weight='bold', pad=20)


def export_final_figures(
    results: Dict,
    output_dir: str,
    figure_configs: List[Dict] = None
) -> List[str]:
    """
    Export all final figures for paper.

    Args:
        results: All experiment results
        output_dir: Output directory
        figure_configs: Optional list of figure configurations

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # Default figure configurations
    if figure_configs is None:
        figure_configs = [
            {'type': 'methodology', 'name': 'fig_methodology'},
            {'type': 'feature_importance', 'name': 'fig_feature_importance'},
            {'type': 'stability', 'name': 'fig_stability'},
            {'type': 'error_distribution', 'name': 'fig_errors'},
            {'type': 'partial_dependence', 'name': 'fig_what_if'},
            {'type': 'model_comparison', 'name': 'fig_comparison'}
        ]

    for config in figure_configs:
        fig_type = config.get('type', 'generic')
        fig_name = config.get('name', f'figure_{fig_type}')

        save_path = os.path.join(output_dir, f'{fig_name}.pdf')

        try:
            if fig_type == 'methodology':
                arch = config.get('architecture', 'multitask')
                fig = create_methodology_diagram(arch, save_path)
            else:
                # Placeholder for other figure types
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, f'Placeholder for {fig_type}',
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close(fig)
            saved_files.append(save_path)

        except Exception as e:
            print(f"Error generating {fig_name}: {e}")

    return saved_files


def create_results_summary(
    experiment_results: Dict[str, Any],
    output_path: str,
    include_metadata: bool = True
) -> str:
    """
    Create comprehensive results summary document.

    Args:
        experiment_results: All experiment results
        output_path: Path to save summary
        include_metadata: Whether to include experiment metadata

    Returns:
        Path to saved summary
    """
    summary = {
        'generated_at': datetime.now().isoformat(),
        'experiments': {}
    }

    for exp_name, results in experiment_results.items():
        exp_summary = {
            'metrics': {},
            'stability': {},
            'best_model': None
        }

        # Extract key metrics
        for key, value in results.items():
            if isinstance(value, dict) and 'mean' in value:
                exp_summary['metrics'][key] = {
                    'mean': value['mean'],
                    'std': value.get('std', 0),
                    'n_folds': len(value.get('all', []))
                }
            elif isinstance(value, (int, float)):
                exp_summary['metrics'][key] = value

        summary['experiments'][exp_name] = exp_summary

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return output_path

