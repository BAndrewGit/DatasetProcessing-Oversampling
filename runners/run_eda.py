"""
EDA Runner for Dataset Analysis
Generates comprehensive exploratory data analysis plots for the original dataset.
Integrates with the main pipeline.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def create_eda_directories(output_dir: str) -> dict:
    """Create EDA output directory structure."""
    subdirs = {
        'univariate': os.path.join(output_dir, 'univariate'),
        'bivariate': os.path.join(output_dir, 'bivariate'),
        'target_analysis': os.path.join(output_dir, 'target_analysis'),
        'distributions': os.path.join(output_dir, 'distributions'),
        'correlations': os.path.join(output_dir, 'correlations'),
        'summary': os.path.join(output_dir, 'summary')
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    return subdirs


def plot_dataset_overview(df: pd.DataFrame, output_dir: str):
    """Generate dataset overview plots."""
    print("Generating dataset overview...")

    # 1. Missing values heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_cols = missing_pct[missing_pct > 0]

    if len(missing_cols) > 0:
        missing_cols.head(20).plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Column (Top 20)')
        for i, v in enumerate(missing_cols.head(20)):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    else:
        ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
        ax.set_title('Missing Values Analysis')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary', 'missing_values.png'), dpi=150)
    plt.close()

    # 2. Data types distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    dtype_counts = df.dtypes.astype(str).value_counts()
    dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=plt.cm.Pastel1.colors)
    ax.set_title('Data Types Distribution')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary', 'data_types.png'), dpi=150)
    plt.close()

    # 3. Dataset shape info
    info = {
        'n_samples': int(len(df)),
        'n_features': int(len(df.columns)),
        'n_numeric': int(len(df.select_dtypes(include=[np.number]).columns)),
        'n_categorical': int(len(df.select_dtypes(exclude=[np.number]).columns)),
        'memory_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        'missing_total': int(df.isnull().sum().sum()),
        'missing_pct': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    }

    with open(os.path.join(output_dir, 'summary', 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  Dataset: {info['n_samples']} samples, {info['n_features']} features")
    return info


def _safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name)


def plot_numeric_distributions(df: pd.DataFrame, output_dir: str, max_cols: int = 30, max_detail: int = 20):
    """Generate distribution plots for numeric columns."""
    print("Generating numeric distributions...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Skip ID-like columns
    skip_patterns = ['_id', 'index', 'unnamed']
    numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in skip_patterns)]

    if len(numeric_cols) == 0:
        print("  No numeric columns found")
        return

    # Limit to max_cols
    if len(numeric_cols) > max_cols:
        print(f"  Limiting to {max_cols} numeric columns")
        numeric_cols = numeric_cols[:max_cols]

    # Grid of histograms
    n_cols_grid = 4
    n_rows = (len(numeric_cols) + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols_grid == 1 else axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(col[:25], fontsize=10)
            ax.tick_params(axis='both', labelsize=8)

            # Add stats
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1)

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Numeric Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions', 'numeric_distributions_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Individual detailed plots for top features
    for col in numeric_cols[:max_detail]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        data = df[col].dropna()

        # Histogram with KDE
        ax1.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        try:
            data.plot(kind='kde', ax=ax1, color='red', linewidth=2)
        except Exception:
            pass
        ax1.set_title(f'{col} - Distribution')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Density')

        # Boxplot
        ax2.boxplot(data, vert=True)
        ax2.set_title(f'{col} - Boxplot')
        ax2.set_ylabel(col)

        # Add stats text
        stats_text = f"Mean: {data.mean():.3f}\nStd: {data.std():.3f}\nMin: {data.min():.3f}\nMax: {data.max():.3f}"
        ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'univariate', f'{_safe_name(col)}_distribution.png'), dpi=150)
        plt.close()

    print(f"  Generated plots for {len(numeric_cols)} numeric columns")


def plot_categorical_distributions(df: pd.DataFrame, output_dir: str, max_cols: int = 15):
    """Generate distribution plots for categorical/binary columns."""
    print("Generating categorical distributions...")

    # Find binary columns (0/1 or True/False)
    binary_cols = []
    for col in df.columns:
        unique = df[col].dropna().unique()
        if len(unique) <= 2 and set(unique).issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)

    if len(binary_cols) == 0:
        print("  No binary columns found")
        return

    # Limit
    if len(binary_cols) > max_cols:
        binary_cols = binary_cols[:max_cols]

    # Calculate proportions
    proportions = {}
    for col in binary_cols:
        prop = df[col].mean()
        proportions[col] = prop

    # Sort by proportion
    sorted_cols = sorted(proportions.items(), key=lambda x: x[1], reverse=True)

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(binary_cols) * 0.4)))

    cols = [c[0] for c in sorted_cols]
    props = [c[1] for c in sorted_cols]

    colors = ['green' if p > 0.5 else 'coral' for p in props]
    bars = ax.barh(range(len(cols)), props, color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c[:30] for c in cols], fontsize=9)
    ax.set_xlabel('Proportion (1 / True)')
    ax.set_title('Binary Feature Proportions')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)

    # Add percentage labels
    for i, (bar, prop) in enumerate(zip(bars, props)):
        ax.text(prop + 0.02, i, f'{prop*100:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions', 'binary_features.png'), dpi=150)
    plt.close()

    print(f"  Generated plots for {len(binary_cols)} binary columns")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str, target_col: str = None):
    """Generate correlation matrix heatmap."""
    print("Generating correlation matrix...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print("  Not enough numeric columns for correlation")
        return

    # Limit columns for readability
    if len(numeric_cols) > 30:
        # If target specified, include it and top correlated
        if target_col and target_col in numeric_cols:
            corr_with_target = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            top_cols = corr_with_target.head(29).index.tolist()
            if target_col not in top_cols:
                top_cols.append(target_col)
            numeric_cols = top_cols
        else:
            numeric_cols = numeric_cols[:30]

    corr = df[numeric_cols].corr()

    # Full correlation heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(numeric_cols) * 0.5), max(10, len(numeric_cols) * 0.4)))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(numeric_cols) <= 15, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlations', 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # High correlations list
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                high_corr.append({
                    'feature_1': corr.columns[i],
                    'feature_2': corr.columns[j],
                    'correlation': float(corr.iloc[i, j])
                })

    high_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)

    with open(os.path.join(output_dir, 'correlations', 'high_correlations.json'), 'w') as f:
        json.dump(high_corr[:20], f, indent=2)

    print(f"  Found {len(high_corr)} highly correlated pairs (|r| > 0.5)")


def plot_target_analysis(df: pd.DataFrame, target_col: str, output_dir: str, task: str = 'regression'):
    """Generate target variable analysis plots."""
    print(f"Generating target analysis for {target_col}...")

    if target_col not in df.columns:
        print(f"  Target {target_col} not found")
        return

    target_data = df[target_col].dropna()

    # Target distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram
    axes[0].hist(target_data, bins=30, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    try:
        target_data.plot(kind='kde', ax=axes[0], color='red', linewidth=2)
    except:
        pass
    axes[0].set_title(f'{target_col} Distribution')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Density')

    # Boxplot
    axes[1].boxplot(target_data)
    axes[1].set_title(f'{target_col} Boxplot')
    axes[1].set_ylabel(target_col)

    # Stats
    stats = {
        'count': len(target_data),
        'mean': target_data.mean(),
        'std': target_data.std(),
        'min': target_data.min(),
        '25%': target_data.quantile(0.25),
        '50%': target_data.quantile(0.50),
        '75%': target_data.quantile(0.75),
        'max': target_data.max()
    }

    stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats.items()])
    axes[2].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                 transform=axes[2].transAxes, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].set_title(f'{target_col} Statistics')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_analysis', f'{target_col}_overview.png'), dpi=150)
    plt.close()

    # Feature correlations with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if len(numeric_cols) > 0:
        corr_with_target = df[numeric_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)

        # Top correlated features
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(corr_with_target))
        top_corr = corr_with_target.head(top_n)

        colors = ['green' if v > 0 else 'coral' for v in top_corr.values]
        bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels([c[:30] for c in top_corr.index], fontsize=9)
        ax.set_xlabel('Correlation')
        ax.set_title(f'Top {top_n} Features Correlated with {target_col}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)

        for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
            ax.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}',
                   va='center', ha='left' if val > 0 else 'right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_analysis', f'{target_col}_correlations.png'), dpi=150)
        plt.close()

        # Scatter plots for top 5
        top5 = corr_with_target.head(5)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for i, (col, corr_val) in enumerate(top5.items()):
            ax = axes[i]
            ax.scatter(df[col], df[target_col], alpha=0.5, s=20)

            # Add regression line
            try:
                z = np.polyfit(df[col].dropna(), df[target_col].dropna(), 1)
                p = np.poly1d(z)
                x_line = np.linspace(df[col].min(), df[col].max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2)
            except:
                pass

            ax.set_xlabel(col[:20])
            ax.set_ylabel(target_col)
            ax.set_title(f'r = {corr_val:.3f}', fontsize=10)

        plt.suptitle(f'Top 5 Features vs {target_col}', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_analysis', f'{target_col}_scatter_top5.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Target analysis complete")


def plot_bivariate_relationships(df: pd.DataFrame, output_dir: str, max_pairs: int = 12, target_col: str = None):
    """Generate bivariate scatter/regression plots for top correlated pairs."""
    print("Generating bivariate relationships...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        # Keep target for correlation ranking, but avoid pair duplicates
        pass

    if len(numeric_cols) < 2:
        print("  Not enough numeric columns for bivariate plots")
        return

    corr = df[numeric_cols].corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    upper.columns = ['feature_1', 'feature_2', 'corr']
    upper['abs_corr'] = upper['corr'].abs()

    # Prefer pairs involving target_col if provided
    if target_col and target_col in numeric_cols:
        target_pairs = upper[(upper['feature_1'] == target_col) | (upper['feature_2'] == target_col)]
        target_pairs = target_pairs.sort_values('abs_corr', ascending=False)
        other_pairs = upper[~((upper['feature_1'] == target_col) | (upper['feature_2'] == target_col))]
        other_pairs = other_pairs.sort_values('abs_corr', ascending=False)
        selected = pd.concat([target_pairs, other_pairs]).head(max_pairs)
    else:
        selected = upper.sort_values('abs_corr', ascending=False).head(max_pairs)

    if selected.empty:
        print("  No bivariate pairs selected")
        return

    # Save selected pairs summary
    try:
        selected[['feature_1', 'feature_2', 'corr']].to_csv(
            os.path.join(output_dir, 'bivariate', 'top_bivariate_pairs.csv'), index=False
        )
    except Exception:
        pass

    for _, row in selected.iterrows():
        f1 = row['feature_1']
        f2 = row['feature_2']
        corr_val = row['corr']

        fig = plt.figure(figsize=(7, 6))
        sns.regplot(x=df[f1], y=df[f2], scatter_kws={"alpha": 0.6, "s": 20}, line_kws={"color": "red"})
        plt.title(f"{f1} vs {f2} (r={corr_val:.2f})", fontsize=12, pad=10)
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.tight_layout()
        fname = f"{_safe_name(f1)}_vs_{_safe_name(f2)}.png"
        plt.savefig(os.path.join(output_dir, 'bivariate', fname), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Generated {len(selected)} bivariate plots")


def generate_eda_summary(df: pd.DataFrame, output_dir: str, target_col: str = None):
    """Generate a summary JSON with all EDA findings."""
    summary = {
        'generated_at': datetime.now().isoformat(),
        'dataset': {
            'n_samples': int(len(df)),
            'n_features': int(len(df.columns)),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        },
        'columns': {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(exclude=[np.number]).columns.tolist()
        },
        'missing': {
            'total': int(df.isnull().sum().sum()),
            'by_column': df.isnull().sum().to_dict()
        },
        'plots_generated': []
    }

    # List all generated plots
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                summary['plots_generated'].append(rel_path)

    with open(os.path.join(output_dir, 'eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_eda(dataset_path: str, output_dir: str, target_col: str = 'Risk_Score', task: str = 'regression'):
    """
    Run full EDA analysis on a dataset.

    Args:
        dataset_path: Path to CSV dataset
        output_dir: Output directory for plots
        target_col: Target column name
        task: 'regression' or 'classification'

    Returns:
        Summary dict
    """
    print(f"\n{'='*60}")
    print("EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Target: {target_col}")

    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Loaded: {len(df)} samples, {len(df.columns)} columns")

    # Create directories
    create_eda_directories(output_dir)

    # Run analyses
    plot_dataset_overview(df, output_dir)
    plot_numeric_distributions(df, output_dir, max_cols=30, max_detail=20)
    plot_categorical_distributions(df, output_dir, max_cols=25)
    plot_correlation_matrix(df, output_dir, target_col)
    plot_bivariate_relationships(df, output_dir, max_pairs=15, target_col=target_col)

    if target_col and target_col in df.columns:
        plot_target_analysis(df, target_col, output_dir, task)

    # Generate summary
    summary = generate_eda_summary(df, output_dir, target_col)

    print(f"\n{'='*60}")
    print(f"EDA COMPLETE - {len(summary['plots_generated'])} plots generated")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run EDA on dataset')
    parser.add_argument('--dataset', required=True, help='Path to CSV dataset')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--target', default='Risk_Score', help='Target column name')
    parser.add_argument('--task', default='regression', choices=['regression', 'classification'])

    args = parser.parse_args()

    run_eda(args.dataset, args.output, args.target, args.task)


if __name__ == '__main__':
    main()
