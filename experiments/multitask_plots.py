# Multi-task Learning Plots Module
# Generates required plots for multi-task experiment analysis
#
# Required plots:
# 1) Ablation Scoreboard - boxplot/bar comparing risk_only, savings_only, multitask
# 2) Learning Curves - per epoch train/val for both tasks
# 3) Gradient Conflict/Dominance - gradient norms and cosine similarity
# 4) Pareto Tradeoff - scatter plot showing risk vs savings tradeoff

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# =============================================================================
# PLOT 1: ABLATION SCOREBOARD
# =============================================================================

def plot_ablation_scoreboard(
    results: Dict,
    output_dir: str,
    filename: str = 'ablation_scoreboard.png'
) -> str:
    """
    Plot 1 (MUST-HAVE): Ablation Scoreboard - Compare 3 models.

    Models:
    - risk_only: Single-task risk regression
    - savings_only: Single-task savings classification
    - multitask: Joint learning of both tasks

    Metrics:
    - Risk: MAE (lower is better), Spearman ρ (higher is better)
    - Savings: Macro-F1 (higher is better)

    Verdict logic:
    - multitask > risk_only on risk AND > savings_only on savings → WIN
    - one improves, other degrades → TRADEOFF
    - both degrade → TRASH (stick to single-task)

    Args:
        results: Dict with 'risk_only', 'savings_only', 'multitask' keys
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define experiments to plot
    experiments = [
        ('risk_only', 'Risk-only\n(baseline)', 'tab:blue'),
        ('savings_only', 'Savings-only\n(baseline)', 'tab:orange'),
        ('multitask', 'Multi-task\n(joint)', 'tab:green')
    ]

    # Metrics configuration
    metrics_config = [
        ('risk_mae', 'Risk MAE ↓', 'lower'),
        ('risk_spearman', 'Risk Spearman ρ ↑', 'higher'),
        ('savings_macro_f1', 'Savings Macro-F1 ↑', 'higher'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Multi-task Ablation Scoreboard', fontsize=14, fontweight='bold')

    verdicts = []  # Collect verdicts for summary

    for ax_idx, (metric_key, metric_label, direction) in enumerate(metrics_config):
        ax = axes[ax_idx]

        exp_names = []
        means = []
        stds = []
        all_values = []
        colors = []

        for exp_key, exp_label, color in experiments:
            if exp_key in results and metric_key in results[exp_key]:
                metric_data = results[exp_key][metric_key]
                mean_val = metric_data.get('mean', 0)
                std_val = metric_data.get('std', 0)
                vals = metric_data.get('all', [])

                exp_names.append(exp_label)
                means.append(mean_val)
                stds.append(std_val)
                all_values.append(vals)
                colors.append(color)

        if not exp_names:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_label)
            continue

        x_pos = np.arange(len(exp_names))

        # Create boxplot if we have fold values
        if all(len(v) > 1 for v in all_values):
            bp = ax.boxplot(all_values, positions=x_pos, widths=0.6, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        else:
            # Bar plot with error bars
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors,
                         edgecolor='black', linewidth=1.2, alpha=0.7)

        # Overlay individual fold values as scatter
        for i, vals in enumerate(all_values):
            if vals:
                jitter = np.random.normal(0, 0.05, len(vals))
                ax.scatter([x_pos[i]] * len(vals) + jitter, vals, color='black',
                          alpha=0.4, s=20, zorder=3)

        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(exp_names, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            y_offset = max(all_values[i]) if all_values[i] else m + s
            ax.text(i, y_offset + 0.02, f'{m:.3f}±{s:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Highlight best model with gold star
        if direction == 'lower':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)

        ax.scatter([x_pos[best_idx]], [means[best_idx]], marker='*', s=300,
                  c='gold', edgecolors='black', linewidth=1.5, zorder=5)

    # Compute overall verdict
    verdict_text, verdict_color = _compute_multitask_verdict(results)

    fig.text(0.5, 0.02, verdict_text, ha='center', va='bottom', fontsize=12,
             fontweight='bold', color=verdict_color,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=verdict_color, alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Ablation scoreboard saved: {filepath}")
    return filepath


def _compute_multitask_verdict(results: Dict) -> Tuple[str, str]:
    """Compute verdict based on multitask vs single-task performance."""

    # Get mean metrics
    def get_mean(exp, metric):
        if exp in results and metric in results[exp]:
            return results[exp][metric].get('mean', None)
        return None

    mt_risk_mae = get_mean('multitask', 'risk_mae')
    ro_risk_mae = get_mean('risk_only', 'risk_mae')
    mt_savings_f1 = get_mean('multitask', 'savings_macro_f1')
    so_savings_f1 = get_mean('savings_only', 'savings_macro_f1')

    if any(x is None for x in [mt_risk_mae, ro_risk_mae, mt_savings_f1, so_savings_f1]):
        return "Insufficient data for verdict", "gray"

    # Check improvements (MAE: lower is better, F1: higher is better)
    risk_improved = mt_risk_mae < ro_risk_mae * 1.02  # Allow 2% margin
    savings_improved = mt_savings_f1 > so_savings_f1 * 0.98  # Allow 2% margin

    risk_better = mt_risk_mae < ro_risk_mae
    savings_better = mt_savings_f1 > so_savings_f1

    if risk_better and savings_better:
        return "✓ WIN: Multi-task beats BOTH single-task baselines!", "green"
    elif risk_improved and savings_improved:
        return "≈ COMPARABLE: Multi-task matches single-task performance", "blue"
    elif (risk_better and not savings_improved) or (savings_better and not risk_improved):
        return "⚠️ TRADEOFF: One task improves, the other degrades", "orange"
    else:
        return "✗ NEGATIVE TRANSFER: Multi-task is worse - use single-task!", "red"


# =============================================================================
# PLOT 2: LEARNING CURVES (PER EPOCH)
# =============================================================================

def plot_learning_curves(
    epoch_logs: List[Dict],
    output_dir: str,
    filename: str = 'learning_curves.png',
    model_name: str = 'multitask'
) -> str:
    """
    Plot 2: Learning curves - train vs val per epoch.

    Two panels:
    - Left: val_risk_mae over epochs
    - Right: val_savings_macro_f1 over epochs

    Info to look for:
    - Overfitting: train good, val bad
    - Instability: large oscillations
    - Task conflict: one improves while the other degrades

    Args:
        epoch_logs: List of dicts with epoch training logs
        output_dir: Output directory
        filename: Output filename
        model_name: Name for the title

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not epoch_logs:
        print("[WARN] No epoch logs provided for learning curves")
        return None

    # Extract data from logs
    epochs = []
    train_losses = []
    val_risk_mae = []
    val_savings_f1 = []
    val_losses = []

    for log in epoch_logs:
        epochs.append(log.get('epoch', len(epochs) + 1))
        train_losses.append(log.get('train_loss', None))
        val_risk_mae.append(log.get('val_risk_mae', log.get('risk_mae', None)))
        val_savings_f1.append(log.get('val_savings_f1', log.get('savings_macro_f1', None)))
        val_losses.append(log.get('val_loss', None))

    # Filter None values
    valid_epochs = [e for e, v in zip(epochs, val_risk_mae) if v is not None]
    valid_risk_mae = [v for v in val_risk_mae if v is not None]
    valid_savings_f1 = [v for v in val_savings_f1 if v is not None]
    valid_train_loss = [v for v in train_losses if v is not None]
    valid_val_loss = [v for v in val_losses if v is not None]

    if not valid_risk_mae and not valid_savings_f1:
        print("[WARN] No valid metrics in epoch logs")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')

    # Left panel: Risk MAE
    ax1 = axes[0]
    if valid_risk_mae:
        ax1.plot(valid_epochs[:len(valid_risk_mae)], valid_risk_mae,
                'b-', linewidth=2, marker='o', markersize=3, label='Val Risk MAE')

        # Add train loss for comparison if available (secondary axis)
        if valid_train_loss:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(valid_epochs[:len(valid_train_loss)], valid_train_loss,
                         'b--', linewidth=1.5, alpha=0.5, label='Train Loss')
            ax1_twin.set_ylabel('Train Loss', color='blue', alpha=0.5)
            ax1_twin.tick_params(axis='y', labelcolor='blue', alpha=0.5)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Val Risk MAE', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_title('Risk Regression (MAE ↓)')

        # Detect overfitting
        if len(valid_risk_mae) > 5:
            early_mae = np.mean(valid_risk_mae[:5])
            late_mae = np.mean(valid_risk_mae[-5:])
            if late_mae > early_mae * 1.1:
                ax1.text(0.02, 0.98, '⚠️ Possible overfitting', transform=ax1.transAxes,
                        fontsize=9, color='red', va='top')

    # Right panel: Savings F1
    ax2 = axes[1]
    if valid_savings_f1:
        ax2.plot(valid_epochs[:len(valid_savings_f1)], valid_savings_f1,
                'r-', linewidth=2, marker='s', markersize=3, label='Val Savings F1')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Savings Macro-F1', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(alpha=0.3)
        ax2.legend(loc='upper left')
        ax2.set_title('Savings Classification (F1 ↑)')

        # Detect instability
        if len(valid_savings_f1) > 5:
            std_f1 = np.std(valid_savings_f1[-10:]) if len(valid_savings_f1) >= 10 else np.std(valid_savings_f1)
            if std_f1 > 0.05:
                ax2.text(0.02, 0.98, '⚠️ High instability', transform=ax2.transAxes,
                        fontsize=9, color='orange', va='top')

    # Detect task conflict
    if valid_risk_mae and valid_savings_f1 and len(valid_risk_mae) > 5:
        # Check if one improves while other degrades in late training
        risk_trend = np.polyfit(range(len(valid_risk_mae[-5:])), valid_risk_mae[-5:], 1)[0]
        f1_trend = np.polyfit(range(len(valid_savings_f1[-5:])), valid_savings_f1[-5:], 1)[0]

        # Risk decreasing (good) but F1 decreasing (bad) OR vice versa
        if (risk_trend < 0 and f1_trend < 0) or (risk_trend > 0 and f1_trend > 0):
            pass  # Both moving same direction (not conflict)
        elif abs(risk_trend) > 0.001 and abs(f1_trend) > 0.001:
            fig.text(0.5, 0.02, '⚠️ Possible task conflict detected (one improves, other degrades)',
                    ha='center', fontsize=10, color='orange')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Learning curves saved: {filepath}")
    return filepath


def plot_learning_curves_comparison(
    all_epoch_logs: Dict[str, List[Dict]],
    output_dir: str,
    filename: str = 'learning_curves_comparison.png'
) -> str:
    """
    Plot learning curves for all 3 models in comparison.

    Args:
        all_epoch_logs: Dict with 'risk_only', 'savings_only', 'multitask' keys
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Curves Comparison (All Models)', fontsize=14, fontweight='bold')

    colors = {'risk_only': 'blue', 'savings_only': 'orange', 'multitask': 'green'}
    labels = {'risk_only': 'Risk-only', 'savings_only': 'Savings-only', 'multitask': 'Multi-task'}

    # Left: Risk MAE
    ax1 = axes[0]
    for model_name, logs in all_epoch_logs.items():
        if not logs:
            continue
        epochs = [log.get('epoch', i+1) for i, log in enumerate(logs)]
        risk_mae = [log.get('val_risk_mae', log.get('risk_mae', None)) for log in logs]
        valid = [(e, v) for e, v in zip(epochs, risk_mae) if v is not None]
        if valid:
            e, v = zip(*valid)
            ax1.plot(e, v, color=colors.get(model_name, 'gray'), linewidth=2,
                    marker='o', markersize=3, label=labels.get(model_name, model_name))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Risk MAE')
    ax1.set_title('Risk MAE ↓')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Savings F1
    ax2 = axes[1]
    for model_name, logs in all_epoch_logs.items():
        if not logs:
            continue
        epochs = [log.get('epoch', i+1) for i, log in enumerate(logs)]
        savings_f1 = [log.get('val_savings_f1', log.get('savings_macro_f1', None)) for log in logs]
        valid = [(e, v) for e, v in zip(epochs, savings_f1) if v is not None]
        if valid:
            e, v = zip(*valid)
            ax2.plot(e, v, color=colors.get(model_name, 'gray'), linewidth=2,
                    marker='s', markersize=3, label=labels.get(model_name, model_name))

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Savings Macro-F1')
    ax2.set_title('Savings F1 ↑')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Learning curves comparison saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 3: GRADIENT CONFLICT / DOMINANCE
# =============================================================================

def plot_gradient_conflict(
    grad_logs: List[Dict],
    output_dir: str,
    filename: str = 'gradient_conflict.png'
) -> str:
    """
    Plot 3: Gradient conflict/dominance analysis.

    Shows per epoch:
    - ||∇ trunk from risk|| (risk gradient norm)
    - ||∇ trunk from savings|| (savings gradient norm)
    - (optional) cosine similarity between gradients

    Info to look for:
    - If one gradient is much larger → it dominates the trunk
    - If cosine < 0 → tasks pull in opposite directions (negative transfer)

    Args:
        grad_logs: List of dicts with gradient info per epoch
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not grad_logs:
        print("[WARN] No gradient logs provided for conflict analysis")
        return None

    # Extract gradient data
    epochs = []
    risk_grad_norms = []
    savings_grad_norms = []
    cosine_sims = []

    for log in grad_logs:
        epochs.append(log.get('epoch', len(epochs) + 1))
        risk_grad_norms.append(log.get('risk_grad_norm', log.get('trunk_grad_from_risk', None)))
        savings_grad_norms.append(log.get('savings_grad_norm', log.get('trunk_grad_from_savings', None)))
        cosine_sims.append(log.get('grad_cosine_sim', log.get('cosine_similarity', None)))

    # Filter None values
    has_norms = any(v is not None for v in risk_grad_norms) and any(v is not None for v in savings_grad_norms)
    has_cosine = any(v is not None for v in cosine_sims)

    if not has_norms and not has_cosine:
        print("[WARN] No gradient norm or cosine similarity data available")
        return None

    n_plots = 2 if has_cosine else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Gradient Conflict Analysis (Multi-task)', fontsize=14, fontweight='bold')

    # Plot 1: Gradient norms
    ax1 = axes[0]

    valid_risk = [(e, v) for e, v in zip(epochs, risk_grad_norms) if v is not None]
    valid_savings = [(e, v) for e, v in zip(epochs, savings_grad_norms) if v is not None]

    if valid_risk:
        e, v = zip(*valid_risk)
        ax1.plot(e, v, 'b-', linewidth=2, marker='o', markersize=3, label='∇ trunk (risk)')

    if valid_savings:
        e, v = zip(*valid_savings)
        ax1.plot(e, v, 'r-', linewidth=2, marker='s', markersize=3, label='∇ trunk (savings)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient L2 Norm')
    ax1.set_title('Gradient Norms per Task')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Compute dominance verdict
    if valid_risk and valid_savings:
        avg_risk = np.mean([v for _, v in valid_risk])
        avg_savings = np.mean([v for _, v in valid_savings])
        ratio = avg_risk / avg_savings if avg_savings > 0 else float('inf')

        if ratio > 2:
            verdict = f"⚠️ Risk dominates ({ratio:.1f}x)"
            color = 'orange'
        elif ratio < 0.5:
            verdict = f"⚠️ Savings dominates ({1/ratio:.1f}x)"
            color = 'orange'
        else:
            verdict = f"✓ Balanced ({ratio:.2f}x)"
            color = 'green'

        ax1.text(0.98, 0.98, verdict, transform=ax1.transAxes, ha='right', va='top',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Cosine similarity (if available)
    if has_cosine:
        ax2 = axes[1]
        valid_cos = [(e, v) for e, v in zip(epochs, cosine_sims) if v is not None]

        if valid_cos:
            e, v = zip(*valid_cos)
            ax2.plot(e, v, 'g-', linewidth=2, marker='^', markersize=4)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero (orthogonal)')
            ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Conflict threshold')

            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Cosine Similarity')
            ax2.set_title('Gradient Direction Similarity')
            ax2.set_ylim(-1.1, 1.1)
            ax2.legend()
            ax2.grid(alpha=0.3)

            # Verdict on conflict
            avg_cos = np.mean([v for _, v in valid_cos])
            if avg_cos < -0.5:
                verdict = f"✗ SEVERE CONFLICT (cos={avg_cos:.2f})"
                color = 'red'
            elif avg_cos < 0:
                verdict = f"⚠️ Mild conflict (cos={avg_cos:.2f})"
                color = 'orange'
            elif avg_cos < 0.5:
                verdict = f"≈ Orthogonal tasks (cos={avg_cos:.2f})"
                color = 'blue'
            else:
                verdict = f"✓ Aligned tasks (cos={avg_cos:.2f})"
                color = 'green'

            ax2.text(0.98, 0.02, verdict, transform=ax2.transAxes, ha='right', va='bottom',
                    fontsize=10, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Gradient conflict saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 4: PARETO TRADEOFF
# =============================================================================

def plot_pareto_tradeoff(
    results: Dict,
    output_dir: str,
    filename: str = 'pareto_tradeoff.png'
) -> str:
    """
    Plot 4: Pareto tradeoff scatter plot.

    Each fold is a point:
    - x = risk_mae (lower is better, so invert or flip axis)
    - y = savings_macro_f1 (higher is better)

    Color by model type:
    - Blue: risk_only
    - Orange: savings_only
    - Green: multitask

    Goal: See if multitask is on the Pareto frontier or dominated.

    Args:
        results: Dict with fold-level metrics for each model
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Pareto Tradeoff: Risk vs Savings Performance', fontsize=14, fontweight='bold')

    models = [
        ('risk_only', 'Risk-only', 'tab:blue', 'o', 80),
        ('savings_only', 'Savings-only', 'tab:orange', 's', 80),
        ('multitask', 'Multi-task', 'tab:green', '^', 120)
    ]

    all_points = []

    for model_key, model_label, color, marker, size in models:
        if model_key not in results:
            continue

        model_results = results[model_key]

        # Get fold-level values
        risk_mae_all = model_results.get('risk_mae', {}).get('all', [])
        savings_f1_all = model_results.get('savings_macro_f1', {}).get('all', [])

        if not risk_mae_all or not savings_f1_all:
            # Use mean values if fold data not available
            risk_mae = model_results.get('risk_mae', {}).get('mean', None)
            savings_f1 = model_results.get('savings_macro_f1', {}).get('mean', None)
            if risk_mae is not None and savings_f1 is not None:
                ax.scatter([risk_mae], [savings_f1], c=color, marker=marker, s=size * 2,
                          label=f'{model_label} (mean)', edgecolors='black', linewidth=1.5)
                all_points.append((risk_mae, savings_f1, model_key))
            continue

        # Make sure same length
        n_points = min(len(risk_mae_all), len(savings_f1_all))
        risk_mae_all = risk_mae_all[:n_points]
        savings_f1_all = savings_f1_all[:n_points]

        # Plot each fold
        ax.scatter(risk_mae_all, savings_f1_all, c=color, marker=marker, s=size,
                  label=model_label, edgecolors='black', linewidth=0.5, alpha=0.7)

        # Store for Pareto analysis
        for r, s in zip(risk_mae_all, savings_f1_all):
            all_points.append((r, s, model_key))

        # Add mean point with larger marker
        mean_risk = np.mean(risk_mae_all)
        mean_savings = np.mean(savings_f1_all)
        ax.scatter([mean_risk], [mean_savings], c=color, marker=marker, s=size * 3,
                  edgecolors='black', linewidth=2)
        ax.annotate(f'{model_label}\nmean', (mean_risk, mean_savings),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Draw Pareto frontier (approximate)
    if all_points:
        # Sort by risk (ascending) then by savings (descending)
        sorted_points = sorted(all_points, key=lambda p: (p[0], -p[1]))

        # Find Pareto-optimal points (lower risk AND higher savings not dominated)
        pareto_points = []
        max_savings = -float('inf')
        for risk, savings, _ in sorted_points:
            if savings > max_savings:
                pareto_points.append((risk, savings))
                max_savings = savings

        if len(pareto_points) >= 2:
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(pareto_x, pareto_y, 'k--', linewidth=1.5, alpha=0.5, label='Pareto frontier (approx)')

    ax.set_xlabel('Risk MAE ← (lower is better)', fontsize=11)
    ax.set_ylabel('Savings Macro-F1 → (higher is better)', fontsize=11)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Invert x-axis so "better" is upper-right
    ax.invert_xaxis()

    # Add ideal direction arrow
    ax.annotate('', xy=(ax.get_xlim()[1], ax.get_ylim()[1]),
               xytext=(ax.get_xlim()[0], ax.get_ylim()[0]),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.5))
    ax.text(0.95, 0.95, '← IDEAL', transform=ax.transAxes, ha='right', va='top',
           fontsize=10, color='gray', style='italic')

    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Pareto tradeoff saved: {filepath}")
    return filepath


# =============================================================================
# COMBINED PLOT GENERATOR
# =============================================================================

def generate_all_multitask_plots(
    results: Dict,
    epoch_logs: Dict[str, List[Dict]] = None,
    grad_logs: List[Dict] = None,
    output_dir: str = '.',
) -> Dict[str, str]:
    """
    Generate all multitask experiment plots.

    Args:
        results: Aggregated CV results with 'risk_only', 'savings_only', 'multitask' keys
        epoch_logs: Dict of epoch logs per model (optional)
        grad_logs: Gradient conflict logs (optional)
        output_dir: Base output directory

    Returns:
        Dict mapping plot name to filepath
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    generated_plots = {}

    # 1) Ablation Scoreboard (MUST-HAVE)
    try:
        path = plot_ablation_scoreboard(results, plots_dir, 'ablation_scoreboard.png')
        if path:
            generated_plots['ablation_scoreboard'] = path
    except Exception as e:
        print(f"[ERROR] Failed to generate ablation scoreboard: {e}")

    # 2) Learning Curves
    if epoch_logs:
        # Individual model curves
        for model_name, logs in epoch_logs.items():
            if logs:
                try:
                    path = plot_learning_curves(logs, plots_dir, f'learning_curves_{model_name}.png', model_name)
                    if path:
                        generated_plots[f'learning_curves_{model_name}'] = path
                except Exception as e:
                    print(f"[ERROR] Failed to generate learning curves for {model_name}: {e}")

        # Comparison plot
        try:
            path = plot_learning_curves_comparison(epoch_logs, plots_dir, 'learning_curves_comparison.png')
            if path:
                generated_plots['learning_curves_comparison'] = path
        except Exception as e:
            print(f"[ERROR] Failed to generate learning curves comparison: {e}")

    # 3) Gradient Conflict (if data available)
    if grad_logs:
        try:
            path = plot_gradient_conflict(grad_logs, plots_dir, 'gradient_conflict.png')
            if path:
                generated_plots['gradient_conflict'] = path
        except Exception as e:
            print(f"[ERROR] Failed to generate gradient conflict plot: {e}")

    # 4) Pareto Tradeoff
    try:
        path = plot_pareto_tradeoff(results, plots_dir, 'pareto_tradeoff.png')
        if path:
            generated_plots['pareto_tradeoff'] = path
    except Exception as e:
        print(f"[ERROR] Failed to generate Pareto tradeoff: {e}")

    # Save plot manifest
    manifest_path = os.path.join(plots_dir, 'plot_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(generated_plots, f, indent=2)

    print(f"\n[SUMMARY] Generated {len(generated_plots)} multitask plots in {plots_dir}")
    for name, path in generated_plots.items():
        print(f"  - {name}: {os.path.basename(path)}")

    return generated_plots


# =============================================================================
# HELPER: Compute gradient norms with conflict detection
# =============================================================================

def compute_gradient_conflict_metrics(
    model,
    X_batch,
    y_risk_batch,
    y_savings_batch,
    risk_loss_fn,
    savings_loss_fn,
    device: str = 'cpu'
) -> Dict:
    """
    Compute gradient norms and cosine similarity for conflict detection.

    This requires two separate backward passes (with retain_graph=True)
    to get gradients from each task separately.

    Args:
        model: MultiTaskModel
        X_batch: Input batch
        y_risk_batch: Risk targets
        y_savings_batch: Savings targets
        risk_loss_fn: Risk loss function
        savings_loss_fn: Savings loss function
        device: Device

    Returns:
        Dict with 'risk_grad_norm', 'savings_grad_norm', 'cosine_similarity'
    """
    import torch

    model.train()
    model.zero_grad()

    X = torch.FloatTensor(X_batch).to(device)
    y_risk = torch.FloatTensor(y_risk_batch).to(device)
    y_savings = torch.FloatTensor(y_savings_batch).to(device)

    # Forward pass
    risk_pred, savings_logits = model(X)

    # Compute losses
    risk_loss = risk_loss_fn(risk_pred.squeeze(), y_risk)
    savings_loss = savings_loss_fn(savings_logits.squeeze(), y_savings)

    # Get gradients from risk loss
    model.zero_grad()
    risk_loss.backward(retain_graph=True)
    risk_grads = []
    for param in model.shared_trunk.parameters():
        if param.grad is not None:
            risk_grads.append(param.grad.clone().flatten())
    risk_grad_vec = torch.cat(risk_grads) if risk_grads else torch.zeros(1)
    risk_grad_norm = risk_grad_vec.norm().item()

    # Get gradients from savings loss
    model.zero_grad()
    savings_loss.backward()
    savings_grads = []
    for param in model.shared_trunk.parameters():
        if param.grad is not None:
            savings_grads.append(param.grad.clone().flatten())
    savings_grad_vec = torch.cat(savings_grads) if savings_grads else torch.zeros(1)
    savings_grad_norm = savings_grad_vec.norm().item()

    # Compute cosine similarity
    if risk_grad_norm > 0 and savings_grad_norm > 0:
        cosine_sim = torch.dot(risk_grad_vec, savings_grad_vec) / (risk_grad_norm * savings_grad_norm)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = 0.0

    model.zero_grad()

    return {
        'risk_grad_norm': risk_grad_norm,
        'savings_grad_norm': savings_grad_norm,
        'cosine_similarity': cosine_sim
    }


if __name__ == "__main__":
    print("Multi-task Plots Module")
    print("Use generate_all_multitask_plots() to create all plots")
    print("\nRequired plots:")
    print("  1) Ablation Scoreboard - boxplot comparing risk_only, savings_only, multitask")
    print("  2) Learning Curves - train vs val per epoch for both tasks")
    print("  3) Gradient Conflict - gradient norms and cosine similarity")
    print("  4) Pareto Tradeoff - scatter showing risk vs savings performance")

