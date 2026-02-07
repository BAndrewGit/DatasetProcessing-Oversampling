# Domain Transfer Plots Module
# Generates required plots for domain transfer experiment output scaffolding
#
# Required plots:
# 1) Ablation comparison (ADV-only vs Transfer variants) - boxplot/bar with mean±std
# 2) Learning curves with markers (warmup, alignment start, freeze/unfreeze)
# 3) Gradient norms per module (adv_adapter, gmsc_adapter, shared_trunk)
# 4) Alignment dynamics (if alignment enabled)
# 5) Latent embeddings overlap (UMAP/t-SNE)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# =============================================================================
# PLOT 1: ABLATION COMPARISON (BOXPLOT / BAR WITH MEAN±STD)
# =============================================================================

def plot_ablation_comparison(results: Dict, output_dir: str, filename: str = 'ablation_comparison.png') -> str:
    """
    Plot ablation comparison: ADV-only vs Transfer variants.

    Metrics (ADV only):
    - Risk: MAE, Spearman
    - Savings: Macro-F1

    Args:
        results: Dict with 'adv_only' and 'transfer' keys, each containing aggregated metrics
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [DEBUG] Ablation comparison - results keys: {list(results.keys())}")

    # Extract data for plotting
    experiments = []
    labels = []

    if 'adv_only' in results:
        experiments.append(('adv_only', 'ADV-only (baseline)'))
        print(f"  [DEBUG] adv_only metrics: {list(results['adv_only'].keys())}")
    if 'transfer' in results:
        experiments.append(('transfer', 'ADV + GMSC transfer'))
        print(f"  [DEBUG] transfer metrics: {list(results['transfer'].keys())}")
    if 'transfer_no_alignment' in results:
        experiments.append(('transfer_no_alignment', 'Transfer (no align)'))
    if 'transfer_alignment_only' in results:
        experiments.append(('transfer_alignment_only', 'Alignment only'))
    if 'pretrain_finetune' in results:
        experiments.append(('pretrain_finetune', 'Pretrain→Finetune'))

    if len(experiments) == 0:
        print("[WARN] No experiments to plot in ablation comparison")
        return None

    # Metrics to plot
    metrics_config = [
        ('risk_mae', 'Risk MAE ↓', 'lower'),
        ('risk_spearman', 'Risk Spearman ρ ↑', 'higher'),
        ('savings_macro_f1', 'Savings Macro-F1 ↑', 'higher'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Domain Transfer Ablation Comparison (ADV Data Only)', fontsize=14, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for ax_idx, (metric_key, metric_label, direction) in enumerate(metrics_config):
        ax = axes[ax_idx]

        exp_names = []
        means = []
        stds = []
        all_values = []

        for exp_key, exp_label in experiments:
            if exp_key in results and metric_key in results[exp_key]:
                metric_data = results[exp_key][metric_key]
                mean_val = metric_data.get('mean', 0)
                std_val = metric_data.get('std', 0)
                vals = metric_data.get('all', [])

                exp_names.append(exp_label)
                means.append(mean_val)
                stds.append(std_val)
                all_values.append(vals)

        if not exp_names:
            ax.set_visible(False)
            continue

        x_pos = np.arange(len(exp_names))

        # Bar plot with error bars
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(exp_names)],
                      edgecolor='black', linewidth=1.2, alpha=0.8)

        # Overlay individual fold values as scatter
        for i, vals in enumerate(all_values):
            if vals:
                jitter = np.random.normal(0, 0.05, len(vals))
                ax.scatter([i] * len(vals) + jitter, vals, color='black', alpha=0.4, s=20, zorder=3)

        ax.set_xlabel('')
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(exp_names, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Highlight best
        if direction == 'lower':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Ablation comparison saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 2: LEARNING CURVES WITH MARKERS
# =============================================================================

def plot_learning_curves(training_log: List[Dict], config: Dict, output_dir: str,
                         filename: str = 'learning_curves.png') -> str:
    """
    Plot learning curves with vertical markers for key events.

    Markers:
    - Warmup end (when GMSC starts)
    - Alignment start epoch
    - Trunk freeze/unfreeze

    Args:
        training_log: List of dicts with epoch logs from trainer
        config: Training config dict
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not training_log:
        print("[WARN] No training log provided for learning curves")
        return None

    print(f"  [DEBUG] Learning curves: {len(training_log)} log entries")

    # Extract phase-based data
    pretrain_epochs = []
    pretrain_losses = []
    finetune_epochs = []
    finetune_train_losses = []
    finetune_val_maes = []

    for entry in training_log:
        phase = entry.get('phase', 'unknown')
        epoch = entry.get('epoch', 0)

        if phase == 'pretrain':
            pretrain_epochs.append(epoch)
            pretrain_losses.append(entry.get('loss', 0))
        elif phase == 'finetune':
            finetune_epochs.append(epoch)
            finetune_train_losses.append(entry.get('train_loss', 0))
            finetune_val_maes.append(entry.get('val_mae', 0))

    # Alternative: use epoch-based logs from DomainTransferTrainer
    epochs = []
    val_maes = []
    val_f1s = []
    train_losses = []

    for entry in training_log:
        if 'val_mae' in entry:
            epochs.append(entry.get('epoch', len(epochs) + 1))
            val_maes.append(entry.get('val_mae', 0))
            val_f1s.append(entry.get('val_f1', 0))
            train_losses.append(entry.get('train_loss', 0))

    # Create figure with 2 or 3 subplots based on available data
    has_pretrain = len(pretrain_epochs) > 0
    has_finetune = len(finetune_epochs) > 0
    has_joint = len(epochs) > 0

    print(f"  [DEBUG] has_pretrain={has_pretrain}, has_finetune={has_finetune}, has_joint={has_joint}")

    # If no data at all, return None
    if not has_pretrain and not has_finetune and not has_joint:
        print("[WARN] No plottable data in training log for learning curves")
        return None

    n_plots = sum([has_pretrain or has_finetune, has_joint or has_finetune, True])
    fig, axes = plt.subplots(1, min(3, n_plots), figsize=(14, 4))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle('Training Curves with Phase Markers', fontsize=14, fontweight='bold')

    # Config markers
    warmup_epochs = config.get('warmup_epochs', 0)
    alignment_start = config.get('alignment_start_epoch', 0)
    freeze_trunk_epochs = config.get('freeze_trunk_epochs', 0)
    alignment_enabled = config.get('alignment_enabled', False)

    ax_idx = 0

    # Plot 1: Pretrain phase (GMSC loss)
    if has_pretrain:
        ax = axes[ax_idx]
        ax.plot(pretrain_epochs, pretrain_losses, 'b-', linewidth=2, marker='o', markersize=4, label='GMSC BCE Loss')
        ax.set_xlabel('Pretrain Epoch')
        ax.set_ylabel('GMSC Loss')
        ax.set_title('Phase 1: Pretrain on GMSC')
        ax.legend()
        ax.grid(alpha=0.3)
        ax_idx += 1

    # Plot 2: Finetune phase (ADV metrics)
    if has_finetune and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.plot(finetune_epochs, finetune_val_maes, 'r-', linewidth=2, marker='s', markersize=4, label='Val Risk MAE')

        # Add trunk freeze marker
        if freeze_trunk_epochs > 0:
            ax.axvline(x=freeze_trunk_epochs, color='purple', linestyle='--', linewidth=2, label=f'Trunk unfreeze (ep {freeze_trunk_epochs})')

        ax.set_xlabel('Finetune Epoch')
        ax.set_ylabel('Risk MAE')
        ax.set_title('Phase 2: Finetune on ADV')
        ax.legend()
        ax.grid(alpha=0.3)
        ax_idx += 1

    # Plot 3: Joint training or combined view
    if has_joint and ax_idx < len(axes):
        ax = axes[ax_idx]

        # Plot dual y-axis for MAE and F1
        color_mae = 'tab:red'
        color_f1 = 'tab:blue'

        ax.plot(epochs, val_maes, color=color_mae, linewidth=2, marker='o', markersize=3, label='Val Risk MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Risk MAE', color=color_mae)
        ax.tick_params(axis='y', labelcolor=color_mae)

        # Secondary y-axis for F1
        ax2 = ax.twinx()
        ax2.plot(epochs, val_f1s, color=color_f1, linewidth=2, marker='s', markersize=3, label='Val Savings F1')
        ax2.set_ylabel('Savings Macro-F1', color=color_f1)
        ax2.tick_params(axis='y', labelcolor=color_f1)

        # Add phase markers
        if warmup_epochs > 0:
            ax.axvline(x=warmup_epochs, color='orange', linestyle='--', linewidth=2, label=f'Warmup end (ep {warmup_epochs})')

        if alignment_enabled and alignment_start > 0:
            ax.axvline(x=alignment_start, color='green', linestyle=':', linewidth=2, label=f'Alignment start (ep {alignment_start})')

        ax.set_title('Joint Training / Combined View')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3)
    elif ax_idx < len(axes):
        # Hide unused axes
        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Learning curves saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 3: GRADIENT NORMS PER MODULE
# =============================================================================

def plot_gradient_norms(grad_logs: List[Dict], output_dir: str,
                        filename: str = 'gradient_norms.png') -> str:
    """
    Plot gradient norms over epochs for each module.

    Modules:
    - adv_adapter
    - gmsc_adapter
    - shared_trunk

    Goal: Verify GMSC doesn't dominate training (gradient norm should be similar or smaller).

    Args:
        grad_logs: List of dicts with gradient norm data per epoch
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not grad_logs:
        print("[WARN] No gradient logs provided")
        return None

    # Extract gradient norms by module
    epochs = []
    adv_adapter_norms = []
    gmsc_adapter_norms = []
    trunk_norms = []

    for log in grad_logs:
        epochs.append(log.get('epoch', len(epochs) + 1))
        adv_adapter_norms.append(log.get('adv_adapter_grad_norm', 0))
        gmsc_adapter_norms.append(log.get('gmsc_adapter_grad_norm', 0))
        trunk_norms.append(log.get('trunk_grad_norm', log.get('shared_trunk_grad_norm', 0)))

    # If we don't have module-specific norms, try alternative keys
    if sum(trunk_norms) == 0:
        for i, log in enumerate(grad_logs):
            trunk_norms[i] = log.get('trunk_grad_norm', 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Gradient Norms per Module (Transfer Dominance Check)', fontsize=14, fontweight='bold')

    if sum(adv_adapter_norms) > 0:
        ax.plot(epochs, adv_adapter_norms, 'b-', linewidth=2, marker='o', markersize=3, label='ADV Adapter')
    if sum(gmsc_adapter_norms) > 0:
        ax.plot(epochs, gmsc_adapter_norms, 'r-', linewidth=2, marker='s', markersize=3, label='GMSC Adapter')
    if sum(trunk_norms) > 0:
        ax.plot(epochs, trunk_norms, 'g-', linewidth=2, marker='^', markersize=3, label='Shared Trunk')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (L2)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add annotation about dominance
    if sum(gmsc_adapter_norms) > 0 and sum(adv_adapter_norms) > 0:
        avg_gmsc = np.mean(gmsc_adapter_norms)
        avg_adv = np.mean(adv_adapter_norms)
        ratio = avg_gmsc / avg_adv if avg_adv > 0 else float('inf')

        if ratio > 2:
            verdict = f"⚠️ GMSC dominates ({ratio:.1f}x > ADV)"
            color = 'red'
        elif ratio > 1:
            verdict = f"GMSC slightly higher ({ratio:.1f}x)"
            color = 'orange'
        else:
            verdict = f"✓ ADV dominant ({1/ratio:.1f}x > GMSC)"
            color = 'green'

        ax.text(0.98, 0.02, verdict, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Gradient norms saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 4: ALIGNMENT DYNAMICS (IF ENABLED)
# =============================================================================

def plot_alignment_dynamics(alignment_logs: List[Dict], output_dir: str,
                            filename: str = 'alignment_dynamics.png') -> str:
    """
    Plot alignment loss and weight over epochs (only if alignment enabled).

    Args:
        alignment_logs: List of dicts with alignment_loss and alignment_weight per epoch
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        Path to saved plot, or None if alignment not used
    """
    os.makedirs(output_dir, exist_ok=True)

    if not alignment_logs:
        print("[INFO] No alignment logs (alignment disabled) - skipping plot")
        return None

    # Check if we have any alignment data
    has_alignment = any(log.get('alignment_loss', 0) > 0 or log.get('alignment_weight', 0) > 0
                        for log in alignment_logs)

    if not has_alignment:
        print("[INFO] Alignment was disabled - skipping alignment dynamics plot")
        return None

    epochs = []
    alignment_losses = []
    alignment_weights = []

    for log in alignment_logs:
        epochs.append(log.get('epoch', len(epochs) + 1))
        alignment_losses.append(log.get('alignment_loss', 0))
        alignment_weights.append(log.get('alignment_weight', 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Domain Alignment Dynamics', fontsize=14, fontweight='bold')

    # Plot 1: Alignment Loss
    ax1 = axes[0]
    ax1.plot(epochs, alignment_losses, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Alignment Loss (CORAL/MMD)')
    ax1.set_title('Alignment Loss Over Time')
    ax1.grid(alpha=0.3)

    # Plot 2: Alignment Weight (scheduled)
    ax2 = axes[1]
    ax2.plot(epochs, alignment_weights, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Alignment Weight (λ)')
    ax2.set_title('Alignment Weight Schedule')
    ax2.grid(alpha=0.3)

    # Highlight when alignment kicks in
    nonzero_epochs = [e for e, w in zip(epochs, alignment_weights) if w > 0]
    if nonzero_epochs:
        start_epoch = min(nonzero_epochs)
        ax2.axvline(x=start_epoch, color='green', linestyle='--', linewidth=2,
                    label=f'Alignment starts (ep {start_epoch})')
        ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Alignment dynamics saved: {filepath}")
    return filepath


# =============================================================================
# PLOT 5: LATENT EMBEDDINGS OVERLAP (UMAP/t-SNE)
# =============================================================================

def plot_latent_embeddings(adv_embeddings: np.ndarray, gmsc_embeddings: np.ndarray,
                           output_dir: str, method: str = 'umap',
                           filename: str = 'latent_embeddings.png',
                           adv_labels: np.ndarray = None,
                           max_samples: int = 2000) -> str:
    """
    Plot 2D visualization of latent embeddings from shared trunk.

    Color: ADV (blue) vs GMSC (red)
    Goal: See if domains overlap or are separated islands.

    Args:
        adv_embeddings: ADV samples in latent space (n_adv, latent_dim)
        gmsc_embeddings: GMSC samples in latent space (n_gmsc, latent_dim)
        output_dir: Directory to save the plot
        method: 'umap' or 'tsne'
        filename: Output filename
        adv_labels: Optional labels for ADV samples (e.g., risk bins)
        max_samples: Max samples per domain for visualization

    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if adv_embeddings is None or gmsc_embeddings is None:
        print("[WARN] Missing embeddings for latent visualization")
        return None

    # Subsample if too large
    if len(adv_embeddings) > max_samples:
        idx = np.random.choice(len(adv_embeddings), max_samples, replace=False)
        adv_embeddings = adv_embeddings[idx]
        if adv_labels is not None:
            adv_labels = adv_labels[idx]

    if len(gmsc_embeddings) > max_samples:
        idx = np.random.choice(len(gmsc_embeddings), max_samples, replace=False)
        gmsc_embeddings = gmsc_embeddings[idx]

    # Combine for dimensionality reduction
    combined = np.vstack([adv_embeddings, gmsc_embeddings])
    domain_labels = np.array(['ADV'] * len(adv_embeddings) + ['GMSC'] * len(gmsc_embeddings))

    # Apply dimensionality reduction
    if method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embedded = reducer.fit_transform(combined)
            method_name = 'UMAP'
        except ImportError:
            print("[WARN] UMAP not installed, falling back to t-SNE")
            method = 'tsne'

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
        embedded = reducer.fit_transform(combined)
        method_name = 't-SNE'

    # Split back
    adv_2d = embedded[:len(adv_embeddings)]
    gmsc_2d = embedded[len(adv_embeddings):]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'Latent Space Embeddings ({method_name}) - Domain Overlap', fontsize=14, fontweight='bold')

    # Plot GMSC first (background)
    ax.scatter(gmsc_2d[:, 0], gmsc_2d[:, 1], c='red', alpha=0.3, s=20, label='GMSC (aux)', edgecolors='none')

    # Plot ADV on top
    if adv_labels is not None:
        scatter = ax.scatter(adv_2d[:, 0], adv_2d[:, 1], c=adv_labels, cmap='viridis',
                             alpha=0.7, s=30, label='ADV (main)', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='ADV Risk Level')
    else:
        ax.scatter(adv_2d[:, 0], adv_2d[:, 1], c='blue', alpha=0.7, s=30, label='ADV (main)',
                   edgecolors='black', linewidth=0.3)

    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Compute overlap metric (simple: centroid distance / average spread)
    adv_centroid = adv_2d.mean(axis=0)
    gmsc_centroid = gmsc_2d.mean(axis=0)
    centroid_dist = np.linalg.norm(adv_centroid - gmsc_centroid)

    adv_spread = np.std(adv_2d)
    gmsc_spread = np.std(gmsc_2d)
    avg_spread = (adv_spread + gmsc_spread) / 2

    overlap_ratio = 1 - min(1, centroid_dist / (2 * avg_spread))

    if overlap_ratio > 0.7:
        verdict = f"✓ High overlap ({overlap_ratio:.2f})"
        color = 'green'
    elif overlap_ratio > 0.4:
        verdict = f"Moderate overlap ({overlap_ratio:.2f})"
        color = 'orange'
    else:
        verdict = f"⚠️ Separate islands ({overlap_ratio:.2f})"
        color = 'red'

    ax.text(0.02, 0.98, verdict, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Latent embeddings saved: {filepath}")
    return filepath


# =============================================================================
# COMBINED PLOT GENERATOR
# =============================================================================

def generate_all_domain_transfer_plots(
    results: Dict,
    training_logs: List[Dict],
    config: Dict,
    output_dir: str,
    adv_embeddings: np.ndarray = None,
    gmsc_embeddings: np.ndarray = None,
    adv_risk_labels: np.ndarray = None,
    grad_logs: List[Dict] = None,
    alignment_logs: List[Dict] = None
) -> Dict[str, str]:
    """
    Generate all domain transfer plots.

    Args:
        results: Aggregated results from CV
        training_logs: Training logs from trainer
        config: Flattened config dict
        output_dir: Base output directory
        adv_embeddings: ADV samples in latent space
        gmsc_embeddings: GMSC samples in latent space
        adv_risk_labels: Risk labels for ADV samples (optional)
        grad_logs: Gradient norm logs (optional)
        alignment_logs: Alignment logs (optional)

    Returns:
        Dict mapping plot name to filepath
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"  Plots directory: {plots_dir}")
    print(f"  Input data:")
    print(f"    - results keys: {list(results.keys())}")
    print(f"    - training_logs: {len(training_logs) if training_logs else 0} entries")
    print(f"    - grad_logs: {len(grad_logs) if grad_logs else 0} entries")
    print(f"    - alignment_logs: {len(alignment_logs) if alignment_logs else 0} entries")
    print(f"    - adv_embeddings: {adv_embeddings.shape if adv_embeddings is not None else 'None'}")
    print(f"    - gmsc_embeddings: {gmsc_embeddings.shape if gmsc_embeddings is not None else 'None'}")

    generated_plots = {}

    # 1) Ablation comparison
    try:
        print("  Generating ablation comparison...")
        path = plot_ablation_comparison(results, plots_dir, 'ablation_comparison.png')
        if path:
            generated_plots['ablation_comparison'] = path
            print(f"    [OK] Saved: {path}")
        else:
            print("    [X] No plot generated (no data)")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate ablation comparison: {e}")
        traceback.print_exc()

    # 2) Learning curves
    try:
        print("  Generating learning curves...")
        path = plot_learning_curves(training_logs, config, plots_dir, 'learning_curves.png')
        if path:
            generated_plots['learning_curves'] = path
            print(f"    [OK] Saved: {path}")
        else:
            print("    [X] No plot generated (no data)")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate learning curves: {e}")
        traceback.print_exc()

    # 3) Gradient norms
    if grad_logs:
        try:
            print("  Generating gradient norms...")
            path = plot_gradient_norms(grad_logs, plots_dir, 'gradient_norms.png')
            if path:
                generated_plots['gradient_norms'] = path
                print(f"    [OK] Saved: {path}")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to generate gradient norms: {e}")
            traceback.print_exc()
    else:
        print("  Skipping gradient norms (no grad_logs)")

    # 4) Alignment dynamics (only if alignment was used)
    if alignment_logs and config.get('alignment_enabled', False):
        try:
            print("  Generating alignment dynamics...")
            path = plot_alignment_dynamics(alignment_logs, plots_dir, 'alignment_dynamics.png')
            if path:
                generated_plots['alignment_dynamics'] = path
                print(f"    [OK] Saved: {path}")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to generate alignment dynamics: {e}")
            traceback.print_exc()
    else:
        print(f"  Skipping alignment dynamics (alignment_enabled={config.get('alignment_enabled', False)}, logs={len(alignment_logs) if alignment_logs else 0})")

    # 5) Latent embeddings
    if adv_embeddings is not None and gmsc_embeddings is not None:
        try:
            print("  Generating latent embeddings (UMAP)...")
            path = plot_latent_embeddings(
                adv_embeddings, gmsc_embeddings, plots_dir,
                method='umap', filename='latent_embeddings_umap.png',
                adv_labels=adv_risk_labels
            )
            if path:
                generated_plots['latent_embeddings_umap'] = path
                print(f"    [OK] Saved: {path}")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to generate UMAP embeddings: {e}")
            traceback.print_exc()
            # Fallback to t-SNE
            try:
                print("  Trying t-SNE fallback...")
                path = plot_latent_embeddings(
                    adv_embeddings, gmsc_embeddings, plots_dir,
                    method='tsne', filename='latent_embeddings_tsne.png',
                    adv_labels=adv_risk_labels
                )
                if path:
                    generated_plots['latent_embeddings_tsne'] = path
                    print(f"    [OK] Saved: {path}")
            except Exception as e2:
                print(f"[ERROR] Failed to generate t-SNE embeddings: {e2}")
    else:
        print("  Skipping latent embeddings (no embeddings data)")

    # Save plot manifest
    manifest_path = os.path.join(plots_dir, 'plot_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(generated_plots, f, indent=2)

    print(f"\n[SUMMARY] Generated {len(generated_plots)} plots in {plots_dir}")
    for name, path in generated_plots.items():
        print(f"  - {name}: {os.path.basename(path)}")

    return generated_plots


# =============================================================================
# HELPER: Extract embeddings from model
# =============================================================================

def extract_latent_embeddings(model, adv_loader, gmsc_loader, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract latent embeddings from trained model.

    Args:
        model: Trained DomainTransferModel
        adv_loader: DataLoader for ADV data
        gmsc_loader: DataLoader for GMSC data
        device: Device to use

    Returns:
        Tuple of (adv_embeddings, gmsc_embeddings)
    """
    import torch

    model.eval()
    model.to(device)

    adv_embeddings = []
    gmsc_embeddings = []

    with torch.no_grad():
        # ADV embeddings
        for batch in adv_loader:
            features = batch['features'].to(device)
            output = model.forward(features, domain=0)
            adv_embeddings.append(output['shared'].cpu().numpy())

        # GMSC embeddings
        for batch in gmsc_loader:
            features = batch['features'].to(device)
            output = model.forward(features, domain=1)
            gmsc_embeddings.append(output['shared'].cpu().numpy())

    adv_embeddings = np.vstack(adv_embeddings) if adv_embeddings else None
    gmsc_embeddings = np.vstack(gmsc_embeddings) if gmsc_embeddings else None

    return adv_embeddings, gmsc_embeddings

