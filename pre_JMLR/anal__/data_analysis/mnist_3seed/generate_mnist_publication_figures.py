#!/usr/bin/env python3
"""
Generate Nature Communications style publication figures for MNIST 10-task, 3-seed experiments.
Generates:
1. 6-panel main figure (test accuracy, Hamiltonian, grad norm, avg accuracy, BWT, forgetting)
2. Individual metric plots
3. Performance matrix heatmaps
4. Loss component plots

Added by Claude: Publication-quality figures for JMLR paper MNIST results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Nature Communications style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Data directory
DATA_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "publication_plots"

# Pastel color scheme
COLORS = {
    'C1: Baseline': {'line': '#E57373', 'fill': '#FFCDD2'},
    'C2: Heuristics': {'line': '#FFB74D', 'fill': '#FFE0B2'},
    'C3: Arch Search': {'line': '#64B5F6', 'fill': '#BBDEFB'},
    'C4: AWB Full': {'line': '#BA68C8', 'fill': '#E1BEE7'},
}

SHORT_LABELS = {
    'C1: Baseline': 'Baseline',
    'C2: Heuristics': 'Heuristics',
    'C3: Arch Search': 'Arch Search',
    'C4: AWB Full': 'AWB Full',
}

# PKL file patterns for 3 seeds
CONDITIONS = {
    'C1: Baseline': [
        'mnist_condition1_baseline_run0/mnist_condition1_baseline_run0_run0/classification_mnist_cnn_run0_records.pkl',
        'mnist_condition1_baseline_run1/mnist_condition1_baseline_run1_run1/classification_mnist_cnn_run1_records.pkl',
        'mnist_condition1_baseline_run2/mnist_condition1_baseline_run2_run2/classification_mnist_cnn_run2_records.pkl',
    ],
    'C2: Heuristics': [
        'mnist_condition2_heuristics_run0/mnist_condition2_heuristics_run0_run0/classification_mnist_cnn_run0_records.pkl',
        'mnist_condition2_heuristics_run1/mnist_condition2_heuristics_run1_run1/classification_mnist_cnn_run1_records.pkl',
        'mnist_condition2_heuristics_run2/mnist_condition2_heuristics_run2_run2/classification_mnist_cnn_run2_records.pkl',
    ],
    'C3: Arch Search': [
        'mnist_condition3_arch_no_transfer_run0/mnist_condition3_arch_no_transfer_run0_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
        'mnist_condition3_arch_no_transfer_run1/mnist_condition3_arch_no_transfer_run1_awb_run1/classification_mnist_cnn_awb_run1_records.pkl',
        'mnist_condition3_arch_no_transfer_run2/mnist_condition3_arch_no_transfer_run2_awb_run2/classification_mnist_cnn_awb_run2_records.pkl',
    ],
    'C4: AWB Full': [
        'mnist_condition4_awb_full_run0/mnist_condition4_awb_full_run0_run0/classification_mnist_cnn_awb_run0_records.pkl',
        'mnist_condition4_awb_full_run1/mnist_condition4_awb_full_run1_run1/classification_mnist_cnn_awb_run1_records.pkl',
        'mnist_condition4_awb_full_run2/mnist_condition4_awb_full_run2_run2/classification_mnist_cnn_awb_run2_records.pkl',
    ],
}

PAPER_FIG_DIR = Path("/Users/kraghavan/Desktop/JMLR_paper/Allyson-nonsmooth-dynamics/paperFigures")


def load_metrics_from_pkl(pkl_path):
    """Load metrics from pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    metadata = data['metadata']
    epochs_per_task = metadata.get('epochs_per_task', 150)
    n_tasks = metadata['n_tasks']

    metrics = defaultdict(list)
    iterations = []
    task_boundaries = []

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            continue

        task_data = data['tasks'][task_id]
        if task_id > 0:
            task_boundaries.append(task_id * epochs_per_task)

        if 'main_training' in task_data:
            training = task_data['main_training']
            metric_map = {
                'H': 'H',
                'V': 'V',
                'dV': 'dV',
                'grad_norm': 'grad_norm',
                'train_metric': 'train_metric',
                'test_current': 'test_cur',
                'test_experience': 'test_exp',
            }
            for pkl_key, metric_name in metric_map.items():
                if pkl_key in training:
                    metrics[metric_name].extend(training[pkl_key])
            if 'iterations' in training:
                iterations.extend(training['iterations'])

    return metrics, np.array(iterations), task_boundaries, n_tasks, data


def compute_cl_metrics(data):
    """Compute CL metrics from task_performance_matrix for classification (higher is better)."""
    if 'task_performance_matrix' not in data:
        return None

    tpm = data['task_performance_matrix']
    if isinstance(tpm, dict):
        n_tasks = len(tpm)
        matrix = np.zeros((n_tasks, n_tasks))
        for j in range(n_tasks):
            if j in tpm:
                for i_str, val in tpm[j].items():
                    matrix[j, int(i_str)] = val
    else:
        matrix = np.array(tpm)

    if matrix.size == 0:
        return None

    n_tasks = matrix.shape[0]

    # Average accuracy (final row)
    avg_acc = np.mean(matrix[-1, :])

    # BWT for classification: higher is better, so BWT = R_{T-1,i} - R_{i,i}
    # Positive BWT = improved on old tasks
    bwt_values = [matrix[-1, i] - matrix[i, i] for i in range(n_tasks - 1)]
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # Forgetting for classification: max performance - final performance
    forgetting_values = [max(0, np.max(matrix[:, i]) - matrix[-1, i]) for i in range(n_tasks - 1)]
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    # Forward transfer: compare initial performance on task i to random baseline
    # FWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{i-1,i} - baseline)
    # For simplicity, use random baseline of 0.1 (10% for 10-class)
    baseline = 0.1
    fwt_values = [matrix[i-1, i] - baseline for i in range(1, n_tasks)]
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    return {
        'Avg_Acc': avg_acc,
        'BWT': bwt,
        'Forgetting': forgetting,
        'FWT': fwt,
        'matrix': matrix
    }


def aggregate_curves(seed_data_list, metric_key):
    """Aggregate curves across seeds with mean and std."""
    all_values = []
    all_iters = []
    for iters, metrics in seed_data_list:
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            all_iters.append(iters)
            all_values.append(np.array(metrics[metric_key]))
    if not all_iters:
        return None, None, None

    min_len = min(len(v) for v in all_values)
    truncated_iters = all_iters[0][:min_len]
    truncated_values = np.array([v[:min_len] for v in all_values])
    mean_vals = np.mean(truncated_values, axis=0)
    std_vals = np.std(truncated_values, axis=0) if len(truncated_values) > 1 else np.zeros_like(mean_vals)
    return truncated_iters, mean_vals, std_vals


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')


def generate_main_6panel_figure(all_condition_data, agg_metrics, task_boundaries):
    """Generate main 6-panel figure (2 rows, 3 columns)."""
    print("\nGenerating main 6-panel figure...")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Panel (a): Test Accuracy (Experience Replay)
    ax = axes[0, 0]
    add_panel_label(ax, 'a')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for i, boundary in enumerate(task_boundaries[:5]):
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test Accuracy (Exp.)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (b): Hamiltonian Loss
    ax = axes[0, 1]
    add_panel_label(ax, 'b')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'H')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:5]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hamiltonian Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Gradient Norm
    ax = axes[0, 2]
    add_panel_label(ax, 'c')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'grad_norm')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:5]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (d): Average Accuracy Bar Chart
    ax = axes[1, 0]
    add_panel_label(ax, 'd')

    cond_names = [c for c in CONDITIONS.keys() if c in agg_metrics]
    x_pos = np.arange(len(cond_names))
    means = [agg_metrics[c]['Avg_Acc_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Avg_Acc_std'] for c in cond_names]
    colors_list = [COLORS[c]['line'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Avg Accuracy')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    # Panel (e): Backward Transfer Bar Chart
    ax = axes[1, 1]
    add_panel_label(ax, 'e')

    means = [agg_metrics[c]['BWT_mean'] for c in cond_names]
    stds = [agg_metrics[c]['BWT_std'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        offset = 0.005 if mean >= 0 else -0.01
        va = 'bottom' if mean >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
               f'{mean:.3f}', ha='center', va=va, fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('BWT (higher better)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    # Panel (f): Forgetting Bar Chart
    ax = axes[1, 2]
    add_panel_label(ax, 'f')

    means = [agg_metrics[c]['Forgetting_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Forgetting_std'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Forgetting (lower better)')
    ax.set_ylim(0, max(means) * 1.4 if max(means) > 0 else 0.1)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'mnist_10task_3seed_main.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'mnist_10task_3seed_main.pdf', facecolor='white')
    print(f"  Saved: mnist_10task_3seed_main.png/pdf")

    if PAPER_FIG_DIR.exists():
        fig.savefig(PAPER_FIG_DIR / 'mnist_10task_3seed_main.png', dpi=300, facecolor='white')
        fig.savefig(PAPER_FIG_DIR / 'mnist_10task_3seed_main.pdf', facecolor='white')
        print(f"  Also saved to: {PAPER_FIG_DIR}")

    plt.close()


def generate_training_curves_figure(all_condition_data, task_boundaries):
    """Generate detailed training curves figure."""
    print("\nGenerating training curves figure...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel (a): Test Accuracy (Experience)
    ax = axes[0, 0]
    add_panel_label(ax, 'a')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test Accuracy (Experience)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (b): Test Accuracy (Current Task)
    ax = axes[0, 1]
    add_panel_label(ax, 'b')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'test_cur')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test Accuracy (Current Task)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Train Metric
    ax = axes[1, 0]
    add_panel_label(ax, 'c')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'train_metric')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (d): Experience Replay Loss (V)
    ax = axes[1, 1]
    add_panel_label(ax, 'd')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'V')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Experience Replay Loss (V)')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'mnist_training_curves.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'mnist_training_curves.pdf', facecolor='white')
    print(f"  Saved: mnist_training_curves.png/pdf")

    plt.close()


def generate_loss_components_figure(all_condition_data, task_boundaries):
    """Generate loss components figure (H, V, dV, grad_norm)."""
    print("\nGenerating loss components figure...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics_config = [
        ('H', 'Hamiltonian Loss (H)', True, axes[0, 0], 'a'),
        ('V', 'Experience Replay Loss (V)', True, axes[0, 1], 'b'),
        ('dV', 'Regularization (dV)', True, axes[1, 0], 'c'),
        ('grad_norm', 'Gradient Norm', True, axes[1, 1], 'd'),
    ]

    for metric_key, ylabel, use_log, ax, label in metrics_config:
        add_panel_label(ax, label)

        for cond_name, seed_data_list in all_condition_data.items():
            iters, mean_vals, std_vals = aggregate_curves(seed_data_list, metric_key)
            if iters is not None:
                ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                       color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
                if np.any(std_vals > 0):
                    ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                                   color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

        for boundary in task_boundaries:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale('log')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'mnist_loss_components.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'mnist_loss_components.pdf', facecolor='white')
    print(f"  Saved: mnist_loss_components.png/pdf")

    plt.close()


def generate_performance_matrices(all_cl_metrics):
    """Generate performance matrix heatmaps for each condition."""
    print("\nGenerating performance matrix heatmaps...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (cond_name, seed_metrics) in enumerate(all_cl_metrics.items()):
        ax = axes[idx]

        # Average matrix across seeds
        matrices = [m['matrix'] for m in seed_metrics if 'matrix' in m]
        if matrices:
            avg_matrix = np.mean(matrices, axis=0)
            n_tasks = avg_matrix.shape[0]

            im = ax.imshow(avg_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

            # Add value annotations
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if j <= i:  # Only show lower triangle + diagonal
                        val = avg_matrix[i, j]
                        color = 'white' if val < 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=6, color=color)

            ax.set_xticks(range(n_tasks))
            ax.set_yticks(range(n_tasks))
            ax.set_xticklabels([f'T{i}' for i in range(n_tasks)], fontsize=7)
            ax.set_yticklabels([f'T{i}' for i in range(n_tasks)], fontsize=7)
            ax.set_xlabel('Task Evaluated On')
            ax.set_ylabel('After Training Task')
            ax.set_title(SHORT_LABELS[cond_name], fontsize=10, fontweight='bold')

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Accuracy', fontsize=10)

    fig.savefig(OUTPUT_DIR / 'mnist_performance_matrices.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'mnist_performance_matrices.pdf', facecolor='white')
    print(f"  Saved: mnist_performance_matrices.png/pdf")

    plt.close()


def generate_metrics_comparison_figure(agg_metrics):
    """Generate comprehensive metrics comparison bar chart."""
    print("\nGenerating metrics comparison figure...")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    cond_names = list(agg_metrics.keys())
    x_pos = np.arange(len(cond_names))
    colors_list = [COLORS[c]['line'] for c in cond_names]

    metrics_config = [
        ('Avg_Acc', 'Average Accuracy', 'higher better', axes[0]),
        ('BWT', 'Backward Transfer', 'higher better', axes[1]),
        ('Forgetting', 'Forgetting', 'lower better', axes[2]),
        ('FWT', 'Forward Transfer', 'higher better', axes[3]),
    ]

    for metric_key, title, direction, ax in metrics_config:
        means = [agg_metrics[c][f'{metric_key}_mean'] for c in cond_names]
        stds = [agg_metrics[c][f'{metric_key}_std'] for c in cond_names]

        bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                      edgecolor='black', linewidth=0.8, capsize=4,
                      error_kw={'elinewidth': 1.2, 'capthick': 1.2})

        for bar, mean in zip(bars, means):
            offset = max(stds) * 0.3 if mean >= 0 else -max(stds) * 0.5
            va = 'bottom' if mean >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                   f'{mean:.3f}', ha='center', va=va, fontsize=7, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(f'{title}\n({direction})')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'mnist_metrics_comparison.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'mnist_metrics_comparison.pdf', facecolor='white')
    print(f"  Saved: mnist_metrics_comparison.png/pdf")

    plt.close()


def print_metrics_table(agg_metrics):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print("MNIST 10-TASK CONTINUAL LEARNING METRICS (3 seeds)")
    print("="*80)
    print(f"{'Condition':<20} {'Avg Acc':>12} {'BWT':>12} {'Forgetting':>12} {'FWT':>12}")
    print("-"*80)

    for cond_name, metrics in agg_metrics.items():
        avg_acc = f"{metrics['Avg_Acc_mean']:.4f}±{metrics['Avg_Acc_std']:.4f}"
        bwt = f"{metrics['BWT_mean']:.4f}±{metrics['BWT_std']:.4f}"
        forg = f"{metrics['Forgetting_mean']:.4f}±{metrics['Forgetting_std']:.4f}"
        fwt = f"{metrics['FWT_mean']:.4f}±{metrics['FWT_std']:.4f}"
        print(f"{SHORT_LABELS[cond_name]:<20} {avg_acc:>12} {bwt:>12} {forg:>12} {fwt:>12}")

    print("="*80)


def main():
    """Generate all publication figures."""
    print("="*70)
    print("GENERATING MNIST 10-TASK 3-SEED PUBLICATION FIGURES")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_condition_data = {}
    all_cl_metrics = {}
    task_boundaries = None

    for cond_name, pkl_files in CONDITIONS.items():
        all_condition_data[cond_name] = []
        all_cl_metrics[cond_name] = []

        for pkl_file in pkl_files:
            pkl_path = DATA_DIR / pkl_file
            if pkl_path.exists():
                metrics, iters, boundaries, n_tasks, data = load_metrics_from_pkl(pkl_path)
                all_condition_data[cond_name].append((iters, dict(metrics)))
                if task_boundaries is None:
                    task_boundaries = boundaries
                cl_metrics = compute_cl_metrics(data)
                if cl_metrics:
                    all_cl_metrics[cond_name].append(cl_metrics)
                print(f"  Loaded: {pkl_file}")
            else:
                print(f"  WARNING: Not found: {pkl_file}")

    # Aggregate CL metrics across seeds
    agg_metrics = {}
    for cond_name, seed_metrics in all_cl_metrics.items():
        if seed_metrics:
            agg_metrics[cond_name] = {
                'Avg_Acc_mean': np.mean([m['Avg_Acc'] for m in seed_metrics]),
                'Avg_Acc_std': np.std([m['Avg_Acc'] for m in seed_metrics]),
                'BWT_mean': np.mean([m['BWT'] for m in seed_metrics]),
                'BWT_std': np.std([m['BWT'] for m in seed_metrics]),
                'Forgetting_mean': np.mean([m['Forgetting'] for m in seed_metrics]),
                'Forgetting_std': np.std([m['Forgetting'] for m in seed_metrics]),
                'FWT_mean': np.mean([m['FWT'] for m in seed_metrics]),
                'FWT_std': np.std([m['FWT'] for m in seed_metrics]),
            }

    # Print metrics table
    print_metrics_table(agg_metrics)

    # Generate all figures
    generate_main_6panel_figure(all_condition_data, agg_metrics, task_boundaries)
    generate_training_curves_figure(all_condition_data, task_boundaries)
    generate_loss_components_figure(all_condition_data, task_boundaries)
    generate_performance_matrices(all_cl_metrics)
    generate_metrics_comparison_figure(agg_metrics)

    print("\n" + "="*70)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
