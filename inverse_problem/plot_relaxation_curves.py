"""
Create relaxation curve comparison plots: Identified vs Ground Truth.
Shows G(t) and K(t) decay curves over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ground truth material (MAT3.5)
GROUND_TRUTH = {
    'G': np.array([200, 500, 1000]),  # MPa
    'tau_G': np.array([5.3, 50.1, 400.2]),  # s
    'G_inf': 1500,  # MPa
    'K': np.array([500, 700, 567]),  # MPa
    'tau_K': np.array([5.3, 50.1, 400.2]),  # s
    'K_inf': 2000,  # MPa
}


def compute_relaxation_modulus(t, G_params, tau_G, G_inf):
    """
    Compute G(t) = G_inf + sum(G_i * exp(-t/tau_i))

    Args:
        t: Time array [s]
        G_params: Prony moduli [MPa]
        tau_G: Relaxation times [s]
        G_inf: Equilibrium modulus [MPa]

    Returns:
        G(t) at each time point
    """
    G_t = G_inf * np.ones_like(t)

    for G_i, tau_i in zip(G_params, tau_G):
        if G_i > 0:  # Only include non-zero terms
            G_t += G_i * np.exp(-t / tau_i)

    return G_t


def plot_relaxation_comparison(experiment_number: int, lambda_i: float = 0.0, lambda_b: float = 1.0):
    """
    Create relaxation curve comparison plots.

    Args:
        experiment_number: Experiment ID (800, 801, 802)
        lambda_i: Interior equation weight
        lambda_b: Boundary equation weight
    """
    # Load results
    base_dir = Path('./Postprocessing/final_outputs')
    result_dir = base_dir / f'{experiment_number}_nG{150}_nK{150}_tau1-600_li{lambda_i}_lb{lambda_b}_raw'
    result_file = result_dir / 'results.npz'

    if not result_file.exists():
        print(f"ERROR: Results file not found: {result_file}")
        return

    data = np.load(result_file, allow_pickle=True)

    # Extract raw NNLS results (before clustering) for curves
    if 'G_raw_nonzero' in data.keys():
        # Use raw unclustered data for continuous curves
        G_params_raw = data['G_raw_nonzero']
        tau_G_raw = data['tau_G_raw_nonzero']
        K_params_raw = data['K_raw_nonzero']
        tau_K_raw = data['tau_K_raw_nonzero']

        # Get clustered data for bars
        G_params_clustered = data['G_nonzero']
        tau_G_clustered = data['tau_G_nonzero']
        K_params_clustered = data['K_nonzero']
        tau_K_clustered = data['tau_K_nonzero']

        G_inf = data['G_inf']
        K_inf = data['K_inf']
        has_clustering = True
        title_suffix = "(Raw curve + Clustered bars)"
    else:
        # Fallback to non-zero filtered data
        tau_G_all = data['tau_G']
        G_params_all = data['G_params']
        tau_K_all = data['tau_K']
        K_params_all = data['K_params']

        # Filter zeros
        G_mask = G_params_all > 0
        G_params_raw = G_params_all[G_mask]
        tau_G_raw = tau_G_all[G_mask]

        K_mask = K_params_all > 0
        K_params_raw = K_params_all[K_mask]
        tau_K_raw = tau_K_all[K_mask]

        G_inf = data['G_inf']
        K_inf = data['K_inf']
        has_clustering = False
        title_suffix = "(NNLS Solution)"

    # Create time vector (logarithmic spacing)
    t = np.logspace(-2, 3.5, 500)  # 0.01s to ~3162s

    # Compute relaxation curves - Ground truth
    G_t_true = compute_relaxation_modulus(t, GROUND_TRUTH['G'], GROUND_TRUTH['tau_G'], GROUND_TRUTH['G_inf'])
    K_t_true = compute_relaxation_modulus(t, GROUND_TRUTH['K'], GROUND_TRUTH['tau_K'], GROUND_TRUTH['K_inf'])

    # Compute relaxation curves - Identified (using raw unclustered data)
    G_t_ident = compute_relaxation_modulus(t, G_params_raw, tau_G_raw, G_inf)
    K_t_ident = compute_relaxation_modulus(t, K_params_raw, tau_K_raw, K_inf)

    # Set publication-quality style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # --- Plot 1: Deviatoric Relaxation G(t) ---
    ax1.plot(t, G_t_true, 'b--', linewidth=3.0, label='Ground truth', zorder=3)
    ax1.plot(t, G_t_ident, 'r-', linewidth=2.5, label='Identified', zorder=2)

    # Mark relaxation times on ground truth curve
    for i, tau_i in enumerate(GROUND_TRUTH['tau_G']):
        G_at_tau = compute_relaxation_modulus(tau_i, GROUND_TRUTH['G'], GROUND_TRUTH['tau_G'], GROUND_TRUTH['G_inf'])
        ax1.plot(tau_i, G_at_tau, 'bo', markersize=8, zorder=4)
        ax1.text(tau_i*1.3, G_at_tau, f'τ_{i+1}', fontsize=11, fontweight='bold', color='blue')

    # Mark relaxation times on identified curve (raw data - red dots)
    for tau_i in tau_G_raw:
        G_at_tau_ident = compute_relaxation_modulus(tau_i, G_params_raw, tau_G_raw, G_inf)
        ax1.plot(tau_i, G_at_tau_ident, 'ro', markersize=8, zorder=4)

    # Add vertical bars for clustered parameters (if available)
    if has_clustering:
        for i, (tau_c, G_c) in enumerate(zip(tau_G_clustered, G_params_clustered)):
            if tau_c > 0:  # Skip the equilibrium term (tau=0)
                # Compute height of the curve at this tau
                G_at_tau = compute_relaxation_modulus(tau_c, G_params_raw, tau_G_raw, G_inf)
                # Draw vertical bar from bottom to the curve
                ax1.vlines(tau_c, 0, G_at_tau, color='darkred', linewidth=2.5, alpha=0.7,
                          linestyle='-', zorder=5, label='Clustered' if i == 1 else '')

    ax1.set_xscale('log')
    ax1.set_xlabel('tau [s]', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Shear Modulus G(t) [MPa]', fontsize=16, fontweight='bold')
    ax1.set_xlim(1e-2, 2e3)
    # Dynamic y-limit: add margin above max G_0
    G_0_true = GROUND_TRUTH['G_inf'] + np.sum(GROUND_TRUTH['G'])
    G_0_ident = np.sum(G_params_raw)  # G_params_raw already includes G_inf as first element
    G_0_max = max(G_0_true, G_0_ident)
    ax1.set_ylim(1400, G_0_max + 300)
    ax1.set_title('Deviatoric Relaxation', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linewidth=0.8, linestyle='-', zorder=0)

    # Add horizontal lines for G_inf and G_0 AFTER grid (so they appear on top)
    ax1.axhline(GROUND_TRUTH['G_inf'], color='blue', linestyle='--', linewidth=2.0, alpha=0.7,
                label='$G_\\infty$ (true)', zorder=1)
    ax1.axhline(G_0_true, color='blue', linestyle=':', linewidth=2.5, alpha=0.8,
                label='$G_0$ (true)', zorder=1)
    ax1.axhline(G_inf, color='red', linestyle='--', linewidth=2.0, alpha=0.7,
                label='$G_\\infty$ (ident)', zorder=1)
    ax1.axhline(G_0_ident, color='red', linestyle=':', linewidth=2.5, alpha=0.8,
                label='$G_0$ (ident)', zorder=1)
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.92, edgecolor='black', ncol=2)
    ax1.tick_params(labelsize=12, width=1.5, length=6)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # --- Plot 2: Volumetric Relaxation K(t) ---
    ax2.plot(t, K_t_true, 'b--', linewidth=3.0, label='Ground truth', zorder=3)
    ax2.plot(t, K_t_ident, 'r-', linewidth=2.5, label='Identified', zorder=2)

    # Mark relaxation times on ground truth curve
    for i, tau_i in enumerate(GROUND_TRUTH['tau_K']):
        K_at_tau = compute_relaxation_modulus(tau_i, GROUND_TRUTH['K'], GROUND_TRUTH['tau_K'], GROUND_TRUTH['K_inf'])
        ax2.plot(tau_i, K_at_tau, 'bo', markersize=8, zorder=4)
        ax2.text(tau_i*1.3, K_at_tau, f'τ_{i+1}', fontsize=11, fontweight='bold', color='blue')

    # Mark relaxation times on identified curve (raw data - red dots)
    for tau_i in tau_K_raw:
        K_at_tau_ident = compute_relaxation_modulus(tau_i, K_params_raw, tau_K_raw, K_inf)
        ax2.plot(tau_i, K_at_tau_ident, 'ro', markersize=8, zorder=4)

    # Add vertical bars for clustered parameters (if available)
    if has_clustering:
        for i, (tau_c, K_c) in enumerate(zip(tau_K_clustered, K_params_clustered)):
            if tau_c > 0:  # Skip the equilibrium term (tau=0)
                # Compute height of the curve at this tau
                K_at_tau = compute_relaxation_modulus(tau_c, K_params_raw, tau_K_raw, K_inf)
                # Draw vertical bar from bottom to the curve
                ax2.vlines(tau_c, 0, K_at_tau, color='darkred', linewidth=2.5, alpha=0.7,
                          linestyle='-', zorder=5, label='Clustered' if i == 1 else '')

    ax2.set_xscale('log')
    ax2.set_xlabel('tau [s]', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Bulk Modulus K(t) [MPa]', fontsize=16, fontweight='bold')
    ax2.set_xlim(1e-2, 2e3)
    # Dynamic y-limit: add margin above max K_0
    K_0_true = GROUND_TRUTH['K_inf'] + np.sum(GROUND_TRUTH['K'])
    K_0_ident = np.sum(K_params_raw)  # K_params_raw already includes K_inf as first element
    K_0_max = max(K_0_true, K_0_ident)
    ax2.set_ylim(1900, K_0_max + 300)
    ax2.set_title('Volumetric Relaxation', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linewidth=0.8, linestyle='-', zorder=0)

    # Add horizontal lines for K_inf and K_0 AFTER grid (so they appear on top)
    ax2.axhline(GROUND_TRUTH['K_inf'], color='blue', linestyle='--', linewidth=2.0, alpha=0.7,
                label='$K_\\infty$ (true)', zorder=1)
    ax2.axhline(K_0_true, color='blue', linestyle=':', linewidth=2.5, alpha=0.8,
                label='$K_0$ (true)', zorder=1)
    ax2.axhline(K_inf, color='red', linestyle='--', linewidth=2.0, alpha=0.7,
                label='$K_\\infty$ (ident)', zorder=1)
    ax2.axhline(K_0_ident, color='red', linestyle=':', linewidth=2.5, alpha=0.8,
                label='$K_0$ (ident)', zorder=1)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.92, edgecolor='black', ncol=2)
    ax2.tick_params(labelsize=12, width=1.5, length=6)

    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    # Overall title
    fig.suptitle(f'Relaxation Moduli - Experiment {experiment_number} ' +
                 f'(λ_i={lambda_i}, λ_b={lambda_b}) {title_suffix}',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save with high quality
    output_file = result_dir / 'relaxation_curves_comparison.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print(f"EXPERIMENT {experiment_number} - Relaxation Curves")
    print("="*70)
    print(f"\nRaw identified parameters (before clustering):")
    print(f"  G: {len(G_params_raw)} terms, tau_G: {tau_G_raw.tolist()}")
    print(f"  K: {len(K_params_raw)} terms, tau_K: {tau_K_raw.tolist()}")
    if has_clustering:
        print(f"\nClustered parameters (after):")
        print(f"  G: {len(G_params_clustered)} terms, tau_G: {tau_G_clustered.tolist()}")
        print(f"  K: {len(K_params_clustered)} terms, tau_K: {tau_K_clustered.tolist()}")
    print("="*70)


def plot_multi_experiment_comparison(lambda_i: float = 0.0, lambda_b: float = 1.0):
    """
    Create combined relaxation curve comparison for experiments 800, 801, 802.

    Overlays all three experiments on the same plot with ground truth.

    Args:
        lambda_i: Interior equation weight
        lambda_b: Boundary equation weight
    """
    # Experiments to compare
    experiments = [800, 801, 802]

    # Colors and line styles for each experiment
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    linestyles = ['-', '--', '-.']  # Solid, Dashed, Dash-dot
    linewidths = [3.0, 2.8, 2.6]  # Slightly different widths

    # Load data for all experiments
    all_data = {}
    base_dir = Path('./Postprocessing/final_outputs')

    for exp in experiments:
        result_dir = base_dir / f'{exp}_nG150_nK150_tau1-600_li{lambda_i}_lb{lambda_b}_raw'
        result_file = result_dir / 'results.npz'

        if not result_file.exists():
            print(f"WARNING: Results file not found for experiment {exp}: {result_file}")
            continue

        data = np.load(result_file, allow_pickle=True)

        # Extract raw NNLS results
        if 'G_raw_nonzero' in data.keys():
            G_params_raw = data['G_raw_nonzero']
            tau_G_raw = data['tau_G_raw_nonzero']
            K_params_raw = data['K_raw_nonzero']
            tau_K_raw = data['tau_K_raw_nonzero']
            G_inf = data['G_inf']
            K_inf = data['K_inf']
        else:
            # Fallback
            tau_G_all = data['tau_G']
            G_params_all = data['G_params']
            tau_K_all = data['tau_K']
            K_params_all = data['K_params']

            G_mask = G_params_all > 0
            G_params_raw = G_params_all[G_mask]
            tau_G_raw = tau_G_all[G_mask]

            K_mask = K_params_all > 0
            K_params_raw = K_params_all[K_mask]
            tau_K_raw = tau_K_all[K_mask]

            G_inf = data['G_inf']
            K_inf = data['K_inf']

        all_data[exp] = {
            'G_params': G_params_raw,
            'tau_G': tau_G_raw,
            'K_params': K_params_raw,
            'tau_K': tau_K_raw,
            'G_inf': G_inf,
            'K_inf': K_inf
        }

    if len(all_data) == 0:
        print("ERROR: No data loaded for any experiment!")
        return

    # Create time vector (logarithmic spacing)
    t = np.logspace(-2, 3.5, 500)  # 0.01s to ~3162s

    # Compute ground truth curves
    G_t_true = compute_relaxation_modulus(t, GROUND_TRUTH['G'], GROUND_TRUTH['tau_G'], GROUND_TRUTH['G_inf'])
    K_t_true = compute_relaxation_modulus(t, GROUND_TRUTH['K'], GROUND_TRUTH['tau_K'], GROUND_TRUTH['K_inf'])

    # Set publication-quality style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # --- Plot 1: Deviatoric Relaxation G(t) ---
    # Ground truth
    ax1.plot(t, G_t_true, 'k:', linewidth=4.0, label='Ground truth', zorder=15, alpha=0.8)

    # Mark ground truth relaxation times
    for i, tau_i in enumerate(GROUND_TRUTH['tau_G']):
        G_at_tau = compute_relaxation_modulus(tau_i, GROUND_TRUTH['G'], GROUND_TRUTH['tau_G'], GROUND_TRUTH['G_inf'])
        ax1.plot(tau_i, G_at_tau, 'ko', markersize=10, zorder=16, markerfacecolor='white', markeredgewidth=2)
        ax1.text(tau_i*1.3, G_at_tau, f'τ_{i+1}', fontsize=11, fontweight='bold', color='black')

    # Plot identified curves for each experiment (reverse order so 800 is on top)
    for idx, (exp, color, ls, lw) in enumerate(zip(reversed(experiments), reversed(colors),
                                                     reversed(linestyles), reversed(linewidths))):
        if exp not in all_data:
            continue

        data = all_data[exp]
        G_t_ident = compute_relaxation_modulus(t, data['G_params'], data['tau_G'], data['G_inf'])
        ax1.plot(t, G_t_ident, color=color, linestyle=ls, linewidth=lw,
                label=f'Exp {exp}', zorder=10-idx, alpha=0.95)

        # Mark relaxation times with distinct markers
        markers = ['s', '^', 'D']  # Square, Triangle, Diamond
        marker = markers[experiments.index(exp)]
        for tau_i in data['tau_G']:
            if tau_i > 0:  # Skip tau=0 (G_inf)
                G_at_tau = compute_relaxation_modulus(tau_i, data['G_params'], data['tau_G'], data['G_inf'])
                ax1.plot(tau_i, G_at_tau, marker=marker, color=color, markersize=7,
                        zorder=10-idx, alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)

    ax1.set_xscale('log')
    ax1.set_xlabel('tau [s]', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Shear Modulus G(t) [MPa]', fontsize=16, fontweight='bold')
    ax1.set_xlim(1e-2, 2e3)
    ax1.set_ylim(1400, 3500)
    ax1.set_title('Deviatoric Relaxation - All Experiments', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linewidth=0.8, linestyle='-', zorder=0)

    # Add horizontal reference lines
    G_0_true = GROUND_TRUTH['G_inf'] + np.sum(GROUND_TRUTH['G'])
    ax1.axhline(GROUND_TRUTH['G_inf'], color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax1.axhline(G_0_true, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)

    ax1.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black')
    ax1.tick_params(labelsize=12, width=1.5, length=6)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # --- Plot 2: Volumetric Relaxation K(t) ---
    # Ground truth
    ax2.plot(t, K_t_true, 'k:', linewidth=4.0, label='Ground truth', zorder=15, alpha=0.8)

    # Mark ground truth relaxation times
    for i, tau_i in enumerate(GROUND_TRUTH['tau_K']):
        K_at_tau = compute_relaxation_modulus(tau_i, GROUND_TRUTH['K'], GROUND_TRUTH['tau_K'], GROUND_TRUTH['K_inf'])
        ax2.plot(tau_i, K_at_tau, 'ko', markersize=10, zorder=16, markerfacecolor='white', markeredgewidth=2)
        ax2.text(tau_i*1.3, K_at_tau, f'τ_{i+1}', fontsize=11, fontweight='bold', color='black')

    # Plot identified curves for each experiment (reverse order so 800 is on top)
    for idx, (exp, color, ls, lw) in enumerate(zip(reversed(experiments), reversed(colors),
                                                     reversed(linestyles), reversed(linewidths))):
        if exp not in all_data:
            continue

        data = all_data[exp]
        K_t_ident = compute_relaxation_modulus(t, data['K_params'], data['tau_K'], data['K_inf'])
        ax2.plot(t, K_t_ident, color=color, linestyle=ls, linewidth=lw,
                label=f'Exp {exp}', zorder=10-idx, alpha=0.95)

        # Mark relaxation times with distinct markers
        markers = ['s', '^', 'D']  # Square, Triangle, Diamond
        marker = markers[experiments.index(exp)]
        for tau_i in data['tau_K']:
            if tau_i > 0:  # Skip tau=0 (K_inf)
                K_at_tau = compute_relaxation_modulus(tau_i, data['K_params'], data['tau_K'], data['K_inf'])
                ax2.plot(tau_i, K_at_tau, marker=marker, color=color, markersize=7,
                        zorder=10-idx, alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_xscale('log')
    ax2.set_xlabel('tau [s]', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Bulk Modulus K(t) [MPa]', fontsize=16, fontweight='bold')
    ax2.set_xlim(1e-2, 2e3)
    ax2.set_ylim(1900, 4000)
    ax2.set_title('Volumetric Relaxation - All Experiments', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linewidth=0.8, linestyle='-', zorder=0)

    # Add horizontal reference lines
    K_0_true = GROUND_TRUTH['K_inf'] + np.sum(GROUND_TRUTH['K'])
    ax2.axhline(GROUND_TRUTH['K_inf'], color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax2.axhline(K_0_true, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)

    ax2.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black')
    ax2.tick_params(labelsize=12, width=1.5, length=6)

    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    # Overall title
    fig.suptitle(f'Relaxation Moduli - Experiments 800, 801, 802 (λ_i={lambda_i}, λ_b={lambda_b})',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save with high quality
    output_dir = base_dir / 'combined_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'relaxation_curves_all_experiments_li{lambda_i}_lb{lambda_b}.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved combined plot: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print(f"COMBINED COMPARISON - Experiments 800, 801, 802 (λ_i={lambda_i}, λ_b={lambda_b})")
    print("="*70)
    for exp in experiments:
        if exp in all_data:
            data = all_data[exp]
            print(f"\nExperiment {exp}:")
            print(f"  G_inf: {data['G_inf']:.2f} MPa")
            print(f"  K_inf: {data['K_inf']:.2f} MPa")
            print(f"  G terms: {len(data['G_params'])}")
            print(f"  K terms: {len(data['K_params'])}")
    print("="*70)


if __name__ == "__main__":
    # Configuration for a single experiment
    experiment_number = 809  # Change this to 800, 801, or 802
    lambda_i = 1.0  # Interior equation weight
    lambda_b = 1.0  # Boundary equation weight

    print(f"\n\nProcessing Experiment {experiment_number} with lambda_i={lambda_i}, lambda_b={lambda_b}...")
    plot_relaxation_comparison(experiment_number, lambda_i, lambda_b)
