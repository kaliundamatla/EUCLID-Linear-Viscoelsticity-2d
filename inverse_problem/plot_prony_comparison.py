"""
Create Prony series comparison plots: Identified vs Ground Truth.
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


def plot_prony_comparison(experiment_number: int, lambda_i: float = 0.0, lambda_b: float = 1.0):
    """
    Create Prony series comparison plots.

    Args:
        experiment_number: Experiment ID (800, 801, 802)
        lambda_i: Interior equation weight
        lambda_b: Boundary equation weight
    """
    # Load results
    base_dir = Path('./Postprocessing/final_outputs')
    result_dir = base_dir / f'{experiment_number}_nG150_nK150_tau1-600_li{lambda_i}_lb{lambda_b}_raw'
    result_file = result_dir / 'results.npz'

    if not result_file.exists():
        print(f"ERROR: Results file not found: {result_file}")
        return

    data = np.load(result_file, allow_pickle=True)

    # Extract raw NNLS results (unclustered)
    tau_G_all = data['tau_G']
    G_params = data['G_params']
    tau_K_all = data['tau_K']
    K_params = data['K_params']

    # Filter out zero parameters to get sparse solution
    G_mask = G_params > 0
    tau_G_nz = tau_G_all[G_mask]
    G_nz = G_params[G_mask]

    K_mask = K_params > 0
    tau_K_nz = tau_K_all[K_mask]
    K_nz = K_params[K_mask]

    # Set publication-quality style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Create figure with high DPI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

    # --- Plot 1: G_α (Deviatoric) ---
    # Ground truth (orange bars)
    ax1.bar(GROUND_TRUTH['tau_G'], GROUND_TRUTH['G'],
            width=GROUND_TRUTH['tau_G']*0.25,
            color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=2.0,
            label='Ground truth', zorder=3)

    # Identified parameters (blue bars)
    ax1.bar(tau_G_nz, G_nz,
            width=tau_G_nz*0.3,
            color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=2.0,
            label='Identified', zorder=3)

    # Inset for small values
    ax1_inset = ax1.inset_axes([0.12, 0.58, 0.35, 0.37])
    ax1_inset.bar(GROUND_TRUTH['tau_G'], GROUND_TRUTH['G'],
                  width=GROUND_TRUTH['tau_G']*0.25,
                  color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1_inset.bar(tau_G_nz, G_nz,
                  width=tau_G_nz*0.3,
                  color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1_inset.set_xlim(1, 10)
    ax1_inset.set_ylim(0, 200)
    ax1_inset.grid(True, alpha=0.4, linewidth=0.8, linestyle='--')
    ax1_inset.tick_params(labelsize=10, width=1.2)
    ax1_inset.spines['top'].set_linewidth(1.5)
    ax1_inset.spines['right'].set_linewidth(1.5)
    ax1_inset.spines['bottom'].set_linewidth(1.5)
    ax1_inset.spines['left'].set_linewidth(1.5)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\tau_{G\alpha}$ [s]', fontsize=16, fontweight='bold')
    ax1.set_ylabel(r'$G_{\alpha}$ [MPa]', fontsize=16, fontweight='bold')
    ax1.set_xlim(0.8, 650)
    ax1.set_ylim(0, 1250)

    # Set custom x-axis ticks with actual values
    ax1.set_xticks([1, 5, 10, 50, 100, 400, 600])
    ax1.set_xticklabels(['1', '5', '10', '50', '100', '400', '600'])

    ax1.grid(True, alpha=0.4, which='both', linewidth=0.8, linestyle='--')
    ax1.legend(fontsize=13, loc='upper right', framealpha=0.98, edgecolor='black', fancybox=False)
    ax1.tick_params(labelsize=12, width=1.5, length=6)

    # Thicker spines
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # --- Plot 2: K_α (Volumetric) ---
    # Ground truth (orange bars)
    ax2.bar(GROUND_TRUTH['tau_K'], GROUND_TRUTH['K'],
            width=GROUND_TRUTH['tau_K']*0.25,
            color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=2.0,
            label='Ground truth', zorder=3)

    # Identified parameters (blue bars)
    ax2.bar(tau_K_nz, K_nz,
            width=tau_K_nz*0.3,
            color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=2.0,
            label='Identified', zorder=3)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\tau_{K\alpha}$ [s]', fontsize=16, fontweight='bold')
    ax2.set_ylabel(r'$K_{\alpha}$ [MPa]', fontsize=16, fontweight='bold')
    ax2.set_xlim(0.8, 650)
    ax2.set_ylim(0, 850)

    # Set custom x-axis ticks with actual values
    ax2.set_xticks([1, 5, 10, 50, 100, 400, 600])
    ax2.set_xticklabels(['1', '5', '10', '50', '100', '400', '600'])

    ax2.grid(True, alpha=0.4, which='both', linewidth=0.8, linestyle='--')
    ax2.legend(fontsize=13, loc='upper right', framealpha=0.98, edgecolor='black', fancybox=False)
    ax2.tick_params(labelsize=12, width=1.5, length=6)

    # Thicker spines
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    # Title
    fig.suptitle(f'Prony Series Identification - Experiment {experiment_number} (λ_i={lambda_i}, λ_b={lambda_b})',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save with high quality
    output_file = result_dir / 'prony_series_comparison.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")

    plt.show()

    # Print summary
    print("\n" + "="*70)
    print(f"EXPERIMENT {experiment_number} - Prony Series Comparison")
    print("="*70)
    print(f"\nDeviatoric (G):")
    print(f"  Identified: {len(G_nz)} terms")
    print(f"    tau_G = {tau_G_nz.tolist()}")
    print(f"    G     = {G_nz.tolist()}")
    print(f"  Ground truth: {len(GROUND_TRUTH['G'])} terms")
    print(f"    tau_G = {GROUND_TRUTH['tau_G'].tolist()}")
    print(f"    G     = {GROUND_TRUTH['G'].tolist()}")

    print(f"\nVolumetric (K):")
    print(f"  Identified: {len(K_nz)} terms")
    print(f"    tau_K = {tau_K_nz.tolist()}")
    print(f"    K     = {K_nz.tolist()}")
    print(f"  Ground truth: {len(GROUND_TRUTH['K'])} terms")
    print(f"    tau_K = {GROUND_TRUTH['tau_K'].tolist()}")
    print(f"    K     = {GROUND_TRUTH['K'].tolist()}")
    print("="*70)


if __name__ == "__main__":
    # Create plots for all experiments
    experiments = [809]
    lambda_configs = [(1.0, 1.0)]

    for exp in experiments:
        for li, lb in lambda_configs:
            print(f"\n\nProcessing Experiment {exp} with lambda_i={li}, lambda_b={lb}...")
            plot_prony_comparison(exp, li, lb)
