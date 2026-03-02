"""
Script to read and analyze results.npz file from inverse problem.

Usage:
    python read_results_npz.py
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_results_npz(npz_file_path):
    """
    Read and display contents of results.npz file.

    Args:
        npz_file_path: Path to results.npz file
    """
    print("="*70)
    print("READING RESULTS.NPZ FILE")
    print("="*70)
    print(f"\nFile: {npz_file_path}")

    # Check if file exists
    if not Path(npz_file_path).exists():
        print(f"\n[ERROR] File not found: {npz_file_path}")
        return None

    # Load the .npz file
    data = np.load(npz_file_path, allow_pickle=True)

    print("\n" + "="*70)
    print("CONTENTS OF RESULTS.NPZ")
    print("="*70)

    # List all arrays in the file
    print(f"\nArrays in file: {list(data.keys())}")
    print(f"Number of arrays: {len(data.keys())}")

    # Display each array
    print("\n" + "="*70)
    print("ARRAY DETAILS")
    print("="*70)

    results = {}

    for key in data.keys():
        arr = data[key]
        results[key] = arr

        print(f"\n[{key}]")
        print(f"  Type: {type(arr)}")

        if isinstance(arr, np.ndarray):
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")

            # Show statistics for numeric arrays
            if np.issubdtype(arr.dtype, np.number):
                print(f"  Min: {arr.min():.6e}")
                print(f"  Max: {arr.max():.6e}")
                print(f"  Mean: {arr.mean():.6e}")
                print(f"  Std: {arr.std():.6e}")
                print(f"  Non-zero: {np.count_nonzero(arr)}/{arr.size}")

                # Show first few values if 1D
                if arr.ndim == 1 and len(arr) <= 20:
                    print(f"  Values: {arr}")
                elif arr.ndim == 1:
                    print(f"  First 5: {arr[:5]}")
                    print(f"  Last 5: {arr[-5:]}")
        else:
            print(f"  Value: {arr}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Common expected arrays
    expected_keys = ['theta', 'tau_G', 'tau_K', 'G_params', 'K_params',
                     'G_inf', 'K_inf', 'residual_norm', 'cost', 'metadata']

    print("\nExpected arrays:")
    for key in expected_keys:
        status = "✓" if key in results else "✗"
        print(f"  {status} {key}")

    return results


def plot_prony_series(results, save_dir=None):
    """
    Plot identified Prony series parameters.

    Args:
        results: Dictionary from read_results_npz()
        save_dir: Optional directory to save plots
    """
    print("\n" + "="*70)
    print("PLOTTING PRONY SERIES")
    print("="*70)

    # Extract parameters
    tau_G = results.get('tau_G')
    tau_K = results.get('tau_K')
    G_params = results.get('G_params')
    K_params = results.get('K_params')
    G_inf = results.get('G_inf')
    K_inf = results.get('K_inf')

    if tau_G is None or G_params is None:
        print("\n[ERROR] Required arrays not found in results")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Shear modulus G
    ax1.semilogx(tau_G, G_params, 'o-', markersize=8, linewidth=2,
                 label=f'G_i (non-zero: {np.count_nonzero(G_params)}/{len(G_params)})')

    if G_inf is not None and G_inf > 0:
        ax1.axhline(G_inf, color='red', linestyle='--', linewidth=2,
                    label=f'G_∞ = {G_inf:.2f} MPa')

    ax1.set_xlabel('Relaxation Time τ_G [s]', fontsize=12)
    ax1.set_ylabel('Shear Modulus G_i [MPa]', fontsize=12)
    ax1.set_title('Shear Prony Series', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)

    # Plot 2: Bulk modulus K
    if tau_K is not None and K_params is not None:
        ax2.semilogx(tau_K, K_params, 's-', markersize=8, linewidth=2, color='orange',
                     label=f'K_i (non-zero: {np.count_nonzero(K_params)}/{len(K_params)})')

        if K_inf is not None and K_inf > 0:
            ax2.axhline(K_inf, color='red', linestyle='--', linewidth=2,
                        label=f'K_∞ = {K_inf:.2f} MPa')

        ax2.set_xlabel('Relaxation Time τ_K [s]', fontsize=12)
        ax2.set_ylabel('Bulk Modulus K_i [MPa]', fontsize=12)
        ax2.set_title('Bulk Prony Series', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'prony_series.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {save_path}")

    plt.show()


def plot_relaxation_modulus(results, save_dir=None):
    """
    Plot relaxation modulus G(t) and K(t) from Prony parameters.

    Args:
        results: Dictionary from read_results_npz()
        save_dir: Optional directory to save plots
    """
    print("\n" + "="*70)
    print("PLOTTING RELAXATION MODULI")
    print("="*70)

    # Extract parameters
    tau_G = results.get('tau_G')
    tau_K = results.get('tau_K')
    G_params = results.get('G_params')
    K_params = results.get('K_params')
    G_inf = results.get('G_inf', 0)
    K_inf = results.get('K_inf', 0)

    if tau_G is None or G_params is None:
        print("\n[ERROR] Required arrays not found")
        return

    # Time vector for plotting
    t = np.logspace(-2, 4, 1000)  # 0.01s to 10000s

    # Calculate G(t) = G_∞ + Σ G_i * exp(-t/τ_i)
    G_t = np.ones_like(t) * G_inf
    for i, (tau_i, G_i) in enumerate(zip(tau_G, G_params)):
        if G_i > 0:  # Only non-zero terms
            G_t += G_i * np.exp(-t / tau_i)

    # Calculate K(t) = K_∞ + Σ K_i * exp(-t/τ_i)
    if tau_K is not None and K_params is not None:
        K_t = np.ones_like(t) * K_inf
        for i, (tau_i, K_i) in enumerate(zip(tau_K, K_params)):
            if K_i > 0:
                K_t += K_i * np.exp(-t / tau_i)
    else:
        K_t = None

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: G(t)
    ax1.semilogx(t, G_t, 'b-', linewidth=2.5, label='G(t)')
    ax1.axhline(G_inf, color='red', linestyle='--', linewidth=1.5,
                label=f'G_∞ = {G_inf:.2f} MPa', alpha=0.7)
    ax1.axhline(G_t[0], color='green', linestyle='--', linewidth=1.5,
                label=f'G(0) = {G_t[0]:.2f} MPa', alpha=0.7)

    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Shear Relaxation Modulus G(t) [MPa]', fontsize=12)
    ax1.set_title('Shear Relaxation Function', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)

    # Plot 2: K(t)
    if K_t is not None:
        ax2.semilogx(t, K_t, 'orange', linewidth=2.5, label='K(t)')
        ax2.axhline(K_inf, color='red', linestyle='--', linewidth=1.5,
                    label=f'K_∞ = {K_inf:.2f} MPa', alpha=0.7)
        ax2.axhline(K_t[0], color='green', linestyle='--', linewidth=1.5,
                    label=f'K(0) = {K_t[0]:.2f} MPa', alpha=0.7)

        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Bulk Relaxation Modulus K(t) [MPa]', fontsize=12)
        ax2.set_title('Bulk Relaxation Function', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'relaxation_moduli.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Path to results.npz
    RESULTS_FILE = Path("Postprocessing/final_outputs/904_nG150_nK150_tau1-1200_li0.0_lb1.0_raw/results.npz")

    # Read results
    results = read_results_npz(RESULTS_FILE)

    if results is not None:
        # Plot Prony series
        plot_prony_series(results, save_dir=RESULTS_FILE.parent)

        # Plot relaxation moduli
        plot_relaxation_modulus(results, save_dir=RESULTS_FILE.parent)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
