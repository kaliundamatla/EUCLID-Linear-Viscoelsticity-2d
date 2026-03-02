"""
Comprehensive analysis linking strain heterogeneity to identification error.

This script:
1. Computes strain heterogeneity metrics (H_dev, H_vol) for experiments 902, 903, 904
2. Links these to the verification error from the inverse problem
3. Creates publication-quality figures for thesis

Key insight: More geometric heterogeneity → More strain field heterogeneity
           → Better identifiability → Lower error
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inverse_problem.core.geometry import Triangle3Node, Node


def setup_thesis_style():
    """Setup matplotlib style for thesis-quality figures."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.linewidth'] = 0.8


def load_experiment_data(experiment_dir: Path):
    """Load experimental/synthetic DIC data."""
    import pandas as pd

    data_dir = Path(experiment_dir)

    # Load coordinates
    coord_file = data_dir / "coord.csv"
    nodes = pd.read_csv(coord_file, header=None, skiprows=1)
    coord = nodes.values

    # Load connectivity
    conne_file = data_dir / "conne.txt"
    data = []
    with open(conne_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = re.split(r'[,\s]+', line)
            try:
                nums = [int(c) for c in cols if c]
                if len(nums) >= 4:
                    data.append(nums[:4])
            except (ValueError, IndexError):
                continue

    conne_raw = pd.DataFrame(data)
    conne = conne_raw.iloc[:, 1:4].values.astype(np.int64) - 1

    # Load displacements
    u_file = data_dir / "U.csv"
    U = pd.read_csv(u_file, header=None).values

    # Load time
    time_file = data_dir / "time.csv"
    time = pd.read_csv(time_file, header=None).values.flatten()

    return {
        'coord': coord,
        'conne': conne,
        'U': U,
        'time': time,
        'nnodes': coord.shape[0],
        'nelements': conne.shape[0]
    }


def compute_element_strains(U_timestep, coord, conne, n_nodes):
    """Compute strain fields at element centroids."""
    n_elements = len(conne)
    nodes_xy = coord[:, 1:3]

    Ux = U_timestep[:n_nodes]
    Uy = U_timestep[n_nodes:]

    eps_xx = np.zeros(n_elements)
    eps_yy = np.zeros(n_elements)
    gamma_xy = np.zeros(n_elements)

    for ie in range(n_elements):
        node_ids = conne[ie]
        nodes = [Node(int(node_ids[i]), nodes_xy[node_ids[i], 0], nodes_xy[node_ids[i], 1])
                 for i in range(3)]
        elem = Triangle3Node(ie, nodes)

        u_elem = np.zeros(6)
        for i, nid in enumerate(node_ids):
            u_elem[2*i] = Ux[nid]
            u_elem[2*i+1] = Uy[nid]

        strain = elem.Be @ u_elem
        eps_xx[ie] = strain[0]
        eps_yy[ie] = strain[1]
        gamma_xy[ie] = strain[2]

    return {'eps_xx': eps_xx, 'eps_yy': eps_yy, 'gamma_xy': gamma_xy}


def compute_J2_strain(eps_xx, eps_yy, gamma_xy):
    """Compute deviatoric strain intensity (J2)."""
    eps_mean = (eps_xx + eps_yy) / 2.0
    e_xx = eps_xx - eps_mean
    e_yy = eps_yy - eps_mean
    e_zz = -(e_xx + e_yy)
    J2 = np.sqrt(0.5 * (e_xx**2 + e_yy**2 + e_zz**2) + (gamma_xy/2)**2)
    return J2


def compute_heterogeneity_metrics(field):
    """Compute heterogeneity index H = std/mean."""
    mean_abs = np.mean(np.abs(field))
    std_field = np.std(field)
    return std_field / mean_abs if mean_abs > 1e-12 else 0.0


def analyze_geometry(experiment_dir: Path, time_fraction: float = 0.5):
    """
    Analyze strain heterogeneity at a given time fraction.

    Args:
        experiment_dir: Path to experiment data
        time_fraction: Fraction of total time (0.0 to 1.0)
    """
    print(f"\nAnalyzing: {experiment_dir.name}")

    data = load_experiment_data(experiment_dir)

    # Use time fraction for consistent comparison
    t_idx = int(time_fraction * (len(data['time']) - 1))
    t_actual = data['time'][t_idx]
    print(f"  Using timestep {t_idx}, t = {t_actual:.1f} s (fraction = {time_fraction})")

    U_t = data['U'][:, t_idx]

    strains = compute_element_strains(U_t, data['coord'], data['conne'], data['nnodes'])

    J2 = compute_J2_strain(strains['eps_xx'], strains['eps_yy'], strains['gamma_xy'])
    eps_vol = strains['eps_xx'] + strains['eps_yy']

    H_dev = compute_heterogeneity_metrics(J2)
    H_vol = compute_heterogeneity_metrics(eps_vol)

    print(f"  H_dev = {H_dev:.4f}, H_vol = {H_vol:.4f}")

    # Create triangulation for plotting
    nodes_xy = data['coord'][:, 1:3]
    triang = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], data['conne'])

    # Element to nodal averaging for smooth plotting
    def element_to_nodal(element_values, conne, n_nodes):
        nodal_sum = np.zeros(n_nodes)
        nodal_count = np.zeros(n_nodes)
        for ie, elem_nodes in enumerate(conne):
            for node_id in elem_nodes:
                nodal_sum[node_id] += element_values[ie]
                nodal_count[node_id] += 1
        nodal_count[nodal_count == 0] = 1
        return nodal_sum / nodal_count

    J2_nodal = element_to_nodal(J2, data['conne'], data['nnodes'])
    eps_vol_nodal = element_to_nodal(eps_vol, data['conne'], data['nnodes'])

    return {
        'H_dev': H_dev,
        'H_vol': H_vol,
        'J2': J2,
        'eps_vol': eps_vol,
        'J2_nodal': J2_nodal,
        'eps_vol_nodal': eps_vol_nodal,
        'triangulation': triang,
        't_actual': t_actual,
        'data': data
    }


def plot_heterogeneity_vs_error(results: dict, errors: dict, save_dir: Path):
    """
    Create correlation plot between heterogeneity and verification error.

    Fig 5.x: The key figure linking heterogeneity to identifiability.
    """
    setup_thesis_style()

    geometries = ['902', '903', '904']
    geometry_labels = {
        '902': 'Solid\n(no holes)',
        '903': '1 Ellipse\nhole',
        '904': '3 Circular\nholes'
    }
    colors = {'902': '#e74c3c', '903': '#f39c12', '904': '#27ae60'}
    markers = {'902': 's', '903': 'o', '904': '^'}

    H_dev = [results[g]['H_dev'] for g in geometries]
    H_vol = [results[g]['H_vol'] for g in geometries]
    error = [errors[g] for g in geometries]

    # Combined heterogeneity metric
    H_combined = [(h_d + h_v) / 2 for h_d, h_v in zip(H_dev, H_vol)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --- Panel (a): H_dev vs Error ---
    ax = axes[0]
    for i, g in enumerate(geometries):
        ax.scatter(H_dev[i], error[i], s=150, c=colors[g], marker=markers[g],
                   edgecolors='black', linewidth=1.5, zorder=5, label=geometry_labels[g])

    # Trend line
    z = np.polyfit(H_dev, error, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(H_dev)*0.9, max(H_dev)*1.1, 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)

    # Correlation coefficient
    r = np.corrcoef(H_dev, error)[0, 1]
    ax.text(0.95, 0.95, f'$r = {r:.3f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Deviatoric Heterogeneity $H_{dev}$')
    ax.set_ylabel('Relative Error [%]')
    ax.set_title('(a) $H_{dev}$ vs Error')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95)

    # --- Panel (b): H_vol vs Error ---
    ax = axes[1]
    for i, g in enumerate(geometries):
        ax.scatter(H_vol[i], error[i], s=150, c=colors[g], marker=markers[g],
                   edgecolors='black', linewidth=1.5, zorder=5, label=geometry_labels[g])

    z = np.polyfit(H_vol, error, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(H_vol)*0.9, max(H_vol)*1.1, 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)

    r = np.corrcoef(H_vol, error)[0, 1]
    ax.text(0.95, 0.95, f'$r = {r:.3f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Volumetric Heterogeneity $H_{vol}$')
    ax.set_ylabel('Relative Error [%]')
    ax.set_title('(b) $H_{vol}$ vs Error')
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Bar chart with dual y-axis ---
    ax = axes[2]
    x = np.arange(len(geometries))
    width = 0.25

    bars1 = ax.bar(x - width, H_dev, width, label='$H_{dev}$',
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x, H_vol, width, label='$H_{vol}$',
                   color='#9b59b6', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Heterogeneity Index $H$')
    ax.set_xticks(x)
    ax.set_xticklabels([geometry_labels[g].replace('\n', ' ') for g in geometries])

    # Secondary y-axis for error
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, error, width, label='Error',
                    color='#e74c3c', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Relative Error [%]')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95)

    ax.set_title('(c) Summary Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = 'heterogeneity_vs_error_correlation.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()


def plot_strain_field_comparison(results: dict, save_dir: Path):
    """
    Create side-by-side strain field comparison for 902, 903, 904.

    Shows how geometric features (holes) create strain concentrations.
    """
    setup_thesis_style()

    geometries = ['902', '903', '904']
    titles = {
        '902': 'Solid (no holes)',
        '903': 'Ellipse hole',
        '904': 'Three holes'
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Find global colorbar limits
    J2_min = min(results[g]['J2_nodal'].min() for g in geometries)
    J2_max = max(results[g]['J2_nodal'].max() for g in geometries)
    vol_abs_max = max(max(abs(results[g]['eps_vol_nodal'].min()),
                         abs(results[g]['eps_vol_nodal'].max())) for g in geometries)

    tpc_J2 = None
    tpc_vol = None

    for col, geom in enumerate(geometries):
        data = results[geom]
        triang = data['triangulation']
        J2_nodal = data['J2_nodal']
        eps_vol_nodal = data['eps_vol_nodal']
        H_dev = data['H_dev']
        H_vol = data['H_vol']

        # Row 0: J2 deviatoric strain
        ax = axes[0, col]
        tpc_J2 = ax.tripcolor(triang, J2_nodal, shading='gouraud', cmap='hot',
                              vmin=J2_min, vmax=J2_max)
        ax.set_aspect('equal')
        ax.set_title(f'{titles[geom]}\n$H_{{dev}} = {H_dev:.3f}$', fontsize=11)
        if col == 0:
            ax.set_ylabel('$J_2$ strain\n\n$y$ [mm]', fontsize=11)
        ax.set_xlabel('$x$ [mm]', fontsize=10)

        # Row 1: Volumetric strain
        ax = axes[1, col]
        tpc_vol = ax.tripcolor(triang, eps_vol_nodal, shading='gouraud', cmap='RdBu_r',
                               vmin=-vol_abs_max, vmax=vol_abs_max)
        ax.set_aspect('equal')
        ax.set_title(f'$H_{{vol}} = {H_vol:.3f}$', fontsize=11)
        if col == 0:
            ax.set_ylabel('$\\varepsilon_{vol}$\n\n$y$ [mm]', fontsize=11)
        ax.set_xlabel('$x$ [mm]', fontsize=10)

    # Colorbars
    fig.subplots_adjust(right=0.88)

    cbar_ax1 = fig.add_axes([0.90, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(tpc_J2, cax=cbar_ax1)
    cbar1.set_label('$J_2$ [-]', fontsize=11)

    cbar_ax2 = fig.add_axes([0.90, 0.10, 0.02, 0.35])
    cbar2 = fig.colorbar(tpc_vol, cax=cbar_ax2)
    cbar2.set_label('$\\varepsilon_{vol}$ [-]', fontsize=11)

    plt.suptitle('Strain Field Heterogeneity Comparison', fontsize=13, fontweight='bold', y=0.98)

    filename = 'strain_field_comparison.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()


def plot_comprehensive_summary(results: dict, errors: dict, geometry_info: dict, save_dir: Path):
    """
    Create the main summary figure for thesis.

    Layout: 2x2
        (a) Geometry schematic comparison
        (b) Strain field comparison (J2 only)
        (c) Heterogeneity bar chart
        (d) Error vs heterogeneity scatter
    """
    setup_thesis_style()

    geometries = ['902', '903', '904']
    geometry_labels = ['Solid', '1 Hole', '3 Holes']
    colors = {'902': '#e74c3c', '903': '#f39c12', '904': '#27ae60'}

    H_dev = [results[g]['H_dev'] for g in geometries]
    H_vol = [results[g]['H_vol'] for g in geometries]
    error = [errors[g] for g in geometries]

    fig = plt.figure(figsize=(14, 10))

    # Create grid spec
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.35)

    # --- Row 0: Strain fields for each geometry ---
    J2_min = min(results[g]['J2_nodal'].min() for g in geometries)
    J2_max = max(results[g]['J2_nodal'].max() for g in geometries)

    for col, (geom, label) in enumerate(zip(geometries, geometry_labels)):
        ax = fig.add_subplot(gs[0, col])
        data = results[geom]
        triang = data['triangulation']
        tpc = ax.tripcolor(triang, data['J2_nodal'], shading='gouraud', cmap='hot',
                          vmin=J2_min, vmax=J2_max)
        ax.set_aspect('equal')
        ax.set_title(f'{geom}: {label}\n$H_{{dev}} = {data["H_dev"]:.3f}$', fontsize=11)
        ax.set_xlabel('$x$ [mm]')
        if col == 0:
            ax.set_ylabel('$y$ [mm]')

    # Colorbar for strain fields
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = fig.colorbar(tpc, cax=cbar_ax)
    cbar.set_label('$J_2$ strain [-]')

    # --- Row 1, Col 0: Heterogeneity bar chart ---
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(geometries))
    width = 0.35

    bars1 = ax.bar(x - width/2, H_dev, width, label='$H_{dev}$',
                   color='#3498db', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, H_vol, width, label='$H_{vol}$',
                   color='#9b59b6', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Heterogeneity Index $H$')
    ax.set_xlabel('Specimen Geometry')
    ax.set_xticks(x)
    ax.set_xticklabels(geometry_labels)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('(a) Strain Heterogeneity', fontsize=11)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # --- Row 1, Col 1: Error bar chart ---
    ax = fig.add_subplot(gs[1, 1])
    bars = ax.bar(x, error, 0.6, color=[colors[g] for g in geometries],
                  edgecolor='black', alpha=0.85)

    ax.set_ylabel('Relative Error [%]')
    ax.set_xlabel('Specimen Geometry')
    ax.set_xticks(x)
    ax.set_xticklabels(geometry_labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('(b) Verification Error', fontsize=11)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # --- Row 1, Col 2: Correlation scatter ---
    ax = fig.add_subplot(gs[1, 2])

    # Use H_dev (deviatoric heterogeneity)
    for i, (g, label) in enumerate(zip(geometries, geometry_labels)):
        ax.scatter(H_dev[i], error[i], s=200, c=colors[g], marker='o',
                   edgecolors='black', linewidth=2, zorder=5, label=label)

    # Trend line
    z = np.polyfit(H_dev, error, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(H_dev)*1.15, 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.6, linewidth=1.5, label='Trend')

    # Correlation coefficient
    r = np.corrcoef(H_dev, error)[0, 1]
    ax.text(0.05, 0.95, f'$r = {r:.3f}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel('Deviatoric Heterogeneity $H_{dev}$')
    ax.set_ylabel('Relative Error [%]')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_title('(c) Heterogeneity vs Error', fontsize=11)
    ax.set_xlim([0, max(H_dev)*1.15])
    ax.set_ylim([0, max(error)*1.15])

    plt.suptitle('Effect of Geometric Heterogeneity on Identification Accuracy',
                 fontsize=14, fontweight='bold', y=0.98)

    filename = 'comprehensive_heterogeneity_analysis.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Main analysis linking heterogeneity to identification error."""

    print("="*70)
    print("HETEROGENEITY vs ERROR ANALYSIS")
    print("="*70)

    # Define paths
    base_dir = Path(__file__).parent.parent / "synthetic_data"

    experiment_dirs = {
        '902': base_dir / '902',
        '903': base_dir / '903',
        '904': base_dir / '904'
    }

    # Error values from verification (from verification_metrics.txt)
    errors = {
        '902': 8.70,  # % relative error
        '903': 3.55,  # %
        '904': 2.56   # %
    }

    # Geometry info
    geometry_info = {
        '902': {'name': 'Solid rectangle', 'holes': 0},
        '903': {'name': 'Ellipse hole', 'holes': 1},
        '904': {'name': 'Three circular holes', 'holes': 3}
    }

    # Output directory
    output_dir = Path(__file__).parent / "heterogeneity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each geometry using consistent time fraction
    results = {}
    time_fraction = 0.5  # Use 50% of total time for consistent comparison

    for geom_id, exp_dir in experiment_dirs.items():
        if exp_dir.exists():
            results[geom_id] = analyze_geometry(exp_dir, time_fraction)
        else:
            print(f"Warning: {exp_dir} not found, skipping {geom_id}")

    if len(results) < 3:
        print("\nWarning: Not all geometries found.")
        return

    # Create visualizations
    print("\n" + "-"*70)
    print("Generating figures...")
    print("-"*70)

    # 1. Correlation plot: heterogeneity vs error
    plot_heterogeneity_vs_error(results, errors, output_dir)

    # 2. Strain field comparison
    plot_strain_field_comparison(results, output_dir)

    # 3. Comprehensive summary figure
    plot_comprehensive_summary(results, errors, geometry_info, output_dir)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Geometry':<15} {'Holes':<8} {'H_dev':<10} {'H_vol':<10} {'Error [%]':<10}")
    print("-"*70)
    for geom_id in sorted(results.keys()):
        r = results[geom_id]
        info = geometry_info[geom_id]
        e = errors[geom_id]
        print(f"{geom_id} ({info['name'][:10]+'...' if len(info['name'])>10 else info['name']})"[:15].ljust(15) +
              f"{info['holes']:<8} {r['H_dev']:<10.4f} {r['H_vol']:<10.4f} {e:<10.2f}")

    # Compute correlations
    H_dev = [results[g]['H_dev'] for g in sorted(results.keys())]
    H_vol = [results[g]['H_vol'] for g in sorted(results.keys())]
    error_vals = [errors[g] for g in sorted(results.keys())]

    r_dev = np.corrcoef(H_dev, error_vals)[0, 1]
    r_vol = np.corrcoef(H_vol, error_vals)[0, 1]

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"""
Correlation Analysis:
  - Correlation(H_dev, Error) = {r_dev:.4f}
  - Correlation(H_vol, Error) = {r_vol:.4f}

Both show NEGATIVE correlation: higher heterogeneity -> lower error

Interpretation:
  - Solid specimen (902): Low strain heterogeneity (H_dev={results['902']['H_dev']:.3f})
    -> Poor identifiability -> High error ({errors['902']:.1f}%)

  - Specimen with holes (903, 904): High strain heterogeneity
    -> Stress concentrations around holes create varied strain patterns
    -> Better identifiability of viscoelastic parameters
    -> Lower error ({errors['903']:.1f}%, {errors['904']:.1f}%)

  The 3-hole geometry (904) achieves the lowest error ({errors['904']:.1f}%)
  because multiple stress concentrations provide the richest information
  for separating deviatoric (G) and volumetric (K) behavior.
""")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
