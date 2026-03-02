"""
Strain heterogeneity analysis for identifiability study.

Computes and visualizes strain field heterogeneity across different specimen geometries
(902, 903, 904) to demonstrate why heterogeneous strain fields improve identifiability.

Metrics computed:
    - Deviatoric strain intensity (J2 or equivalent strain)
    - Volumetric strain (εvol = εxx + εyy)
    - Heterogeneity indices: Hdev = std(J2)/mean(J2), Hvol = std(εvol)/mean(|εvol|)
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
    conne = conne_raw.iloc[:, 1:4].values.astype(np.int64) - 1  # 0-indexed

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
    """
    Compute strain fields (εxx, εyy, γxy) at element centroids.

    Args:
        U_timestep: Displacement at one timestep (2*n_nodes,) in separated format [Ux; Uy]
        coord: Node coordinates (n_nodes, 4+)
        conne: Connectivity (n_elements, 3)
        n_nodes: Number of nodes

    Returns:
        strains: dict with 'eps_xx', 'eps_yy', 'gamma_xy' arrays (n_elements,)
    """
    n_elements = len(conne)
    nodes_xy = coord[:, 1:3]

    # Convert separated format to interleaved for element extraction
    Ux = U_timestep[:n_nodes]
    Uy = U_timestep[n_nodes:]

    eps_xx = np.zeros(n_elements)
    eps_yy = np.zeros(n_elements)
    gamma_xy = np.zeros(n_elements)

    for ie in range(n_elements):
        node_ids = conne[ie]

        # Create nodes for Triangle3Node
        nodes = [Node(int(node_ids[i]), nodes_xy[node_ids[i], 0], nodes_xy[node_ids[i], 1])
                 for i in range(3)]

        # Create element (computes Be matrix)
        elem = Triangle3Node(ie, nodes)

        # Extract element displacements [u1, v1, u2, v2, u3, v3]
        u_elem = np.zeros(6)
        for i, nid in enumerate(node_ids):
            u_elem[2*i] = Ux[nid]
            u_elem[2*i+1] = Uy[nid]

        # Compute strains: ε = Be @ u
        strain = elem.Be @ u_elem
        eps_xx[ie] = strain[0]
        eps_yy[ie] = strain[1]
        gamma_xy[ie] = strain[2]

    return {
        'eps_xx': eps_xx,
        'eps_yy': eps_yy,
        'gamma_xy': gamma_xy
    }


def compute_J2_strain(eps_xx, eps_yy, gamma_xy):
    """
    Compute deviatoric strain intensity (J2 invariant).

    J2 = sqrt(0.5 * (e1^2 + e2^2 + e3^2) + (gamma_xy/2)^2)

    For plane stress deviatoric: e1 = eps_xx - eps_m, e2 = eps_yy - eps_m
    where eps_m = (eps_xx + eps_yy) / 2 (assuming eps_zz = 0 for simplicity)

    Simplified: J2 = sqrt((eps_xx - eps_yy)^2 / 4 + (gamma_xy/2)^2)
               = 0.5 * sqrt((eps_xx - eps_yy)^2 + gamma_xy^2)

    Args:
        eps_xx, eps_yy, gamma_xy: Strain components (n_elements,)

    Returns:
        J2: Deviatoric strain intensity (n_elements,)
    """
    # Mean strain (in-plane only)
    eps_mean = (eps_xx + eps_yy) / 2.0

    # Deviatoric strains
    e_xx = eps_xx - eps_mean
    e_yy = eps_yy - eps_mean

    # J2 invariant (second invariant of deviatoric strain tensor)
    # J2 = sqrt(0.5 * (e_xx^2 + e_yy^2 + e_zz^2) + eps_xy^2)
    # For plane stress with e_zz = -e_xx - e_yy:
    e_zz = -(e_xx + e_yy)  # Trace-free condition

    J2 = np.sqrt(0.5 * (e_xx**2 + e_yy**2 + e_zz**2) + (gamma_xy/2)**2)

    return J2


def compute_volumetric_strain(eps_xx, eps_yy):
    """
    Compute volumetric (dilatational) strain.

    εvol = εxx + εyy (in 2D plane stress, εzz contributes to out-of-plane thinning)

    Args:
        eps_xx, eps_yy: Strain components (n_elements,)

    Returns:
        eps_vol: Volumetric strain (n_elements,)
    """
    return eps_xx + eps_yy


def compute_heterogeneity_metrics(field, field_name='field'):
    """
    Compute heterogeneity index for a strain field.

    H = std(field) / mean(|field|)

    Args:
        field: Strain field array (n_elements,)
        field_name: Name for reporting

    Returns:
        H: Heterogeneity index
    """
    mean_abs = np.mean(np.abs(field))
    std_field = np.std(field)

    if mean_abs > 1e-12:
        H = std_field / mean_abs
    else:
        H = 0.0

    return H


def element_to_nodal_average(element_values, conne, n_nodes):
    """
    Convert element-centered values to nodal values by averaging.

    Each node gets the average of all elements it belongs to.

    Args:
        element_values: Array of shape (n_elements,)
        conne: Connectivity array (n_elements, 3)
        n_nodes: Number of nodes

    Returns:
        nodal_values: Array of shape (n_nodes,)
    """
    nodal_sum = np.zeros(n_nodes)
    nodal_count = np.zeros(n_nodes)

    for ie, elem_nodes in enumerate(conne):
        for node_id in elem_nodes:
            nodal_sum[node_id] += element_values[ie]
            nodal_count[node_id] += 1

    # Avoid division by zero
    nodal_count[nodal_count == 0] = 1
    nodal_values = nodal_sum / nodal_count

    return nodal_values


def setup_thesis_style():
    """Setup matplotlib style for thesis-quality figures."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 11
    plt.rcParams['axes.linewidth'] = 0.8


def plot_heterogeneity_comparison(results: dict, save_dir: Path, t_target: float = 603.0):
    """
    Create multi-panel figure comparing strain heterogeneity across geometries.

    Layout: 2 rows x 3 columns
        Row 1: J2 deviatoric strain maps for 902, 903, 904
        Row 2: Volumetric strain maps for 902, 903, 904

    Plus summary metrics table.

    Args:
        results: dict with keys '902', '903', '904', each containing strain data
        save_dir: Output directory
        t_target: Target time for snapshot (seconds)
    """
    setup_thesis_style()

    geometries = ['902', '903', '904']

    # Create figure: 2 rows (J2, εvol) x 3 columns (geometries)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Find global colorbar limits (use nodal values for plotting)
    J2_min = min(results[g]['J2_nodal'].min() for g in geometries)
    J2_max = max(results[g]['J2_nodal'].max() for g in geometries)
    vol_min = min(results[g]['eps_vol_nodal'].min() for g in geometries)
    vol_max = max(results[g]['eps_vol_nodal'].max() for g in geometries)

    # For volumetric strain, use symmetric limits centered at 0
    vol_abs_max = max(abs(vol_min), abs(vol_max))

    # Store colorbars for later
    tpc_J2 = None
    tpc_vol = None

    for col, geom in enumerate(geometries):
        data = results[geom]
        triang = data['triangulation']
        J2_nodal = data['J2_nodal']
        eps_vol_nodal = data['eps_vol_nodal']
        H_dev = data['H_dev']
        H_vol = data['H_vol']

        # Row 0: J2 deviatoric strain (smooth Gouraud shading)
        ax = axes[0, col]
        tpc_J2 = ax.tripcolor(triang, J2_nodal, shading='gouraud', cmap='hot',
                              vmin=J2_min, vmax=J2_max)
        ax.set_aspect('equal')
        ax.set_title(f'{geom}\n$H_{{dev}} = {H_dev:.3f}$', fontsize=10)
        if col == 0:
            ax.set_ylabel('$J_2$ strain\n\n$y$ [mm]', fontsize=10)
        ax.set_xlabel('$x$ [mm]', fontsize=9)

        # Row 1: Volumetric strain (smooth Gouraud shading)
        ax = axes[1, col]
        tpc_vol = ax.tripcolor(triang, eps_vol_nodal, shading='gouraud', cmap='RdBu_r',
                               vmin=-vol_abs_max, vmax=vol_abs_max)
        ax.set_aspect('equal')
        ax.set_title(f'$H_{{vol}} = {H_vol:.3f}$', fontsize=10)
        if col == 0:
            ax.set_ylabel('$\\varepsilon_{vol}$\n\n$y$ [mm]', fontsize=10)
        ax.set_xlabel('$x$ [mm]', fontsize=9)

    # Add colorbars
    fig.subplots_adjust(right=0.88)

    # J2 colorbar (right side, top half)
    cbar_ax1 = fig.add_axes([0.90, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(tpc_J2, cax=cbar_ax1)
    cbar1.set_label('$J_2$ [-]', fontsize=10)

    # Volumetric colorbar (right side, bottom half)
    cbar_ax2 = fig.add_axes([0.90, 0.10, 0.02, 0.35])
    cbar2 = fig.colorbar(tpc_vol, cax=cbar_ax2)
    cbar2.set_label('$\\varepsilon_{vol}$ [-]', fontsize=10)

    plt.suptitle(f'Strain Heterogeneity at $t = {t_target:.0f}$ s', fontsize=12, fontweight='bold')

    # Save figure
    filename = f'strain_heterogeneity_comparison_t{int(t_target)}.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()


def plot_heterogeneity_bar_chart(results: dict, save_dir: Path, t_target: float = 1200.0):
    """
    Create bar chart comparing heterogeneity metrics across geometries.

    Args:
        results: dict with keys '902', '903', '904'
        save_dir: Output directory
        t_target: Target time (for labeling)
    """
    setup_thesis_style()

    geometries = ['902', '903', '904']
    H_dev = [results[g]['H_dev'] for g in geometries]
    H_vol = [results[g]['H_vol'] for g in geometries]

    x = np.arange(len(geometries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    bars1 = ax.bar(x - width/2, H_dev, width, label='$H_{dev}$', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, H_vol, width, label='$H_{vol}$', color='#3498db', edgecolor='black')

    ax.set_ylabel('Heterogeneity Index $H$', fontsize=10)
    ax.set_xlabel('Specimen Geometry', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(geometries)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.title(f'Strain Field Heterogeneity at $t = {t_target:.0f}$ s', fontsize=11)
    plt.tight_layout()

    filename = f'heterogeneity_metrics_t{int(t_target)}.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()


def analyze_geometry(experiment_dir: Path, t_target: float = 603.0):
    """
    Analyze strain heterogeneity for a single geometry.

    Args:
        experiment_dir: Path to experiment data
        t_target: Target time for analysis (seconds)

    Returns:
        dict: Results including strains and metrics
    """
    print(f"\nAnalyzing: {experiment_dir.name}")

    # Load data
    data = load_experiment_data(experiment_dir)

    # Find closest timestep to t_target
    t_idx = np.argmin(np.abs(data['time'] - t_target))
    t_actual = data['time'][t_idx]
    print(f"  Using timestep {t_idx}, t = {t_actual:.1f} s")

    # Extract displacement at this time
    U_t = data['U'][:, t_idx]

    # Compute strain fields
    strains = compute_element_strains(
        U_t, data['coord'], data['conne'], data['nnodes']
    )

    # Compute J2 and volumetric strain
    J2 = compute_J2_strain(strains['eps_xx'], strains['eps_yy'], strains['gamma_xy'])
    eps_vol = compute_volumetric_strain(strains['eps_xx'], strains['eps_yy'])

    # Compute heterogeneity metrics
    H_dev = compute_heterogeneity_metrics(J2, 'J2')
    H_vol = compute_heterogeneity_metrics(eps_vol, 'eps_vol')

    print(f"  J2 range: [{J2.min():.6f}, {J2.max():.6f}]")
    print(f"  eps_vol range: [{eps_vol.min():.6f}, {eps_vol.max():.6f}]")
    print(f"  H_dev = {H_dev:.4f}")
    print(f"  H_vol = {H_vol:.4f}")

    # Convert element values to nodal values for smooth plotting
    J2_nodal = element_to_nodal_average(J2, data['conne'], data['nnodes'])
    eps_vol_nodal = element_to_nodal_average(eps_vol, data['conne'], data['nnodes'])

    # Create triangulation for plotting
    nodes_xy = data['coord'][:, 1:3]
    triang = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], data['conne'])

    return {
        'strains': strains,
        'J2': J2,
        'eps_vol': eps_vol,
        'J2_nodal': J2_nodal,
        'eps_vol_nodal': eps_vol_nodal,
        'H_dev': H_dev,
        'H_vol': H_vol,
        'triangulation': triang,
        't_actual': t_actual,
        'data': data
    }


def main():
    """Main analysis comparing 902, 903, 904 geometries."""

    print("="*70)
    print("STRAIN HETEROGENEITY ANALYSIS")
    print("="*70)

    # Define paths
    base_dir = Path(__file__).parent.parent / "synthetic_data"

    # Experiment directories
    experiment_dirs = {
        '902': base_dir / '902',
        '903': base_dir / '903',
        '904': base_dir / '904'
    }

    # Target time for analysis
    t_target = 1200  # seconds

    # Output directory
    output_dir = Path(__file__).parent / "heterogeneity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each geometry
    results = {}
    for geom_id, exp_dir in experiment_dirs.items():
        if exp_dir.exists():
            results[geom_id] = analyze_geometry(exp_dir, t_target)
        else:
            print(f"Warning: {exp_dir} not found, skipping {geom_id}")

    if len(results) < 3:
        print("\nWarning: Not all geometries found. Proceeding with available data.")

    if not results:
        print("Error: No data found!")
        return

    # Create comparison plots
    print("\nGenerating comparison figures...")
    plot_heterogeneity_comparison(results, output_dir, t_target)
    plot_heterogeneity_bar_chart(results, output_dir, t_target)

    # Print summary table
    print("\n" + "="*70)
    print("HETEROGENEITY SUMMARY")
    print("="*70)
    print(f"\nTime: t = {t_target:.0f} s")
    print(f"\n{'Geometry':<12} {'H_dev':<12} {'H_vol':<12} {'mean(J2)':<15} {'std(J2)':<15}")
    print("-"*70)
    for geom_id in sorted(results.keys()):
        r = results[geom_id]
        print(f"{geom_id:<12} {r['H_dev']:<12.4f} {r['H_vol']:<12.4f} "
              f"{np.mean(r['J2']):<15.6e} {np.std(r['J2']):<15.6e}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Higher heterogeneity (H) values indicate more varied strain distributions,
which provide richer information for material parameter identification.

- H_dev: Heterogeneity of deviatoric (shear) strain
  Higher values indicate more varied shear deformation patterns,
  improving sensitivity to shear modulus (G) identification.

- H_vol: Heterogeneity of volumetric strain
  Higher values indicate more varied compression/tension patterns,
  improving sensitivity to bulk modulus (K) identification.

Geometries with holes (903, 904) typically show higher heterogeneity
due to stress concentrations around the hole boundaries.
""")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
