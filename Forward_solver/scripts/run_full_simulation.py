"""
Full-scale realistic forward simulation.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from datetime import datetime

from Forward_solver.core.mesh import MeshGenerator, Node
from Forward_solver.core.material import create_reference_material
from Forward_solver.core.data_generation import SyntheticDataGenerator

# Complex geometry support (Phase 2 & 3)
try:
    from Forward_solver.core.geometry_builder import GeometryBuilder, load_config
    from Forward_solver.core.mesh_converter import MeshConverter
    COMPLEX_GEOMETRY_AVAILABLE = True
except ImportError:
    COMPLEX_GEOMETRY_AVAILABLE = False


def plot_mesh(mesh, save_dir: Path):
    """Plot and save mesh visualization."""
    fig, ax = plt.subplots(figsize=(8, 12))

    # Plot mesh with darker lines
    ax.triplot(mesh.triangulation, 'k-', lw=0.8, alpha=1.0)

    # Highlight boundaries
    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])

    # Bottom (fixed)
    bottom_nodes = np.where(mesh.coord[:, 3] == 2)[0]
    ax.plot(nodes_xy[bottom_nodes, 0], nodes_xy[bottom_nodes, 1],
            'bs', markersize=6, label='Bottom (fixed)', markerfacecolor='blue')

    # Top (loaded)
    top_nodes = np.where(mesh.coord[:, 3] == 1)[0]
    ax.plot(nodes_xy[top_nodes, 0], nodes_xy[top_nodes, 1],
            'r^', markersize=8, label='Top (loaded)', markerfacecolor='red')

    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(f'Mesh: {len(mesh.nodes)} nodes, {len(mesh.conne)} elements',
                 fontsize=14, fontweight='bold')

    # Set axis limits with small padding to avoid overlap
    # Axes start at 0 but add padding to create space around mesh
    x_max = nodes_xy[:, 0].max()
    y_max = nodes_xy[:, 1].max()
    padding = 1.0  # 1mm padding
    ax.set_xlim([-padding, x_max + padding])
    ax.set_ylim([-padding, y_max + padding])

    # Set equal aspect ratio AFTER setting limits
    ax.set_aspect('equal', adjustable='box')

    # Hide right and top spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Remove grid for cleaner look like the reference image
    ax.grid(False)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_dir / 'mesh.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: mesh.png")
    plt.close()


def plot_displacement_evolution(time, U_history, mesh, save_dir: Path):
    """Plot displacement evolution at key nodes."""

    # Find key monitoring nodes
    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])

    # Top center
    top_center = np.where(
        (np.abs(nodes_xy[:, 0] - mesh.width/2) < 0.5) &
        (np.abs(nodes_xy[:, 1] - mesh.height) < 0.5)
    )[0]

    # Mid center - find closest node to center
    mid_y_target = mesh.height / 2
    mid_x_target = mesh.width / 2

    # Find nodes near mid-height
    mid_candidates = np.where(np.abs(nodes_xy[:, 1] - mid_y_target) < 2.5)[0]
    if len(mid_candidates) > 0:
        # Among candidates, find closest to x-center
        distances = np.abs(nodes_xy[mid_candidates, 0] - mid_x_target)
        mid_center = mid_candidates[np.argmin(distances):np.argmin(distances)+1]
    else:
        mid_center = np.array([])

    # Bottom center (should be ~zero)
    bottom_center = np.where(
        (np.abs(nodes_xy[:, 0] - mesh.width/2) < 0.5) &
        (np.abs(nodes_xy[:, 1]) < 0.5)
    )[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Vertical displacement at top center
    if len(top_center) > 0:
        node_id = top_center[0]
        u_y = U_history[2*node_id + 1, :]

        axes[0, 0].plot(time, u_y, 'o-', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Time [s]', fontsize=11)
        axes[0, 0].set_ylabel('Vertical Displacement [mm]', fontsize=11)
        axes[0, 0].set_title(f'Top Center Node (y={mesh.height:.1f}mm)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Vertical displacement at mid center
    if len(mid_center) > 0:
        node_id = mid_center[0]
        u_y = U_history[2*node_id + 1, :]

        axes[0, 1].plot(time, u_y, 'o-', color='orange', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Time [s]', fontsize=11)
        axes[0, 1].set_ylabel('Vertical Displacement [mm]', fontsize=11)
        axes[0, 1].set_title(f'Mid Center Node (y={mesh.height/2:.1f}mm)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Horizontal displacement at top center
    if len(top_center) > 0:
        node_id = top_center[0]
        u_x = U_history[2*node_id, :]

        axes[1, 0].plot(time, u_x, 'o-', color='green', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Time [s]', fontsize=11)
        axes[1, 0].set_ylabel('Horizontal Displacement [mm]', fontsize=11)
        axes[1, 0].set_title('Top Center Node - Horizontal', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Max displacement over time
    max_disp = np.max(np.abs(U_history), axis=0)

    axes[1, 1].plot(time, max_disp, 'o-', color='red', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Time [s]', fontsize=11)
    axes[1, 1].set_ylabel('Max |Displacement| [mm]', fontsize=11)
    axes[1, 1].set_title('Maximum Displacement Magnitude', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'displacement_evolution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: displacement_evolution.png")
    plt.close()


def plot_stress_strain(time, U_history, mesh, material, save_dir: Path):
    """Plot stress and strain evolution at element centroids."""

    # Create elements
    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])

    # Select an element for plotting
    # Strategy: Pick element at mid-height but offset from center to avoid holes
    # Use x = 3/4 * width (right side) to avoid center holes
    centroids = []
    for elem_nodes in mesh.conne:
        centroid = np.mean(nodes_xy[elem_nodes], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Target: Mid-height, 3/4 width (avoids center holes)
    target = np.array([mesh.width * 0.75, mesh.height * 0.5])
    distances = np.linalg.norm(centroids - target, axis=1)
    mid_element_id = np.argmin(distances)

    # Compute stress and strain for this element over time
    n_timesteps = U_history.shape[1]
    stress_xx = np.zeros(n_timesteps)
    stress_yy = np.zeros(n_timesteps)
    stress_xy = np.zeros(n_timesteps)
    strain_xx = np.zeros(n_timesteps)
    strain_yy = np.zeros(n_timesteps)
    strain_xy = np.zeros(n_timesteps)

    # Get element coordinates
    elem_node_ids = mesh.conne[mid_element_id]
    elem_coords = nodes_xy[elem_node_ids]  # (3, 2)

    # Compute strain-displacement matrix B (constant for linear triangle)
    x1, y1 = elem_coords[0]
    x2, y2 = elem_coords[1]
    x3, y3 = elem_coords[2]

    # Area and Jacobian
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(detJ)

    # Shape function derivatives
    dN1_dx = (y2 - y3) / detJ
    dN2_dx = (y3 - y1) / detJ
    dN3_dx = (y1 - y2) / detJ
    dN1_dy = (x3 - x2) / detJ
    dN2_dy = (x1 - x3) / detJ
    dN3_dy = (x2 - x1) / detJ

    # Strain-displacement matrix B (3 x 6)
    Be = np.array([
        [dN1_dx, 0,      dN2_dx, 0,      dN3_dx, 0     ],
        [0,      dN1_dy, 0,      dN2_dy, 0,      dN3_dy],
        [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]
    ])

    for nt in range(n_timesteps):
        # Get displacement for this element
        U_elem = np.zeros(6)
        for i, node_id in enumerate(elem_node_ids):
            U_elem[2*i] = U_history[2*node_id, nt]
            U_elem[2*i+1] = U_history[2*node_id+1, nt]

        # Compute strain: ε = B * U
        strain = Be @ U_elem  # (3,)
        strain_xx[nt] = strain[0]
        strain_yy[nt] = strain[1]
        strain_xy[nt] = strain[2]

        # Compute stress (elastic approximation for visualization)
        # σ = G_eff * D_mu * ε_dev + K_eff * θ * m
        G_eff = material.G_inf + np.sum(material.G) * 0.5  # Approximate
        K_eff = material.K_inf + np.sum(material.K) * 0.5

        # Deviatoric strain
        theta = strain[0] + strain[1]  # volumetric strain
        eps_dev = strain - 0.5 * theta * np.array([1, 1, 0])

        # Stress
        stress_dev = G_eff * material.Dmu @ eps_dev
        stress_vol = K_eff * theta * np.array([1, 1, 0])
        stress = stress_dev + stress_vol

        stress_xx[nt] = stress[0]
        stress_yy[nt] = stress[1]
        stress_xy[nt] = stress[2]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Stress components
    axes[0, 0].plot(time, stress_xx, 'o-', label='σxx', linewidth=2, markersize=3)
    axes[0, 0].plot(time, stress_yy, 'o-', label='σyy', linewidth=2, markersize=3)
    axes[0, 0].plot(time, stress_xy, 'o-', label='σxy', linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Time [s]', fontsize=11)
    axes[0, 0].set_ylabel('Stress [MPa]', fontsize=11)
    axes[0, 0].set_title(f'Stress Evolution (Element {mid_element_id})', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Strain components
    axes[0, 1].plot(time, strain_xx, 'o-', label='εxx', linewidth=2, markersize=3)
    axes[0, 1].plot(time, strain_yy, 'o-', label='εyy', linewidth=2, markersize=3)
    axes[0, 1].plot(time, strain_xy, 'o-', label='εxy', linewidth=2, markersize=3)
    axes[0, 1].set_xlabel('Time [s]', fontsize=11)
    axes[0, 1].set_ylabel('Strain [-]', fontsize=11)
    axes[0, 1].set_title(f'Strain Evolution (Element {mid_element_id})', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Stress-strain curve (σyy vs εyy)
    axes[1, 0].plot(strain_yy, stress_yy, 'o-', linewidth=2, markersize=3, color='purple')
    axes[1, 0].set_xlabel('Strain εyy [-]', fontsize=11)
    axes[1, 0].set_ylabel('Stress σyy [MPa]', fontsize=11)
    axes[1, 0].set_title('Stress-Strain Curve (Vertical)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Von Mises stress
    vm_stress = np.sqrt(stress_xx**2 - stress_xx*stress_yy + stress_yy**2 + 3*stress_xy**2)
    axes[1, 1].plot(time, vm_stress, 'o-', linewidth=2, markersize=3, color='red')
    axes[1, 1].set_xlabel('Time [s]', fontsize=11)
    axes[1, 1].set_ylabel('Von Mises Stress [MPa]', fontsize=11)
    axes[1, 1].set_title('Von Mises Stress Evolution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'stress_strain_evolution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: stress_strain_evolution.png")
    plt.close()


def plot_field_contours(time, U_history, mesh, material, save_dir: Path):
    """Plot stress and strain field contours at final timestep."""

    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])
    n_elements = len(mesh.conne)

    # Use final timestep for field visualization
    nt = -1

    # Compute stress and strain at each element centroid
    stress_xx = np.zeros(n_elements)
    stress_yy = np.zeros(n_elements)
    stress_xy = np.zeros(n_elements)
    strain_xx = np.zeros(n_elements)
    strain_yy = np.zeros(n_elements)
    strain_xy = np.zeros(n_elements)
    centroids = np.zeros((n_elements, 2))

    G_eff = material.G_inf + np.sum(material.G) * 0.5
    K_eff = material.K_inf + np.sum(material.K) * 0.5

    for ie, elem_node_ids in enumerate(mesh.conne):
        elem_coords = nodes_xy[elem_node_ids]
        centroids[ie] = np.mean(elem_coords, axis=0)

        # Compute B matrix
        x1, y1 = elem_coords[0]
        x2, y2 = elem_coords[1]
        x3, y3 = elem_coords[2]

        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        dN1_dx = (y2 - y3) / detJ
        dN2_dx = (y3 - y1) / detJ
        dN3_dx = (y1 - y2) / detJ
        dN1_dy = (x3 - x2) / detJ
        dN2_dy = (x1 - x3) / detJ
        dN3_dy = (x2 - x1) / detJ

        Be = np.array([
            [dN1_dx, 0,      dN2_dx, 0,      dN3_dx, 0     ],
            [0,      dN1_dy, 0,      dN2_dy, 0,      dN3_dy],
            [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]
        ])

        # Get displacement
        U_elem = np.zeros(6)
        for i, node_id in enumerate(elem_node_ids):
            U_elem[2*i] = U_history[2*node_id, nt]
            U_elem[2*i+1] = U_history[2*node_id+1, nt]

        # Compute strain
        strain = Be @ U_elem
        strain_xx[ie] = strain[0]
        strain_yy[ie] = strain[1]
        strain_xy[ie] = strain[2]

        # Compute stress
        theta = strain[0] + strain[1]
        eps_dev = strain - 0.5 * theta * np.array([1, 1, 0])
        stress_dev = G_eff * material.Dmu @ eps_dev
        stress_vol = K_eff * theta * np.array([1, 1, 0])
        stress = stress_dev + stress_vol

        stress_xx[ie] = stress[0]
        stress_yy[ie] = stress[1]
        stress_xy[ie] = stress[2]

    # Von Mises stress
    vm_stress = np.sqrt(stress_xx**2 - stress_xx*stress_yy + stress_yy**2 + 3*stress_xy**2)

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Plot stress fields
    for idx, (field, title) in enumerate([
        (stress_xx, 'σxx [MPa]'),
        (stress_yy, 'σyy [MPa]'),
        (stress_xy, 'σxy [MPa]'),
        (vm_stress, 'Von Mises [MPa]'),
        (strain_xx, 'εxx [-]'),
        (strain_yy, 'εyy [-]')
    ]):
        ax = axes.flatten()[idx]

        # Tricontourf for smooth contours
        tcf = ax.tricontourf(centroids[:, 0], centroids[:, 1], field,
                             levels=15, cmap='RdYlBu_r')
        ax.triplot(mesh.triangulation, 'k-', lw=0.3, alpha=0.2)

        ax.set_xlabel('X [mm]', fontsize=10)
        ax.set_ylabel('Y [mm]', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

        # Colorbar
        cbar = plt.colorbar(tcf, ax=ax)
        cbar.ax.tick_params(labelsize=9)

    plt.suptitle(f'Stress and Strain Fields at t={time[nt]:.2f}s',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_dir / 'field_contours.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: field_contours.png")
    plt.close()


def plot_deformed_shapes(time, U_history, mesh, save_dir: Path, timesteps_to_plot=None):
    """Plot deformed mesh at selected timesteps."""

    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])

    if timesteps_to_plot is None:
        # Select 6 evenly spaced timesteps
        n_plots = min(6, len(time))
        timesteps_to_plot = np.linspace(0, len(time)-1, n_plots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()

    # Magnification factor for visualization
    scale_factor = 10  # Amplify displacements for visibility

    for idx, nt in enumerate(timesteps_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Original mesh
        ax.triplot(mesh.triangulation, 'k-', lw=0.3, alpha=0.3, label='Original')

        # Deformed mesh
        U_x = U_history[::2, nt]
        U_y = U_history[1::2, nt]

        deformed_x = nodes_xy[:, 0] + scale_factor * U_x
        deformed_y = nodes_xy[:, 1] + scale_factor * U_y

        # Create triangulation for deformed mesh
        import matplotlib.tri as mtri
        tri_deformed = mtri.Triangulation(deformed_x, deformed_y, mesh.conne)
        ax.triplot(tri_deformed, 'r-', lw=0.5, alpha=0.7, label='Deformed')

        max_disp = np.max(np.abs(U_history[:, nt]))
        ax.set_title(f't={time[nt]:.2f}s (max |u|={max_disp:.4f}mm)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('X [mm]', fontsize=10)
        ax.set_ylabel('Y [mm]', fontsize=10)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    # Add note about scale factor
    fig.text(0.5, 0.02, f'Note: Displacements magnified {scale_factor}× for visualization',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_dir / 'deformed_shapes.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: deformed_shapes.png")
    plt.close()


def plot_material_relaxation(material, save_dir: Path):
    """Plot material relaxation functions."""

    # Time vector for plotting
    t_plot = np.logspace(-2, 3, 200)  # 0.01s to 1000s

    # Compute relaxation moduli
    G_t = material.G_inf + np.sum(material.G[:, None] * np.exp(-t_plot / material.tau_G[:, None]), axis=0)
    K_t = material.K_inf + np.sum(material.K[:, None] * np.exp(-t_plot / material.tau_K[:, None]), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Shear modulus
    ax1.semilogx(t_plot, G_t, 'b-', linewidth=2.5, label='G(t)')
    ax1.axhline(material.G_inf, color='r', linestyle='--', linewidth=1.5, label=f'G_∞={material.G_inf} MPa')
    ax1.axhline(material.G0, color='g', linestyle='--', linewidth=1.5, label=f'G_0={material.G0} MPa')

    # Mark relaxation times
    for i, tau in enumerate(material.tau_G):
        G_tau = material.G_inf + np.sum(material.G * np.exp(-tau / material.tau_G))
        ax1.plot(tau, G_tau, 'ko', markersize=8)
        ax1.text(tau, G_tau, f'  τ_{i+1}', fontsize=9, va='center')

    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Shear Modulus G(t) [MPa]', fontsize=12)
    ax1.set_title('Deviatoric Relaxation', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=10)

    # Bulk modulus
    ax2.semilogx(t_plot, K_t, 'b-', linewidth=2.5, label='K(t)')
    ax2.axhline(material.K_inf, color='r', linestyle='--', linewidth=1.5, label=f'K_∞={material.K_inf} MPa')
    ax2.axhline(material.K0, color='g', linestyle='--', linewidth=1.5, label=f'K_0={material.K0} MPa')

    # Mark relaxation times
    for i, tau in enumerate(material.tau_K):
        K_tau = material.K_inf + np.sum(material.K * np.exp(-tau / material.tau_K))
        ax2.plot(tau, K_tau, 'ko', markersize=8)
        ax2.text(tau, K_tau, f'  τ_{i+1}', fontsize=9, va='center')

    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Bulk Modulus K(t) [MPa]', fontsize=12)
    ax2.set_title('Volumetric Relaxation', fontsize=13, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / 'material_relaxation.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: material_relaxation.png")
    plt.close()


def plot_displacement_field(time, U_history, mesh, save_dir: Path, timestep: int = -1):
    """
    Plot Ux and Uy on the actual FE mesh to avoid artificial X-patterns.
    """
    import matplotlib.pyplot as plt

    # Select time step (default: last)
    if timestep < 0:
        timestep = len(time) - 1

    # Node coordinates and triangulation from mesh
    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])
    triang = mesh.triangulation

    # Extract nodal displacements at this time step
    Ux = U_history[0::2, timestep]
    Uy = U_history[1::2, timestep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # --- Ux ---
    tpc1 = ax1.tripcolor(triang, Ux, shading="gouraud", cmap="viridis")
    cbar1 = plt.colorbar(tpc1, ax=ax1)
    cbar1.set_label("Ux [mm]")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_title(f"Horizontal Displacement (Ux) at t={time[timestep]:.1f}s")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)

    # --- Uy ---
    tpc2 = ax2.tripcolor(triang, Uy, shading="gouraud", cmap="viridis")
    cbar2 = plt.colorbar(tpc2, ax=ax2)
    cbar2.set_label("Uy [mm]")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_title(f"Vertical Displacement (Uy) at t={time[timestep]:.1f}s")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "displacement_field.png", dpi=300, bbox_inches="tight")
    print(" Saved: displacement_field.png")
    plt.close()



def create_simulation_report(config, results_dir: Path):
    """Create a detailed simulation report."""

    report_file = results_dir / 'SIMULATION_REPORT.txt'

    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FORWARD SIMULATION - FULL SCALE REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Mesh:\n")
        f.write(f"  Domain: {config['width']}mm × {config['height']}mm\n")
        f.write(f"  Nodes: {config['nx']} × {config['ny']} = {config['nx']*config['ny']}\n")
        f.write(f"  Elements: ~{2*(config['nx']-1)*(config['ny']-1)} (triangles)\n\n")

        f.write(f"Material ( Reference):\n")
        f.write(f"  Deviatoric:\n")
        f.write(f"    G_inf = {config['material'].G_inf} MPa\n")
        f.write(f"    G = {config['material'].G} MPa\n")
        f.write(f"    tau_G = {config['material'].tau_G} s\n")
        f.write(f"    G0 = {config['material'].G0} MPa\n\n")
        f.write(f"  Volumetric:\n")
        f.write(f"    K_inf = {config['material'].K_inf} MPa\n")
        f.write(f"    K = {config['material'].K} MPa\n")
        f.write(f"    tau_K = {config['material'].tau_K} s\n")
        f.write(f"    K0 = {config['material'].K0} MPa\n\n")

        f.write(f"Time Integration:\n")
        f.write(f"  Time step: dt = {config['dt']} s\n")
        f.write(f"  Number of steps: {config['n_timesteps']}\n")
        f.write(f"  Total simulation time: {config['n_timesteps']*config['dt']} s\n\n")

        f.write(f"Loading:\n")
        f.write(f"  Type: Distributed load on top surface\n")
        f.write(f"  Magnitude: {config['load']} MPa\n\n")

        f.write(f"Boundary Conditions:\n")
        f.write(f"  Bottom: v = 0 (all nodes), u = 0 (node 2 only)\n")
        f.write(f"  Top: Loaded\n")
        f.write(f"  Sides: Free\n\n")

        f.write("="*70 + "\n")
        f.write("FILES GENERATED\n")
        f.write("="*70 + "\n\n")

        f.write("Data files (inverse problem format):\n")
        f.write(f"  - coord.csv\n")
        f.write(f"  - conne.txt\n")
        f.write(f"  - U.csv\n")
        f.write(f"  - F.csv\n")
        f.write(f"  - time.csv\n")
        f.write(f"  - ground_truth_material.txt\n\n")

        f.write("Visualizations:\n")
        f.write(f"  - mesh.png\n")
        f.write(f"  - displacement_evolution.png\n")
        f.write(f"  - deformed_shapes.png\n")
        f.write(f"  - material_relaxation.png\n\n")

        f.write("="*70 + "\n")
        f.write("USAGE\n")
        f.write("="*70 + "\n\n")

        f.write("To test inverse problem with this synthetic data:\n\n")
        f.write("  cd ../inverse_problem\n")
        f.write(f"  python inverse_problem.py --experiment ../Forward_solver/{results_dir.name}\n\n")

        f.write("Compare identified parameters with ground_truth_material.txt\n")

    print(f"  Saved: SIMULATION_REPORT.txt")


def run_full_simulation(
    width: float = 20.0,
    height: float = 50.0,
    nx: int = 11,
    ny: int = 26,
    dt: float = 0.1,
    n_timesteps: int = 100,
    load: float = 50.0,
    experiment_id: int = 800,
    output_name: str = None,
    holes: list = None,
    mesh_import_dir: Path = None,
    use_complex_geometry: bool = None,
    mesh_size_outer: float = None,
    mesh_size_hole: float = None
):
    """
    Run complete full-scale forward simulation.

    Supports two modes:
    - Simple geometry (800-802): Uses mesh.py for rectangular domains
    - Complex geometry (804-806): Uses pygmsh + meshio for holes/ellipses

    Args:
        width: Domain width [mm] 
        height: Domain height [mm] 
        nx: Nodes in x-direction 
        ny: Nodes in y-direction 
        dt: Time step [s] 
        n_timesteps: Number of timesteps 
        load: Distributed traction [N/mm] 
              Sign convention:
              - Positive (+50) = TENSION (upward traction on top surface)
              - Negative (-50) = COMPRESSION (downward traction on top surface)
               uses q0=50 (positive) for TENSION test
        experiment_id: Experiment number for output folder (default: 800)
        output_name: Optional custom output name (overrides experiment_id)
        holes: DEPRECATED for 804+. List of (center_x, center_y, radius) tuples
               Only used for experiments < 804. For complex geometries, use
               configs/geometry_config_XXX.py instead.
        mesh_import_dir: Optional path to import existing mesh (coord.csv, conne.txt)
                        If provided, bypasses mesh generation and uses pre-existing mesh
                        Example: Path('./synthetic_data/817') for  imported geometry
        use_complex_geometry: Force complex geometry mode (auto-detected if None)
                             If None, automatically detects based on experiment_id:
                             - experiment_id >= 804: Complex mode (pygmsh)
                             - experiment_id < 804: Simple mode (mesh.py)
    """

    # Auto-detect complex geometry mode
    if use_complex_geometry is None:
        use_complex_geometry = experiment_id >= 804

    print("="*70)
    print(f"FORWARD SIMULATION - EXPERIMENT {experiment_id}")
    print("="*70)

    if use_complex_geometry:
        print(f"\nMode: COMPLEX GEOMETRY (pygmsh + meshio)")
        if not COMPLEX_GEOMETRY_AVAILABLE:
            raise ImportError(
                "Complex geometry mode requires geometry_builder.py and mesh_converter.py. "
                "These modules were not found."
            )
    else:
        print(f"\nMode: SIMPLE GEOMETRY (internal mesh generator)")
    print("\nConfiguration:")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Mesh: {width}mm × {height}mm, {nx}×{ny} nodes")
    print(f"  Time: {n_timesteps} steps of {dt}s = {n_timesteps*dt}s total")
    print(f"  Load: {load} N/mm (tensile creep test)")
    print("="*70)

    # Export to synthetic_data/{experiment_id}/ for direct use by inverse problem
    if output_name is None:
        output_dir = Path(__file__).parent.parent.parent / "synthetic_data"
        experiment_name = str(experiment_id)
    else:
        # Custom output name (for backwards compatibility)
        output_dir = Path(__file__).parent.parent / "results"
        experiment_name = output_name

    data_dir = output_dir / experiment_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {data_dir}")

    # Step 1: Create or import mesh
    if use_complex_geometry and mesh_import_dir is None:
        # COMPLEX GEOMETRY BRANCH (804-806)
        print("\n[1/6] Building complex geometry with pygmsh...")

        # Load configuration
        print(f"  Loading configuration for experiment {experiment_id}...")
        config = load_config(experiment_id)

        # Override simulation params if provided
        if dt != 1.0:
            config['simulation_params']['dt'] = dt
        if n_timesteps != 600:
            config['simulation_params']['n_timesteps'] = n_timesteps
        if load != 50.0:
            config['simulation_params']['load'] = load

        # Override mesh parameters if provided
        if mesh_size_outer is not None:
            config['mesh_params']['mesh_size_outer'] = mesh_size_outer
        if mesh_size_hole is not None:
            config['mesh_params']['mesh_size_hole'] = mesh_size_hole

        # Update domain dimensions
        width = config['domain']['width']
        height = config['domain']['height']
        dt = config['simulation_params']['dt']
        n_timesteps = config['simulation_params']['n_timesteps']
        load = config['simulation_params']['load']

        # Build geometry with pygmsh
        builder = GeometryBuilder(config)
        mesh_obj = builder.build()

        # Convert to EUCLID format
        print(f"\n  Converting to EUCLID format...")
        temp_dir = Path(__file__).parent.parent.parent / "synthetic_data" / f"{experiment_id}_temp"
        converter = MeshConverter(mesh_obj, width, height)
        coord, conne = converter.convert()
        converter.save(coord, conne, temp_dir)

        # Save mesh visualization
        builder.save_mesh(mesh_obj, temp_dir)

        # Use converted mesh
        mesh_import_dir = temp_dir

        print(f"  [OK] Complex geometry created and converted")

    if mesh_import_dir is not None:
        # Import pre-existing mesh
        print("\n[1/6] Importing pre-existing mesh...")
        mesh_import_dir = Path(mesh_import_dir)
        print(f"  Loading from: {mesh_import_dir}")

        from Forward_solver.core.mesh import MeshLoader
        coord, conne = MeshLoader.load(
            mesh_import_dir / 'coord.csv',
            mesh_import_dir / 'conne.txt'
        )

        # Create minimal mesh object for compatibility
        # Extract nodes from coord
        nodes_xy = coord[:, 1:3]  # x, y columns

        # Create a minimal MeshGenerator-compatible object
        mesh = MeshGenerator(width=width, height=height, nx=nx, ny=ny)
        mesh.coord = coord
        mesh.conne = conne
        mesh.nodes = [Node(i, nodes_xy[i, 0], nodes_xy[i, 1]) for i in range(len(nodes_xy))]
        mesh.triangulation = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], conne)

        print(f"  [OK] Mesh imported: {len(mesh.nodes)} nodes, {len(conne)} elements")
    else:
        # Generate new mesh (simple geometry)
        print("\n[1/6] Creating simple mesh...")
        mesh = MeshGenerator(width=width, height=height, nx=nx, ny=ny)
        coord, conne = mesh.generate()
        print(f"  [OK] Mesh created: {len(mesh.nodes)} nodes, {len(conne)} elements")

        if holes:
            print(f"  [WARNING] 'holes' parameter is deprecated for experiment {experiment_id}.")
            print(f"            For complex geometries, use experiment_id >= 804 with geometry configs.")

    # Step 2: Create material
    print("\n[2/6] Loading material ...")
    material = create_reference_material()
    print(f"  [OK] Material loaded: {material}")

    # Step 3: Run simulation
    print("\n[3/6] Running forward simulation...")
    print(f"  This may take a few minutes for {n_timesteps} timesteps...")

    generator = SyntheticDataGenerator(
        mesh=mesh,
        material=material,
        dt=dt,
        n_timesteps=n_timesteps,
        load_magnitude=load
    )

    time, U_history, F_history = generator.generate()
    print(f"  [OK] Simulation complete")

    # Step 4: Export data
    print("\n[4/6] Exporting data files...")
    generator.export(output_dir, experiment_name=experiment_name)
    print(f"  [OK] Data exported to: {data_dir}")

    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    viz_dir = data_dir

    plot_mesh(mesh, viz_dir)
    plot_material_relaxation(material, viz_dir)
    plot_displacement_field(time, U_history, mesh, viz_dir)
    plot_displacement_evolution(time, U_history, mesh, viz_dir)
    plot_stress_strain(time, U_history, mesh, material, viz_dir)
    plot_field_contours(time, U_history, mesh, material, viz_dir)
    plot_deformed_shapes(time, U_history, mesh, viz_dir)

    print(f"  [OK] All visualizations saved")

    # Step 6: Create report
    print("\n[6/6] Creating simulation report...")
    config = {
        'width': width,
        'height': height,
        'nx': nx,
        'ny': ny,
        'material': material,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'load': load
    }
    create_simulation_report(config, viz_dir)
    print(f"  [OK] Report created")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print(f"\nSynthetic data saved to: {data_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review data files in {data_dir}:")
    print(f"     - coord.csv, conne.txt (mesh)")
    print(f"     - U.csv (displacements)")
    print(f"     - F.csv (forces - 4 rows: top, bottom, zeros)")
    print(f"     - time.csv (time vector)")
    print(f"  2. Check plots and SIMULATION_REPORT.txt")
    print(f"  3. Run inverse problem:")
    print(f"     cd ../inverse_problem")
    print(f"     python run_experiment.py {experiment_id}")
    print("="*70)

    return data_dir


if __name__ == "__main__":
    """
    Edit the parameters below and run: python run_full_simulation.py
    """

    # ====================================================================
    # CONFIGURATION - EDIT THESE PARAMETERS
    # ====================================================================

    # Experiment ID (800-802: simple geometry, 804-806: complex geometry)
    experiment_id = 806

    # Time parameters
    dt = 1.0              # Time step [s]
    n_timesteps = 600     # Number of timesteps

    # Domain size (optional - uses config default if None)
    width = 20.0          # Domain width [mm]
    height = 60.0         # Domain height [mm]
    load = 50.0           # Load magnitude [N/mm]

    # Simple geometry mesh parameters (only for experiments 800-802)
    nx = 12                # Grid points in x-direction
    ny = 30               # Grid points in y-direction

    # Complex geometry mesh parameters (only for experiments 804-806)
    mesh_size_outer = 2.0  # Outer boundary element size [mm]
    mesh_size_hole = 0.5   # Hole boundary element size [mm]
                           # Smaller = smoother circles, more elements
                           # 0.15 = very smooth (~13k elements)
                           # 0.3  = moderate (~2-3k elements)
                           # 0.5  = coarse (~800-1k elements)

    # ====================================================================
    # RUN SIMULATION
    # ====================================================================

    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT {experiment_id}")
    print("="*70)

    result = run_full_simulation(
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        dt=dt,
        n_timesteps=n_timesteps,
        load=load,
        experiment_id=experiment_id,
        mesh_size_outer=mesh_size_outer,
        mesh_size_hole=mesh_size_hole
    )

    if result:
        print("\n" + "="*70)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SIMULATION FAILED")
        print("="*70)
        sys.exit(1)
