"""
Verification mode for forward solver.

This script validates identified material parameters by:
1. Loading F.csv (time-varying forces from experiment/inverse problem)
2. Loading identified material parameters
3. Running forward solver with these forces to compute U_predicted
4. Comparing U_predicted with U_measured from experiment
5. Reporting error metrics

This does NOT modify the existing forward solver pipeline.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Forward_solver.core.mesh import MeshGenerator, MeshLoader, Node
from Forward_solver.core.material import ViscoelasticMaterial
from Forward_solver.core.time_integration import ForwardTimeIntegrator
from Forward_solver.core.assembly import ForwardAssembler
import matplotlib.tri as mtri


class VerificationSolver:
    """
    Forward solver that uses pre-computed force vectors from F.csv.

    Unlike the standard forward solver which computes forces from boundary conditions,
    this solver reads time-varying forces directly from a file.
    """

    def __init__(self, mesh, material: ViscoelasticMaterial, time: np.ndarray,
                 F_nodal: np.ndarray, U_measured: np.ndarray = None,
                 bc_type: str = 'roller'):
        """
        Initialize verification solver.

        Args:
            mesh: MeshGenerator or compatible mesh object
            material: ViscoelasticMaterial with identified parameters
            time: Time vector (n_timesteps,)
            F_nodal: Nodal force vector (n_dofs, n_timesteps) - from F.csv expansion
            U_measured: Optional measured displacement (2*n_nodes, n_timesteps) for comparison
            bc_type: Boundary condition type ('clamped' for experimental, 'roller' for synthetic)
        """
        self.mesh = mesh
        self.material = material
        self.time = time
        self.F_nodal = F_nodal
        self.U_measured = U_measured
        self.bc_type = bc_type

        self.n_nodes = len(mesh.nodes)
        self.n_dofs = 2 * self.n_nodes
        self.n_timesteps = len(time)

        # Compute dt from time vector
        self.dt = np.zeros(self.n_timesteps)
        self.dt[0] = time[1] - time[0] if len(time) > 1 else 1.0
        self.dt[1:] = time[1:] - time[:-1]

        # Initialize components
        self.integrator = ForwardTimeIntegrator(mesh, material)
        self.assembler = ForwardAssembler(mesh, material, self.integrator)

        # Identify boundary conditions (fixed DOFs)
        self._setup_boundary_conditions(bc_type=bc_type)

        # Storage for results
        self.U_predicted = np.zeros((self.n_dofs, self.n_timesteps))

        print(f"\nVerification Solver Initialized:")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  DOFs: {self.n_dofs}")
        print(f"  Timesteps: {self.n_timesteps}")
        print(f"  Time range: [{time[0]:.2f}, {time[-1]:.2f}] s")
        print(f"  dt range: [{self.dt.min():.4f}, {self.dt.max():.4f}] s")

    def _setup_boundary_conditions(self, bc_type: str = 'clamped'):
        """
        Setup boundary conditions from mesh.

        Args:
            bc_type: Boundary condition type
                - 'clamped': Fully clamped at both top and bottom (experimental setup)
                    * Bottom: Ux=0, Uy=0 for all nodes
                    * Top: Ux=0 for all nodes (clamped horizontally, force applied vertically)
                - 'roller': Minimal constraints (original synthetic data setup)
                    * Bottom: Uy=0 for all nodes (rollers)
                    * Single pin: Ux=0 for one node (prevent rigid body motion)
        """
        # Identify boundary nodes from coord array
        self.bottom_nodes = np.where(self.mesh.coord[:, 3] == 2)[0]
        self.top_nodes = np.where(self.mesh.coord[:, 3] == 1)[0]

        # Fixed DOFs
        fixed_dofs = []

        if bc_type == 'clamped':
            # ===== CLAMPED BOUNDARY CONDITIONS (matching experimental setup) =====
            # Bottom: Ux=0, Uy=0 for ALL bottom nodes (fully clamped)
            dof_bottom_x = 2 * self.bottom_nodes       # Ux DOFs
            dof_bottom_y = 2 * self.bottom_nodes + 1   # Uy DOFs
            fixed_dofs.extend(dof_bottom_x)
            fixed_dofs.extend(dof_bottom_y)

            # Top: Ux=0 for ALL top nodes (clamped horizontally)
            # Note: Uy is free (force-controlled from F.csv)
            dof_top_x = 2 * self.top_nodes             # Ux DOFs
            fixed_dofs.extend(dof_top_x)

            print(f"  BC Type: CLAMPED (experimental setup)")
            print(f"    Bottom: Ux=0, Uy=0 for {len(self.bottom_nodes)} nodes")
            print(f"    Top: Ux=0 for {len(self.top_nodes)} nodes (Uy free, force-controlled)")

        elif bc_type == 'roller':
            # ===== ROLLER BOUNDARY CONDITIONS (minimal constraints) =====
            # Bottom: Uy = 0 for all nodes (rollers)
            dof_bottom_y = 2 * self.bottom_nodes + 1
            fixed_dofs.extend(dof_bottom_y)

            # Bottom 2nd node: Ux = 0 (prevent horizontal rigid body motion)
            if len(self.bottom_nodes) >= 2:
                dof_bottom_x_2nd = 2 * self.bottom_nodes[1]
                fixed_dofs.append(dof_bottom_x_2nd)

            print(f"  BC Type: ROLLER (minimal constraints)")
            print(f"    Bottom: Uy=0 for {len(self.bottom_nodes)} nodes")
            print(f"    Single pin: Ux=0 at node {self.bottom_nodes[1] if len(self.bottom_nodes) >= 2 else 'N/A'}")

        else:
            raise ValueError(f"Unknown BC type: {bc_type}. Use 'clamped' or 'roller'.")

        self.fixed_dofs = np.array(fixed_dofs, dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(self.n_dofs), self.fixed_dofs)

        print(f"  Fixed DOFs: {len(self.fixed_dofs)}")
        print(f"  Free DOFs: {len(self.free_dofs)}")

    def solve_timestep_0(self):
        """Solve first timestep with elastic response."""
        dt = self.dt[0]

        # Assemble elastic stiffness
        K_global = self.assembler.assemble_global_stiffness(dt, is_first_timestep=True)

        # Get force vector from F_nodal
        F = self.F_nodal[:, 0]

        # Apply boundary conditions
        K_free = K_global[self.free_dofs, :][:, self.free_dofs]
        F_free = F[self.free_dofs]

        # Solve
        U_free = spla.spsolve(K_free, F_free)

        # Expand to full DOF vector
        U = np.zeros(self.n_dofs)
        U[self.free_dofs] = U_free

        # Store
        self.U_predicted[:, 0] = U

        # Compute beta
        self.integrator.compute_beta_first_timestep(U, dt, timestep=0)
        self.integrator.finalize_timestep()

        return U

    def solve_timestep(self, nt: int):
        """Solve timestep n > 0 with viscoelastic response."""
        dt = self.dt[nt]

        # Assemble viscoelastic stiffness
        K_global = self.assembler.assemble_global_stiffness(dt, is_first_timestep=False)

        # Assemble history force
        F_hist = self.assembler.assemble_history_force()

        # Get external force from F_nodal
        F_ext = self.F_nodal[:, nt]

        # Total force
        F = F_ext + F_hist

        # Apply boundary conditions
        K_free = K_global[self.free_dofs, :][:, self.free_dofs]
        F_free = F[self.free_dofs]

        # Solve
        U_free = spla.spsolve(K_free, F_free)

        # Expand
        U = np.zeros(self.n_dofs)
        U[self.free_dofs] = U_free

        # Store
        self.U_predicted[:, nt] = U

        # Compute beta
        self.integrator.compute_beta_timestep(U, dt, timestep=nt)
        self.integrator.finalize_timestep()

        return U

    def solve(self):
        """Run complete verification simulation."""
        print("\n" + "="*70)
        print("RUNNING VERIFICATION SIMULATION")
        print("="*70)

        # Timestep 0
        print(f"\nTimestep 0 (t={self.time[0]:.2f}s)...")
        self.solve_timestep_0()

        # Remaining timesteps
        for nt in range(1, self.n_timesteps):
            self.solve_timestep(nt)

            if nt % 100 == 0 or nt == self.n_timesteps - 1:
                max_u = np.abs(self.U_predicted[:, nt]).max()
                print(f"  Timestep {nt}/{self.n_timesteps} (t={self.time[nt]:.2f}s) - Max |U|: {max_u:.6e} mm")

        print("\nSimulation complete!")
        return self.U_predicted

    def compute_errors(self):
        """
        Compute error metrics between predicted and measured displacements.

        Returns:
            dict: Error metrics
        """
        if self.U_measured is None:
            raise ValueError("U_measured not provided. Cannot compute errors.")

        # Convert U_predicted from interleaved to separated format if needed
        # U_predicted is (2*n_nodes, n_timesteps) in interleaved format [u0x, u0y, u1x, u1y, ...]
        # U_measured is (2*n_nodes, n_timesteps) in separated format [u0x, u1x, ..., u0y, u1y, ...]

        U_pred_x = self.U_predicted[0::2, :]  # (n_nodes, n_timesteps)
        U_pred_y = self.U_predicted[1::2, :]  # (n_nodes, n_timesteps)
        U_pred_separated = np.vstack([U_pred_x, U_pred_y])  # (2*n_nodes, n_timesteps)

        # Reference frame correction: shift predicted displacements so that t=0 matches measured (zero)
        # This accounts for the fact that DIC uses the first frame as reference (U=0 at t=0)
        # while the solver computes actual displacement from non-zero initial force
        U_pred_initial = U_pred_separated[:, 0:1]  # Keep as 2D for broadcasting
        U_pred_separated = U_pred_separated - U_pred_initial

        print(f"  Reference frame correction applied: subtracted initial displacement")
        print(f"    Initial Ux range: [{U_pred_initial[:self.n_nodes].min():.6f}, {U_pred_initial[:self.n_nodes].max():.6f}] mm")
        print(f"    Initial Uy range: [{U_pred_initial[self.n_nodes:].min():.6f}, {U_pred_initial[self.n_nodes:].max():.6f}] mm")

        # Compute errors
        error = U_pred_separated - self.U_measured

        # RMSE per timestep
        rmse_per_timestep = np.sqrt(np.mean(error**2, axis=0))

        # Overall RMSE
        rmse_total = np.sqrt(np.mean(error**2))

        # Relative error (normalized by max displacement)
        max_measured = np.abs(self.U_measured).max()
        relative_error = rmse_total / max_measured if max_measured > 0 else np.inf

        # Max absolute error
        max_abs_error = np.abs(error).max()

        # Error per component (using reference-frame-corrected predictions)
        error_x = U_pred_separated[:self.n_nodes, :] - self.U_measured[:self.n_nodes, :]
        error_y = U_pred_separated[self.n_nodes:, :] - self.U_measured[self.n_nodes:, :]
        rmse_x = np.sqrt(np.mean(error_x**2))
        rmse_y = np.sqrt(np.mean(error_y**2))

        # Component-wise RMSE per timestep
        rmse_x_per_timestep = np.sqrt(np.mean(error_x**2, axis=0))
        rmse_y_per_timestep = np.sqrt(np.mean(error_y**2, axis=0))

        errors = {
            'rmse_total': rmse_total,
            'rmse_per_timestep': rmse_per_timestep,
            'rmse_x_per_timestep': rmse_x_per_timestep,
            'rmse_y_per_timestep': rmse_y_per_timestep,
            'relative_error': relative_error,
            'max_abs_error': max_abs_error,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'error_field': error,
            'U_pred_separated': U_pred_separated
        }

        return errors


def load_experiment_data(experiment_dir: Path):
    import pandas as pd
    """Load experimental data using robust parsing matching your data loader."""
    print(f"Loading experiment data from: {experiment_dir}")
    
    data_dir = experiment_dir
    
    # Load coordinates (skip header row 0, no pandas header)
    coord_file = data_dir / "coord.csv"
    if not coord_file.exists():
        raise FileNotFoundError(f"coord.csv not found in {data_dir}")
    nodes = pd.read_csv(coord_file, header=None, skiprows=1)
    coord = nodes.values
    
    # Load connectivity (robust multi-delimiter parsing)
    conne_file = data_dir / "conne.txt"
    if not conne_file.exists():
        raise FileNotFoundError(f"conne.txt not found in {data_dir}")
    
    import re
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
                    data.append(nums[:4])  # Take first 4 columns
            except (ValueError, IndexError):
                continue
    
    if not data:
        raise ValueError(f"conne.txt unreadable at {conne_file}")
    
    conne_raw = pd.DataFrame(data)
    conne = conne_raw.iloc[:, 1:4].values.astype(np.int64)  # Columns 1:4 (0-indexed 1,2,3)
    conne = conne - 1  # Convert to 0-indexed
    
    # Load displacements (pandas fallback to numpy)
    u_file = data_dir / "U.csv"
    if not u_file.exists():
        raise FileNotFoundError(f"U.csv not found in {data_dir}")
    try:
        U = pd.read_csv(u_file, header=None).values
    except:
        U = np.loadtxt(u_file, delimiter=',', skiprows=0)
    
    # Load forces (pandas fallback) - keep in kN as stored in F.csv
    f_file = data_dir / "F.csv"
    if not f_file.exists():
        raise FileNotFoundError(f"F.csv not found in {data_dir}")
    try:
        F_reduced = pd.read_csv(f_file, header=None).values
    except:
        F_reduced = np.loadtxt(f_file, delimiter=',', skiprows=0)
    # F_reduced is in kN - DO NOT convert here, expand_forces_to_nodal handles units
    
    # Load time (flatten)
    time_file = data_dir / "time.csv"
    if not time_file.exists():
        raise FileNotFoundError(f"time.csv not found in {data_dir}")
    try:
        time = pd.read_csv(time_file, header=None).values.flatten()
    except:
        time = np.loadtxt(time_file, delimiter=',', skiprows=0)
    
    nnodes = coord.shape[0]
    ntimesteps = len(time)
    
    print(f"[OK] Loaded: {nnodes} nodes, {conne.shape[0]} elements, {ntimesteps} timesteps")
    print(f"U shape: {U.shape}")
    print(f"F_reduced shape: {F_reduced.shape}")
    print(f"F_reduced[0] range: [{F_reduced[0,:].min():.4f}, {F_reduced[0,:].max():.4f}] kN")
    print(f"F_reduced[0] = [{F_reduced[0,:].min()*1000:.1f}, {F_reduced[0,:].max()*1000:.1f}] N")
    print(f"U_measured max Uy: {U[nnodes:,:].max():.6f} mm")
    
    return {
        'coord': coord,
        'conne': conne,
        'U': U,
        'F_reduced': F_reduced,
        'time': time,
        'nnodes': nnodes,
        'ntimesteps': ntimesteps
    }


def expand_forces_to_nodal(F_reduced: np.ndarray, mesh, time: np.ndarray):
    """
    Expand reduced force representation to full nodal force vector.

    F_reduced format (4 rows):
        Row 0: Applied load (constant or time-varying)
        Row 1: Reaction at bottom (not used for forward solve)
        Row 2-3: Zeros

    The applied load is distributed to top boundary nodes using
    proper edge integration (matching the original forward solver).

    Args:
        F_reduced: (4, n_timesteps) reduced force array
        mesh: Mesh object with nodes and connectivity
        time: Time vector

    Returns:
        F_nodal: (2*n_nodes, n_timesteps) nodal force array
    """
    n_nodes = len(mesh.nodes)
    n_dofs = 2 * n_nodes
    n_timesteps = len(time)

    F_nodal = np.zeros((n_dofs, n_timesteps))

    # Identify top boundary nodes and elements
    top_nodes = np.where(mesh.coord[:, 3] == 1)[0]
    nodes_xy = np.array([[n.x, n.y] for n in mesh.nodes])

    # Find top edge elements
    top_elements = []
    for ie, element_nodes in enumerate(mesh.conne):
        n_top = 0
        top_node_indices = []
        for local_idx, node_idx in enumerate(element_nodes):
            if node_idx in top_nodes:
                n_top += 1
                top_node_indices.append(local_idx)

        if n_top >= 2:
            top_elements.append({
                'element_id': ie,
                'nodes': element_nodes,
                'top_local_indices': top_node_indices
            })

    # Compute total top edge length
    total_edge_length = 0.0
    for elem_info in top_elements:
        element_nodes = elem_info['nodes']
        top_local = elem_info['top_local_indices']
        if len(top_local) >= 2:
            p1 = nodes_xy[element_nodes[top_local[0]]]
            p2 = nodes_xy[element_nodes[top_local[1]]]
            total_edge_length += np.linalg.norm(p2 - p1)

    print(f"  Top edge length: {total_edge_length:.2f} mm")
    print(f"  Top elements: {len(top_elements)}")

    # For each timestep, distribute the applied load to top nodes
    for nt in range(n_timesteps):
        # F_reduced[0, nt] is total applied force in kN (from F.csv)
        # Convert to traction: q = F_total_N / edge_length [N/mm]
        F_total_kN = F_reduced[0, nt]
        F_total_N = F_total_kN * 1000.0  # kN to N

        if total_edge_length > 0:
            traction = F_total_N / total_edge_length  # N/mm = MPa (for 1mm thickness)
        else:
            traction = 0.0

        # Traction vector (vertical only for tensile test)
        q = np.array([0.0, traction])

        # Distribute to nodes
        for elem_info in top_elements:
            element_nodes = elem_info['nodes']
            coords = nodes_xy[element_nodes]
            top_local = elem_info['top_local_indices']

            if len(top_local) < 2:
                continue

            # Edge length
            p1 = coords[top_local[0]]
            p2 = coords[top_local[1]]
            edge_length = np.linalg.norm(p2 - p1)

            # Force per node (half each)
            force_per_node = q * edge_length / 2.0

            # Assemble
            for local_idx in top_local:
                global_node = element_nodes[local_idx]
                dof_x = 2 * global_node
                dof_y = 2 * global_node + 1

                F_nodal[dof_x, nt] += force_per_node[0]
                F_nodal[dof_y, nt] += force_per_node[1]

    print(f"  F_nodal shape: {F_nodal.shape}")
    print(f"  F_nodal[y] sum at t=0: {F_nodal[1::2, 0].sum():.2f} N (should match F_reduced[0,0]*1000)")
    print(f"  F_nodal[y] sum at t=10: {F_nodal[1::2, 10].sum():.2f} N")
    print(f"  Traction at t=10: {F_reduced[0,10]*1000/total_edge_length:.2f} N/mm = MPa")

    return F_nodal


def create_material_from_identified(identified_params: dict) -> ViscoelasticMaterial:
    """
    Create ViscoelasticMaterial from identified parameters.

    Args:
        identified_params: Dictionary with G_inf, G, tau_G, K_inf, K, tau_K

    Returns:
        ViscoelasticMaterial instance
    """
    return ViscoelasticMaterial(
        G_inf=identified_params['G_inf'],
        G=np.array(identified_params['G']),
        tau_G=np.array(identified_params['tau_G']),
        K_inf=identified_params['K_inf'],
        K=np.array(identified_params['K']),
        tau_K=np.array(identified_params['tau_K'])
    )


def plot_verification_results(solver: VerificationSolver, errors: dict, save_dir: Path):
    """
    Create plots comparing predicted and measured displacements.
    """
    time = solver.time
    U_pred = errors['U_pred_separated']
    U_meas = solver.U_measured
    n_nodes = solver.n_nodes

    # Find representative nodes
    nodes_xy = np.array([[n.x, n.y] for n in solver.mesh.nodes])

    # Top center node
    top_center = np.argmin(np.abs(nodes_xy[:, 0] - solver.mesh.width/2) +
                          np.abs(nodes_xy[:, 1] - solver.mesh.height))

    # Mid center node
    mid_center = np.argmin(np.abs(nodes_xy[:, 0] - solver.mesh.width/2) +
                          np.abs(nodes_xy[:, 1] - solver.mesh.height/2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Uy at top center
    ax = axes[0, 0]
    ax.plot(time, U_meas[n_nodes + top_center, :], 'b-', linewidth=2, label='Measured')
    ax.plot(time, U_pred[n_nodes + top_center, :], 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Uy [mm]')
    ax.set_title(f'Top Center Node (id={top_center})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Uy at mid center
    ax = axes[0, 1]
    ax.plot(time, U_meas[n_nodes + mid_center, :], 'b-', linewidth=2, label='Measured')
    ax.plot(time, U_pred[n_nodes + mid_center, :], 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Uy [mm]')
    ax.set_title(f'Mid Center Node (id={mid_center})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: RMSE over time
    ax = axes[0, 2]
    ax.plot(time, errors['rmse_per_timestep'], 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RMSE [mm]')
    ax.set_title('RMSE per Timestep')
    ax.grid(True, alpha=0.3)

    # Plot 4: Error at top center
    ax = axes[1, 0]
    error_top = U_pred[n_nodes + top_center, :] - U_meas[n_nodes + top_center, :]
    ax.plot(time, error_top, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [mm]')
    ax.set_title('Error at Top Center')
    ax.grid(True, alpha=0.3)

    # Plot 5: Max displacement comparison
    ax = axes[1, 1]
    max_meas = np.max(np.abs(U_meas), axis=0)
    max_pred = np.max(np.abs(U_pred), axis=0)
    ax.plot(time, max_meas, 'b-', linewidth=2, label='Measured')
    ax.plot(time, max_pred, 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Max |U| [mm]')
    ax.set_title('Maximum Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Error metrics summary
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = (
        f"Error Metrics Summary\n"
        f"{'='*30}\n\n"
        f"RMSE (total): {errors['rmse_total']:.6e} mm\n"
        f"RMSE (Ux):    {errors['rmse_x']:.6e} mm\n"
        f"RMSE (Uy):    {errors['rmse_y']:.6e} mm\n"
        f"Max Error:    {errors['max_abs_error']:.6e} mm\n"
        f"Relative:     {errors['relative_error']*100:.2f} %\n"
    )
    ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle('Verification: Predicted vs Measured Displacements', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'verification_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: verification_comparison.png")
    plt.close()


def plot_displacement_field_comparison(solver: VerificationSolver, errors: dict, save_dir: Path,
                                        timesteps_to_plot: list = None):
    """
    Create heatmap comparison of measured vs predicted displacement fields.

    Shows side-by-side contour plots of Ux, Uy for measured, predicted, and error.
    """
    U_pred = errors['U_pred_separated']
    U_meas = solver.U_measured
    n_nodes = solver.n_nodes
    time = solver.time

    nodes_xy = np.array([[n.x, n.y] for n in solver.mesh.nodes])
    triang = solver.mesh.triangulation

    # Select timesteps to plot
    if timesteps_to_plot is None:
        # Plot at 3 key times: early, middle, end
        n_t = len(time)
        timesteps_to_plot = [
            min(10, n_t-1),      # Early (after ramp)
            n_t // 2,            # Middle
            n_t - 1              # End
        ]

    for t_idx in timesteps_to_plot:
        t_val = time[t_idx]

        # Extract displacements at this timestep
        Ux_meas = U_meas[:n_nodes, t_idx]
        Uy_meas = U_meas[n_nodes:, t_idx]
        Ux_pred = U_pred[:n_nodes, t_idx]
        Uy_pred = U_pred[n_nodes:, t_idx]

        # Compute errors
        Ux_error = Ux_pred - Ux_meas
        Uy_error = Uy_pred - Uy_meas

        # Create figure with 3 rows (Measured, Predicted, Error) x 2 cols (Ux, Uy)
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))

        # Common colorbar limits for Ux
        vmin_x = min(Ux_meas.min(), Ux_pred.min())
        vmax_x = max(Ux_meas.max(), Ux_pred.max())

        # Common colorbar limits for Uy
        vmin_y = min(Uy_meas.min(), Uy_pred.min())
        vmax_y = max(Uy_meas.max(), Uy_pred.max())

        # Error limits (symmetric around 0)
        err_max_x = max(abs(Ux_error.min()), abs(Ux_error.max()))
        err_max_y = max(abs(Uy_error.min()), abs(Uy_error.max()))

        # Row 0: Measured
        ax = axes[0, 0]
        tpc = ax.tripcolor(triang, Ux_meas, shading='gouraud', cmap='viridis',
                          vmin=vmin_x, vmax=vmax_x)
        plt.colorbar(tpc, ax=ax, label='Ux [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title('Measured Ux')
        ax.set_aspect('equal')

        ax = axes[0, 1]
        tpc = ax.tripcolor(triang, Uy_meas, shading='gouraud', cmap='viridis',
                          vmin=vmin_y, vmax=vmax_y)
        plt.colorbar(tpc, ax=ax, label='Uy [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title('Measured Uy')
        ax.set_aspect('equal')

        # Row 1: Predicted
        ax = axes[1, 0]
        tpc = ax.tripcolor(triang, Ux_pred, shading='gouraud', cmap='viridis',
                          vmin=vmin_x, vmax=vmax_x)
        plt.colorbar(tpc, ax=ax, label='Ux [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title('Predicted Ux')
        ax.set_aspect('equal')

        ax = axes[1, 1]
        tpc = ax.tripcolor(triang, Uy_pred, shading='gouraud', cmap='viridis',
                          vmin=vmin_y, vmax=vmax_y)
        plt.colorbar(tpc, ax=ax, label='Uy [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title('Predicted Uy')
        ax.set_aspect('equal')

        # Row 2: Error (diverging colormap centered at 0)
        ax = axes[2, 0]
        tpc = ax.tripcolor(triang, Ux_error, shading='gouraud', cmap='RdBu_r',
                          vmin=-err_max_x, vmax=err_max_x)
        plt.colorbar(tpc, ax=ax, label='Error [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(f'Error Ux (RMSE={np.sqrt(np.mean(Ux_error**2)):.4f} mm)')
        ax.set_aspect('equal')

        ax = axes[2, 1]
        tpc = ax.tripcolor(triang, Uy_error, shading='gouraud', cmap='RdBu_r',
                          vmin=-err_max_y, vmax=err_max_y)
        plt.colorbar(tpc, ax=ax, label='Error [mm]')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(f'Error Uy (RMSE={np.sqrt(np.mean(Uy_error**2)):.4f} mm)')
        ax.set_aspect('equal')

        plt.suptitle(f'Displacement Field Comparison at t = {t_val:.1f} s (timestep {t_idx})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f'field_comparison_t{t_idx:03d}.png'
        plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    # Also create a summary figure showing error magnitude over the entire field at final time
    t_idx = timesteps_to_plot[-1]
    t_val = time[t_idx]

    Ux_error = U_pred[:n_nodes, t_idx] - U_meas[:n_nodes, t_idx]
    Uy_error = U_pred[n_nodes:, t_idx] - U_meas[n_nodes:, t_idx]
    U_error_magnitude = np.sqrt(Ux_error**2 + Uy_error**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Error magnitude
    ax = axes[0]
    tpc = ax.tripcolor(triang, U_error_magnitude, shading='gouraud', cmap='hot')
    plt.colorbar(tpc, ax=ax, label='|Error| [mm]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'Error Magnitude at t={t_val:.1f}s')
    ax.set_aspect('equal')

    # Relative error (as percentage of local displacement magnitude)
    U_meas_magnitude = np.sqrt(U_meas[:n_nodes, t_idx]**2 + U_meas[n_nodes:, t_idx]**2)
    relative_error_field = np.divide(U_error_magnitude, U_meas_magnitude,
                                     out=np.zeros_like(U_error_magnitude),
                                     where=U_meas_magnitude > 1e-10) * 100

    ax = axes[1]
    tpc = ax.tripcolor(triang, relative_error_field, shading='gouraud', cmap='YlOrRd',
                      vmin=0, vmax=min(50, relative_error_field.max()))
    plt.colorbar(tpc, ax=ax, label='Relative Error [%]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'Relative Error at t={t_val:.1f}s')
    ax.set_aspect('equal')

    # Histogram of errors
    ax = axes[2]
    ax.hist(U_error_magnitude, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(U_error_magnitude), color='red', linestyle='--',
               label=f'Mean: {np.mean(U_error_magnitude):.4f} mm')
    ax.axvline(np.median(U_error_magnitude), color='green', linestyle='--',
               label=f'Median: {np.median(U_error_magnitude):.4f} mm')
    ax.set_xlabel('Error Magnitude [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Error Summary at t = {t_val:.1f} s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'error_summary.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: error_summary.png")
    plt.close()


# ============================================================================
# THESIS-QUALITY PLOTTING FUNCTIONS
# ============================================================================

def setup_thesis_style():
    """Setup matplotlib style for thesis-quality figures."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['lines.linewidth'] = 1.5


def plot_thesis_temporal_verification(solver: VerificationSolver, errors: dict,
                                       save_dir: Path, experiment_id: int = 904):
    """
    Fig 5.4-2: Temporal verification plot for thesis.

    Layout:
        Top: 1-2 node curves (measured vs predicted), Uy only
        Bottom: RMSE vs time (global)

    Size: 152mm × 100mm (≈ 6" × 3.9")
    """
    setup_thesis_style()

    time = solver.time
    U_pred = errors['U_pred_separated']
    U_meas = solver.U_measured
    n_nodes = solver.n_nodes

    # Find representative nodes
    nodes_xy = np.array([[n.x, n.y] for n in solver.mesh.nodes])

    # Top center node (main node of interest)
    top_center = np.argmin(np.abs(nodes_xy[:, 0] - solver.mesh.width/2) +
                          np.abs(nodes_xy[:, 1] - solver.mesh.height))

    # Mid center node
    mid_center = np.argmin(np.abs(nodes_xy[:, 0] - solver.mesh.width/2) +
                          np.abs(nodes_xy[:, 1] - solver.mesh.height/2))

    # Figure size: 152mm × 100mm ≈ 6" × 3.9"
    fig, axes = plt.subplots(2, 1, figsize=(6, 3.9), dpi=300)

    # --- Top: Node displacement curves (Uy only) ---
    ax = axes[0]

    # Top center node
    ax.plot(time, U_meas[n_nodes + top_center, :], 'b-', linewidth=1.5,
            label=f'Measured (node {top_center})')
    ax.plot(time, U_pred[n_nodes + top_center, :], 'r--', linewidth=1.5,
            label=f'Predicted (node {top_center})')

    # Mid center node (different markers)
    ax.plot(time, U_meas[n_nodes + mid_center, :], 'b-', linewidth=1.0, alpha=0.6,
            label=f'Measured (node {mid_center})')
    ax.plot(time, U_pred[n_nodes + mid_center, :], 'r--', linewidth=1.0, alpha=0.6,
            label=f'Predicted (node {mid_center})')

    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('$u_y$ [mm]')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([time[0], time[-1]])

    # --- Bottom: RMSE vs time ---
    ax = axes[1]
    ax.plot(time, errors['rmse_per_timestep'], 'k-', linewidth=1.5)
    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('RMSE [mm]')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([time[0], time[-1]])

    # Add mean RMSE line
    mean_rmse = np.mean(errors['rmse_per_timestep'])
    ax.axhline(mean_rmse, color='red', linestyle='--', linewidth=1.0, alpha=0.7,
               label=f'Mean: {mean_rmse:.4f} mm')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')

    plt.tight_layout()

    filename = f'verification_comparison_{experiment_id}.png'
    plt.savefig(save_dir / filename, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved thesis figure: {filename}")
    plt.close()


def plot_thesis_spatial_error(solver: VerificationSolver, errors: dict,
                               save_dir: Path, experiment_id: int = 904):
    """
    Fig 5.4-3: Spatial error at final time for thesis.

    Layout:
        Left: absolute error map |uy_pred - uy_meas| at final time
        Right: histogram of error magnitude

    Size: 152mm × 90mm (≈ 6" × 3.5")
    Uses tripcolor with gouraud shading (same as field snapshots).
    """
    setup_thesis_style()

    U_pred = errors['U_pred_separated']
    U_meas = solver.U_measured
    n_nodes = solver.n_nodes
    time = solver.time
    triang = solver.mesh.triangulation

    # Final timestep
    t_idx = len(time) - 1
    t_val = time[t_idx]

    # Compute Uy error at final time
    Uy_meas = U_meas[n_nodes:, t_idx]
    Uy_pred = U_pred[n_nodes:, t_idx]
    Uy_error = np.abs(Uy_pred - Uy_meas)

    # Figure size: 152mm × 90mm ≈ 6" × 3.5"
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4), dpi=300,
                             gridspec_kw={'width_ratios': [1.3, 1]})

    # --- Left: Absolute error map using tripcolor ---
    ax = axes[0]
    tpc = ax.tripcolor(triang, Uy_error, shading='gouraud', cmap='hot',
                       vmin=0, vmax=Uy_error.max())
    cbar = plt.colorbar(tpc, ax=ax, label='$|u_y^{\\mathrm{pred}} - u_y^{\\mathrm{meas}}|$ [mm]')
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlabel('$x$ [mm]', fontsize=10)
    ax.set_ylabel('$y$ [mm]', fontsize=10)
    ax.set_title(f'Absolute error at $t = {t_val:.0f}$ s', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=9)

    # --- Right: Histogram of error magnitude ---
    ax = axes[1]
    ax.hist(Uy_error, bins=30, color='steelblue', edgecolor='black',
            alpha=0.8, linewidth=0.5)

    mean_err = np.mean(Uy_error)
    median_err = np.median(Uy_error)

    ax.axvline(mean_err, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_err:.4f} mm')
    ax.axvline(median_err, color='green', linestyle='-.', linewidth=1.5,
               label=f'Median: {median_err:.4f} mm')

    ax.set_xlabel('$|\\Delta u_y|$ [mm]', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Error distribution', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()

    filename = f'error_summary_{experiment_id}.png'
    plt.savefig(save_dir / filename, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved thesis figure: {filename}")
    plt.close()


def plot_thesis_field_snapshots(solver: VerificationSolver, errors: dict,
                                 save_dir: Path, experiment_id: int = 904,
                                 times_to_plot: list = None):
    """
    Fig 5.4-4: Field snapshots combined for thesis.

    Layout: 3 rows × 3 columns
        Rows: t=10s, t=603s, t=1205s (or specified times)
        Columns: Measured uy, Predicted uy, Residual

    Uses same tripcolor approach as the working plot_displacement_field_comparison.
    """
    setup_thesis_style()

    U_pred = errors['U_pred_separated']
    U_meas = solver.U_measured
    n_nodes = solver.n_nodes
    time = solver.time
    triang = solver.mesh.triangulation

    # Find closest timesteps to requested times
    if times_to_plot is None:
        times_to_plot = [10, 603, 1205]

    timestep_indices = []
    actual_times = []
    for t_target in times_to_plot:
        idx = np.argmin(np.abs(time - t_target))
        timestep_indices.append(idx)
        actual_times.append(time[idx])

    # Compute global limits for consistent colorbars
    all_Uy_meas = []
    all_Uy_pred = []
    all_residuals = []

    for t_idx in timestep_indices:
        Uy_meas = U_meas[n_nodes:, t_idx]
        Uy_pred = U_pred[n_nodes:, t_idx]
        all_Uy_meas.append(Uy_meas)
        all_Uy_pred.append(Uy_pred)
        all_residuals.append(Uy_pred - Uy_meas)

    vmin_disp = min(np.min(all_Uy_meas), np.min(all_Uy_pred))
    vmax_disp = max(np.max(all_Uy_meas), np.max(all_Uy_pred))
    res_max = max(np.abs(np.min(all_residuals)), np.abs(np.max(all_residuals)))
    vmin_res, vmax_res = -res_max, res_max

    # Column headers
    col_titles = ['Measured $u_y$', 'Predicted $u_y$', 'Residual']

    # Figure: 3 rows x 3 cols + colorbar row
    fig, axes = plt.subplots(3, 3, figsize=(12, 16))

    # Store last mappables for colorbars
    tpc_disp = None
    tpc_res = None

    for row_idx, (t_idx, t_actual) in enumerate(zip(timestep_indices, actual_times)):
        Uy_meas = U_meas[n_nodes:, t_idx]
        Uy_pred = U_pred[n_nodes:, t_idx]
        residual = Uy_pred - Uy_meas

        # Column 0: Measured
        ax = axes[row_idx, 0]
        tpc = ax.tripcolor(triang, Uy_meas, shading='gouraud', cmap='viridis',
                          vmin=vmin_disp, vmax=vmax_disp)
        ax.set_aspect('equal')
        ax.set_ylabel(f'$t = {t_actual:.0f}$ s\n\nY [mm]', fontsize=10)
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=11, fontweight='bold')
        if row_idx == len(timestep_indices) - 1:
            ax.set_xlabel('X [mm]', fontsize=10)

        # Column 1: Predicted
        ax = axes[row_idx, 1]
        tpc_disp = ax.tripcolor(triang, Uy_pred, shading='gouraud', cmap='viridis',
                                vmin=vmin_disp, vmax=vmax_disp)
        ax.set_aspect('equal')
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=11, fontweight='bold')
        if row_idx == len(timestep_indices) - 1:
            ax.set_xlabel('X [mm]', fontsize=10)

        # Column 2: Residual
        ax = axes[row_idx, 2]
        tpc_res = ax.tripcolor(triang, residual, shading='gouraud', cmap='RdBu_r',
                               vmin=vmin_res, vmax=vmax_res)
        ax.set_aspect('equal')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=11, fontweight='bold')
        if row_idx == len(timestep_indices) - 1:
            ax.set_xlabel('X [mm]', fontsize=10)

    # Add colorbars
    # Displacement colorbar (spans first two columns)
    fig.subplots_adjust(bottom=0.08, right=0.92)
    cbar_ax1 = fig.add_axes([0.1, 0.03, 0.5, 0.015])
    cbar1 = fig.colorbar(tpc_disp, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('$u_y$ [mm]', fontsize=10)

    # Residual colorbar
    cbar_ax2 = fig.add_axes([0.68, 0.03, 0.24, 0.015])
    cbar2 = fig.colorbar(tpc_res, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('$\\Delta u_y$ [mm]', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    filename = f'field_comparison_{experiment_id}_combined.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved thesis figure: {filename}")
    plt.close()


def plot_thesis_figures(solver: VerificationSolver, errors: dict, save_dir: Path,
                        experiment_id: int = 904, times_to_plot: list = None):
    """
    Generate all thesis-quality figures for verification results.

    Args:
        solver: VerificationSolver instance
        errors: Dictionary of error metrics
        save_dir: Output directory
        experiment_id: Experiment number for filenames
        times_to_plot: List of times in seconds for field snapshots
    """
    print("\nGenerating thesis-quality figures...")

    # Fig 5.4-2: Temporal verification
    plot_thesis_temporal_verification(solver, errors, save_dir, experiment_id)

    # Fig 5.4-3: Spatial error at final time
    plot_thesis_spatial_error(solver, errors, save_dir, experiment_id)

    # Fig 5.4-4: Field snapshots combined
    if times_to_plot is None:
        times_to_plot = [10, 603, 1205]
    plot_thesis_field_snapshots(solver, errors, save_dir, experiment_id, times_to_plot)

    print("  Thesis figures complete!")


def plot_bc_comparison(time: np.ndarray, errors_roller: dict, errors_clamped: dict,
                        save_dir: Path, experiment_id: int = 904):
    """
    Plot RMSE comparison between roller and clamped boundary conditions.

    Creates a figure with:
        (a) RMSE(t) overlay (roller vs clamped)
        (b) RMSE_x(t) overlay (roller vs clamped)

    Args:
        time: Time vector
        errors_roller: Error dictionary from roller BC verification
        errors_clamped: Error dictionary from clamped BC verification
        save_dir: Output directory
        experiment_id: Experiment number for filename
    """
    setup_thesis_style()

    # Figure size: wider to accommodate legend outside
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

    # Colors
    color_roller = '#2ecc71'   # Green
    color_clamped = '#e74c3c'  # Red

    # --- (a) RMSE(t) comparison ---
    ax = axes[0]
    ax.plot(time, errors_roller['rmse_per_timestep'], color=color_roller,
            linewidth=1.5, label='Roller BC')
    ax.plot(time, errors_clamped['rmse_per_timestep'], color=color_clamped,
            linewidth=1.5, linestyle='--', label='Clamped BC')

    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('RMSE [mm]')
    ax.set_title('(a) Global RMSE$(t)$', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([time[0], time[-1]])

    # --- (b) RMSE_x(t) comparison ---
    ax = axes[1]
    ax.plot(time, errors_roller['rmse_x_per_timestep'], color=color_roller,
            linewidth=1.5, label='Roller BC')
    ax.plot(time, errors_clamped['rmse_x_per_timestep'], color=color_clamped,
            linewidth=1.5, linestyle='--', label='Clamped BC')

    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('RMSE$_x$ [mm]')
    ax.set_title('(b) $x$-component RMSE$_x(t)$', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([time[0], time[-1]])

    # Single legend outside, below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, framealpha=0.95,
               edgecolor='black', fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    filename = f'bc_comparison_{experiment_id}.png'
    plt.savefig(save_dir / filename, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved BC comparison figure: {filename}")
    plt.close()


def run_bc_comparison(experiment_dir: Path, identified_params: dict,
                       output_dir: Path = None, experiment_id: int = 904):
    """
    Run verification with both roller and clamped BCs and create comparison plot.

    Args:
        experiment_dir: Directory with experimental data
        identified_params: Dictionary with identified material parameters
        output_dir: Optional output directory for results
        experiment_id: Experiment number for filenames

    Returns:
        dict: Results from both BC types
    """
    print("="*70)
    print("BOUNDARY CONDITION COMPARISON")
    print("="*70)

    # Run with roller BC
    print("\n>>> Running with ROLLER boundary conditions...")
    results_roller = run_verification(
        experiment_dir=experiment_dir,
        identified_params=identified_params,
        output_dir=output_dir,
        bc_type='roller',
        experiment_id=experiment_id
    )

    # Run with clamped BC
    print("\n>>> Running with CLAMPED boundary conditions...")
    results_clamped = run_verification(
        experiment_dir=experiment_dir,
        identified_params=identified_params,
        output_dir=output_dir,
        bc_type='clamped',
        experiment_id=experiment_id
    )

    # Create comparison plot
    save_dir = results_roller['output_dir']
    time = results_roller['solver'].time

    plot_bc_comparison(
        time=time,
        errors_roller=results_roller['errors'],
        errors_clamped=results_clamped['errors'],
        save_dir=save_dir,
        experiment_id=experiment_id
    )

    # Print comparison summary
    print("\n" + "="*70)
    print("BC COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<25} {'Roller':<15} {'Clamped':<15} {'Difference':<15}")
    print("-"*70)

    for metric in ['rmse_total', 'rmse_x', 'rmse_y', 'max_abs_error']:
        val_r = results_roller['errors'][metric]
        val_c = results_clamped['errors'][metric]
        diff = val_c - val_r
        print(f"{metric:<25} {val_r:<15.6e} {val_c:<15.6e} {diff:<+15.6e}")

    print("="*70)

    return {
        'roller': results_roller,
        'clamped': results_clamped
    }


def run_verification(
    experiment_dir: Path,
    identified_params: dict,
    output_dir: Path = None,
    bc_type: str = 'roller',
    experiment_id: int = 904,
    times_to_plot: list = None
):
    """
    Run verification of identified parameters.

    Args:
        experiment_dir: Directory with experimental data (coord.csv, U.csv, F.csv, etc.)
        identified_params: Dictionary with identified material parameters
        output_dir: Optional output directory for results
        bc_type: Boundary condition type ('clamped' for experimental, 'roller' for synthetic)
        experiment_id: Experiment number for thesis figure filenames
        times_to_plot: List of times (in seconds) for field snapshot plots

    Returns:
        dict: Verification results including errors
    """
    print("="*70)
    print("VERIFICATION MODE")
    print("="*70)
    print(f"\nExperiment: {experiment_dir}")
    print(f"Identified Parameters:")
    print(f"  G_inf = {identified_params['G_inf']:.2f} MPa")
    print(f"  G = {identified_params['G']} MPa")
    print(f"  tau_G = {identified_params['tau_G']} s")
    print(f"  K_inf = {identified_params['K_inf']:.2f} MPa")
    print(f"  K = {identified_params['K']} MPa")
    print(f"  tau_K = {identified_params['tau_K']} s")

    # Load experiment data
    data = load_experiment_data(Path(experiment_dir))

    # Create mesh object
    print("\nCreating mesh...")
    coord = data['coord']
    conne = data['conne']

    nodes_xy = coord[:, 1:3]
    width = nodes_xy[:, 0].max() - nodes_xy[:, 0].min()
    height = nodes_xy[:, 1].max() - nodes_xy[:, 1].min()

    # Create minimal mesh object
    mesh = MeshGenerator(width=width, height=height, nx=10, ny=10)
    mesh.coord = coord
    mesh.conne = conne
    mesh.nodes = [Node(i, nodes_xy[i, 0], nodes_xy[i, 1]) for i in range(len(nodes_xy))]
    mesh.triangulation = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], conne)
    mesh.width = width
    mesh.height = height

    print(f"  Mesh: {len(mesh.nodes)} nodes, {len(conne)} elements")
    print(f"  Domain: {width:.2f} x {height:.2f} mm")

    # Create material from identified parameters
    print("\nCreating material from identified parameters...")
    material = create_material_from_identified(identified_params)
    print(f"  {material}")

    # Expand F_reduced to nodal forces
    print("\nExpanding forces to nodal vector...")
    F_nodal = expand_forces_to_nodal(data['F_reduced'], mesh, data['time'])

    # Create and run verification solver
    solver = VerificationSolver(
        mesh=mesh,
        material=material,
        time=data['time'],
        F_nodal=F_nodal,
        U_measured=data['U'],
        bc_type=bc_type
    )

    U_predicted = solver.solve()

    # Compute errors
    print("\nComputing errors...")
    errors = solver.compute_errors()

    print(f"\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"\nError Metrics:")
    print(f"  RMSE (total): {errors['rmse_total']:.6e} mm")
    print(f"  RMSE (Ux):    {errors['rmse_x']:.6e} mm")
    print(f"  RMSE (Uy):    {errors['rmse_y']:.6e} mm")
    print(f"  Max Error:    {errors['max_abs_error']:.6e} mm")
    print(f"  Relative:     {errors['relative_error']*100:.2f} %")

    # Save results and plots
    if output_dir is None:
        output_dir = Path(experiment_dir) / 'verification_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Save error metrics
    with open(output_dir / 'verification_metrics.txt', 'w') as f:
        f.write("Verification Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment: {experiment_dir}\n")
        f.write(f"Boundary Conditions: {bc_type.upper()}\n\n")
        f.write("Identified Parameters:\n")
        f.write(f"  G_inf = {identified_params['G_inf']:.4f} MPa\n")
        f.write(f"  G = {identified_params['G']} MPa\n")
        f.write(f"  tau_G = {identified_params['tau_G']} s\n")
        f.write(f"  K_inf = {identified_params['K_inf']:.4f} MPa\n")
        f.write(f"  K = {identified_params['K']} MPa\n")
        f.write(f"  tau_K = {identified_params['tau_K']} s\n\n")
        f.write("Error Metrics:\n")
        f.write(f"  RMSE (total): {errors['rmse_total']:.6e} mm\n")
        f.write(f"  RMSE (Ux):    {errors['rmse_x']:.6e} mm\n")
        f.write(f"  RMSE (Uy):    {errors['rmse_y']:.6e} mm\n")
        f.write(f"  Max Error:    {errors['max_abs_error']:.6e} mm\n")
        f.write(f"  Relative:     {errors['relative_error']*100:.2f} %\n")

    # Save predicted displacements
    np.savetxt(output_dir / 'U_predicted.csv', errors['U_pred_separated'], delimiter=',', fmt='%.12e')

    # Create plots
    print("\nGenerating plots...")
    plot_verification_results(solver, errors, output_dir)

    # Create displacement field comparison heatmaps
    print("\nGenerating displacement field comparisons...")
    plot_displacement_field_comparison(solver, errors, output_dir)

    # Generate thesis-quality figures
    plot_thesis_figures(solver, errors, output_dir, experiment_id, times_to_plot)

    print(f"\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

    return {
        'errors': errors,
        'solver': solver,
        'output_dir': output_dir
    }


# ============================================================================
# MAIN - Example usage
# ============================================================================
if __name__ == "__main__":
    """
    Verify identified parameters for experiment 904.

    Edit the parameters below and run:
        python run_verification.py
    """

    # ========================================================================
    # CONFIGURATION - EDIT THESE
    # ========================================================================

    # Experiment number (for thesis figure filenames)
    experiment_id = 904

    # Experiment directory (contains coord.csv, U.csv, F.csv, time.csv)
    experiment_dir = Path(__file__).parent.parent / "synthetic_data" / f"{experiment_id}"

    # Identified material parameters (from inverse problem output)
    # Example: Parameters for experiment 904
    identified_params = {
        'G_inf': 225.67,       # MPa
        'G': [7.27, 29.16, 27.38, 9.70, 31.56],  # MPa
        'tau_G': [55.48, 102.71, 190.12, 351.94, 1205.99],  # seconds
        'K_inf': 421.18,       # MPa
        'K': [33.14, 51.80, 3.22, 54.08],   # MPa
        'tau_K': [102.71, 190.13, 351.94, 651.49]  # seconds
    }

    # Times for field snapshot plots (Fig 5.4-4)
    times_to_plot = [10, 603, 1205]  # seconds

    # Output directory (None = experiment_dir/verification_results)
    output_dir = None

    # Boundary condition type
    bc_type = 'roller'  # 'clamped' for experimental, 'roller' for synthetic

    # ========================================================================
    # RUN BC COMPARISON (roller vs clamped)
    # ========================================================================

    results = run_bc_comparison(
        experiment_dir=experiment_dir,
        identified_params=identified_params,
        output_dir=output_dir,
        experiment_id=experiment_id
    )

    print(f"\nResults saved to: {results['roller']['output_dir']}")
    print(f"\nThesis figures generated:")
    print(f"  - bc_comparison_{experiment_id}.png (BC comparison)")
    print(f"  - verification_comparison_{experiment_id}.png (Fig 5.4-2)")
    print(f"  - error_summary_{experiment_id}.png (Fig 5.4-3)")
    print(f"  - field_comparison_{experiment_id}_combined.png (Fig 5.4-4)")
