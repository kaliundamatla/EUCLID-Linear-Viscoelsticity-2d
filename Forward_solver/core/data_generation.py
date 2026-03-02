"""
Synthetic data generator for validation pipeline.
Exports forward solver results in inverse problem format.

Generates the 5 required files:
- coord.csv: Node coordinates with boundary flags
- conne.txt: Element connectivity
- U.csv: Displacement field history
- F.csv: Force field history
- time.csv: Time vector
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .mesh import MeshGenerator
from .material import ViscoelasticMaterial
from .solver import ForwardSolver


class SyntheticDataGenerator:
    """
    Generates synthetic experimental data from forward simulation.

    Exports results in the exact format expected by inverse_problem pipeline.
    """

    def __init__(self,
                 mesh: MeshGenerator,
                 material: ViscoelasticMaterial,
                 dt: float,
                 n_timesteps: int,
                 load_magnitude: float = 50.0):
        """
        Initialize synthetic data generator.

        Args:
            mesh: MeshGenerator instance
            material: ViscoelasticMaterial (ground truth)
            dt: Time step size [s]
            n_timesteps: Number of timesteps
            load_magnitude: Applied load [MPa]
        """
        self.mesh = mesh
        self.material = material
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.load_magnitude = load_magnitude

        # Create and run forward solver
        print("="*70)
        print("SYNTHETIC DATA GENERATION")
        print("="*70)
        print(f"\nGround Truth Material:")
        print(f"  {material}")

        self.solver = ForwardSolver(
            mesh=mesh,
            material=material,
            dt=dt,
            n_timesteps=n_timesteps,
            load_magnitude=load_magnitude
        )

    def generate(self):
        """
        Run forward simulation to generate synthetic data.

        Returns:
            (time, U, F): Time vector, displacement history, force history
        """
        time, U_history, F_history = self.solver.solve()
        return time, U_history, F_history

    def export(self, output_dir: Path, experiment_name: str = "synthetic"):
        """
        Export synthetic data in inverse problem format.

        Creates directory structure:
            output_dir/
                experiment_name/
                    coord.csv
                    conne.txt
                    U.csv
                    F.csv
                    time.csv
                    ground_truth_material.txt

        Args:
            output_dir: Output directory path
            experiment_name: Experiment subdirectory name
        """
        # Create output directory
        exp_dir = Path(output_dir) / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"EXPORTING SYNTHETIC DATA")
        print(f"{'='*70}")
        print(f"Output directory: {exp_dir}")

        # 1. Export mesh files (coord.csv, conne.txt)
        print("\n1. Exporting mesh files...")
        self.mesh.export(exp_dir)

        # 2. Export displacement field U.csv
        print("\n2. Exporting displacement field (U.csv)...")
        U_file = exp_dir / "U.csv"

        # Convert from INTERLEAVED to SEPARATED format to match dataset 713
        # Solver stores: [u0x, u0y, u1x, u1y, ..., unx, uny] (interleaved)
        # Inverse problem needs: [u0x, u1x, ..., unx, u0y, u1y, ..., uny] (separated)
        n_nodes = len(self.mesh.nodes)
        U_interleaved = self.solver.U_history  # (2*n_nodes, n_timesteps)

        # Extract x and y components
        U_x = U_interleaved[0::2, :]  # Even indices: 0, 2, 4, ... (x-displacements)
        U_y = U_interleaved[1::2, :]  # Odd indices: 1, 3, 5, ... (y-displacements)

        # Stack as [Ux; Uy] (separated format)
        U_separated = np.vstack([U_x, U_y])

        np.savetxt(U_file, U_separated, delimiter=',', fmt='%.12e')
        print(f"   U.csv: {U_separated.shape} (2*nNodes x nTimesteps)")
        print(f"   Format: SEPARATED [Ux(0-{n_nodes-1}); Uy(0-{n_nodes-1})] - matches dataset 713")

        # 3. Export force field F.csv
        print("\n3. Exporting force field (F.csv)...")
        F_file = exp_dir / "F.csv"

        # Row 0: Constant applied load = edge_length * load_magnitude / 1000
        # Row 1: Reaction forces at bottom (time-varying)
        # Row 2-3: Zeros

        # Compute top edge length (sum of all top edge element lengths)
        nodes_xy = np.array([[n.x, n.y] for n in self.mesh.nodes])
        top_edge_length = 0.0

        for elem_info in self.solver.loading.top_elements:
            element_nodes = elem_info['nodes']
            top_local_indices = elem_info['top_local_indices']

            if len(top_local_indices) >= 2:
                # Get coordinates of top edge nodes
                node1_idx = element_nodes[top_local_indices[0]]
                node2_idx = element_nodes[top_local_indices[1]]
                p1 = nodes_xy[node1_idx]
                p2 = nodes_xy[node2_idx]
                edge_len = np.linalg.norm(p2 - p1)
                top_edge_length += edge_len

        #  Force(1,:) = 20*q0*ones(size(time))/1000
        # where 20 = Lx (width), q0 = load_magnitude
        # Python: use computed edge length instead of width (more accurate for Delaunay mesh)
        constant_applied_load = top_edge_length * self.load_magnitude / 1000.0

        print(f"   Top edge length: {top_edge_length:.2f} mm")
        print(f"   Load magnitude: {self.load_magnitude} N/mm")
        print(f"   Constant applied load: {constant_applied_load:.6f} kN")

        # Compute reaction forces at bottom boundary
        #  RBottomY = KKold(dof_bottomY,:)*Z - Bet(dof_bottomY)
        # This is: K_elastic * U - F_hist
        bottom_nodes = np.where(self.mesh.coord[:, 3] == 2)[0]
        bottom_y_dofs = 2 * bottom_nodes + 1

        # Get elastic stiffness (same for all timesteps)
        K_elastic = self.solver.assembler.assemble_global_stiffness(self.dt, is_first_timestep=True)

        F_reduced = np.zeros((4, self.solver.n_timesteps))

        for nt in range(self.solver.n_timesteps):
            # Row 0: Constant applied load (same for all timesteps)
            F_reduced[0, nt] = constant_applied_load

            # Row 1: Reaction force at bottom = K*U - F_hist
            U = self.solver.U_history[:, nt]

            # Reaction = K*U (force needed to maintain current displacement)
            reaction_total = K_elastic[bottom_y_dofs, :].dot(U)

            # Subtract history force if present (for t > 0)
            if nt > 0:
                # Need to recompute history force for this timestep
                # Use the integrator's stored beta values
                F_hist_full = np.zeros(self.solver.n_dofs)
                # This is complex - for now, use simpler approach:
                # Reaction = total force at bottom DOFs
                # Since bottom is fixed (v=0), reaction = -F_hist[bottom_y_dofs]
                F_hist_full = self.solver.F_history[:, nt] - self.solver.F_ext_history[:, nt]
                reaction_total = reaction_total - F_hist_full[bottom_y_dofs]

            # Sum reactions and convert to kN
            F_reduced[1, nt] = np.sum(reaction_total) / 1000.0

            # Rows 2-3: zeros 

        np.savetxt(F_file, F_reduced, delimiter=',', fmt='%.12e')
        print(f"   F.csv: {F_reduced.shape} (4 force components x nTimesteps)")
        print(f"   Row 0: Constant applied load = {constant_applied_load:.6f} kN")
        print(f"   Row 1: Bottom reaction forces (time-varying)")
        print(f"   Row 1 range: [{F_reduced[1,:].min():.6f}, {F_reduced[1,:].max():.6f}] kN")
        print(f"   Rows 2-3: Zeros")

        # 4. Export time vector time.csv
        print("\n4. Exporting time vector (time.csv)...")
        time_file = exp_dir / "time.csv"
        np.savetxt(time_file, self.solver.time, delimiter=',', fmt='%.6f')
        print(f"   time.csv: {len(self.solver.time)} timesteps")

        # 5. Save ground truth material parameters
        print("\n5. Saving ground truth material parameters...")
        material_file = exp_dir / "ground_truth_material.txt"
        with open(material_file, 'w') as f:
            f.write("Ground Truth Material Parameters\n")
            f.write("="*50 + "\n\n")
            f.write(f"Deviatoric (Shear):\n")
            f.write(f"  G_inf = {self.material.G_inf} MPa\n")
            f.write(f"  G = {self.material.G} MPa\n")
            f.write(f"  tau_G = {self.material.tau_G} s\n")
            f.write(f"  G0 = {self.material.G0} MPa (instantaneous)\n\n")
            f.write(f"Volumetric (Bulk):\n")
            f.write(f"  K_inf = {self.material.K_inf} MPa\n")
            f.write(f"  K = {self.material.K} MPa\n")
            f.write(f"  tau_K = {self.material.tau_K} s\n")
            f.write(f"  K0 = {self.material.K0} MPa (instantaneous)\n\n")
            f.write(f"Prony Series:\n")
            f.write(f"  nG = {self.material.nG}\n")
            f.write(f"  nK = {self.material.nK}\n")

        print(f"   ground_truth_material.txt saved")

        # 6. Create summary file
        print("\n6. Creating summary...")
        summary_file = exp_dir / "SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write("Synthetic Data Generation Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated by: Forward Solver\n")
            f.write(f"Date: {Path.cwd()}\n\n")

            f.write(f"Mesh:\n")
            f.write(f"  Nodes: {len(self.mesh.nodes)}\n")
            f.write(f"  Elements: {len(self.mesh.conne)}\n")
            f.write(f"  Domain: {self.mesh.width}mm x {self.mesh.height}mm\n\n")

            f.write(f"Time discretization:\n")
            f.write(f"  dt = {self.dt} s\n")
            f.write(f"  n_timesteps = {self.n_timesteps}\n")
            f.write(f"  Total time = {self.n_timesteps * self.dt} s\n\n")

            f.write(f"Loading:\n")
            f.write(f"  Distributed load on top: {self.load_magnitude} MPa\n\n")

            f.write(f"Material (Ground Truth):\n")
            f.write(f"  {self.material}\n\n")

            f.write(f"Files generated:\n")
            f.write(f"  - coord.csv: {len(self.mesh.nodes)} nodes\n")
            f.write(f"  - conne.txt: {len(self.mesh.conne)} elements\n")
            f.write(f"  - U.csv: {self.solver.U_history.shape}\n")
            f.write(f"  - F.csv: {self.solver.F_history.shape}\n")
            f.write(f"  - time.csv: {len(self.solver.time)} timesteps\n")
            f.write(f"  - ground_truth_material.txt\n")

        print(f"   SUMMARY.txt saved")

        print(f"\n{'='*70}")
        print(f"Export complete! Files saved to:")
        print(f"  {exp_dir}")
        print(f"{'='*70}")


# ========== Testing & Example Usage ==========
if __name__ == "__main__":
    """
    Generate synthetic data for validation testing.
    Run: python data_generation.py
    """
    from mesh import MeshGenerator
    from material import create_reference_material

    print("="*70)
    print("SYNTHETIC DATA GENERATION - EXAMPLE")
    print("="*70)

    # Configuration
    width = 20  # mm
    height = 50  # mm
    nx = 11
    ny = 26
    dt = 0.1  # seconds
    n_timesteps = 10  # Keep small for testing
    load = 50.0  # MPa
    output_dir = Path("./synthetic_data")

    # 1. Create mesh
    print("\n[1/4] Creating mesh...")
    mesh = MeshGenerator(width=width, height=height, nx=nx, ny=ny)
    coord, conne = mesh.generate()

    # 2. Create ground truth material 
    print("\n[2/4] Creating ground truth material...")
    material = create_reference_material()

    # 3. Generate synthetic data
    print("\n[3/4] Running forward simulation...")
    generator = SyntheticDataGenerator(
        mesh=mesh,
        material=material,
        dt=dt,
        n_timesteps=n_timesteps,
        load_magnitude=load
    )

    time, U, F = generator.generate()

    # 4. Export to inverse problem format
    print("\n[4/4] Exporting data...")
    generator.export(output_dir, experiment_name="test_001")

    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nYou can now test the inverse problem with:")
    print(f"  cd ../inverse_problem")
    print(f"  python inverse_problem.py --experiment synthetic_data/test_001")
    print("="*70)
