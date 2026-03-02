"""
Forward solver engine for viscoelastic FEM simulation.
Main time-stepping loop with boundary conditions and loading.
"""

import numpy as np
import scipy.sparse.linalg as spla
from typing import Callable, Optional
from pathlib import Path

from .mesh import MeshGenerator
from .material import ViscoelasticMaterial
from .time_integration import ForwardTimeIntegrator
from .assembly import ForwardAssembler


class BoundaryConditions:
    """
    Manages boundary conditions for forward solver.
    """

    def __init__(self, mesh):
        """
        Initialize boundary conditions from mesh.

        Args:
            mesh: MeshGenerator instance with coord array
        """
        self.mesh = mesh
        self.n_nodes = len(mesh.nodes)
        self.n_dofs = 2 * self.n_nodes

        # Identify boundary nodes from mesh.coord
        # coord columns: [id, x, y, verticalGDL, horizontalGDL]
        self.bottom_nodes = np.where(mesh.coord[:, 3] == 2)[0]  # verticalGDL == 2
        self.top_nodes = np.where(mesh.coord[:, 3] == 1)[0]     # verticalGDL == 1
        self.left_nodes = np.where(mesh.coord[:, 4] == 4)[0]    # horizontalGDL == 4
        self.right_nodes = np.where(mesh.coord[:, 4] == 3)[0]   # horizontalGDL == 3

        print(f"Boundary Conditions:")
        print(f"  Bottom nodes: {len(self.bottom_nodes)}")
        print(f"  Top nodes: {len(self.top_nodes)}")
        print(f"  Left nodes: {len(self.left_nodes)}")
        print(f"  Right nodes: {len(self.right_nodes)}")

    def get_fixed_dofs(self) -> np.ndarray:
        """
        Get DOFs with prescribed displacement (Dirichlet BCs).

        Tensile creep test boundary conditions 
        - Bottom edge: Uy=0 for ALL nodes (prevent vertical rigid body motion)
        - Bottom 2nd node: Ux=0 (prevent horizontal rigid body motion)

        This is the minimal constraint set for a tensile test.

        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []

        # Bottom edge: fix vertical displacement for ALL nodes (Uy=0)
        dof_bottom_y = 2 * self.bottom_nodes + 1  # Uy = 0 for all bottom nodes
        fixed_dofs.extend(dof_bottom_y)

        # Bottom 2nd node ONLY: fix horizontal displacement (Ux=0) to prevent rigid motion
        if len(self.bottom_nodes) >= 2:
            dof_bottom_x_2nd = 2 * self.bottom_nodes[1]  # Ux = 0 for 2nd bottom node (index 1)
            fixed_dofs.append(dof_bottom_x_2nd)

        return np.array(fixed_dofs, dtype=int)

    def get_free_dofs(self) -> np.ndarray:
        """
        Get free (unconstrained) DOFs.

        Returns:
            Array of free DOF indices
        """
        all_dofs = np.arange(self.n_dofs)
        fixed_dofs = self.get_fixed_dofs()
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        return free_dofs


class LoadingProtocol:
    """
    Defines loading protocol for forward simulation.

    Applies traction (distributed load) on top boundary elements.
    """

    def __init__(self, mesh, load_magnitude: float = 50.0):
        """
        Initialize loading protocol with proper edge traction.

        Args:
            mesh: MeshGenerator instance
            load_magnitude: Traction magnitude [N/mm] (force per unit length)
        """
        self.mesh = mesh
        self.load_magnitude = load_magnitude
        self.n_dofs = 2 * len(mesh.nodes)

        # Identify top nodes and top edge elements
        self.top_nodes = np.where(mesh.coord[:, 3] == 1)[0]
        self._find_top_edge_elements()

        print(f"Loading Protocol:")
        print(f"  Traction on top edge: {load_magnitude} N/mm")
        print(f"  Top nodes: {len(self.top_nodes)}")
        print(f"  Top edge elements: {len(self.top_elements)}")

    def _find_top_edge_elements(self):
        """
        Find elements on the top boundary.

        An element is on the top if it has at least 2 nodes on the top boundary.
        Uses verticalGDL labels instead of y-coordinates for robustness with
        imported meshes that may have irregular node spacing.
        """
        self.top_elements = []

        for ie, element_nodes in enumerate(self.mesh.conne):
            # Check how many nodes are on top boundary (using pre-computed top_nodes)
            n_top = 0
            top_node_indices = []
            for local_idx, node_idx in enumerate(element_nodes):
                if node_idx in self.top_nodes:
                    n_top += 1
                    top_node_indices.append(local_idx)

            # If 2 or more nodes on top, this element has a top edge
            if n_top >= 2:
                self.top_elements.append({
                    'element_id': ie,
                    'nodes': element_nodes,
                    'top_local_indices': top_node_indices
                })

    def compute_force_vector(self, time: float) -> np.ndarray:
        """
        Compute force vector with proper edge traction integration.

        - Traction q = [0, q0]' (no horizontal, vertical traction)
        - Edge integration using linear shape functions
        - fe = [0; 0; N2'*q*w*L; N3'*q*w*L] for edge 2-3

        Args:
            time: Current time [s]

        Returns:
            F: (n_dofs,) force vector
        """
        F = np.zeros(self.n_dofs)

        nodes_xy = np.array([[n.x, n.y] for n in self.mesh.nodes])

        # Traction vector: [qx, qy] = [0, q0]
        q = np.array([0.0, self.load_magnitude])

        for elem_info in self.top_elements:
            element_nodes = elem_info['nodes']
            coords = nodes_xy[element_nodes]  # (3, 2)

            # Find the top edge (edge between two top nodes)
            top_local = elem_info['top_local_indices']
            if len(top_local) < 2:
                continue

            # Get the two nodes on the top edge
            n1_local = top_local[0]
            n2_local = top_local[1]

            # Edge length
            p1 = coords[n1_local]
            p2 = coords[n2_local]
            edge_length = np.linalg.norm(p2 - p1)

            # Linear shape functions integrated over edge
            # For a 1D edge with 2 nodes: ∫N1 = L/2, ∫N2 = L/2
            # Force contribution = (shape function integral) * traction * edge_length

            # Each of the two nodes gets half the traction load
            force_per_node = q * edge_length / 2.0  # [fx, fy]

            # Assemble into global force vector
            for local_idx in top_local:
                global_node = element_nodes[local_idx]
                dof_x = 2 * global_node
                dof_y = 2 * global_node + 1

                F[dof_x] += force_per_node[0]  # x-component
                F[dof_y] += force_per_node[1]  # y-component

        return F


class ForwardSolver:
    """
    Main forward solver engine.

    Performs time-stepping simulation with viscoelastic material.
    """

    def __init__(self,
                 mesh: MeshGenerator,
                 material: ViscoelasticMaterial,
                 dt: float,
                 n_timesteps: int,
                 load_magnitude: float = 50.0):
        """
        Initialize forward solver.

        Args:
            mesh: MeshGenerator instance
            material: ViscoelasticMaterial with known parameters
            dt: Time step size [s]
            n_timesteps: Number of timesteps
            load_magnitude: Distributed load on top [MPa]
        """
        self.mesh = mesh
        self.material = material
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.n_dofs = 2 * len(mesh.nodes)

        # Initialize components
        self.integrator = ForwardTimeIntegrator(mesh, material)
        self.assembler = ForwardAssembler(mesh, material, self.integrator)
        self.bc = BoundaryConditions(mesh)
        self.loading = LoadingProtocol(mesh, load_magnitude)

        # Storage for results
        self.time = np.zeros(n_timesteps)
        self.U_history = np.zeros((self.n_dofs, n_timesteps))  # Displacement history
        self.F_history = np.zeros((self.n_dofs, n_timesteps))  # Total force history (F_ext + F_hist)
        self.F_ext_history = np.zeros((self.n_dofs, n_timesteps))  # External force only (for F.csv export)

        # Free and fixed DOFs
        self.free_dofs = self.bc.get_free_dofs()
        self.fixed_dofs = self.bc.get_fixed_dofs()

        print(f"\nForward Solver Initialized:")
        print(f"  Time steps: {n_timesteps}, dt = {dt}s")
        print(f"  Total time: {n_timesteps * dt}s")
        print(f"  DOFs: {self.n_dofs} ({len(self.free_dofs)} free, {len(self.fixed_dofs)} fixed)")

    def solve_timestep_0(self):
        """
        Solve first timestep (t=0) with elastic response.
        """
        print(f"\nTimestep 0 (t=0.0s):")

        # Assemble elastic stiffness
        K_global = self.assembler.assemble_global_stiffness(self.dt, is_first_timestep=True)

        # Get force vector
        F = self.loading.compute_force_vector(0.0)

        # Apply boundary conditions (reduce to free DOFs)
        K_free = K_global[self.free_dofs, :][:, self.free_dofs]
        F_free = F[self.free_dofs]

        # Solve: K * U = F
        print(f"  Solving system: {K_free.shape[0]} DOFs...")
        U_free = spla.spsolve(K_free, F_free)

        # Expand to full DOF vector
        U = np.zeros(self.n_dofs)
        U[self.free_dofs] = U_free

        # Store results
        self.time[0] = 0.0
        self.U_history[:, 0] = U
        self.F_history[:, 0] = F
        self.F_ext_history[:, 0] = F  # At t=0, F_ext = F (no history force yet)

        # Compute beta coefficients
        self.integrator.compute_beta_first_timestep(U, self.dt, timestep=0)
        self.integrator.finalize_timestep()

        print(f"  Max displacement: {np.abs(U).max():.6e} mm")

    def solve_timestep(self, nt: int):
        """
        Solve timestep n > 0 with viscoelastic response.

        Args:
            nt: Timestep index (> 0)
        """
        t = nt * self.dt

        # Assemble viscoelastic stiffness
        K_global = self.assembler.assemble_global_stiffness(self.dt, is_first_timestep=False)

        # Assemble history force
        F_hist = self.assembler.assemble_history_force()

        # Get external force
        F_ext = self.loading.compute_force_vector(t)

        # Total force: F = F_ext + F_hist
        F = F_ext + F_hist

        # Apply boundary conditions
        K_free = K_global[self.free_dofs, :][:, self.free_dofs]
        F_free = F[self.free_dofs]

        # Solve: K * U = F
        U_free = spla.spsolve(K_free, F_free)

        # Expand to full DOF vector
        U = np.zeros(self.n_dofs)
        U[self.free_dofs] = U_free

        # Store results
        self.time[nt] = t
        self.U_history[:, nt] = U
        self.F_history[:, nt] = F  # Total force (F_ext + F_hist)
        self.F_ext_history[:, nt] = F_ext  # External force only

        # Compute beta coefficients for next timestep
        self.integrator.compute_beta_timestep(U, self.dt, timestep=nt)
        self.integrator.finalize_timestep()

        # Print progress every 100 timesteps or at the last timestep
        if nt % 100 == 0 or nt == self.n_timesteps - 1:
            print(f"Timestep {nt} (t={t:.3f}s, progress: {100*nt/self.n_timesteps:.1f}%) - Max displacement: {np.abs(U).max():.6e} mm")

    def solve(self):
        """
        Run complete time-stepping simulation.

        Returns:
            (time, U_history, F_history): Time array, displacement history, force history
        """
        print("="*70)
        print("STARTING FORWARD SIMULATION")
        print("="*70)

        # Solve first timestep
        self.solve_timestep_0()

        # Solve remaining timesteps
        for nt in range(1, self.n_timesteps):
            self.solve_timestep(nt)

        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)

        return self.time, self.U_history, self.F_history


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test forward solver with simple mesh.
    Run: python solver.py
    """
    from mesh import MeshGenerator
    from material import create_simple_test_material
    import matplotlib.pyplot as plt

    print("="*70)
    print("FORWARD SOLVER TEST")
    print("="*70)

    # Create mesh
    print("\n1. Creating mesh...")
    mesh = MeshGenerator(width=20, height=50, nx=11, ny=26)
    coord, conne = mesh.generate()

    # Create material
    print("\n2. Creating material...")
    material = create_simple_test_material()

    # Create solver
    print("\n3. Creating solver...")
    dt = 0.1  # seconds
    n_timesteps = 5  # Short simulation for testing
    solver = ForwardSolver(
        mesh=mesh,
        material=material,
        dt=dt,
        n_timesteps=n_timesteps,
        load_magnitude=-50.0
    )

    # Run simulation
    print("\n4. Running simulation...")
    time, U_history, F_history = solver.solve()

    # Plot results
    print("\n5. Plotting results...")

    # Displacement at top center node
    top_center_node = np.where(
        (np.abs(mesh.coord[:, 1] - mesh.width/2) < 0.1) &  # x ~ width/2
        (np.abs(mesh.coord[:, 2] - mesh.height) < 0.1)     # y ~ height
    )[0]

    if len(top_center_node) > 0:
        node_id = top_center_node[0]
        u_x = U_history[2*node_id, :]
        u_y = U_history[2*node_id + 1, :]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(time, u_x, 'o-', label='u_x')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Horizontal displacement [mm]')
        ax1.set_title(f'Top Center Node (id={node_id}) Displacement History')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(time, u_y, 'o-', label='u_y', color='red')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Vertical displacement [mm]')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    print("\n" + "="*70)
    print("Forward solver test complete!")
    print("="*70)
