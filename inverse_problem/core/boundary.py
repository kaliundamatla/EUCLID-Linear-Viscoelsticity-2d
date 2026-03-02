"""
Boundary condition handling and final system assembly.
Supports both interior and boundary equation assembly with configurable weights.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    Defines interface for different BC strategies.
    """
    
    @abstractmethod
    def get_boundary_edges(self, mesh) -> Dict[str, np.ndarray]:
        """
        Returns dictionary of boundary edge names to DOF arrays.
        
        Returns:
            Dict mapping edge_name -> DOF array
        """
        pass
    
    @abstractmethod
    def get_force_values(self, edge_name: str, timestep: int, force_data: np.ndarray) -> float:
        """
        Returns force value for given edge at given timestep.
        
        Args:
            edge_name: Name of boundary edge
            timestep: Time index
            force_data: Force array from experimental data
        
        Returns:
            Force value (scalar)
        """
        pass


class TopBottomForce(BoundaryCondition):
    """
    Standard boundary condition: forces on top and bottom edges.
    
    Extracted from trail_inv.py Block 7 (current implementation).
    """
    
    def get_boundary_edges(self, mesh) -> Dict[str, np.ndarray]:
        """
        Define boundary edges including hole boundaries.

        Returns dict with keys:
        - 'bottom_x', 'bottom_y'
        - 'left_x', 'left_y'
        - 'right_x', 'right_y'
        - 'top_x', 'top_y'
        - 'hole_5_x', 'hole_5_y' (if hole with label 5 exists)
        - 'hole_7_x', 'hole_7_y' (if hole with label 7 exists)
        - etc.
        """
        # Get outer boundary nodes
        top_nodes = mesh.get_boundary_nodes('top')
        bottom_nodes = mesh.get_boundary_nodes('bottom')
        right_nodes = mesh.get_boundary_nodes('right')
        left_nodes = mesh.get_boundary_nodes('left')

        # Extract DOFs (EXACT from trail_inv.py)
        edges = {}

        # Bottom
        edges['bottom_x'] = np.array([2 * n.id for n in bottom_nodes], dtype=np.int64)
        edges['bottom_y'] = np.array([2 * n.id + 1 for n in bottom_nodes], dtype=np.int64)

        # Left
        edges['left_x'] = np.array([2 * n.id for n in left_nodes], dtype=np.int64)
        edges['left_y'] = np.array([2 * n.id + 1 for n in left_nodes], dtype=np.int64)

        # Right
        edges['right_x'] = np.array([2 * n.id for n in right_nodes], dtype=np.int64)
        edges['right_y'] = np.array([2 * n.id + 1 for n in right_nodes], dtype=np.int64)

        # Top
        edges['top_x'] = np.array([2 * n.id for n in top_nodes], dtype=np.int64)
        edges['top_y'] = np.array([2 * n.id + 1 for n in top_nodes], dtype=np.int64)

        # Add hole boundaries (traction-free boundaries)
        hole_labels = mesh.get_unique_hole_labels()
        for hole_label in hole_labels:
            hole_nodes = mesh.get_hole_boundary_nodes(hole_label)
            edges[f'hole_{hole_label}_x'] = np.array([2 * n.id for n in hole_nodes], dtype=np.int64)
            edges[f'hole_{hole_label}_y'] = np.array([2 * n.id + 1 for n in hole_nodes], dtype=np.int64)

        return edges
    
    def get_force_values(self, edge_name: str, timestep: int, force_data: np.ndarray) -> float:
       
        if edge_name == 'top_y':
            return force_data[0, timestep]  # Top applied force
        elif edge_name == 'bottom_y':
            return force_data[1, timestep]  # Bottom reaction force
        elif edge_name.startswith('hole_'):
            return 0.0  # Holes are traction-free boundaries
        else:
            return 0.0  # Other edges have zero force

class BottomForceBC(BoundaryCondition):
    """
    Experiment 998:
    - Bottom: Force applied (Neumann BC)
    - Top: Free boundary (in ROI, below clamp)
    - Top/Bottom: Both have DIC displacement data
    """
    
    def get_boundary_edges(self, mesh):
        bottom_nodes = mesh.get_boundary_nodes('bottom')
        top_nodes = mesh.get_boundary_nodes('top')

        edges = {}
        # Bottom edges - force application
        edges['bottom_x'] = np.array([2 * n.id for n in bottom_nodes])
        edges['bottom_y'] = np.array([2 * n.id + 1 for n in bottom_nodes])

        # Top edges - still define them for completeness (force = 0)
        edges['top_x'] = np.array([2 * n.id for n in top_nodes])
        edges['top_y'] = np.array([2 * n.id + 1 for n in top_nodes])

        # Left/Right edges
        left_nodes = mesh.get_boundary_nodes('left')
        right_nodes = mesh.get_boundary_nodes('right')
        edges['left_x'] = np.array([2 * n.id for n in left_nodes])
        edges['left_y'] = np.array([2 * n.id + 1 for n in left_nodes])
        edges['right_x'] = np.array([2 * n.id for n in right_nodes])
        edges['right_y'] = np.array([2 * n.id + 1 for n in right_nodes])

        # Add hole boundaries (traction-free)
        hole_labels = mesh.get_unique_hole_labels()
        for hole_label in hole_labels:
            hole_nodes = mesh.get_hole_boundary_nodes(hole_label)
            edges[f'hole_{hole_label}_x'] = np.array([2 * n.id for n in hole_nodes], dtype=np.int64)
            edges[f'hole_{hole_label}_y'] = np.array([2 * n.id + 1 for n in hole_nodes], dtype=np.int64)

        return edges
    
    def get_force_values(self, edge_name: str, timestep: int, force_data: np.ndarray) -> float:
        """
        Force convention for exp 998:
        - F[0, t] = top force (should be 0 or small, since top is below clamp)
        - F[1, t] = bottom Y force (applied)
        - Hole boundaries are traction-free (force = 0)
        """

        if edge_name == 'top_y':
            return force_data[0, timestep]
        elif edge_name == 'bottom_y':
            return force_data[1, timestep]  # Bottom force applied
        elif edge_name.startswith('hole_'):
            return 0.0  # Holes are traction-free boundaries
        else:
            return 0.0


class BoundaryAssembler:
    """
    Assembles final system matrices A_exp and R_exp.
    
    Supports both interior and boundary assembly with configurable weights.
    Extracted from trail_inv.py Block 7.
    """
    
    def __init__(self, 
                 mesh, 
                 system_assembler, 
                 exp_data,
                 boundary_condition: BoundaryCondition,
                 lambda_interior: float = 0.0,
                 lambda_boundary: float = 1.0):
        """
        Initialize boundary assembler.
        
        Args:
            mesh: Mesh object
            system_assembler: SystemAssembler with computed ae matrices
            exp_data: ExperimentData object
            boundary_condition: BoundaryCondition strategy
            lambda_interior: Weight for interior equations (λ_i)
            lambda_boundary: Weight for boundary equations (λ_r)
        """
        self.mesh = mesh
        self.system_assembler = system_assembler
        self.exp_data = exp_data
        self.bc = boundary_condition
        self.lambda_i = lambda_interior
        self.lambda_r = lambda_boundary
        
        # Get boundary edge DOFs
        self.edge_dofs = self.bc.get_boundary_edges(mesh)
        
        print(f"✓ Boundary assembler initialized:")
        print(f"  λ_interior = {lambda_interior}, λ_boundary = {lambda_boundary}")
        print(f"  Number of edges: {len(self.edge_dofs)}")
    
    def _assemble_interior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble interior equations (EXCLUDING boundary DOFs).


        A_int @ θ ≈ R_int (measured displacements at interior nodes only)

        Returns:
            (A_int, R_int) matrices
        """
        n_time = self.exp_data.n_timesteps
        n_params = self.system_assembler.material.n_params
        n_dofs = 2 * self.mesh.n_nodes

        print("\nAssembling interior equations...")

        # Get boundary DOFs to exclude 
        edge_dofs = self.bc.get_boundary_edges(self.mesh)
        external_dofs = set()
        for dofs in edge_dofs.values():
            external_dofs.update(dofs)
        external_dofs = sorted(list(external_dofs))

        # Create index vector 
        # True for interior DOFs, False for boundary DOFs
        index_vector = np.ones(n_dofs, dtype=bool)
        index_vector[external_dofs] = False

        print(f"  Total DOFs: {n_dofs}")
        print(f"  Boundary DOFs: {len(external_dofs)}")
        print(f"  Interior DOFs: {index_vector.sum()}")

        # Initialize
        A_int_list = []
        R_int_list = []

        for t in range(1, n_time):  #  for t = 1:timesteps (includes all frames)
            # For each DOF, sum contributions from all adjacent elements
            A_int_t = np.zeros((n_dofs, n_params))
            R_int_t = np.zeros(n_dofs)  # Will be zero for interior (equilibrium)

            for e, element in enumerate(self.mesh.elements):
                ae_t = self.system_assembler.ae[e][t]

                # Get element DOFs
                elem_dofs = element.get_global_dofs()

                # Add contribution to each DOF
                for local_idx, global_dof in enumerate(elem_dofs):
                    A_int_t[global_dof, :] += ae_t[local_idx, :]

            # Remove boundary DOFs 
            A_int_t = A_int_t[index_vector, :]
            R_int_t = R_int_t[index_vector]

            A_int_list.append(A_int_t * self.lambda_i)
            R_int_list.append(R_int_t * self.lambda_i)

            if (t + 1) % 200 == 0 or t == 1:
                print(f"  ✓ Timestep {t+1}/{n_time}")


        A_int = np.vstack(A_int_list)
        R_int = np.hstack(R_int_list)

        print(f"✓ Interior assembly complete: A_int shape={A_int.shape}")
        print(f"  Used timesteps: {1} to {n_time-1} ({n_time-1} frames)")
        print(f"  (Excluded {len(external_dofs)} boundary DOFs)")
        return A_int, R_int
    
    def _assemble_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble boundary equations (only boundary DOFs).

        EXACT extraction from trail_inv.py Block 7 (lines ~600-680).
        Uses FIXED edge order for outer boundaries, appends hole boundaries.

        Returns:
            (A_bnd, R_bnd) matrices stacked over time
        """
        n_time = self.exp_data.n_timesteps
        n_params = self.system_assembler.material.n_params

        
        edge_list = [
            ('bottom_x', self.edge_dofs['bottom_x']),
            ('bottom_y', self.edge_dofs['bottom_y']),
            ('left_x', self.edge_dofs['left_x']),
            ('left_y', self.edge_dofs['left_y']),
            ('right_x', self.edge_dofs['right_x']),
            ('right_y', self.edge_dofs['right_y']),
            ('top_x', self.edge_dofs['top_x']),
            ('top_y', self.edge_dofs['top_y'])
        ]

        # Append hole boundaries dynamically (traction-free)
        hole_labels = self.mesh.get_unique_hole_labels()
        for hole_label in hole_labels:
            edge_list.append((f'hole_{hole_label}_x', self.edge_dofs[f'hole_{hole_label}_x']))
            edge_list.append((f'hole_{hole_label}_y', self.edge_dofs[f'hole_{hole_label}_y']))

        n_edges = len(edge_list)
    
        print("\nAssembling boundary equations...")
        print(f"  Edges: {[name for name, _ in edge_list]}")
    
        # Initialize (EXACT from trail_inv.py)
        A_bnd = [np.zeros((n_edges, n_params)) for _ in range(1, n_time)]
        R_bnd = [np.zeros(n_edges) for _ in range(1, n_time)]
    
        # Set force boundary conditions - only for t=1 onwards(EXACT from trail_inv.py)
        for t_idx, t in enumerate(range(1, n_time)):  # ← CHANGED
            for edge_idx, (edge_name, _) in enumerate(edge_list):
                R_bnd[t_idx][edge_idx] = self.bc.get_force_values(
                    edge_name, t, self.exp_data.F
                )
    
        # Assembly loop - only for t=1 onwards
        print("  Assembling contributions from elements...")
        for t_idx, t in enumerate(range(1, n_time)):  # ← CHANGED
            for e, element in enumerate(self.mesh.elements):
                ae_et = self.system_assembler.ae[e][t]
        
                # For each node in element (EXACT from trail_inv.py)
                for nN in range(element.n_nodes):
                    node = element.nodes[nN]
                    dof_x = 2 * node.id
                    dof_y = 2 * node.id + 1
            
                    # Check which edge this DOF belongs to (FIXED ORDER)
                    for edge_idx, (edge_name, edge_dofs) in enumerate(edge_list):
                        if dof_x in edge_dofs:
                            A_bnd[t_idx][edge_idx, :] += ae_et[2*nN, :]
                        if dof_y in edge_dofs:
                            A_bnd[t_idx][edge_idx, :] += ae_et[2*nN + 1, :]
    
            if (t + 1) % 200 == 0 or t == 1:  # ← CHANGED
                print(f"  ✓ Timestep {t+1}/{n_time}")

        # Stack over time (EXACT from trail_inv.py)
        A_bnd_stacked = np.vstack([A_bnd[t_idx] * self.lambda_r for t_idx in range(len(A_bnd))])  # ← CHANGED
        R_bnd_stacked = np.hstack([R_bnd[t_idx] * self.lambda_r for t_idx in range(len(R_bnd))])  # ← CHANGED

        print(f"✓ Boundary assembly complete: A_bnd shape={A_bnd_stacked.shape}")
        print(f"  Used timesteps: {1} to {n_time-1} ({n_time-1} frames)")
        return A_bnd_stacked, R_bnd_stacked
    
    def assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble final system combining interior and boundary.
        
        Returns:
            (A_exp, R_exp) - Final overdetermined system
        """
        print("\n" + "="*60)
        print("FINAL SYSTEM ASSEMBLY")
        print("="*60)
        
        components = []
        
        # Assemble interior if λ_i > 0
        if self.lambda_i > 0:
            A_int, R_int = self._assemble_interior()
            components.append(("Interior", A_int, R_int))
        else:
            print("\nSkipping interior assembly (λ_i = 0)")
        
        # Assemble boundary if λ_r > 0
        if self.lambda_r > 0:
            A_bnd, R_bnd = self._assemble_boundary()
            components.append(("Boundary", A_bnd, R_bnd))
        else:
            print("\nSkipping boundary assembly (λ_r = 0)")
        
        # Combine
        if len(components) == 0:
            raise ValueError("Both λ_i and λ_r are zero - no equations to solve!")
        elif len(components) == 1:
            name, A_exp, R_exp = components[0]
            print(f"\nUsing only {name} equations")
        else:
            print("\nCombining interior + boundary equations...")
            A_exp = np.vstack([comp[1] for comp in components])
            R_exp = np.hstack([comp[2] for comp in components])
        
        print("\n" + "="*60)
        print("FINAL SYSTEM")
        print("="*60)
        print(f"A_exp shape: {A_exp.shape}")
        print(f"R_exp shape: {R_exp.shape}")
        print(f"System: {A_exp.shape[0]} equations, {A_exp.shape[1]} unknowns")
        print(f"Overdetermined by factor: {A_exp.shape[0] / A_exp.shape[1]:.1f}x")
        print("="*60)
        
        return A_exp, R_exp


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test boundary assembly with experiment 713.
    Run: python boundary.py
    """
    from data import ExperimentData
    from geometry import Mesh
    from material import MaterialModel
    from history import BetaComputer
    from assembly import SystemAssembler
    import time
    
    try:
        print("="*60)
        print("BOUNDARY ASSEMBLY TEST")
        print("="*60)
        
        start_time = time.time()
        
        # Load and setup (reduced model for speed)
        exp_data = ExperimentData(713)
        mesh = Mesh(exp_data.coord, exp_data.conne)
        
        print("\nUsing reduced model (nG=10, nK=10)")
        material = MaterialModel(n_maxwell_shear=10, n_maxwell_bulk=10)
        
        # Compute beta and assemble
        beta_computer = BetaComputer(mesh, material)
        history = beta_computer.compute(exp_data)
        
        system_assembler = SystemAssembler(mesh, material, exp_data, history)
        system_assembler.assemble()
        
        # Test different configurations
        print("\n" + "="*60)
        print("TEST 1: Boundary only (λ_i=0, λ_r=1) - CURRENT")
        print("="*60)
        
        bc = TopBottomForce()
        assembler1 = BoundaryAssembler(
            mesh, system_assembler, exp_data, bc,
            lambda_interior=0.0,
            lambda_boundary=1.0
        )
        A_exp1, R_exp1 = assembler1.assemble()
        
        print("\n" + "="*60)
        print("TEST 2: Interior only (λ_i=1, λ_r=0)")
        print("="*60)
        
        assembler2 = BoundaryAssembler(
            mesh, system_assembler, exp_data, bc,
            lambda_interior=1.0,
            lambda_boundary=0.0
        )
        A_exp2, R_exp2 = assembler2.assemble()
        
        print("\n" + "="*60)
        print("TEST 3: Combined (λ_i=0.5, λ_r=1)")
        print("="*60)
        
        assembler3 = BoundaryAssembler(
            mesh, system_assembler, exp_data, bc,
            lambda_interior=0.5,
            lambda_boundary=1.0
        )
        A_exp3, R_exp3 = assembler3.assemble()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Boundary only:  A_exp {A_exp1.shape}, R_exp {R_exp1.shape}")
        print(f"Interior only:  A_exp {A_exp2.shape}, R_exp {R_exp2.shape}")
        print(f"Combined:       A_exp {A_exp3.shape}, R_exp {R_exp3.shape}")
        print(f"\nTotal time: {elapsed:.1f}s")
        
        print("\n✓ boundary.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()