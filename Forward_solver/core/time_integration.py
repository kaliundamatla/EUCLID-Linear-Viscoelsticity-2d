"""
Time integration for forward viscoelastic solver.
Computes history variables (beta coefficients) using implicit midpoint rule.

Uses inverse_problem geometry module for element operations.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import from inverse_problem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inverse_problem.core.geometry import Triangle3Node
from .material import ViscoelasticMaterial


class HistoryVariables:
    """
    Storage for viscoelastic history variables at element level.

    For each element, stores:
    - epsG: Deviatoric strain (3,)
    - epsvG: Viscous deviatoric strain (3, nG)
    - tht: Volumetric strain (scalar)
    - thtv: Viscous volumetric strain (nK,)
    - betaG: Deviatoric beta coefficients (3, nG)
    - betaK: Volumetric beta coefficients (nK,)
    """

    def __init__(self, n_elements: int, nG: int, nK: int):
        """
        Initialize storage for all elements.

        Args:
            n_elements: Number of finite elements
            nG: Number of deviatoric Maxwell elements
            nK: Number of volumetric Maxwell elements
        """
        self.n_elements = n_elements
        self.nG = nG
        self.nK = nK

        # Current timestep values
        self.epsG = np.zeros((3, n_elements))           # (3, nEl)
        self.epsvG = np.zeros((3, nG, n_elements))      # (3, nG, nEl)
        self.tht = np.zeros((1, n_elements))            # (1, nEl)
        self.thtv = np.zeros((1, nK, n_elements))       # (1, nK, nEl)
        self.betaG = np.zeros((3, nG, n_elements))      # (3, nG, nEl)
        self.betaK = np.zeros((1, nK, n_elements))      # (1, nK, nEl)

        # Previous timestep values (for time integration)
        self.epsG_nm1 = np.zeros((3, n_elements))
        self.epsvG_nm1 = np.zeros((3, nG, n_elements))
        self.tht_nm1 = np.zeros((1, n_elements))
        self.thtv_nm1 = np.zeros((1, nK, n_elements))

    def update_timestep(self):
        """
        Copy current values to previous timestep storage.
        Call this at the END of each time step.
        """
        self.epsG_nm1[:] = self.epsG
        self.epsvG_nm1[:] = self.epsvG
        self.tht_nm1[:] = self.tht
        self.thtv_nm1[:] = self.thtv


class ForwardTimeIntegrator:
    """
    Time integration for forward solver.

    Computes beta coefficients and history variables for viscoelastic evolution.
    """

    def __init__(self, mesh, material: ViscoelasticMaterial):
        """
        Initialize time integrator.

        Args:
            mesh: Mesh object from mesh.py (has nodes, conne)
            material: ViscoelasticMaterial with known parameters
        """
        self.mesh = mesh
        self.material = material
        self.n_elements = len(mesh.conne)

        # Create Triangle3Node elements for each mesh element
        self.elements = []
        for ie, element_nodes in enumerate(mesh.conne):
            # Get node objects
            nodes = [mesh.nodes[node_id] for node_id in element_nodes]
            elem = Triangle3Node(ie, nodes)
            self.elements.append(elem)

        # Initialize history variables
        self.history = HistoryVariables(
            n_elements=self.n_elements,
            nG=material.nG,
            nK=material.nK
        )

        print(f"Time Integrator Initialized:")
        print(f"  Elements: {self.n_elements}")
        print(f"  Maxwell branches: nG={material.nG}, nK={material.nK}")

    def compute_element_strains(self, U: np.ndarray, element_id: int):
        """
        Compute deviatoric and volumetric strains for an element.

        Args:
            U: Global displacement vector (2*nNodes,)
            element_id: Element index

        Returns:
            (epsG, tht): Deviatoric strain (3,) and volumetric strain (scalar)
        """
        element = self.elements[element_id]
        node_indices = self.mesh.conne[element_id]

        # Extract element DOFs from global displacement
        # DOF ordering: [u1, v1, u2, v2, u3, v3]
        dof_indices = np.zeros(6, dtype=int)
        for i, node_idx in enumerate(node_indices):
            dof_indices[2*i] = 2 * node_idx      # u_i
            dof_indices[2*i+1] = 2 * node_idx + 1  # v_i

        u_elem = U[dof_indices]  # (6,)

        # Compute strains using element matrices
        epsG = element.Bd @ u_elem      # (3,) deviatoric strain
        tht = element.b @ u_elem        # (1,) volumetric strain

        return epsG, tht

    def compute_beta_first_timestep(self, U: np.ndarray, dt: float, timestep: int = 0):
        """
        Compute beta coefficients for the FIRST timestep (t=0).

        At t=0: beta = [dt/(2*tau + dt)] * strain

        Args:
            U: Global displacement vector at t=0 (2*nNodes,)
            dt: Time step size [s]
            timestep: Timestep index (for storage)
        """
        print(f"Computing beta for first timestep (t=0)...")

        for ie in range(self.n_elements):
            # Compute strains
            epsG, tht = self.compute_element_strains(U, ie)

            # Store strains
            self.history.epsG[:, ie] = epsG
            self.history.tht[0, ie] = tht

            
            # betaG = [dt/(2*tau_G + dt)] * epsG (broadcast over Maxwell branches)
            weight_G = dt / (2 * self.material.tau_G + dt)  # (nG,)
            self.history.betaG[:, :, ie] = np.outer(epsG, weight_G)  # (3, nG)

            # betaK = [dt/(2*tau_K + dt)] * tht (broadcast over Maxwell branches)
            weight_K = dt / (2 * self.material.tau_K + dt)  # (nK,)
            self.history.betaK[0, :, ie] = tht * weight_K  # (nK,)

            # Viscous strains at t=0
            self.history.epsvG[:, :, ie] = self.history.betaG[:, :, ie]
            self.history.thtv[0, :, ie] = self.history.betaK[0, :, ie]

    def compute_beta_timestep(self, U: np.ndarray, dt: float, timestep: int):
        """
        Compute beta coefficients for timestep n > 0.

        Implicit midpoint rule (trapezoidal):
            beta^(n-1) = [dt/(2*tau+dt)] * (eps^(n-1) - eps_v^(n-1))
                         + [2*tau/(2*tau+dt)] * eps_v^(n-1)


        Args:
            U: Global displacement vector at timestep n (2*nNodes,)
            dt: Time step size [s]
            timestep: Timestep index (for logging)
        """
        if timestep == 0:
            raise ValueError("Use compute_beta_first_timestep for t=0")

        for ie in range(self.n_elements):
            # Compute current strains
            epsG, tht = self.compute_element_strains(U, ie)

            # Store current strains
            self.history.epsG[:, ie] = epsG
            self.history.tht[0, ie] = tht

            # Get previous timestep values
            epsG_nm1 = self.history.epsG_nm1[:, ie]          # (3,)
            epsvG_nm1 = self.history.epsvG_nm1[:, :, ie]    # (3, nG)
            tht_nm1 = self.history.tht_nm1[0, ie]            # scalar
            thtv_nm1 = self.history.thtv_nm1[0, :, ie]      # (nK,)

            # Deviatoric: beta_G^(n-1) = w1 * (eps_G^(n-1) - eps_vG^(n-1)) + w2 * eps_vG^(n-1)
            w1_G = dt / (2 * self.material.tau_G + dt)          # (nG,)
            w2_G = (2 * self.material.tau_G) / (2 * self.material.tau_G + dt)  # (nG,)

            # Broadcast: (3,1) * (nG,) operations
            term1_G = np.outer(epsG_nm1, w1_G) - epsvG_nm1 * w1_G  # (3, nG)
            term2_G = epsvG_nm1 * w2_G  # (3, nG)
            self.history.betaG[:, :, ie] = term1_G + term2_G

            # Volumetric: beta_K^(n-1) = w1 * (tht^(n-1) - tht_v^(n-1)) + w2 * tht_v^(n-1)
            w1_K = dt / (2 * self.material.tau_K + dt)          # (nK,)
            w2_K = (2 * self.material.tau_K) / (2 * self.material.tau_K + dt)  # (nK,)

            self.history.betaK[0, :, ie] = (
                w1_K * (tht_nm1 - thtv_nm1) + w2_K * thtv_nm1
            )

            # eps_vG^n = [dt/(2*tau+dt)] * eps_G^n + beta_G^(n-1)
            self.history.epsvG[:, :, ie] = (
                np.outer(epsG, w1_G) + self.history.betaG[:, :, ie]
            )

            # tht_v^n = [dt/(2*tau+dt)] * tht^n + beta_K^(n-1)
            self.history.thtv[0, :, ie] = (
                w1_K * tht + self.history.betaK[0, :, ie]
            )

    def finalize_timestep(self):
        """
        Finalize timestep by copying current to previous.
        Call this at the END of each successful timestep.
        """
        self.history.update_timestep()

    def get_beta_arrays(self):
        """
        Get current beta coefficients for all elements.

        Returns:
            (betaG, betaK):
                betaG: (3, nG, nEl) deviatoric beta
                betaK: (1, nK, nEl) volumetric beta
        """
        return self.history.betaG, self.history.betaK


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test time integration with simple mesh and displacement.
    Run: python time_integration.py
    """
    from mesh import MeshGenerator
    from material import create_simple_test_material

    print("="*70)
    print("TIME INTEGRATION TEST")
    print("="*70)

    # Create simple mesh
    print("\n1. Creating test mesh...")
    gen = MeshGenerator(width=10, height=10, nx=3, ny=3)
    coord, conne = gen.generate()
    print(f"   Mesh: {len(gen.nodes)} nodes, {len(conne)} elements")

    # Create material
    print("\n2. Creating material...")
    material = create_simple_test_material()

    # Create time integrator
    print("\n3. Initializing time integrator...")
    integrator = ForwardTimeIntegrator(gen, material)

    # Test with dummy displacement
    print("\n4. Testing beta computation...")
    n_nodes = len(gen.nodes)
    dt = 0.1  # seconds

    # Create synthetic displacement (linear ramp in y-direction)
    U_test = np.zeros(2 * n_nodes)
    for i, node in enumerate(gen.nodes):
        U_test[2*i] = 0.001 * node.x      # Small x displacement
        U_test[2*i+1] = 0.01 * node.y     # Larger y displacement (stretch)

    # Timestep 0
    print("\n   Computing beta for t=0...")
    integrator.compute_beta_first_timestep(U_test, dt, timestep=0)
    betaG_0, betaK_0 = integrator.get_beta_arrays()

    print(f"\n   Beta at t=0:")
    print(f"     Element 0, betaG[0,0,0] = {betaG_0[0, 0, 0]:.6e}")
    print(f"     Element 0, betaK[0,0,0] = {betaK_0[0, 0, 0]:.6e}")

    # Finalize timestep 0
    integrator.finalize_timestep()

    # Timestep 1 (with slightly increased displacement)
    print("\n   Computing beta for t=1...")
    U_test *= 1.1  # 10% increase
    integrator.compute_beta_timestep(U_test, dt, timestep=1)
    betaG_1, betaK_1 = integrator.get_beta_arrays()

    print(f"\n   Beta at t=1:")
    print(f"     Element 0, betaG[0,0,0] = {betaG_1[0, 0, 0]:.6e}")
    print(f"     Element 0, betaK[0,0,0] = {betaK_1[0, 0, 0]:.6e}")

    # Finalize timestep 1
    integrator.finalize_timestep()

    # Verify shapes
    print(f"\n5. Verification:")
    print(f"     betaG shape: {betaG_1.shape} (expected: (3, {material.nG}, {len(conne)}))")
    print(f"     betaK shape: {betaK_1.shape} (expected: (1, {material.nK}, {len(conne)}))")

    assert betaG_1.shape == (3, material.nG, len(conne)), "betaG shape mismatch"
    assert betaK_1.shape == (1, material.nK, len(conne)), "betaK shape mismatch"

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
