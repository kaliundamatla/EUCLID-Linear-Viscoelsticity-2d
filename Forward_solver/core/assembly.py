"""
Assembly module for forward viscoelastic solver.
Computes element stiffness matrices and history force vectors.

Uses inverse_problem geometry module for element operations.
"""

import numpy as np
import scipy.sparse as sp
from .time_integration import ForwardTimeIntegrator, HistoryVariables
from .material import ViscoelasticMaterial


class ForwardAssembler:
    """
    Assembles global stiffness matrix and force vectors for forward solver.

    """

    def __init__(self, mesh, material: ViscoelasticMaterial, integrator: ForwardTimeIntegrator):
        """
        Initialize assembler.

        Args:
            mesh: Mesh object from mesh.py
            material: ViscoelasticMaterial with known parameters
            integrator: ForwardTimeIntegrator instance
        """
        self.mesh = mesh
        self.material = material
        self.integrator = integrator
        self.n_nodes = len(mesh.nodes)
        self.n_elements = len(mesh.conne)
        self.n_dofs = 2 * self.n_nodes

        print(f"Assembler Initialized:")
        print(f"  DOFs: {self.n_dofs} (2 x {self.n_nodes} nodes)")
        print(f"  Elements: {self.n_elements}")

    def compute_element_stiffness(self, element_id: int, dt: float, is_first_timestep: bool = False):
        """
        Compute element stiffness matrix for viscoelastic material.

        For t=0 (elastic):
            K_e = (G_0) * B' * D_mu * B_d + (K_0) * b' * b

        For t>0 (viscoelastic):
            K_e = [G_inf + sum(G) - sum(G * dt/(2*tau_G + dt))] * B' * D_mu * B_d
                + [K_inf + sum(K) - sum(K * dt/(2*tau_K + dt))] * b' * b


        Args:
            element_id: Element index
            dt: Time step size [s]
            is_first_timestep: If True, use elastic stiffness (t=0)

        Returns:
            K_elem: (6, 6) element stiffness matrix
        """
        element = self.integrator.elements[element_id]

        # Gauss point weight for single centroid integration
        w_gp = 0.5
        detJ = element.detJ

        # Get strain-displacement matrices
        B = element.Be    # (3, 6) full strain-displacement
        Bd = element.Bd   # (3, 6) deviatoric part
        b = element.b     # (6,) volumetric part

        # Material stiffness operators
        Dmu = self.material.Dmu  # (3, 3) deviatoric operator

        if is_first_timestep:
            # Elastic stiffness at t=0 
            G_eff = self.material.G0  # G_inf + sum(G)
            K_eff = self.material.K0  # K_inf + sum(K)
        else:
            # Viscoelastic stiffness for t>0 
            # Effective moduli with relaxation correction
            weight_G = dt / (2 * self.material.tau_G + dt)  # (nG,)
            weight_K = dt / (2 * self.material.tau_K + dt)  # (nK,)

            G_eff = self.material.G_inf + np.sum(self.material.G) - np.sum(self.material.G * weight_G)
            K_eff = self.material.K_inf + np.sum(self.material.K) - np.sum(self.material.K * weight_K)

        # Deviatoric stiffness: K_G = G_eff * B' * D_mu * B_d * detJ
        K_dev = G_eff * (B.T @ Dmu @ Bd) * w_gp * detJ  # (6, 6)

        # Volumetric stiffness: K_K = K_eff * b' * b * detJ
        K_vol = K_eff * np.outer(b, b) * w_gp * detJ  # (6, 6)

        # Total element stiffness
        K_elem = K_dev + K_vol

        return K_elem

    def compute_element_history_force(self, element_id: int):
        """
        Compute element history force vector from beta coefficients.

        F_hist = B' * D_mu * sum(G_alpha * beta_G) + b' * sum(K_alpha * beta_K)


        Args:
            element_id: Element index

        Returns:
            F_elem: (6,) element history force vector
        """
        element = self.integrator.elements[element_id]

        # Gauss point weight
        w_gp = 0.5
        detJ = element.detJ

        # Get matrices
        B = element.Be    # (3, 6)
        Dmu = self.material.Dmu  # (3, 3)
        b = element.b     # (6,)

        # Get beta coefficients for this element
        betaG = self.integrator.history.betaG[:, :, element_id]  # (3, nG)
        betaK = self.integrator.history.betaK[0, :, element_id]  # (nK,)

        # Deviatoric history force: B' * D_mu * sum(G_alpha * beta_G_alpha)
        # sum over Maxwell branches: (3, nG) * (nG,) = (3,)
        stress_dev = betaG @ self.material.G  # (3,) weighted sum over branches
        F_dev = (B.T @ Dmu @ stress_dev) * w_gp * detJ  # (6,)

        # Volumetric history force: b' * sum(K_alpha * beta_K_alpha)
        stress_vol = betaK @ self.material.K  # scalar
        F_vol = b * stress_vol * w_gp * detJ  # (6,)

        # Total history force
        F_elem = F_dev + F_vol

        return F_elem

    def assemble_global_stiffness(self, dt: float, is_first_timestep: bool = False):
        """
        Assemble global stiffness matrix.

        Args:
            dt: Time step size [s]
            is_first_timestep: If True, use elastic stiffness

        Returns:
            K_global: (n_dofs, n_dofs) sparse global stiffness matrix
        """
        # Use COO format for efficient assembly
        row_indices = []
        col_indices = []
        values = []

        for ie in range(self.n_elements):
            # Compute element stiffness
            K_elem = self.compute_element_stiffness(ie, dt, is_first_timestep)

            # Get global DOF indices for this element
            node_indices = self.mesh.conne[ie]  # (3,)
            dof_indices = np.zeros(6, dtype=int)
            for i, node_idx in enumerate(node_indices):
                dof_indices[2*i] = 2 * node_idx
                dof_indices[2*i+1] = 2 * node_idx + 1

            # Add to global matrix (expand element matrix into global indices)
            for i in range(6):
                for j in range(6):
                    row_indices.append(dof_indices[i])
                    col_indices.append(dof_indices[j])
                    values.append(K_elem[i, j])

        # Create sparse matrix
        K_global = sp.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(self.n_dofs, self.n_dofs)
        ).tocsr()

        return K_global

    def assemble_history_force(self):
        """
        Assemble global history force vector.

        Returns:
            F_hist: (n_dofs,) global history force vector
        """
        F_hist = np.zeros(self.n_dofs)

        for ie in range(self.n_elements):
            # Compute element history force
            F_elem = self.compute_element_history_force(ie)

            # Get global DOF indices
            node_indices = self.mesh.conne[ie]
            dof_indices = np.zeros(6, dtype=int)
            for i, node_idx in enumerate(node_indices):
                dof_indices[2*i] = 2 * node_idx
                dof_indices[2*i+1] = 2 * node_idx + 1

            # Add to global vector
            F_hist[dof_indices] += F_elem

        return F_hist


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test assembly with simple mesh.
    Run: python assembly.py
    """
    from mesh import MeshGenerator
    from material import create_simple_test_material
    from time_integration import ForwardTimeIntegrator

    print("="*70)
    print("ASSEMBLY TEST")
    print("="*70)

    # Create mesh
    print("\n1. Creating test mesh...")
    gen = MeshGenerator(width=10, height=10, nx=3, ny=3)
    coord, conne = gen.generate()

    # Create material
    print("\n2. Creating material...")
    material = create_simple_test_material()

    # Create time integrator
    print("\n3. Creating time integrator...")
    integrator = ForwardTimeIntegrator(gen, material)

    # Create assembler
    print("\n4. Creating assembler...")
    assembler = ForwardAssembler(gen, material, integrator)

    # Test element stiffness
    print("\n5. Testing element stiffness computation...")
    dt = 0.1
    K_elem_0 = assembler.compute_element_stiffness(0, dt, is_first_timestep=True)
    print(f"   Element 0 stiffness (t=0):")
    print(f"     Shape: {K_elem_0.shape}")
    print(f"     Diagonal: {np.diag(K_elem_0)}")
    print(f"     Symmetric: {np.allclose(K_elem_0, K_elem_0.T)}")

    K_elem_1 = assembler.compute_element_stiffness(0, dt, is_first_timestep=False)
    print(f"\n   Element 0 stiffness (t>0):")
    print(f"     Shape: {K_elem_1.shape}")
    print(f"     Different from t=0: {not np.allclose(K_elem_0, K_elem_1)}")

    # Test global assembly
    print("\n6. Testing global stiffness assembly...")
    K_global = assembler.assemble_global_stiffness(dt, is_first_timestep=True)
    print(f"   Global stiffness:")
    print(f"     Shape: {K_global.shape}")
    print(f"     Non-zeros: {K_global.nnz}")
    print(f"     Symmetric: {np.allclose(K_global.toarray(), K_global.T.toarray())}")
    print(f"     Positive diagonal: {np.all(K_global.diagonal() > 0)}")

    # Test history force (need to compute beta first)
    print("\n7. Testing history force assembly...")
    n_nodes = len(gen.nodes)
    U_test = np.zeros(2 * n_nodes)
    for i, node in enumerate(gen.nodes):
        U_test[2*i+1] = 0.01 * node.y  # Vertical displacement

    # Compute beta for t=0
    integrator.compute_beta_first_timestep(U_test, dt)
    integrator.finalize_timestep()

    # Compute beta for t=1
    integrator.compute_beta_timestep(U_test * 1.1, dt, timestep=1)

    # Assemble history force
    F_hist = assembler.assemble_history_force()
    print(f"   History force:")
    print(f"     Shape: {F_hist.shape}")
    print(f"     Norm: {np.linalg.norm(F_hist):.6e}")
    print(f"     Max magnitude: {np.abs(F_hist).max():.6e}")

    print("\n" + "="*70)
    print("All assembly tests passed!")
    print("="*70)
