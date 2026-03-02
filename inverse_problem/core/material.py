"""
Material model definition for linear viscoelasticity.
Defines Prony series structure and projection operators.
"""

import numpy as np


class MaterialModel:
    """
    Defines the linear viscoelastic material model structure.
    
    Uses generalized Maxwell model (Prony series):
    - G(t) = G_∞ + Σ G_i * exp(-t/τ_G_i)  [Deviatoric/Shear]
    - K(t) = K_∞ + Σ K_i * exp(-t/τ_K_i)  [Volumetric/Bulk]
    
    This class defines the STRUCTURE (tau values, projections),
    not the actual moduli values (G, K are unknowns to be identified).
    
    Extracted from trail_inv.py Block 2.
    """
    
    def __init__(self, 
                 n_maxwell_shear: int = 150,
                 n_maxwell_bulk: int = 150,
                 tau_min: float = 1.0,
                 tau_max: float = 600.0,
                 plane_stress: bool = True):
        """
        Initialize material model structure.
        
        Args:
            n_maxwell_shear: Number of Maxwell elements for deviatoric response (nG)
            n_maxwell_bulk: Number of Maxwell elements for volumetric response (nK)
            tau_min: Minimum relaxation time [seconds]
            tau_max: Maximum relaxation time [seconds]
            plane_stress: If True, use plane stress formulation; if False, plane strain
        """
        self.nG = n_maxwell_shear
        self.nK = n_maxwell_bulk
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.plane_stress = plane_stress
        
        # Generate logarithmically-spaced relaxation times
        self.tau_G = np.logspace(np.log10(tau_min), np.log10(tau_max), self.nG)
        self.tau_K = np.logspace(np.log10(tau_min), np.log10(tau_max), self.nK)
        
        # Projection matrices for stress/strain decomposition
        self._setup_projections()
        
        print(f"✓ Material model: nG={self.nG}, nK={self.nK}, "
              f"τ ∈ [{tau_min}, {tau_max}]s, "
              f"{'plane stress' if plane_stress else 'plane strain'}")
    
    def _setup_projections(self):
        """
        Setup projection matrices for deviatoric-volumetric split.
        
        From trail_inv.py Block 3.
        Plane stress: 3 components [ε11, ε22, γ12]
        Plane strain: 4 components [ε11, ε22, ε33, γ12]
        """
        if self.plane_stress:
            # Plane stress (3 components)
            self.m = np.array([1, 1, 0]).reshape(-1, 1)  # Trace vector
            self.Idev = np.eye(3) - 0.5 * (self.m @ self.m.T)  # Deviatoric projector
            self.Ivol = 0.5 * (self.m @ self.m.T)              # Volumetric projector
            self.Dmu = np.array([[2, 0, 0],    # Deviatoric stiffness operator
                                 [0, 2, 0],
                                 [0, 0, 1]])
        else:
            # Plane strain (4 components)
            self.m = np.array([1, 1, 1, 0]).reshape(-1, 1)
            self.Idev = np.eye(4) - (1/3) * (self.m @ self.m.T)
            self.Ivol = (1/3) * (self.m @ self.m.T)
            self.Dmu = np.array([[2, 0, 0, 0],
                                 [0, 2, 0, 0],
                                 [0, 0, 2, 0],
                                 [0, 0, 0, 1]])
    
    @property
    def n_params(self) -> int:
        """
        Total number of parameters to identify.
        
        Returns:
            (nG + 1) + (nK + 1) = total Prony parameters
            +1 for G_∞ and K_∞ (equilibrium moduli)
        """
        return (self.nG + 1) + (self.nK + 1)
    
    @property
    def tau_full(self) -> np.ndarray:
        """
        Full relaxation time array (for reference).
        
        Returns:
            [0, τ_G_1, ..., τ_G_nG, 0, τ_K_1, ..., τ_K_nK]
            (zeros represent G_∞ and K_∞ which have no relaxation)
        """
        return np.concatenate(([0], self.tau_G, [0], self.tau_K))
    
    def get_tau_G_range(self) -> tuple:
        """Get min and max of deviatoric relaxation times."""
        return (self.tau_G[0], self.tau_G[-1])
    
    def get_tau_K_range(self) -> tuple:
        """Get min and max of volumetric relaxation times."""
        return (self.tau_K[0], self.tau_K[-1])
    
    def __repr__(self):
        return (f"MaterialModel(nG={self.nG}, nK={self.nK}, "
                f"τ_G ∈ [{self.tau_G[0]:.2f}, {self.tau_G[-1]:.2f}]s, "
                f"τ_K ∈ [{self.tau_K[0]:.2f}, {self.tau_K[-1]:.2f}]s, "
                f"{self.n_params} parameters)")


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test material model creation.
    Run: python material.py
    """
    
    try:
        print("="*60)
        print("MATERIAL MODEL TEST")
        print("="*60)
        
        # Create material model with default settings
        material = MaterialModel(n_maxwell_shear=150, n_maxwell_bulk=150)
        
        print(f"\n{material}")
        print(f"\nTotal parameters to identify: {material.n_params}")
        print(f"  G parameters: {material.nG + 1} (G_∞ + {material.nG} Maxwell)")
        print(f"  K parameters: {material.nK + 1} (K_∞ + {material.nK} Maxwell)")
        
        print(f"\nDeviatoric relaxation times:")
        print(f"  τ_G[0] = {material.tau_G[0]:.4f} s")
        print(f"  τ_G[74] = {material.tau_G[74]:.4f} s (middle)")
        print(f"  τ_G[-1] = {material.tau_G[-1]:.4f} s")
        
        print(f"\nVolumetric relaxation times:")
        print(f"  τ_K[0] = {material.tau_K[0]:.4f} s")
        print(f"  τ_K[74] = {material.tau_K[74]:.4f} s (middle)")
        print(f"  τ_K[-1] = {material.tau_K[-1]:.4f} s")
        
        print(f"\nProjection matrices:")
        print(f"  Idev shape: {material.Idev.shape}")
        print(f"  Ivol shape: {material.Ivol.shape}")
        print(f"  Dmu shape: {material.Dmu.shape}")
        print(f"  m shape: {material.m.shape}")
        
        print(f"\nTrace vector m:")
        print(f"  {material.m.flatten()}")
        
        print(f"\nDeviatoric projector Idev:")
        print(material.Idev)
        
        print(f"\nVolumetric projector Ivol:")
        print(material.Ivol)
        
        print(f"\nDeviatoric stiffness Dmu:")
        print(material.Dmu)
        
        # Verify projections are orthogonal
        print(f"\nVerification:")
        print(f"  Idev + Ivol ≈ I? {np.allclose(material.Idev + material.Ivol, np.eye(3))}")
        print(f"  Idev @ Ivol ≈ 0? {np.allclose(material.Idev @ material.Ivol, 0)}")
        
        # Test with different settings
        print("\n" + "="*60)
        print("Testing different configurations:")
        print("="*60)
        
        # Fewer Maxwell elements
        material_small = MaterialModel(n_maxwell_shear=10, n_maxwell_bulk=10, 
                                       tau_min=0.1, tau_max=100)
        print(f"\n{material_small}")
        
        # Plane strain
        material_strain = MaterialModel(plane_stress=False)
        print(f"\n{material_strain}")
        print(f"  Plane strain: m shape = {material_strain.m.shape}, "
              f"Idev shape = {material_strain.Idev.shape}")
        
        print("\n✓ material.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()