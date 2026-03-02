"""
Material model for forward viscoelastic solver.
Defines known Prony series parameters for synthetic data generation.

Unlike inverse_problem/material.py which defines STRUCTURE (tau values),
this module defines KNOWN material parameters (G, K moduli) for forward simulation.
"""

import numpy as np
from typing import Dict, Tuple


class ViscoelasticMaterial:
    """
    Complete material definition with known Prony series parameters.

    Parameters are stored in ABSOLUTE form (MPa), not normalized.
    """

    def __init__(self,
                 G: np.ndarray,
                 tau_G: np.ndarray,
                 G_inf: float,
                 K: np.ndarray,
                 tau_K: np.ndarray,
                 K_inf: float,
                 plane_stress: bool = True):
        """
        Initialize viscoelastic material with known parameters.

        Args:
            G: Deviatoric Prony moduli [MPa], shape (nG,)
            tau_G: Deviatoric relaxation times [s], shape (nG,)
            G_inf: Equilibrium shear modulus [MPa]
            K: Volumetric Prony moduli [MPa], shape (nK,)
            tau_K: Volumetric relaxation times [s], shape (nK,)
            K_inf: Equilibrium bulk modulus [MPa]
            plane_stress: If True, use plane stress (default); else plane strain
        """
        # Validate inputs
        if len(G) != len(tau_G):
            raise ValueError(f"G and tau_G must have same length: {len(G)} vs {len(tau_G)}")
        if len(K) != len(tau_K):
            raise ValueError(f"K and tau_K must have same length: {len(K)} vs {len(tau_K)}")

        # Store parameters
        self.G = np.asarray(G, dtype=float)
        self.tau_G = np.asarray(tau_G, dtype=float)
        self.G_inf = float(G_inf)

        self.K = np.asarray(K, dtype=float)
        self.tau_K = np.asarray(tau_K, dtype=float)
        self.K_inf = float(K_inf)

        self.plane_stress = plane_stress

        # Compute derived quantities
        self.nG = len(self.G)
        self.nK = len(self.K)
        self.G0 = self.G_inf + np.sum(self.G)  # Instantaneous shear modulus
        self.K0 = self.K_inf + np.sum(self.K)  # Instantaneous bulk modulus

        # Normalized coefficients (Abaqus form)
        self.g = self.G / self.G0  # Normalized deviatoric coefficients
        self.k = self.K / self.K0  # Normalized volumetric coefficients

        # Setup projection operators
        self._setup_projections()

        print(f"Viscoelastic Material Initialized:")
        print(f"  Deviatoric: nG={self.nG}, G0={self.G0:.2f} MPa, G_inf={self.G_inf:.2f} MPa")
        print(f"  Volumetric: nK={self.nK}, K0={self.K0:.2f} MPa, K_inf={self.K_inf:.2f} MPa")
        print(f"  tau_G range: [{self.tau_G.min():.2f}, {self.tau_G.max():.2f}] s")
        print(f"  tau_K range: [{self.tau_K.min():.2f}, {self.tau_K.max():.2f}] s")
        print(f"  Formulation: {'Plane stress' if plane_stress else 'Plane strain'}")

    def _setup_projections(self):
        """
        Setup projection matrices for deviatoric-volumetric decomposition.

        Plane stress: 3 components 
        Plane strain: 4 components 

        """
        if self.plane_stress:
            # Trace vector for plane stress
            self.m = np.array([1, 1, 0]).reshape(-1, 1)

            # Deviatoric projector: I_dev = I - 0.5*(m�m)
            self.Idev = np.eye(3) - 0.5 * (self.m @ self.m.T)

            # Volumetric projector: I_vol = 0.5*(m�m)
            self.Ivol = 0.5 * (self.m @ self.m.T)

            # Deviatoric stiffness operator (2� factor)
            self.Dmu = np.array([[2, 0, 0],
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
    def n_components(self) -> int:
        """Number of stress/strain components (3 for plane stress, 4 for plane strain)."""
        return 3 if self.plane_stress else 4

    @property
    def n_params(self) -> int:
        """Total number of Prony parameters."""
        return (self.nG + 1) + (self.nK + 1)

    def get_exponential_factors(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exponential decay factors for time step dt.

        exp_G[i] = exp(-dt / tau_G[i])
        exp_K[i] = exp(-dt / tau_K[i])

        Used in history variable updates (Eq 3.42-3.43 from theory PDF).

        Args:
            dt: Time step size [s]

        Returns:
            (exp_G, exp_K): Exponential factors for deviatoric and volumetric
        """
        exp_G = np.exp(-dt / self.tau_G)
        exp_K = np.exp(-dt / self.tau_K)
        return exp_G, exp_K

    def get_integration_weights(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute integration weights for history force computation.

        For implicit midpoint rule (trapezoidal):
            w_G[i] = 2 * G[i] / tau_G[i] * (1 - exp_G[i]) / (1 + exp_G[i])
            w_K[i] = 2 * K[i] / tau_K[i] * (1 - exp_K[i]) / (1 + exp_K[i])

        Args:
            dt: Time step size [s]

        Returns:
            (w_G, w_K): Integration weights for deviatoric and volumetric
        """
        exp_G, exp_K = self.get_exponential_factors(dt)

        # Deviatoric weights
        w_G = 2 * self.G / self.tau_G * (1 - exp_G) / (1 + exp_G)

        # Volumetric weights
        w_K = 2 * self.K / self.tau_K * (1 - exp_K) / (1 + exp_K)

        return w_G, w_K

    def to_dict(self) -> Dict:
        """
        Export material parameters as dictionary.

        Useful for saving/loading material definitions.

        Returns:
            Dictionary with all material parameters
        """
        return {
            'G': self.G.tolist(),
            'tau_G': self.tau_G.tolist(),
            'G_inf': self.G_inf,
            'K': self.K.tolist(),
            'tau_K': self.tau_K.tolist(),
            'K_inf': self.K_inf,
            'plane_stress': self.plane_stress,
            'G0': self.G0,
            'K0': self.K0,
            'nG': self.nG,
            'nK': self.nK
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ViscoelasticMaterial':
        """
        Create material from dictionary.

        Args:
            data: Dictionary with material parameters

        Returns:
            ViscoelasticMaterial instance
        """
        return cls(
            G=np.array(data['G']),
            tau_G=np.array(data['tau_G']),
            G_inf=data['G_inf'],
            K=np.array(data['K']),
            tau_K=np.array(data['tau_K']),
            K_inf=data['K_inf'],
            plane_stress=data.get('plane_stress', True)
        )

    def __repr__(self):
        return (f"ViscoelasticMaterial(nG={self.nG}, nK={self.nK}, "
                f"G0={self.G0:.1f} MPa, K0={self.K0:.1f} MPa)")


def create_reference_material() -> ViscoelasticMaterial:
    """

    This is the ground truth material for validation testing.

    Returns:
        ViscoelasticMaterial with parameters
    """
    # From  lines 9-22
    G = np.array([200, 500, 1000])        # [MPa]
    tau_G = np.array([5.3, 50.1, 400.2])  # [s]
    G_inf = 1500                           # [MPa]

    K = np.array([500, 700, 567])         # [MPa]
    tau_K = np.array([5.3, 50.1, 400.2])  # [s]
    K_inf = 2000                           # [MPa]

    return ViscoelasticMaterial(
        G=G, tau_G=tau_G, G_inf=G_inf,
        K=K, tau_K=tau_K, K_inf=K_inf,
        plane_stress=True
    )


def create_simple_test_material() -> ViscoelasticMaterial:
    """
    Create simple material for quick testing (single Maxwell element).

    Returns:
        ViscoelasticMaterial with single Prony term
    """
    return ViscoelasticMaterial(
        G=np.array([500]),
        tau_G=np.array([10.0]),
        G_inf=1000,
        K=np.array([800]),
        tau_K=np.array([10.0]),
        K_inf=1500,
        plane_stress=True
    )


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test material model creation and methods.
    Run: python material.py
    """

    print("="*70)
    print("MATERIAL MODEL TEST")
    print("="*70)
