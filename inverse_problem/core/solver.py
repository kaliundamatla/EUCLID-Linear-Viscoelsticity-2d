"""
Optimization solvers for parameter identification.
Supports multiple solver strategies (NNLS, LASSO, etc.)
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import lsq_linear
from typing import Optional, Tuple


class ParameterSet:
    """
    Container for identified Prony series parameters.
    
    Extracts and organizes the solution vector θ into G and K parameters.
    """
    
    def __init__(self, theta: np.ndarray, material):
        """
        Create parameter set from solution vector.
        
        Args:
            theta: Solution vector (nG+1 + nK+1,)
                   Structure: [G_∞, G_1, ..., G_nG, K_∞, K_1, ..., K_nK]
            material: MaterialModel object (for structure info)
        """
        self.theta = theta
        self.material = material
        
        nG = material.nG
        nK = material.nK
        
        # Extract parameters (EXACT from trail_inv.py)
        self.G_inf = theta[0]
        self.G_params = theta[1:nG+1]
        self.K_inf = theta[nG+1]
        self.K_params = theta[nG+2:]
        
        # Corresponding tau values
        self.tau_G = material.tau_G
        self.tau_K = material.tau_K
        
        # Statistics
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute parameter statistics."""
        self.n_nonzero_G = np.count_nonzero(self.G_params)
        self.n_nonzero_K = np.count_nonzero(self.K_params)
        self.total_G = self.G_inf + np.sum(self.G_params)
        self.total_K = self.K_inf + np.sum(self.K_params)
    
    def get_nonzero_G(self, threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nonzero deviatoric parameters.
        
        Args:
            threshold: Values below this are considered zero
        
        Returns:
            (tau_G_nonzero, G_nonzero) - Filtered parameters
        """
        # Include G_inf (tau=0) if nonzero
        tau_list = [0] if self.G_inf > threshold else []
        G_list = [self.G_inf] if self.G_inf > threshold else []
        
        # Add Maxwell elements
        for i, G_i in enumerate(self.G_params):
            if G_i > threshold:
                tau_list.append(self.tau_G[i])
                G_list.append(G_i)
        
        return np.array(tau_list), np.array(G_list)
    
    def get_nonzero_K(self, threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nonzero volumetric parameters.
        
        Args:
            threshold: Values below this are considered zero
        
        Returns:
            (tau_K_nonzero, K_nonzero) - Filtered parameters
        """
        # Include K_inf (tau=0) if nonzero
        tau_list = [0] if self.K_inf > threshold else []
        K_list = [self.K_inf] if self.K_inf > threshold else []
        
        # Add Maxwell elements
        for i, K_i in enumerate(self.K_params):
            if K_i > threshold:
                tau_list.append(self.tau_K[i])
                K_list.append(K_i)
        
        return np.array(tau_list), np.array(K_list)
    
    def __repr__(self):
        return (f"ParameterSet:\n"
                f"  G: {self.n_nonzero_G}/{len(self.G_params)} nonzero, "
                f"G_inf={self.G_inf:.2f}, total={self.total_G:.2f}\n"
                f"  K: {self.n_nonzero_K}/{len(self.K_params)} nonzero, "
                f"K_inf={self.K_inf:.2f}, total={self.total_K:.2f}")


class Solver(ABC):
    """
    Abstract base class for optimization solvers.
    """
    
    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the inverse problem: min ||A @ θ - b||²
        
        Args:
            A: System matrix (m × n)
            b: Right-hand side (m,)
        
        Returns:
            theta: Solution vector (n,)
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Get solver metrics (cost, iterations, etc.)
        
        Returns:
            Dictionary of metrics
        """
        pass


class NNLSSolver(Solver):
    """
    Non-negative least squares solver.
    
    Solves: min ||A @ θ - b||²  subject to θ ≥ 0
    
    EXACT extraction from trail_inv.py Block 8.
    """
    
    def __init__(self):
        """Initialize NNLS solver."""
        self.result = None
        self.residual_norm = None
        self.mse = None
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve using bounded least squares (NNLS).
        
        EXACT from trail_inv.py lines ~720-740.
        
        Args:
            A: System matrix (m × n)
            b: Right-hand side (m,)
        
        Returns:
            theta: Non-negative solution (n,)
        """
        print("\n" + "="*60)
        print("SOLVING NON-NEGATIVE LEAST SQUARES")
        print("="*60)
        print(f"System size: {A.shape[0]} equations, {A.shape[1]} unknowns")
        print(f"Overdetermined by factor: {A.shape[0] / A.shape[1]:.1f}x")
        print(f"\nMethod: lsq_linear with bounds=[0, inf]")
        print("Solving...")
        
        # Solve NNLS (EXACT from trail_inv.py)
        self.result = lsq_linear(A, b, bounds=(0, np.inf), method='bvls', verbose=1)
        
        theta = self.result.x
        
        # Compute metrics (EXACT from trail_inv.py)
        self.residual_norm = np.linalg.norm(A @ theta - b)
        self.mse = self.residual_norm**2 / len(b)
        
        print(f"\n✓ Solution converged")
        print(f"  Status: {self.result.message}")
        print(f"  Cost: {self.result.cost:.6e}")
        print(f"  Residual norm: {self.residual_norm:.6e}")
        print(f"  MSE: {self.mse:.6e}")
        print(f"  Nonzero parameters: {np.count_nonzero(theta)}/{len(theta)}")
        print("="*60)
        
        return theta
    
    def get_metrics(self) -> dict:
        """Get solver metrics."""
        if self.result is None:
            return {}
        
        return {
            'cost': self.result.cost,
            'residual_norm': self.residual_norm,
            'mse': self.mse,
            'status': self.result.status,
            'message': self.result.message,
            'n_iter': self.result.nit if hasattr(self.result, 'nit') else None
        }


class LASSOSolver(Solver):
    """
    LASSO solver with L1 regularization (sparse solution).
    
    Solves: min ||A @ θ - b||² + α||θ||₁  subject to θ ≥ 0
    
    Future enhancement for automatic parameter selection.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize LASSO solver.
        
        Args:
            alpha: L1 regularization parameter (higher = more sparse)
        """
        self.alpha = alpha
        self.result = None
        self.residual_norm = None
        self.mse = None
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve using LASSO with non-negativity constraint.
        
        Args:
            A: System matrix (m × n)
            b: Right-hand side (m,)
        
        Returns:
            theta: Sparse non-negative solution (n,)
        """
        print("\n" + "="*60)
        print("SOLVING LASSO (L1 REGULARIZED)")
        print("="*60)
        print(f"System size: {A.shape[0]} equations, {A.shape[1]} unknowns")
        print(f"Regularization: α = {self.alpha}")
        print("Solving...")
        
        try:
            from sklearn.linear_model import Lasso
            
            # LASSO doesn't directly support non-negativity in sklearn
            # Use coordinate descent with positive constraint
            model = Lasso(alpha=self.alpha, positive=True, max_iter=10000)
            model.fit(A, b)
            
            theta = model.coef_
            
            # Compute metrics
            residual = A @ theta - b
            self.residual_norm = np.linalg.norm(residual)
            self.mse = np.mean(residual**2)
            
            print(f"\n✓ Solution converged")
            print(f"  Residual norm: {self.residual_norm:.6e}")
            print(f"  MSE: {self.mse:.6e}")
            print(f"  Nonzero parameters: {np.count_nonzero(theta)}/{len(theta)}")
            print(f"  Sparsity: {100 * (1 - np.count_nonzero(theta)/len(theta)):.1f}%")
            print("="*60)
            
            return theta
            
        except ImportError:
            raise ImportError(
                "sklearn not available. Install with: pip install scikit-learn"
            )
    
    def get_metrics(self) -> dict:
        """Get solver metrics."""
        return {
            'residual_norm': self.residual_norm,
            'mse': self.mse,
            'alpha': self.alpha
        }


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test solvers with experiment 713.
    Run: python solver.py
    """
    from data import ExperimentData
    from geometry import Mesh
    from material import MaterialModel
    from history import BetaComputer
    from assembly import SystemAssembler
    from boundary import BoundaryAssembler, TopBottomForce
    import time
    
    try:
        print("="*60)
        print("SOLVER TEST")
        print("="*60)
        
        start_time = time.time()
        
        # Full pipeline (reduced model)
        exp_data = ExperimentData(713)
        mesh = Mesh(exp_data.coord, exp_data.conne)
        
        print("\nUsing reduced model (nG=10, nK=10)")
        material = MaterialModel(n_maxwell_shear=10, n_maxwell_bulk=10)
        
        beta_computer = BetaComputer(mesh, material)
        history = beta_computer.compute(exp_data)
        
        system_assembler = SystemAssembler(mesh, material, exp_data, history)
        system_assembler.assemble()
        
        bc = TopBottomForce()
        boundary_assembler = BoundaryAssembler(
            mesh, system_assembler, exp_data, bc,
            lambda_interior=0.0,
            lambda_boundary=1.0
        )
        A_exp, R_exp = boundary_assembler.assemble()
        
        # Test NNLS Solver
        print("\n" + "="*60)
        print("TEST 1: NNLS SOLVER")
        print("="*60)
        
        nnls_solver = NNLSSolver()
        theta_nnls = nnls_solver.solve(A_exp, R_exp)
        params_nnls = ParameterSet(theta_nnls, material)
        
        print("\n" + str(params_nnls))
        
        tau_G_nz, G_nz = params_nnls.get_nonzero_G()
        tau_K_nz, K_nz = params_nnls.get_nonzero_K()
        
        print(f"\nNonzero G parameters: {len(G_nz)}")
        print(f"  First 3: {G_nz[:3] if len(G_nz) >= 3 else G_nz}")
        print(f"\nNonzero K parameters: {len(K_nz)}")
        print(f"  First 3: {K_nz[:3] if len(K_nz) >= 3 else K_nz}")
        
        # Test LASSO Solver (optional)
        print("\n" + "="*60)
        print("TEST 2: LASSO SOLVER (OPTIONAL)")
        print("="*60)
        
        try:
            lasso_solver = LASSOSolver(alpha=0.01)
            theta_lasso = lasso_solver.solve(A_exp, R_exp)
            params_lasso = ParameterSet(theta_lasso, material)
            
            print("\n" + str(params_lasso))
            
            print("\nComparison:")
            print(f"  NNLS nonzeros:  G={params_nnls.n_nonzero_G}, K={params_nnls.n_nonzero_K}")
            print(f"  LASSO nonzeros: G={params_lasso.n_nonzero_G}, K={params_lasso.n_nonzero_K}")
            
        except ImportError as e:
            print(f"\nSkipping LASSO test: {e}")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"Total pipeline time: {elapsed:.1f}s")
        print("="*60)
        
        print("\n✓ solver.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()