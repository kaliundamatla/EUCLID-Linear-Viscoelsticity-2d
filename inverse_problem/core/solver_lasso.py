# Create: inverse_problem/solver_lasso.py

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import time

class LassoSolver:
    """
    LASSO solver for EUCLID inverse problem with non-negativity.
    """
    
    def __init__(self, alpha=1e-3, max_iter=10000, normalize=True):
        """
        Parameters:
        -----------
        alpha : float
            Regularization strength (larger = more sparse)
            Start with 1e-3, adjust if needed
        max_iter : int
            Maximum iterations
        normalize : bool
            Whether to normalize A matrix (recommended for ill-conditioned systems)
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.scaler = None
        
    def solve(self, A, R, verbose=True):
        """
        Solve: min ||Ax - R||² + alpha*||x||₁  subject to x ≥ 0
        
        Returns:
        --------
        theta : ndarray
            Solution vector (non-negative)
        info : dict
            Solution information
        """
        t_start = time.time()
        
        if verbose:
            print("="*60)
            print("SOLVING WITH LASSO (L1 REGULARIZATION)")
            print("="*60)
            print(f"System size: {A.shape[0]} equations, {A.shape[1]} unknowns")
            print(f"Regularization: alpha={self.alpha}")
            print(f"Max iterations: {self.max_iter}")
            print("="*60)
        
        # Optional: Normalize A for better conditioning
        if self.normalize:
            if verbose:
                print("Normalizing system matrix...")
            self.scaler = StandardScaler(with_mean=False)
            A_scaled = self.scaler.fit_transform(A)
        else:
            A_scaled = A
        
        # Create LASSO model with non-negativity
        model = Lasso(
            alpha=self.alpha,
            fit_intercept=False,  # No intercept (material parameters)
            positive=True,        # Force non-negativity
            max_iter=self.max_iter,
            warm_start=False,
            selection='random',   # Faster for large problems
            tol=1e-4
        )
        
        if verbose:
            print("Fitting LASSO model...")
        
        # Fit model
        model.fit(A_scaled, R)
        
        # Get solution
        theta_scaled = model.coef_
        
        # Unscale if needed
        if self.normalize:
            theta = theta_scaled / self.scaler.scale_
        else:
            theta = theta_scaled
        
        # Compute residual
        residual = R - A @ theta
        residual_norm = np.linalg.norm(residual)
        mse = np.mean(residual**2)
        
        # Count non-zero terms
        n_nonzero = np.sum(theta > 1e-10)
        
        t_solve = time.time() - t_start
        
        if verbose:
            print("="*60)
            print("✓ Solution converged")
            print(f"  Iterations: {model.n_iter_}")
            print(f"  Residual norm: {residual_norm:.6e}")
            print(f"  MSE: {mse:.6e}")
            print(f"  Nonzero parameters: {n_nonzero}/{len(theta)}")
            print(f"  Solution time: {t_solve:.1f}s")
            print("="*60)
        
        # Package info
        info = {
            'success': True,
            'n_iter': model.n_iter_,
            'residual_norm': residual_norm,
            'mse': mse,
            'n_nonzero': n_nonzero,
            'time': t_solve,
            'alpha': self.alpha
        }
        
        return theta, info
