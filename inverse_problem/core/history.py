"""
History data computation for inverse problem.
Computes beta coefficients from displacement field.
"""

import numpy as np
from typing import List
from pathlib import Path

# Import the external beta computation function
from .beta_computation import inverse_problem_input_realData_noEps33


class HistoryData:
    """
    Container for beta coefficients (viscoelastic history variables).
    
    Storage format:
    - beta_dev[t]: (3, nG, nEl) array for timestep t
    - beta_vol[t]: (1, nK, nEl) array for timestep t
    """
    
    def __init__(self, n_timesteps: int, n_elements: int, 
                 n_maxwell_shear: int, n_maxwell_bulk: int):
        """
        Initialize storage for beta coefficients.
        
        Args:
            n_timesteps: Number of time steps
            n_elements: Number of finite elements
            n_maxwell_shear: Number of deviatoric Maxwell elements (nG)
            n_maxwell_bulk: Number of volumetric Maxwell elements (nK)
        """
        self.n_timesteps = n_timesteps
        self.n_elements = n_elements
        self.nG = n_maxwell_shear
        self.nK = n_maxwell_bulk
        
        # Storage: list of arrays (one per timestep)
        # beta_dev[t].shape = (3, nG, nEl)
        # beta_vol[t].shape = (1, nK, nEl)
        self.beta_dev: List[np.ndarray] = []
        self.beta_vol: List[np.ndarray] = []
    
    def set_timestep(self, t: int, beta_dev_t: np.ndarray, beta_vol_t: np.ndarray):
        """
        Store beta coefficients for a specific timestep.
        
        Args:
            t: Timestep index
            beta_dev_t: Deviatoric beta (3, nG, nEl)
            beta_vol_t: Volumetric beta (1, nK, nEl)
        """
        assert beta_dev_t.shape == (3, self.nG, self.n_elements), \
            f"beta_dev shape {beta_dev_t.shape} != expected (3, {self.nG}, {self.n_elements})"
        assert beta_vol_t.shape == (1, self.nK, self.n_elements), \
            f"beta_vol shape {beta_vol_t.shape} != expected (1, {self.nK}, {self.n_elements})"
        
        if t >= len(self.beta_dev):
            self.beta_dev.append(beta_dev_t)
            self.beta_vol.append(beta_vol_t)
        else:
            self.beta_dev[t] = beta_dev_t
            self.beta_vol[t] = beta_vol_t
    
    def get_element_dev(self, element_id: int, gamma: int) -> np.ndarray:
        """
        Get deviatoric beta history for a specific element and Maxwell branch.
        
        Args:
            element_id: Element index
            gamma: Maxwell element index (0 to nG-1)
        
        Returns:
            Array of shape (3, n_timesteps) - beta values over time
        """
        return np.array([self.beta_dev[t][:, gamma, element_id] 
                        for t in range(self.n_timesteps)]).T
    
    def get_element_vol(self, element_id: int, gamma: int) -> np.ndarray:
        """
        Get volumetric beta history for a specific element and Maxwell branch.
        
        Args:
            element_id: Element index
            gamma: Maxwell element index (0 to nK-1)
        
        Returns:
            Array of shape (n_timesteps,) - beta values over time
        """
        return np.array([self.beta_vol[t][0, gamma, element_id] 
                        for t in range(self.n_timesteps)])
    
    def __repr__(self):
        return (f"HistoryData: {self.n_timesteps} timesteps, "
                f"{self.n_elements} elements, nG={self.nG}, nK={self.nK}")


class BetaComputer:
    """
    Computes beta coefficients by calling external function.
    
    Wraps inverse_problem_input_realData_noEps33 in clean OOP interface.
    """
    
    def __init__(self, mesh, material):
        """
        Initialize beta computer.
        
        Args:
            mesh: Mesh object (from geometry.py)
            material: MaterialModel object (from material.py)
        """
        self.mesh = mesh
        self.material = material
    
    def compute(self, exp_data) -> HistoryData:
        """
        Compute beta coefficients for all timesteps and elements.
        
        Args:
            exp_data: ExperimentData object (from data.py)
        
        Returns:
            HistoryData object containing all beta coefficients
        """
        print(f"Computing β coefficients...")
        print(f"  Elements: {self.mesh.n_elements}")
        print(f"  Timesteps: {exp_data.n_timesteps}")
        print(f"  Maxwell branches: nG={self.material.nG}, nK={self.material.nK}")
        
        # Prepare inputs for external function
        name_dir = str(exp_data.base_path / str(exp_data.experiment_number))
        
        # Call external function
        betGnm1_list, betKnm1_list = inverse_problem_input_realData_noEps33(
            nameDir=name_dir,
            time=exp_data.time,
            dt=exp_data.dt,
            coord=exp_data.coord,
            U=exp_data.U,
            conne=exp_data.conne,
            Nel=self.mesh.n_elements,
            NMeG=self.material.nG,
            NMeK=self.material.nK,
            tauG=self.material.tau_G,
            tauK=self.material.tau_K
        )
        
        # Package into HistoryData object
        history = HistoryData(
            n_timesteps=exp_data.n_timesteps,
            n_elements=self.mesh.n_elements,
            n_maxwell_shear=self.material.nG,
            n_maxwell_bulk=self.material.nK
        )
        
        # Store all timesteps
        for t in range(exp_data.n_timesteps):
            history.set_timestep(t, betGnm1_list[t], betKnm1_list[t])
        
        print(f"✓ β coefficients computed successfully")
        return history


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test beta computation with experiment 713.
    Run: python history.py
    
    WARNING: This is SLOW (several minutes)
    """
    from data import ExperimentData
    from geometry import Mesh
    from material import MaterialModel
    import time
    
    try:
        print("="*60)
        print("BETA COMPUTATION TEST")
        print("="*60)
        print("⚠️  Warning: This will take several minutes...")
        
        start_time = time.time()
        
        # Load data
        exp_data = ExperimentData(713)
        
        # Create mesh
        mesh = Mesh(exp_data.coord, exp_data.conne)
        
        # Create material (use smaller nG, nK for faster testing)
        print("\nUsing reduced model for testing (nG=10, nK=10)")
        material = MaterialModel(n_maxwell_shear=10, n_maxwell_bulk=10)
        
        # Compute beta
        beta_computer = BetaComputer(mesh, material)
        history = beta_computer.compute(exp_data)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(history)
        print(f"\nComputation time: {elapsed:.1f} seconds")
        
        # Check some values
        print("\nSample values:")
        print(f"  β_dev[0][0, 0, 0] (t=0, gamma=0, elem=0):")
        print(f"    {history.beta_dev[0][:, 0, 0]}")
        print(f"  β_vol[0][0, 0, 0] (t=0, gamma=0, elem=0):")
        print(f"    {history.beta_vol[0][0, 0, 0]:.6e}")
        
        # Plot evolution for element 0, gamma 0
        print(f"\n  β_dev[t][0, 0, 0] for first 5 timesteps:")
        for t in range(min(5, exp_data.n_timesteps)):
            print(f"    t={t}: {history.beta_dev[t][0, 0, 0]:.6e}")
        
        print("\n✓ history.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()