"""
System assembly for inverse problem.
Computes ae matrices for all elements and timesteps.
"""

import numpy as np
from typing import Optional


class TimeCoefficients:
    """
    Computes BGt and BKt time-dependent coefficients.
    
    Extracted from trail_inv.py Block 4 (lines ~280-310).
    """
    
    def __init__(self, time: np.ndarray, tau_G: np.ndarray, tau_K: np.ndarray):
        """
        Compute BGt and BKt coefficients for all timesteps.
        
        Args:
            time: Time vector (n_timesteps,)
            tau_G: Deviatoric relaxation times (nG,)
            tau_K: Volumetric relaxation times (nK,)
        """
        self.time = time
        self.tau_G = tau_G
        self.tau_K = tau_K
        
        # Compute dt array (EXACT from trail_inv.py)
        self.dt = np.ones_like(time)
        self.dt[1:] = time[1:] - time[:-1]
        
        nG = len(tau_G)
        nK = len(tau_K)
        n_time = len(time)
        
        # Initialize BGt and BKt (EXACT from trail_inv.py)
        BGt_1 = np.zeros((nG + 1, n_time))
        BKt_1 = np.zeros((nK + 1, n_time))
        
        # Compute BGt (EXACT from trail_inv.py lines ~285-290)
        for t in range(n_time):
            for gamma in range(1, nG + 1):
                BGt_1[gamma, t] = 1 - (self.dt[t] / (2 * tau_G[gamma - 1] + self.dt[t]))
        
        self.BGt = BGt_1.copy()
        self.BGt[0, :] = 1  # G_inf term
        
        # Compute BKt (EXACT from trail_inv.py lines ~292-297)
        for t in range(n_time):
            for gamma in range(1, nK + 1):
                BKt_1[gamma, t] = 1 - (self.dt[t] / (2 * tau_K[gamma - 1] + self.dt[t]))
        
        self.BKt = BKt_1.copy()
        self.BKt[0, :] = 1  # K_inf term
        
        print(f"✓ Time coefficients: BGt shape={self.BGt.shape}, BKt shape={self.BKt.shape}")


class ElementMatrixComputer:
    """
    Computes ae matrix for a single element at a single timestep.
    
    Extracted from trail_inv.py Block 6 (lines ~450-520).
    """
    
    def __init__(self, mesh, material, exp_data, history, time_coeff):
        """Initialize with all dependencies."""
        self.mesh = mesh
        self.material = material
        self.exp_data = exp_data
        self.history = history
        self.time_coeff = time_coeff
        self.w_gp = 0.5
    
    def compute_ae(self, e: int, t: int) -> np.ndarray:
        """
        Compute ae matrix for element e at timestep t.
        
        Args:
            e: Element index
            t: Timestep index
        
        Returns:
            ae matrix of shape (6, n_params)
        """
        element = self.mesh.elements[e]
        Be_e = element.Be
        Bd_e = element.Bd
        b_e = element.b
        detJ_e = element.detJ
        
        # Get displacement vector
        Ue_t = np.zeros(2 * element.n_nodes)
        for i, node in enumerate(element.nodes):
            Ue_t[2*i] = self.exp_data.U[node.id, t]
            Ue_t[2*i + 1] = self.exp_data.U[node.id + self.mesh.n_nodes, t]
        
        # Get time coefficients
        BGt_t = self.time_coeff.BGt[:, t]
        BKt_t = self.time_coeff.BKt[:, t]
        
        # Get history
        beta_dev_prev = self.history.beta_dev[t-1] if t > 0 else None
        beta_vol_prev = self.history.beta_vol[t-1] if t > 0 else None
        
        nG = self.material.nG
        nK = self.material.nK
        
        # ========== DEVIATORIC ==========
        temp1 = Bd_e @ Ue_t
        temp2 = self.material.Dmu @ temp1
        temp3 = Be_e.T @ temp2
        temp4 = temp3 * self.w_gp * detJ_e
        
        aeG = temp4[:, np.newaxis] * BGt_t
        
        # Subtract history
        if beta_dev_prev is not None:
            for gamma in range(1, nG + 1):
                beta_hist = beta_dev_prev[:, gamma - 1, e]
                temp_hist = self.material.Dmu @ beta_hist
                hist_contrib = Be_e.T @ temp_hist * self.w_gp * detJ_e
                aeG[:, gamma] -= hist_contrib
        
        # ========== VOLUMETRIC ==========
        temp1 = np.outer(b_e, b_e) @ Ue_t
        temp2 = temp1 * self.w_gp * detJ_e
        
        aeK = temp2[:, np.newaxis] * BKt_t
        
        # Subtract history
        if beta_vol_prev is not None:
            for gamma in range(1, nK + 1):
                beta_hist = beta_vol_prev[0, gamma - 1, e]
                hist_contrib = b_e * beta_hist * self.w_gp * detJ_e
                aeK[:, gamma] -= hist_contrib
        
        # Combine
        return np.hstack([aeG, aeK])


class SystemAssembler:
    """Orchestrates computation of ae matrices."""
    
    def __init__(self, mesh, material, exp_data, history):
        self.mesh = mesh
        self.material = material
        self.exp_data = exp_data
        self.history = history
        
        # Precompute time coefficients
        print("Computing time coefficients (BGt, BKt)...")
        self.time_coeff = TimeCoefficients(
            exp_data.time, 
            material.tau_G, 
            material.tau_K
        )
        
        # Element matrix computer
        self.matrix_computer = ElementMatrixComputer(
            mesh, material, exp_data, history, self.time_coeff
        )
        
        # Storage
        self.ae = [[None for _ in range(exp_data.n_timesteps)] 
                   for _ in range(mesh.n_elements)]
        
        print(f"✓ System assembler initialized")
    
    def assemble(self):
        """Compute all ae matrices."""
        print("\nAssembling element matrices ae[e][t]...")
        print(f"  Elements: {self.mesh.n_elements}")
        print(f"  Timesteps: {self.exp_data.n_timesteps}")
        print(f"  ⚠️  Skipping t=0 (reference frame with zero displacement)") #added new to remove zero displacement at t=0 
        print(f"  This will take a few minutes...")
        
        timesteps = self.exp_data.n_timesteps
        
        for t in range(1,timesteps): # start from 1, not 0
            for e in range(self.mesh.n_elements):
                self.ae[e][t] = self.matrix_computer.compute_ae(e, t)
            
            if (t + 1) % 200 == 0 or t == 0:
                print(f"  ✓ Timestep {t+1}/{timesteps}")
        n_dofs_per_elem = self.mesh.elements[0].n_dof  # Gets 6
        for e in range(self.mesh.n_elements):
            self.ae[e][0] = np.zeros((n_dofs_per_elem, self.material.n_params))
        
        print(f"✓ Assembly complete")
        print(f"  ⚠️  Note: ae[e][0] = 0 for all elements (reference frame)")
        print(f"  ae[0][1] shape: {self.ae[0][1].shape}")
        print(f"  ae[0][1] nonzeros: {np.count_nonzero(self.ae[0][1])}")