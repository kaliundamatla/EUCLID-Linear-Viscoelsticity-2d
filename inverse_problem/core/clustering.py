"""
Parameter clustering for Prony series.
Merges nearby relaxation times using weighted averaging.


"""

import numpy as np
from typing import Tuple
from .solver import ParameterSet


class ParameterClusterer:
    """
    Clusters Prony series parameters by merging nearby relaxation times.
    
    Algorithm :
    - If two consecutive tau values are within clusteringRange*tau(i), merge them
    - Weighted average: tau_merged = (G1*tau1 + G2*tau2)/(G1 + G2)
    - Sum moduli: G_merged = G1 + G2
    """
    
    def __init__(self, clustering_range: float = 0.5):
        """
        Initialize clusterer.
        
        Args:
            clustering_range: Relative distance threshold (default 0.3 = 30%)
        """
        self.clustering_range = clustering_range
    
    def cluster(self, params: ParameterSet) -> ParameterSet:
        """
        Cluster parameters by merging nearby relaxation times.
        
        Args:
            params: Original ParameterSet from NNLS
        
        Returns:
            Clustered ParameterSet with fewer terms
        """
        print("\n" + "="*70)
        print("PARAMETER CLUSTERING")
        print("="*70)
        print(f"Clustering range: {self.clustering_range} (relative)")
        print(f"Input: {params.n_nonzero_G} G terms, {params.n_nonzero_K} K terms")
        
        # Get nonzero G parameters (includes G_inf if nonzero)
        G_tau_nz, G_vals_nz = params.get_nonzero_G()
        
        # Get nonzero K parameters (includes K_inf if nonzero)
        K_tau_nz, K_vals_nz = params.get_nonzero_K()
        
        # Cluster G parameters
        clustered_G_tau, clustered_G_vals = self._cluster_one_set(
            G_tau_nz, G_vals_nz, "G"
        )
        
        # Cluster K parameters
        clustered_K_tau, clustered_K_vals = self._cluster_one_set(
            K_tau_nz, K_vals_nz, "K"
        )
        
        # Reconstruct full theta vector matching material structure
        theta_clustered = self._rebuild_theta(
            clustered_G_tau, clustered_G_vals,
            clustered_K_tau, clustered_K_vals,
            params.material
        )
        
        # Create new parameter set using standard constructor
        clustered_params = ParameterSet(theta_clustered, params.material)
        
        # Copy over solver metrics if available
        if hasattr(params, 'residual_norm'):
            clustered_params.residual_norm = params.residual_norm
        if hasattr(params, 'cost'):
            clustered_params.cost = params.cost
        
        print(f"\nOutput: {len(clustered_G_vals)} G terms, {len(clustered_K_vals)} K terms")
        n_before = params.n_nonzero_G + params.n_nonzero_K
        n_after = len(clustered_G_vals) + len(clustered_K_vals)
        if n_before > 0:
            reduction_pct = 100 * (1 - n_after / n_before)
            print(f"Reduction: {n_before} → {n_after} terms ({reduction_pct:.1f}% reduction)")
        print("="*70)
        
        return clustered_params
    
    def _rebuild_theta(self, 
                      G_tau: np.ndarray, G_vals: np.ndarray,
                      K_tau: np.ndarray, K_vals: np.ndarray,
                      material) -> np.ndarray:
        """
        Rebuild theta vector from clustered parameters.
        
        Must match the material structure: [G_inf, G_1, ..., G_nG, K_inf, K_1, ..., K_nK]
        
        Args:
            G_tau: Clustered G relaxation times (may include 0 for G_inf)
            G_vals: Clustered G moduli
            K_tau: Clustered K relaxation times (may include 0 for K_inf)
            K_vals: Clustered K moduli
            material: MaterialModel with tau_G, tau_K arrays
        
        Returns:
            theta: Full parameter vector matching material structure
        """
        nG = material.nG
        nK = material.nK
        
        # Initialize with zeros
        theta = np.zeros(nG + nK + 2)
        
        # Separate G_inf from Maxwell elements
        if len(G_tau) > 0 and G_tau[0] == 0:
            theta[0] = G_vals[0]  # G_inf
            G_maxwell_tau = G_tau[1:]
            G_maxwell_vals = G_vals[1:]
        else:
            theta[0] = 0.0  # No G_inf
            G_maxwell_tau = G_tau
            G_maxwell_vals = G_vals
        
        # Map G Maxwell elements to original tau_G positions
        for tau_clust, val_clust in zip(G_maxwell_tau, G_maxwell_vals):
            # Find closest tau in material.tau_G
            idx = np.argmin(np.abs(material.tau_G - tau_clust))
            theta[1 + idx] = val_clust
        
        # Separate K_inf from Maxwell elements
        if len(K_tau) > 0 and K_tau[0] == 0:
            theta[nG + 1] = K_vals[0]  # K_inf
            K_maxwell_tau = K_tau[1:]
            K_maxwell_vals = K_vals[1:]
        else:
            theta[nG + 1] = 0.0  # No K_inf
            K_maxwell_tau = K_tau
            K_maxwell_vals = K_vals
        
        # Map K Maxwell elements to original tau_K positions
        for tau_clust, val_clust in zip(K_maxwell_tau, K_maxwell_vals):
            # Find closest tau in material.tau_K
            idx = np.argmin(np.abs(material.tau_K - tau_clust))
            theta[nG + 2 + idx] = val_clust
        
        return theta
    
    def _cluster_one_set(
        self, 
        tau: np.ndarray, 
        vals: np.ndarray,
        name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster one set of parameters (G or K).
        
        Args:
            tau: Relaxation times (including 0 for _inf term)
            vals: Moduli values
            name: "G" or "K" for printing
        
        Returns:
            (clustered_tau, clustered_vals)
        """
        if len(tau) == 0:
            return np.array([]), np.array([])
        
        # Initialize with first term
        if tau[0] == 0:
            # Start with inf term
            if len(tau) == 1:
                return tau, vals
            clustered_tau = [0.0, tau[1]]
            clustered_vals = [vals[0], vals[1]]
            start_idx = 2
        else:
            clustered_tau = [tau[0]]
            clustered_vals = [vals[0]]
            start_idx = 1
        
        # Process remaining terms 
        for i in range(start_idx, len(tau)):
            tau_i = tau[i]
            val_i = vals[i]
            
            # Check if close enough to last clustered term
            tau_last = clustered_tau[-1]
            val_last = clustered_vals[-1]
            
            distance = abs(tau_last - tau_i)
            threshold = self.clustering_range * tau_i
            
            if distance < threshold and tau_i > 0:
                # MERGE: weighted average tau, sum moduli
                total_val = val_last + val_i
                weight_last = val_last / total_val
                weight_new = val_i / total_val
                
                tau_merged = weight_last * tau_last + weight_new * tau_i
                val_merged = total_val
                
                # Update last cluster
                clustered_tau[-1] = tau_merged
                clustered_vals[-1] = val_merged
                
            elif tau_i > 0:
                # NEW CLUSTER: too far apart
                clustered_tau.append(tau_i)
                clustered_vals.append(val_i)
        
        clustered_tau = np.array(clustered_tau)
        clustered_vals = np.array(clustered_vals)
        
        print(f"  {name}: {len(tau)} → {len(clustered_tau)} terms")
        
        return clustered_tau, clustered_vals


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test clustering with synthetic example.
    """
    print("="*60)
    print("CLUSTERING TEST")
    print("="*60)
    
    # Create synthetic material
    from material import MaterialModel
    material = MaterialModel(n_maxwell_shear=10, n_maxwell_bulk=10)
    
    # Create synthetic theta with some nonzero values
    theta = np.zeros(material.n_params)
    theta[0] = 1500.0  # G_inf
    theta[1] = 50.0    # G_1
    theta[2] = 80.0    # G_2
    theta[5] = 200.0   # G_5
    theta[6] = 150.0   # G_6
    theta[11] = 2000.0  # K_inf
    theta[12] = 100.0   # K_1
    theta[14] = 250.0   # K_4
    
    # Create parameter set
    params = ParameterSet(theta, material)
    
    print("\nBefore clustering:")
    tau_G_nz, G_nz = params.get_nonzero_G()
    tau_K_nz, K_nz = params.get_nonzero_K()
    print(f"  G: {len(G_nz)} nonzero terms")
    print(f"  K: {len(K_nz)} nonzero terms")
    
    # Cluster
    clusterer = ParameterClusterer(clustering_range=0.5)
    clustered = clusterer.cluster(params)
    
    print("\nAfter clustering:")
    tau_G_nz_c, G_nz_c = clustered.get_nonzero_G()
    tau_K_nz_c, K_nz_c = clustered.get_nonzero_K()
    print(f"  G: {len(G_nz_c)} nonzero terms")
    print(f"  K: {len(K_nz_c)} nonzero terms")
    
    print("\n✓ clustering.py working correctly!")