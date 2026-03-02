"""
Comprehensive visualization for inverse problem results.
Generates detailed plots at each stage of the pipeline.
"""

import numpy as np
import matplotlib

from .solver import ParameterSet
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inverse_problem import InverseProblem


matplotlib.use('Agg')  # Non-interactive backend for HPC/server
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
from typing import Optional

class InverseProblemVisualizer:
    """
    Creates publication-quality visualizations for inverse problem.
    
    Usage:
        viz = InverseProblemVisualizer(problem, output_dir)
        viz.plot_all()  # Generate all plots
        
        # Or individual plots:
        viz.plot_data_validation()
        viz.plot_mesh_quality()
        # etc.
    """
    
    def __init__(self, inverse_problem, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            inverse_problem: InverseProblem instance (after run())
            output_dir: Directory to save figures
        """
        self.problem = inverse_problem
        self.output_dir = Path(output_dir) / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract commonly used data
        self.exp_data = inverse_problem.exp_data
        self.mesh = inverse_problem.mesh
        self.material = inverse_problem.material
        self.history = inverse_problem.history
        self.system_assembler = inverse_problem.system_assembler
        self.boundary_assembler = inverse_problem.boundary_assembler
        self.parameters = inverse_problem.parameters
        self.A_exp = inverse_problem.A_exp
        self.R_exp = inverse_problem.R_exp
        
        # Coordinate arrays for plotting
        self.xs = self.exp_data.coord[:, 1]
        self.ys = self.exp_data.coord[:, 2]
        self.conne = self.exp_data.conne
        
        print(f"✓ Visualizer initialized: {self.output_dir}")
    
    def _savefig(self, filename: str):
        """Save figure with standard settings."""
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    # ========== BLOCK 0: Preprocessing Visualization (NEW) ==========

    def plot_preprocessing(self):
        """Block 0: Preprocessing visualization (correction and filtering effects)."""
        print("\n[0/9] Preprocessing Visualization Plot...")
    
        # Check if any preprocessing was applied
        has_correction = hasattr(self.exp_data, 'displacement_corrected') and self.exp_data.displacement_corrected
        has_filtering = hasattr(self.exp_data, 'filtering_applied') and self.exp_data.filtering_applied
        has_original = hasattr(self.exp_data, 'U_original')
    
        if not (has_correction or has_filtering):
            print("  ⚠️  No preprocessing applied - skipping preprocessing visualization")
            return
    
        if not has_original:
            print("  ⚠️  Original data not stored - skipping preprocessing visualization")
            return
    
        fig = plt.figure(figsize=(18, 12))
    
        # Get data for comparison
        U_original = self.exp_data.U_original
        U_final = self.exp_data.U
        n_nodes = self.mesh.n_nodes
        time_vec = self.exp_data.time
    
        # Select sample nodes for time history (top, center, bottom)
        top_nodes = self.mesh.get_boundary_nodes('top')
        top_node_id = top_nodes[len(top_nodes)//2].id if len(top_nodes) > 0 else 0
        center_node_id = n_nodes // 2
    
        # Panel 1: Y-displacement field (ORIGINAL)
        ax1 = plt.subplot(3, 3, 1)
        triangulation = tri.Triangulation(self.xs, self.ys, self.conne)
        Uy_original = U_original[n_nodes:, -1]
        tcf = ax1.tricontourf(triangulation, Uy_original, levels=20, cmap='RdBu_r')
        ax1.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax1, label='Uy [mm]')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title('ORIGINAL: Uy at final time')
        ax1.set_aspect('equal')
    
        # Panel 2: Y-displacement field (PREPROCESSED)
        ax2 = plt.subplot(3, 3, 2)
        Uy_final = U_final[n_nodes:, -1]
        tcf = ax2.tricontourf(triangulation, Uy_final, levels=20, cmap='RdBu_r')
        ax2.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax2, label='Uy [mm]')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_title('PREPROCESSED: Uy at final time')
        ax2.set_aspect('equal')
    
        # Panel 3: Difference field
        ax3 = plt.subplot(3, 3, 3)
        Uy_diff = Uy_final - Uy_original
        tcf = ax3.tricontourf(triangulation, Uy_diff, levels=20, cmap='seismic')
        ax3.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax3, label='Δ Uy [mm]')
        ax3.set_xlabel('X [mm]')
        ax3.set_ylabel('Y [mm]')
        ax3.set_title(f'DIFFERENCE (max: {np.abs(Uy_diff).max():.4f} mm)')
        ax3.set_aspect('equal')
    
        # Panel 4: Top node Y-displacement history
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(time_vec, U_original[n_nodes + top_node_id, :], 'b-', 
                linewidth=2, alpha=0.7, label='Original')
        ax4.plot(time_vec, U_final[n_nodes + top_node_id, :], 'r-', 
                linewidth=2, alpha=0.7, label='Preprocessed')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Uy [mm]')
        ax4.set_title(f'Top Node (id={top_node_id}): Y-displacement')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
        # Panel 5: Center node Y-displacement history
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(time_vec, U_original[n_nodes + center_node_id, :], 'b-', 
                linewidth=2, alpha=0.7, label='Original')
        ax5.plot(time_vec, U_final[n_nodes + center_node_id, :], 'r-', 
                linewidth=2, alpha=0.7, label='Preprocessed')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Uy [mm]')
        ax5.set_title(f'Center Node (id={center_node_id}): Y-displacement')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
        # Panel 6: Noise analysis (temporal derivative)
        ax6 = plt.subplot(3, 3, 6)
        # Compute temporal derivative (noise indicator)
        dU_dt_original = np.diff(U_original[n_nodes:, :], axis=1)
        dU_dt_final = np.diff(U_final[n_nodes:, :], axis=1)
    
        noise_original = np.std(dU_dt_original, axis=1)
        noise_final = np.std(dU_dt_final, axis=1)
    
        ax6.hist(noise_original, bins=50, alpha=0.6, label='Original', color='blue', edgecolor='black')
        ax6.hist(noise_final, bins=50, alpha=0.6, label='Preprocessed', color='red', edgecolor='black')
        ax6.set_xlabel('Noise Level (std of dU/dt) [mm/step]')
        ax6.set_ylabel('Node Count')
        ax6.set_title('Noise Distribution Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
        # Panel 7: Signal statistics (histograms)
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(U_original[n_nodes:, :].flatten(), bins=100, alpha=0.6, 
                label='Original', color='blue', edgecolor='black', density=True)
        ax7.hist(U_final[n_nodes:, :].flatten(), bins=100, alpha=0.6, 
                label='Preprocessed', color='red', edgecolor='black', density=True)
        ax7.set_xlabel('Y-displacement [mm]')
        ax7.set_ylabel('Probability Density')
        ax7.set_title('Displacement Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
        # Panel 8: Correction effect on top boundary
        if has_correction:
            ax8 = plt.subplot(3, 3, 8)
            top_node_ids = np.array([n.id for n in top_nodes], dtype=np.int64)
        
            # Original top displacements
            top_Uy_original = np.mean(U_original[n_nodes + top_node_ids, :], axis=0)
            top_Uy_final = np.mean(U_final[n_nodes + top_node_ids, :], axis=0)
        
            ax8.plot(time_vec, top_Uy_original, 'b-', linewidth=2, label='Original (rigid body motion)')
            ax8.plot(time_vec, top_Uy_final, 'r-', linewidth=2, label='Corrected (≈0)')
            ax8.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel('Mean Top Uy [mm]')
            ax8.set_title('Displacement Correction Effect')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8 = plt.subplot(3, 3, 8)
            ax8.axis('off')
            ax8.text(0.5, 0.5, 'Displacement correction\nnot applied', 
                    ha='center', va='center', fontsize=12)
    
        # Panel 9: Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
    
        # Compute statistics
        noise_reduction = (1 - np.mean(noise_final) / np.mean(noise_original)) * 100 if has_filtering else 0.0
        rms_change = np.sqrt(np.mean((U_final - U_original)**2))
        signal_range = np.ptp(U_original)
        relative_change = (rms_change / signal_range) * 100
    
        # Max shift from correction
        if has_correction:
            max_shift = np.max(np.abs(U_original - U_final))
        else:
            max_shift = 0.0
    
        summary = f"""
    PREPROCESSING SUMMARY

    Applied Operations:
        Displacement correction: {'✓ YES' if has_correction else '✗ NO'}
        Noise filtering:        {'✓ YES' if has_filtering else '✗ NO'}

    Displacement Statistics:
        Original Uy range:   [{U_original[n_nodes:,:].min():.4f}, {U_original[n_nodes:,:].max():.4f}] mm
        Final Uy range:      [{U_final[n_nodes:,:].min():.4f}, {U_final[n_nodes:,:].max():.4f}] mm
        Max shift:           {max_shift:.4f} mm

    Noise Analysis:
        Original noise (avg): {np.mean(noise_original):.2e} mm/step
        Final noise (avg):    {np.mean(noise_final):.2e} mm/step
        Noise reduction:      {noise_reduction:.1f}%

    Signal Preservation:
        RMS change:          {rms_change:.2e} mm
        Relative change:     {relative_change:.2f}%
        Signal range:        {signal_range:.4f} mm

    Quality Assessment:
        {'✓ Excellent' if relative_change < 2 else '⚠️ Significant' if relative_change < 5 else '❌ Large'}: Signal change {relative_change:.1f}%
        {'✓ Effective' if noise_reduction > 10 else '⚠️ Minimal' if noise_reduction > 0 else 'N/A'}: Noise reduction
    """
    
        ax9.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
        plt.tight_layout()
        self._savefig('00_preprocessing.png')
    
    # ========== BLOCK 1: Data Validation ==========
    
    def plot_data_validation(self):
        """Block 1: Data loading and mesh visualization."""
        print("\n[1/8] Data Validation Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        triangulation = tri.Triangulation(self.xs, self.ys, self.conne)
        
        # Panel 1: Mesh geometry
        ax1 = plt.subplot(2, 3, 1)
        ax1.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.6)
        ax1.plot(self.xs, self.ys, 'ro', markersize=2, alpha=0.5)
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title(f'Mesh: {self.mesh.n_nodes} nodes, {self.mesh.n_elements} elements')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Ux displacement at final timestep
        ax2 = plt.subplot(2, 3, 2)
        Ux_final = self.exp_data.U[:self.mesh.n_nodes, -1]
        tcf = ax2.tricontourf(triangulation, Ux_final, levels=20, cmap='RdBu_r')
        ax2.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax2, label='Ux [mm]')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_title(f'Ux at t={self.exp_data.time[-1]:.1f}s')
        ax2.set_aspect('equal')
        
        # Panel 3: Uy displacement at final timestep
        ax3 = plt.subplot(2, 3, 3)
        Uy_final = self.exp_data.U[self.mesh.n_nodes:, -1]
        tcf = ax3.tricontourf(triangulation, Uy_final, levels=20, cmap='RdBu_r')
        ax3.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax3, label='Uy [mm]')
        ax3.set_xlabel('X [mm]')
        ax3.set_ylabel('Y [mm]')
        ax3.set_title(f'Uy at t={self.exp_data.time[-1]:.1f}s')
        ax3.set_aspect('equal')
        
        # Panel 4: Force time history
        ax4 = plt.subplot(2, 3, 4)
        for i in range(self.exp_data.F.shape[0]):
            ax4.plot(self.exp_data.time, self.exp_data.F[i, :], alpha=0.7, linewidth=1, label=f'F{i+1}')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Force [N]')
        ax4.set_title('Boundary Forces vs Time')
        ax4.grid(True, alpha=0.3)
        if self.exp_data.F.shape[0] <= 5:
            ax4.legend(fontsize=8)
        
        # Panel 5: Time vector
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.exp_data.time, 'b.-', markersize=2, linewidth=0.5)
        ax5.set_xlabel('Timestep Index')
        ax5.set_ylabel('Time [s]')
        ax5.set_title(f'Time Vector: {self.exp_data.n_timesteps} steps')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary = f"""
DATA LOADING SUMMARY

Mesh:
  Nodes:       {self.mesh.n_nodes}
  Elements:    {self.mesh.n_elements}
  X range:     [{self.xs.min():.2f}, {self.xs.max():.2f}] mm
  Y range:     [{self.ys.min():.2f}, {self.ys.max():.2f}] mm

Time:
  Timesteps:   {self.exp_data.n_timesteps}
  Duration:    {self.exp_data.time[-1]:.2f} s
  t_start:     {self.exp_data.time[0]:.3f} s
  t_end:       {self.exp_data.time[-1]:.3f} s

Displacements:
  Ux range:    [{self.exp_data.U[:self.mesh.n_nodes,:].min():.4f}, {self.exp_data.U[:self.mesh.n_nodes,:].max():.4f}] mm
  Uy range:    [{self.exp_data.U[self.mesh.n_nodes:,:].min():.4f}, {self.exp_data.U[self.mesh.n_nodes:,:].max():.4f}] mm
  
Forces:
  DOFs:        {self.exp_data.F.shape[0]}
  Max force:   {self.exp_data.F.max():.2f} N
  Min force:   {self.exp_data.F.min():.2f} N

Material:
  tau_G:       [{self.material.tau_G[0]:.1f}, {self.material.tau_G[-1]:.1f}] s (n={self.material.nG})
  tau_K:       [{self.material.tau_K[0]:.1f}, {self.material.tau_K[-1]:.1f}] s (n={self.material.nK})
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top', 
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('01_data_validation.png')
    
    # ========== BLOCK 2: Mesh Quality ==========
    
    def plot_mesh_quality(self):
        """Block 2: Element quality and geometry matrices."""
        print("\n[2/8] Mesh Quality Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        triangulation = tri.Triangulation(self.xs, self.ys, self.conne)
        
        # Extract element properties
        detJ_array = np.array([elem.detJ for elem in self.mesh.elements])
        Area_array = detJ_array / 2.0  # Triangle area = detJ/2
        
        # Panel 1: Mesh colored by Area
        ax1 = plt.subplot(2, 3, 1)
        tcf = ax1.tripcolor(triangulation, Area_array, cmap='viridis', shading='flat')
        ax1.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.3)
        plt.colorbar(tcf, ax=ax1, label='Area [mm²]')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title('Element Areas')
        ax1.set_aspect('equal')
        
        # Panel 2: Area histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(Area_array, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(Area_array.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {Area_array.mean():.4f}')
        ax2.axvline(np.median(Area_array), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(Area_array):.4f}')
        ax2.set_xlabel('Area [mm²]')
        ax2.set_ylabel('Element Count')
        ax2.set_title('Element Area Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Be matrix heatmap (first element)
        elem0 = self.mesh.elements[0]
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(elem0.Be, cmap='RdBu_r', aspect='auto')
        ax3.set_xlabel('DOF')
        ax3.set_ylabel('Strain Component')
        ax3.set_title('Be Matrix (Element 0)')
        ax3.set_yticks(range(elem0.Be.shape[0]))
        ax3.set_yticklabels(['εxx', 'εyy', 'γxy'])
        plt.colorbar(im, ax=ax3)
        
        # Panel 4: Bd matrix heatmap
        ax4 = plt.subplot(2, 3, 4)
        im = ax4.imshow(elem0.Bd, cmap='RdBu_r', aspect='auto')
        ax4.set_xlabel('DOF')
        ax4.set_ylabel('Deviatoric Strain')
        ax4.set_title('Bd Matrix (Element 0)')
        ax4.set_yticks(range(elem0.Bd.shape[0]))
        ax4.set_yticklabels(['εxx', 'εyy', 'γxy'])
        plt.colorbar(im, ax=ax4)
        
        # Panel 5: b vector for multiple elements
        ax5 = plt.subplot(2, 3, 5)
        elem_samples = range(min(10, self.mesh.n_elements))
        for e in elem_samples:
            ax5.plot(range(6), self.mesh.elements[e].b, 'o-', alpha=0.6, markersize=4, label=f'Elem {e}')
        ax5.set_xlabel('DOF Index')
        ax5.set_ylabel('b Value')
        ax5.set_title('Volumetric b Vector (First 10 Elements)')
        ax5.legend(fontsize=7, ncol=2)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        degenerate = detJ_array < 1e-10
        summary = f"""
MESH QUALITY SUMMARY

Element Matrices:
  Total elements:   {self.mesh.n_elements}
  Plane stress:     {self.material.plane_stress}
  
Jacobian Statistics:
  Mean:             {detJ_array.mean():.6f}
  Median:           {np.median(detJ_array):.6f}
  Min:              {detJ_array.min():.6f}
  Max:              {detJ_array.max():.6f}
  Std dev:          {detJ_array.std():.6f}

Area Statistics:
  Mean:             {Area_array.mean():.6f} mm²
  Median:           {np.median(Area_array):.6f} mm²
  Min:              {Area_array.min():.6f} mm²
  Max:              {Area_array.max():.6f} mm²
  Total area:       {Area_array.sum():.2f} mm²
  
Element Quality:
  Degenerate (<1e-10): {np.sum(degenerate)}
  Small (<1e-6):       {np.sum(detJ_array < 1e-6)}
  Good (>1e-6):        {np.sum(detJ_array >= 1e-6)}

Matrix Dimensions:
  Be:               {elem0.Be.shape}
  Bd:               {elem0.Bd.shape}
  b:                {elem0.b.shape}
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('02_mesh_quality.png')
    
    # ========== BLOCK 3: Time Coefficients ==========
    
    def plot_time_coefficients(self):
        """Block 3: BGt and BKt time-dependent coefficients."""
        print("\n[3/8] Time Coefficients Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        
        BGt = self.system_assembler.time_coeff.BGt
        BKt = self.system_assembler.time_coeff.BKt
        time_vec = self.exp_data.time
        
        # Panel 1: BGt coefficients (first 10 gammas)
        ax1 = plt.subplot(2, 3, 1)
        n_plot = min(10, self.material.nG)
        for gamma in range(n_plot):
            ax1.plot(time_vec, BGt[gamma+1, :], alpha=0.7, linewidth=1, label=f'γ={gamma+1}')
        ax1.plot(time_vec, BGt[0, :], 'k--', linewidth=2, label='G_inf')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('BGt Coefficient')
        ax1.set_title(f'BGt Coefficients (first {n_plot} gammas)')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: BKt coefficients (first 10 gammas)
        ax2 = plt.subplot(2, 3, 2)
        n_plot = min(10, self.material.nK)
        for gamma in range(n_plot):
            ax2.plot(time_vec, BKt[gamma+1, :], alpha=0.7, linewidth=1, label=f'γ={gamma+1}')
        ax2.plot(time_vec, BKt[0, :], 'k--', linewidth=2, label='K_inf')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('BKt Coefficient')
        ax2.set_title(f'BKt Coefficients (first {n_plot} gammas)')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: BGt at specific time steps
        ax3 = plt.subplot(2, 3, 3)
        timesteps = self.exp_data.n_timesteps
        t_samples = [0, timesteps//4, timesteps//2, 3*timesteps//4, timesteps-1]
        for t_idx in t_samples:
            ax3.plot(range(self.material.nG+1), BGt[:, t_idx], 'o-', markersize=3, alpha=0.7, 
                    label=f't={time_vec[t_idx]:.1f}s')
        ax3.set_xlabel('Gamma Index')
        ax3.set_ylabel('BGt Value')
        ax3.set_title('BGt Distribution Across Gammas')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: BKt at specific time steps
        ax4 = plt.subplot(2, 3, 4)
        for t_idx in t_samples:
            ax4.plot(range(self.material.nK+1), BKt[:, t_idx], 's-', markersize=3, alpha=0.7, 
                    label=f't={time_vec[t_idx]:.1f}s')
        ax4.set_xlabel('Gamma Index')
        ax4.set_ylabel('BKt Value')
        ax4.set_title('BKt Distribution Across Gammas')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Displacement time history (element 0)
        ax5 = plt.subplot(2, 3, 5)
        elem0 = self.mesh.elements[0]
        dofs = elem0.get_global_dofs()
        for i in range(min(3, elem0.n_nodes)):
            Ux = self.exp_data.U[dofs[2*i], :]
            Uy = self.exp_data.U[dofs[2*i+1], :]
            ax5.plot(time_vec, Ux, alpha=0.7, label=f'Node{i+1}_X')
            ax5.plot(time_vec, Uy, alpha=0.7, linestyle='--', label=f'Node{i+1}_Y')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Displacement [mm]')
        ax5.set_title('Nodal Displacements (Element 0)')
        ax5.legend(fontsize=7, ncol=2)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary = f"""
TIME COEFFICIENTS SUMMARY

BGt Coefficients:
  Shape:        {BGt.shape}
  G_inf (γ=0):  All = {BGt[0, 0]:.4f}
  Range (γ>0):  [{BGt[1:, 0].min():.4f}, {BGt[1:, 0].max():.4f}]

BKt Coefficients:
  Shape:        {BKt.shape}
  K_inf (γ=0):  All = {BKt[0, 0]:.4f}
  Range (γ>0):  [{BKt[1:, 0].min():.4f}, {BKt[1:, 0].max():.4f}]

Displacements:
  Max Ux:       {self.exp_data.U[:self.mesh.n_nodes, :].max():.4f} mm
  Max Uy:       {self.exp_data.U[self.mesh.n_nodes:, :].max():.4f} mm

Time Integration:
  tau_G range:  [{self.material.tau_G[0]:.2f}, {self.material.tau_G[-1]:.2f}] s
  tau_K range:  [{self.material.tau_K[0]:.2f}, {self.material.tau_K[-1]:.2f}] s
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('03_time_coefficients.png')
    
    # ========== BLOCK 4: Beta Coefficients ==========
    
    def plot_beta_coefficients(self):
        """Block 4: History variables (beta coefficients)."""
        print("\n[4/8] Beta Coefficients Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        time_vec = self.exp_data.time
        
        # Panel 1: Beta dev evolution for element 0, gamma 0
        ax1 = plt.subplot(2, 3, 1)
        t_samples = np.linspace(0, self.exp_data.n_timesteps-1, min(100, self.exp_data.n_timesteps), dtype=int)
        beta_11 = [self.history.beta_dev[t][0, 0, 0] for t in t_samples]
        beta_22 = [self.history.beta_dev[t][1, 0, 0] for t in t_samples]
        beta_12 = [self.history.beta_dev[t][2, 0, 0] for t in t_samples]
        ax1.plot(time_vec[t_samples], beta_11, 'b-', linewidth=2, label='β₁₁')
        ax1.plot(time_vec[t_samples], beta_22, 'r-', linewidth=2, label='β₂₂')
        ax1.plot(time_vec[t_samples], beta_12, 'g-', linewidth=2, label='β₁₂')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('β_dev')
        ax1.set_title('β_dev Evolution (Element 0, γ=0)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Beta vol evolution for element 0, gamma 0
        ax2 = plt.subplot(2, 3, 2)
        beta_vol = [self.history.beta_vol[t][0, 0, 0] for t in t_samples]
        ax2.plot(time_vec[t_samples], beta_vol, 'b-', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('β_vol')
        ax2.set_title('β_vol Evolution (Element 0, γ=0)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Beta dev at final time for all gammas
        ax3 = plt.subplot(2, 3, 3)
        t_final = self.exp_data.n_timesteps - 1
        beta_dev_final = self.history.beta_dev[t_final][:, :, 0]  # (3, nG) for element 0
        ax3.plot(beta_dev_final[0, :], 'o-', markersize=4, label='β₁₁')
        ax3.plot(beta_dev_final[1, :], 's-', markersize=4, label='β₂₂')
        ax3.plot(beta_dev_final[2, :], '^-', markersize=4, label='β₁₂')
        ax3.set_xlabel('Maxwell Element γ')
        ax3.set_ylabel('β_dev at final time')
        ax3.set_title(f'β_dev Distribution (Element 0, t={time_vec[-1]:.1f}s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Beta vol at final time for all gammas
        ax4 = plt.subplot(2, 3, 4)
        beta_vol_final = self.history.beta_vol[t_final][0, :, 0]  # (nK,) for element 0
        ax4.plot(beta_vol_final, 'o-', markersize=4, color='purple')
        ax4.set_xlabel('Maxwell Element γ')
        ax4.set_ylabel('β_vol at final time')
        ax4.set_title(f'β_vol Distribution (Element 0, t={time_vec[-1]:.1f}s)')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Beta dev magnitude across elements at t=final
        ax5 = plt.subplot(2, 3, 5)
        elem_samples = range(min(20, self.mesh.n_elements))
        beta_mag = []
        for e in elem_samples:
            beta_e = self.history.beta_dev[t_final][:, 0, e]  # gamma=0
            beta_mag.append(np.linalg.norm(beta_e))
        ax5.plot(elem_samples, beta_mag, 'o-', markersize=4)
        ax5.set_xlabel('Element Index')
        ax5.set_ylabel('||β_dev|| (γ=0)')
        ax5.set_title('β_dev Magnitude Across Elements')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Compute statistics
        beta_dev_all = np.array([self.history.beta_dev[t] for t in range(self.exp_data.n_timesteps)])
        beta_vol_all = np.array([self.history.beta_vol[t] for t in range(self.exp_data.n_timesteps)])
        
        summary = f"""
BETA COEFFICIENTS SUMMARY

History Data:
  Timesteps:      {self.history.n_timesteps}
  Elements:       {self.history.n_elements}
  nG (dev):       {self.history.nG}
  nK (vol):       {self.history.nK}

β_dev Statistics:
  Shape per t:    (3, {self.history.nG}, {self.history.n_elements})
  Range:          [{beta_dev_all.min():.6e}, {beta_dev_all.max():.6e}]
  Mean |β_dev|:   {np.mean(np.abs(beta_dev_all)):.6e}

β_vol Statistics:
  Shape per t:    (1, {self.history.nK}, {self.history.n_elements})
  Range:          [{beta_vol_all.min():.6e}, {beta_vol_all.max():.6e}]
  Mean |β_vol|:   {np.mean(np.abs(beta_vol_all)):.6e}

Sample Values (elem 0, γ=0):
  β_dev[0]:       [{beta_dev_all[0, 0, 0, 0]:.3e}, {beta_dev_all[0, 1, 0, 0]:.3e}, {beta_dev_all[0, 2, 0, 0]:.3e}]
  β_vol[0]:       {beta_vol_all[0, 0, 0, 0]:.3e}
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('04_beta_coefficients.png')
    
    # ========== BLOCK 5: Element Matrices ==========
    
    def plot_element_matrices(self):
        """Block 5: ae matrices assembly."""
        print("\n[5/8] Element Matrices Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        
        ae = self.system_assembler.ae
        time_vec = self.exp_data.time
        
        # Panel 1: ae matrix heatmap (element 0, t=0)
        ax1 = plt.subplot(2, 3, 1)
        im = ax1.imshow(ae[0][0], cmap='RdBu_r', aspect='auto')
        ax1.set_xlabel('Maxwell Element Index')
        ax1.set_ylabel('Element DOF')
        ax1.set_title('ae Matrix (Element 0, t=0)')
        ax1.axvline(self.material.nG + 0.5, color='green', linestyle='--', linewidth=2, label='G/K split')
        plt.colorbar(im, ax=ax1)
        ax1.legend()
        
        # Panel 2: ae matrix heatmap (element 0, t=1)
        ax2 = plt.subplot(2, 3, 2)
        if self.exp_data.n_timesteps > 1:
            im = ax2.imshow(ae[0][1], cmap='RdBu_r', aspect='auto')
            ax2.set_xlabel('Maxwell Element Index')
            ax2.set_ylabel('Element DOF')
            ax2.set_title('ae Matrix (Element 0, t=1)')
            ax2.axvline(self.material.nG + 0.5, color='green', linestyle='--', linewidth=2, label='G/K split')
            plt.colorbar(im, ax=ax2)
            ax2.legend()
        
        # Panel 3: ae magnitude over time
        ax3 = plt.subplot(2, 3, 3)
        t_samples = np.linspace(0, self.exp_data.n_timesteps-1, min(50, self.exp_data.n_timesteps), dtype=int)
        ae_norms = [np.linalg.norm(ae[0][t]) for t in t_samples]
        ax3.plot(time_vec[t_samples], ae_norms, 'b-', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('||ae|| (Element 0)')
        ax3.set_title('ae Matrix Frobenius Norm Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Non-zero entries evolution
        ax4 = plt.subplot(2, 3, 4)
        nz_counts = [np.count_nonzero(ae[0][t]) for t in t_samples]
        ax4.plot(time_vec[t_samples], nz_counts, 'ro-', linewidth=2, markersize=4)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Non-zero Entries')
        ax4.set_title('ae Sparsity Evolution (Element 0)')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: ae magnitude across elements at t=0
        ax5 = plt.subplot(2, 3, 5)
        elem_samples = range(min(20, self.mesh.n_elements))
        ae_mags = [np.linalg.norm(ae[e][0]) for e in elem_samples]
        ax5.plot(elem_samples, ae_mags, 'o-', markersize=4)
        ax5.set_xlabel('Element Index')
        ax5.set_ylabel('||ae|| at t=0')
        ax5.set_title('ae Magnitude Across Elements')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        ae_mag_all = [[np.linalg.norm(ae[e][t]) for t in range(min(10, self.exp_data.n_timesteps))] 
                      for e in range(self.mesh.n_elements)]
        
        summary = f"""
ELEMENT MATRICES SUMMARY

ae Matrix Assembly:
  Elements:         {self.mesh.n_elements}
  Timesteps:        {self.exp_data.n_timesteps}
  Maxwell (G):      {self.material.nG} + 1 (G_inf)
  Maxwell (K):      {self.material.nK} + 1 (K_inf)

Matrix Dimensions:
  ae[e][t]:         {ae[0][0].shape}

ae Statistics (t=0):
  ||ae|| mean:      {np.mean([np.linalg.norm(ae[e][0]) for e in range(self.mesh.n_elements)]):.4e}
  ||ae|| max:       {np.max([np.linalg.norm(ae[e][0]) for e in range(self.mesh.n_elements)]):.4e}
  ||ae|| min:       {np.min([np.linalg.norm(ae[e][0]) for e in range(self.mesh.n_elements)]):.4e}

Non-zero Entries (Element 0):
  t=0:  {np.count_nonzero(ae[0][0])}
  t=1:  {np.count_nonzero(ae[0][1]) if self.exp_data.n_timesteps > 1 else 'N/A'}

Integration:
  w_gp:             0.5
  detJ (elem 0):    {self.mesh.elements[0].detJ:.6f}
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('05_element_matrices.png')
    
    # ========== BLOCK 6: Boundary Assembly ==========
    
    def plot_boundary_assembly(self):
        """Block 6: Boundary condition assembly."""
        print("\n[6/8] Boundary Assembly Plot...")
        
        fig = plt.figure(figsize=(16, 10))
        triangulation = tri.Triangulation(self.xs, self.ys, self.conne)
        
        # Get boundary info
        edge_dofs = self.boundary_assembler.edge_dofs
        edge_names = list(edge_dofs.keys())
        
        # Panel 1: Mesh with boundary DOFs highlighted
        ax1 = plt.subplot(2, 3, 1)
        ax1.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.3)
        ax1.plot(self.xs, self.ys, 'o', color='lightgray', markersize=3, alpha=0.5, label='Internal')
        
        # Highlight boundary nodes
        all_boundary_dofs = np.concatenate([dofs for dofs in edge_dofs.values()])
        boundary_node_ids = np.unique(all_boundary_dofs // 2)
        ax1.plot(self.xs[boundary_node_ids], self.ys[boundary_node_ids], 'ro', markersize=4, label='Boundary')
        
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title(f'Boundary Nodes ({len(boundary_node_ids)} nodes)')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: A_exp sparsity pattern
        ax2 = plt.subplot(2, 3, 2)
        A_exp_sparse = np.abs(self.A_exp) > 1e-10
        # Downsample if too large
        if A_exp_sparse.shape[0] > 1000:
            stride = A_exp_sparse.shape[0] // 1000
            A_exp_sparse = A_exp_sparse[::stride, :]
        im = ax2.imshow(A_exp_sparse, cmap='binary', aspect='auto', interpolation='nearest')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Equation Row (downsampled)')
        ax2.set_title(f'A_exp Sparsity Pattern\n{self.A_exp.shape}')
        ax2.axvline(self.material.nG + 0.5, color='green', linestyle='--', linewidth=2, alpha=0.5)
        
        # Panel 3: R_exp force vector
        ax3 = plt.subplot(2, 3, 3)
        # Downsample if too large
        R_plot = self.R_exp
        if len(R_plot) > 2000:
            stride = len(R_plot) // 2000
            R_plot = R_plot[::stride]
        ax3.plot(R_plot, 'b-', linewidth=0.5)
        ax3.set_xlabel('Equation Row')
        ax3.set_ylabel('Force [N]')
        ax3.set_title('R_exp Force Vector')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Boundary edge DOF counts
        ax4 = plt.subplot(2, 3, 4)
        edge_counts = [len(dofs) for dofs in edge_dofs.values()]
        ax4.bar(range(len(edge_names)), edge_counts, color='steelblue', alpha=0.7)
        ax4.set_xticks(range(len(edge_names)))
        ax4.set_xticklabels(edge_names, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('DOF Count')
        ax4.set_title('DOFs per Boundary Edge')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Panel 5: System conditioning
        ax5 = plt.subplot(2, 3, 5)
        # Compute singular values (expensive, so sample if too large)
        if self.A_exp.shape[0] > 5000:
            A_sample = self.A_exp[::self.A_exp.shape[0]//1000, :]
            s = np.linalg.svd(A_sample, compute_uv=False)
        else:
            s = np.linalg.svd(self.A_exp, compute_uv=False)
        ax5.semilogy(s, 'b-', linewidth=2)
        ax5.set_xlabel('Singular Value Index')
        ax5.set_ylabel('Singular Value')
        ax5.set_title(f'A_exp Singular Values\nCondition: {s[0]/s[-1]:.2e}')
        ax5.grid(True, which='both', alpha=0.3)
        
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        A_exp_nnz = np.count_nonzero(self.A_exp)
        A_exp_sparsity = 100 * (1 - A_exp_nnz / self.A_exp.size)
        
        summary = f"""
BOUNDARY ASSEMBLY SUMMARY

System Configuration:
  Boundary edges:   {len(edge_names)}
  Total boundary DOFs: {len(boundary_node_ids)}
  λ_interior:       {self.boundary_assembler.lambda_i}
  λ_boundary:       {self.boundary_assembler.lambda_r}

Global System:
  A_exp shape:      {self.A_exp.shape}
  R_exp shape:      {self.R_exp.shape}
  Overdetermined:   {self.A_exp.shape[0]/self.A_exp.shape[1]:.1f}x
  
Matrix Statistics:
  Non-zeros:        {A_exp_nnz}
  Sparsity:         {A_exp_sparsity:.1f}%
  ||A_exp||_F:      {np.linalg.norm(self.A_exp, 'fro'):.4e}
  ||R_exp||_2:      {np.linalg.norm(self.R_exp):.4e}

Conditioning:
  σ_max:            {s[0]:.4e}
  σ_min:            {s[-1]:.4e}
  κ(A):             {s[0]/s[-1]:.4e}
"""
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        self._savefig('06_boundary_assembly.png')
    
    # ========== BLOCK 7: Solution ==========
    
    def plot_solution(self):
        """Block 7: NNLS solution and identified parameters."""
        print("\n[7/8] Solution Plot...")
    
        fig = plt.figure(figsize=(16, 10))
    
        theta = self.parameters.theta
        tau = self.material.tau_full
    
        # Separate G and K
        firstKindex = self.material.nG + 1
    
        # Filter nonzero
        nonzero_mask = theta > 1e-10
        theta_nz = theta[nonzero_mask]
        tau_nz = tau[nonzero_mask]
    
        # Panel 1: Full solution
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(theta, '.b', markersize=3)
        ax1.axvline(firstKindex - 0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='G/K split')
        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('θ (Prony Coefficient)')
        ax1.set_title(f'Full Solution ({len(theta)} coefficients)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Panel 2: Non-zero solution
        ax2 = plt.subplot(2, 3, 2)
        if len(theta_nz) > 0:
            ax2.semilogy(tau_nz, theta_nz, '.r', markersize=6)
            if firstKindex < len(tau):
                ax2.axvline(tau[firstKindex], color='green', linestyle='--', linewidth=2, alpha=0.5, label='G/K split')
            ax2.set_xlabel('Relaxation Time τ [s]')
            ax2.set_ylabel('θ (Prony Coefficient)')
            ax2.set_title(f'Non-zero Solution ({len(theta_nz)} coefficients)')
            if len(tau_nz[tau_nz > 0]) > 0:
                ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, which='both', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'All parameters zero', ha='center', va='center', fontsize=12)
            ax2.set_title('No non-zero solution')
    
        # Panel 3: G and K separated
        ax3 = plt.subplot(2, 3, 3)
        tau_G_nz, G_nz = self.parameters.get_nonzero_G()
        tau_K_nz, K_nz = self.parameters.get_nonzero_K()
    
        has_data = False
        if len(G_nz) > 0:
            ax3.semilogy(tau_G_nz, G_nz, 'ob', markersize=6, label=f'G (shear, n={len(G_nz)})')
            has_data = True
        if len(K_nz) > 0:
            ax3.semilogy(tau_K_nz, K_nz, 'sr', markersize=6, label=f'K (bulk, n={len(K_nz)})')
            has_data = True
    
        if has_data:
            ax3.set_xlabel('Relaxation Time τ [s]')
            ax3.set_ylabel('Modulus [MPa]')
            ax3.set_title('Separated G and K Parameters')
            all_tau = np.concatenate([tau_G_nz, tau_K_nz])
            if len(all_tau[all_tau > 0]) > 0:
                ax3.set_xscale('log')
            ax3.legend()
            ax3.grid(True, which='both', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Only equilibrium terms\n(G_∞ and K_∞)', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('No Maxwell Elements Identified')
    
        # Panel 4: Residual analysis
        ax4 = plt.subplot(2, 3, 4)
        residual = self.A_exp @ theta - self.R_exp
        if len(residual) > 2000:
            stride = len(residual) // 2000
            residual_plot = residual[::stride]
        else:
            residual_plot = residual
        ax4.plot(residual_plot, 'b-', linewidth=0.5)
        ax4.set_xlabel('Equation Row')
        ax4.set_ylabel('Residual')
        ax4.set_title(f'Residual Vector\n||r||={np.linalg.norm(residual):.4e}')
        ax4.grid(True, alpha=0.3)
    
        # Panel 5: Residual histogram
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(residual, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Residual Value')
        ax5.set_ylabel('Count')
        ax5.set_title('Residual Distribution')
        ax5.grid(True, axis='y', alpha=0.3)
    
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
    
        metrics = self.problem.solver.get_metrics()
        status_msg = metrics.get('message', 'N/A')
        if isinstance(status_msg, str):
            status_msg = status_msg[:50]
        else:
            status_msg = str(status_msg)[:50]
    
        theta_range = f"[{theta_nz.min():.2e}, {theta_nz.max():.2e}]" if len(theta_nz) > 0 else "N/A"
    
        summary = f"""
    SOLUTION SUMMARY

    NNLS Solution:
        Total parameters:   {len(theta)}
        Non-zero:           {len(theta_nz)}
        Sparsity:           {100*(1-len(theta_nz)/len(theta)):.1f}%
        θ range:            {theta_range}

    Identified Parameters:
        G_∞:                {self.parameters.G_inf:.2f} MPa
        K_∞:                {self.parameters.K_inf:.2f} MPa
        G_0 (total):        {self.parameters.total_G:.2f} MPa
        K_0 (total):        {self.parameters.total_K:.2f} MPa

    Prony Series:
        G terms:            {self.parameters.n_nonzero_G}/{self.material.nG}
        K terms:            {self.parameters.n_nonzero_K}/{self.material.nK}

    Solver Metrics:
        Cost:               {metrics.get('cost', 0):.4e}
        Residual norm:      {metrics.get('residual_norm', 0):.4e}
        MSE:                {metrics.get('mse', 0):.4e}
        Status:             {status_msg}
    """
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
        plt.tight_layout()
        self._savefig('07_solution.png')
    
    # ========== BLOCK 8: Prony Series ==========
    
    def plot_prony_series(self):
        """Block 8: Final Prony series visualization."""
        print("\n[8/8] Prony Series Plot...")
    
        fig = plt.figure(figsize=(16, 10))
    
        tau_G_nz, G_nz = self.parameters.get_nonzero_G()
        tau_K_nz, K_nz = self.parameters.get_nonzero_K()
    
        # Panel 1: G Prony series (stem plot)
        ax1 = plt.subplot(2, 3, 1)
        if len(G_nz) > 0:
            ax1.stem(tau_G_nz, G_nz, basefmt=' ')
            if len(tau_G_nz[tau_G_nz > 0]) > 0:  # Only set log scale if we have positive tau values
                ax1.set_xscale('log')
            ax1.set_xlabel('Relaxation Time τ_G [s]')
            ax1.set_ylabel('G [MPa]')
            ax1.set_title(f'Deviatoric Prony Series\n{len(G_nz)} terms, G_∞={self.parameters.G_inf:.2f} MPa')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Only G_∞ identified', ha='center', va='center')
            ax1.set_title(f'Deviatoric: G_∞={self.parameters.G_inf:.2f} MPa')
    
        # Panel 2: K Prony series (stem plot)
        ax2 = plt.subplot(2, 3, 2)
        if len(K_nz) > 0:
            ax2.stem(tau_K_nz, K_nz, basefmt=' ', linefmt='C1-', markerfmt='C1o')
            if len(tau_K_nz[tau_K_nz > 0]) > 0:  # Only set log scale if we have positive tau values
                ax2.set_xscale('log')
            ax2.set_xlabel('Relaxation Time τ_K [s]')
            ax2.set_ylabel('K [MPa]')
            ax2.set_title(f'Volumetric Prony Series\n{len(K_nz)} terms, K_∞={self.parameters.K_inf:.2f} MPa')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Only K_∞ identified', ha='center', va='center')
            ax2.set_title(f'Volumetric: K_∞={self.parameters.K_inf:.2f} MPa')
    
        # Panel 3: Combined spectrum
        ax3 = plt.subplot(2, 3, 3)
        has_maxwell = False
    
        if len(G_nz) > 1:  # Exclude G_inf
            G_maxwell = G_nz[1:] if len(tau_G_nz) > 0 and tau_G_nz[0] == 0 else G_nz
            tau_G_maxwell = tau_G_nz[1:] if len(tau_G_nz) > 0 and tau_G_nz[0] == 0 else tau_G_nz
            if len(G_maxwell) > 0:
                ax3.semilogy(tau_G_maxwell, G_maxwell, 'ob', markersize=8, label=f'G (n={len(G_maxwell)})')
                has_maxwell = True
    
        if len(K_nz) > 1:  # Exclude K_inf
            K_maxwell = K_nz[1:] if len(tau_K_nz) > 0 and tau_K_nz[0] == 0 else K_nz
            tau_K_maxwell = tau_K_nz[1:] if len(tau_K_nz) > 0 and tau_K_nz[0] == 0 else tau_K_nz
            if len(K_maxwell) > 0:
                ax3.semilogy(tau_K_maxwell, K_maxwell, 'sr', markersize=8, label=f'K (n={len(K_maxwell)})')
                has_maxwell = True
    
        if has_maxwell:
            ax3.set_xscale('log')
            ax3.set_xlabel('Relaxation Time τ [s]')
            ax3.set_ylabel('Modulus [MPa]')
            ax3.set_title('Combined Prony Spectrum')
            ax3.legend()
            ax3.grid(True, which='both', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Only equilibrium terms\n(G_∞ and K_∞)', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('No Maxwell Elements Identified')
    
        # Panel 4: Relaxation function G(t)
        ax4 = plt.subplot(2, 3, 4)
        t_eval = np.logspace(np.log10(0.1), np.log10(self.exp_data.time[-1]), 100)
        G_t = self.parameters.G_inf * np.ones_like(t_eval)
    
        # Add Maxwell contributions if they exist
        if len(G_nz) > 0:
            G_maxwell_indices = range(1, len(G_nz)) if len(tau_G_nz) > 0 and tau_G_nz[0] == 0 else range(len(G_nz))
            for idx in G_maxwell_indices:
                if tau_G_nz[idx] > 0:  # Only add if tau > 0
                    G_t += G_nz[idx] * np.exp(-t_eval / tau_G_nz[idx])
    
        ax4.plot(t_eval, G_t, 'b-', linewidth=2)
        ax4.set_xscale('log')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('G(t) [MPa]')
        ax4.set_title('Shear Relaxation Function')
        ax4.grid(True, which='both', alpha=0.3)
    
        # Panel 5: Relaxation function K(t)
        ax5 = plt.subplot(2, 3, 5)
        K_t = self.parameters.K_inf * np.ones_like(t_eval)
    
        # Add Maxwell contributions if they exist
        if len(K_nz) > 0:
            K_maxwell_indices = range(1, len(K_nz)) if len(tau_K_nz) > 0 and tau_K_nz[0] == 0 else range(len(K_nz))
            for idx in K_maxwell_indices:
                if tau_K_nz[idx] > 0:  # Only add if tau > 0
                    K_t += K_nz[idx] * np.exp(-t_eval / tau_K_nz[idx])
    
        ax5.plot(t_eval, K_t, 'r-', linewidth=2)
        ax5.set_xscale('log')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('K(t) [MPa]')
        ax5.set_title('Bulk Relaxation Function')
        ax5.grid(True, which='both', alpha=0.3)
    
        # Panel 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
    
        # Safe access to tau ranges
        tau_G_range_str = f"[{tau_G_nz.min():.2e}, {tau_G_nz.max():.2e}]" if len(tau_G_nz) > 0 else "N/A"
        tau_K_range_str = f"[{tau_K_nz.min():.2e}, {tau_K_nz.max():.2e}]" if len(tau_K_nz) > 0 else "N/A"
    
        summary = f"""
    PRONY SERIES SUMMARY

    Experiment:           {self.problem.experiment_number}

    Identified Parameters:
    G_∞:                {self.parameters.G_inf:.2f} MPa
    G Maxwell terms:    {self.parameters.n_nonzero_G}
    G_0 (instantaneous): {self.parameters.total_G:.2f} MPa

    K_∞:                {self.parameters.K_inf:.2f} MPa
    K Maxwell terms:    {self.parameters.n_nonzero_K}
    K_0 (instantaneous): {self.parameters.total_K:.2f} MPa

    Relaxation Times:
    τ_G range:          {tau_G_range_str} s
    τ_K range:          {tau_K_range_str} s

    Material Response:
    G(0):               {self.parameters.total_G:.2f} MPa
    G(∞):               {self.parameters.G_inf:.2f} MPa
    K(0):               {self.parameters.total_K:.2f} MPa
    K(∞):               {self.parameters.K_inf:.2f} MPa

    Computation Time:
    Total:              {sum(self.problem.timings.values()):.1f} s
    """
        ax6.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
        plt.tight_layout()
        self._savefig('08_prony_series.png')
    
        # ========== Main Interface ==========
    
    def plot_all(self):
        """Generate all visualization plots."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*70)
        
        self.plot_preprocessing()
        self.plot_data_validation()
        self.plot_mesh_quality()
        self.plot_time_coefficients()
        self.plot_beta_coefficients()
        self.plot_element_matrices()
        self.plot_boundary_assembly()
        self.plot_solution()
        self.plot_prony_series()
        
        print("\n" + "="*70)
        print(f"✓ All 8 visualization sets saved to:")
        print(f"  {self.output_dir}")
        print("="*70)

    def plot_clustering_comparison(
        self,
        params_before: ParameterSet,
        params_after: ParameterSet
        ) -> str:
        """
        Plot before/after clustering comparison.

        Args:
            params_before: Parameters before clustering
            params_after: Parameters after clustering

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Get nonzero parameters for both before and after
        G_tau_before, G_vals_before = params_before.get_nonzero_G()
        K_tau_before, K_vals_before = params_before.get_nonzero_K()
        G_tau_after, G_vals_after = params_after.get_nonzero_G()
        K_tau_after, K_vals_after = params_after.get_nonzero_K()

        # ========== DEVIATORIC (G) ==========
        ax = axes[0]

        # Before clustering (small dots)
        if len(G_tau_before) > 0:
            ax.scatter(G_tau_before, G_vals_before, 
                    s=30, c='lightblue', alpha=0.6, label='Before clustering', 
                    edgecolors='blue', linewidths=0.5, zorder=2)

        # After clustering (stem plot)
        if len(G_tau_after) > 0:
            markerline, stemlines, baseline = ax.stem(
                G_tau_after, G_vals_after,
                linefmt='C0-', markerfmt='C0o', basefmt=' ', label='After clustering'
            )
            markerline.set_markersize(10)
            markerline.set_markerfacecolor('blue')
            markerline.set_markeredgecolor('darkblue')
            markerline.set_markeredgewidth(2)
            stemlines.set_linewidth(2)

        # G_inf marker
        ax.scatter([0.8], [params_after.G_inf], s=200, marker='s',
                c='blue', edgecolors='darkblue', linewidths=2,
                label=f'$G_∞$ = {params_after.G_inf:.1f} MPa', zorder=3)

        ax.set_xscale('log')
        ax.set_xlabel('Relaxation time τ_G [s]', fontsize=13, fontweight='bold')
        ax.set_ylabel('G [MPa]', fontsize=13, fontweight='bold')
        ax.set_title(f'Deviatoric Prony Series (G)\n{len(G_vals_after)} parameters', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_xlim([0.5, 1000])

        # Add reduction text
        reduction_G = len(G_vals_before) - len(G_vals_after)
        ax.text(0.98, 0.02, 
            f'Reduction: {len(G_vals_before)} → {len(G_vals_after)} terms\n'
            f'({reduction_G} terms merged)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ========== VOLUMETRIC (K) ==========
        ax = axes[1]

        # Before clustering (small dots)
        if len(K_tau_before) > 0:
            ax.scatter(K_tau_before, K_vals_before,
                    s=30, c='lightsalmon', alpha=0.6, label='Before clustering',
                    edgecolors='orangered', linewidths=0.5, zorder=2)

        # After clustering (stem plot)
        if len(K_tau_after) > 0:
            markerline, stemlines, baseline = ax.stem(
                K_tau_after, K_vals_after,
                linefmt='C1-', markerfmt='C1o', basefmt=' ', label='After clustering'
            )
            markerline.set_markersize(10)
            markerline.set_markerfacecolor('orangered')
            markerline.set_markeredgecolor('darkred')
            markerline.set_markeredgewidth(2)
            stemlines.set_linewidth(2)

        # K_inf marker
        ax.scatter([0.8], [params_after.K_inf], s=200, marker='s',
                c='orangered', edgecolors='darkred', linewidths=2,
                label=f'$K_∞$ = {params_after.K_inf:.1f} MPa', zorder=3)

        ax.set_xscale('log')
        ax.set_xlabel('Relaxation time τ_K [s]', fontsize=13, fontweight='bold')
        ax.set_ylabel('K [MPa]', fontsize=13, fontweight='bold')
        ax.set_title(f'Volumetric Prony Series (K)\n{len(K_vals_after)} parameters',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_xlim([0.5, 1000])

        # Add reduction text
        reduction_K = len(K_vals_before) - len(K_vals_after)
        ax.text(0.98, 0.02,
            f'Reduction: {len(K_vals_before)} → {len(K_vals_after)} terms\n'
            f'({reduction_K} terms merged)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "09_clustering_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: 09_clustering_comparison.png")

        return str(save_path)

# ========== Convenience Function ==========

def visualize_results(inverse_problem, output_dir: Optional[Path] = None):
    """
    Convenience function to generate all plots.
    
    Args:
        inverse_problem: InverseProblem instance (after run())
        output_dir: Output directory (default: Postprocessing/final_outputs/exp_number)
    
    Usage:
        from visualization import visualize_results
        
        problem = InverseProblem(713)
        problem.run()
        visualize_results(problem)
    """
    if output_dir is None:
        output_dir = Path("../Postprocessing/final_outputs") / str(inverse_problem.experiment_number)
    
    viz = InverseProblemVisualizer(inverse_problem, output_dir)
    viz.plot_all()
    
    return viz


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test visualizations.
    Run: python visualization.py
    
    This requires a completed InverseProblem run.
    """
    print("="*70)
    print("VISUALIZATION MODULE TEST")
    print("="*70)
    print("\nThis module requires a completed InverseProblem run.")
    print("Run inverse_problem.py first, then use:")
    print("  from visualization import visualize_results")
    print("  visualize_results(problem)")
    print("="*70)