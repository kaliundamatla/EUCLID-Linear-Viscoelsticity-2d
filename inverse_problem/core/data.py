"""
Data loading and validation for EUCLID inverse problem.
Handles experimental data input and verification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class ExperimentData:

    
    def __init__(self, experiment_number: int, base_path: Path = None):
        """
        Load experimental data for given experiment number.
        
        Args:
            experiment_number: Experiment ID (e.g., 800, 801, 900)
            base_path: Path to synthetic_data directory (contains both synthetic and real preprocessed data).
                      If None, uses '../synthetic_data' (relative to inverse_problem/)
        """
        self.experiment_number = experiment_number
        
        # Default path: root-level synthetic_data folder (same for synthetic and real data)
        if base_path is None:
            base_path = Path(__file__).parent.parent / "synthetic_data"
        
        self.base_path = Path(base_path)
        
        # Data arrays (will be populated by _load_all)
        self.coord: Optional[np.ndarray] = None
        self.conne: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        
        # Load and validate
        self._load_all()
        self._validate()
    
    def _load_all(self):
        """Load all CSV files from experiment directory."""
        data_dir = self.base_path / str(self.experiment_number)
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Expected structure: {self.base_path}/{{experiment_number}}/\n"
                f"Looking for files: coord.csv, conne.txt, U.csv, F.csv, time.csv"
            )
        
        print(f"Loading data from: {data_dir}")
        
        # Load coordinates
        coord_file = data_dir / "coord.csv"
        if not coord_file.exists():
            raise FileNotFoundError(f"coord.csv not found in {data_dir}")
        nodes = pd.read_csv(coord_file, header=None, skiprows=1)
        self.coord = nodes.values
        
        # Load connectivity (robust multi-delimiter parsing)
        conne_file = data_dir / "conne.txt"
        if not conne_file.exists():
            raise FileNotFoundError(f"conne.txt not found in {data_dir}")
        
        import re
        data = []
        with open(conne_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cols = re.split(r'[,\s]+', line)
                try:
                    nums = [int(c) for c in cols if c]
                    if len(nums) >= 4:
                        data.append(nums[:4])
                except (ValueError, IndexError):
                    continue
        
        if not data:
            raise ValueError(f"conne.txt unreadable at {conne_file}")
        
        conne_raw = pd.DataFrame(data)
        self.conne = conne_raw.iloc[:, 1:4].values.astype(np.int64)
        self.conne = self.conne - 1
        
        # Load displacements (with fallback to numpy)
        u_file = data_dir / "U.csv"
        if not u_file.exists():
            raise FileNotFoundError(f"U.csv not found in {data_dir}")
        try:
            self.U = pd.read_csv(u_file, header=None).values
        except:
            self.U = np.loadtxt(u_file, delimiter=',', skiprows=0)
        
        # Load forces
        f_file = data_dir / "F.csv"
        if not f_file.exists():
            raise FileNotFoundError(f"F.csv not found in {data_dir}")
        try:
            self.F = pd.read_csv(f_file, header=None).values
        except:
            self.F = np.loadtxt(f_file, delimiter=',', skiprows=0)
        self.F = self.F * 1000
        
        # Load time
        time_file = data_dir / "time.csv"
        if not time_file.exists():
            raise FileNotFoundError(f"time.csv not found in {data_dir}")
        try:
            self.time = pd.read_csv(time_file, header=None).values.flatten()
        except:
            self.time = np.loadtxt(time_file, delimiter=',', skiprows=0)
        
        print(f"[OK] Loaded: {self.n_nodes} nodes, {self.n_elements} elements, {self.n_timesteps} timesteps")

    def _validate(self):
        """
        Validate loaded data for consistency.
        """
        # Derived quantities
        nNodes = self.coord.shape[0]
        nElements = self.conne.shape[0]
        timesteps = self.time.size
        
        # Check connectivity not empty FIRST
        if nElements == 0:
            raise ValueError("Connectivity array is empty - check conne.txt format")
        
        # 1. Shape checks
        assert self.U.shape == (2 * nNodes, timesteps), \
            f"U shape {self.U.shape} != expected ({2*nNodes}, {timesteps})"
        
        assert self.F.shape[1] == timesteps, \
            f"F has {self.F.shape[1]} columns but time has {timesteps}"
        
        # 2. Connectivity bounds check
        assert self.conne.min() >= 0, f"Connectivity has negative indices: min={self.conne.min()}"
        assert self.conne.max() < nNodes, \
            f"Connectivity out of range: max={self.conne.max()}, nNodes={nNodes}"
        
        # 3. Time vector checks
        assert np.all(np.isfinite(self.time)), "Time vector contains NaN or Inf"
        assert np.all(np.diff(self.time) > 0), "Time vector is not strictly increasing"
        
        # 4. Coordinate checks
        xs, ys = self.coord[:, 1], self.coord[:, 2]
        assert xs.shape == (nNodes,) and ys.shape == (nNodes,), \
            "Coordinate columns not as expected [id, x, y, ...]"
        
        # 5. Node ID uniqueness (optional but good to check)
        node_ids = self.coord[:, 0].astype(np.int64)
        assert len(np.unique(node_ids)) == nNodes, "Duplicate node IDs found in coord"
        
        print(f"[OK] Validation passed: All data consistent")

    # ========== Alternative constructors ==========

    @classmethod
    def from_real_data(cls, preprocessed_dir: Path):
        """
        Load preprocessed real DIC data.

        This method creates an ExperimentData instance from preprocessed real data
        generated by the DICPreprocessor (preprocessing_real_data.py).

        Args:
            preprocessed_dir: Directory containing preprocessed CSV files
                             (coord.csv, conne.txt, U.csv, F.csv, time.csv, bc.csv)

        Returns:
            ExperimentData instance with loaded real data

        Example:
            >>> from pathlib import Path
            >>> exp_data = ExperimentData.from_real_data(
            ...     Path("./real_data/preprocessed/specimen_001")
            ... )
        """
        preprocessed_dir = Path(preprocessed_dir)

        if not preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_dir}")

        print(f"Loading preprocessed real data from: {preprocessed_dir}")

        # Create a dummy instance without loading
        instance = cls.__new__(cls)
        instance.experiment_number = -1  # Special marker for real data
        instance.base_path = preprocessed_dir.parent

        # Load coordinate data
        coord_file = preprocessed_dir / 'coord.csv'
        if not coord_file.exists():
            raise FileNotFoundError(f"coord.csv not found in {preprocessed_dir}")
        coord_data = np.loadtxt(coord_file, delimiter=',', skiprows=0)
        instance.coord = coord_data  # Already in [id, x, y] format

        # Load connectivity
        conne_file = preprocessed_dir / 'conne.txt'
        if not conne_file.exists():
            raise FileNotFoundError(f"conne.txt not found in {preprocessed_dir}")
        conne_data = np.loadtxt(conne_file, delimiter=' ', skiprows=0, dtype=int)
        instance.conne = conne_data[:, 1:] - 1  # Convert to 0-indexed, skip elem_id

        # Load displacement matrix
        u_file = preprocessed_dir / 'U.csv'
        if not u_file.exists():
            raise FileNotFoundError(f"U.csv not found in {preprocessed_dir}")
        instance.U = np.loadtxt(u_file, delimiter=',', skiprows=0).T  # Transpose to [2N x T]

        # Load force matrix
        f_file = preprocessed_dir / 'F.csv'
        if not f_file.exists():
            raise FileNotFoundError(f"F.csv not found in {preprocessed_dir}")
        instance.F = np.loadtxt(f_file, delimiter=',', skiprows=0).T  # Transpose to [4 x T]
        # NOTE: Real data forces are already in Newtons (N), no conversion needed

        # Load time vector
        time_file = preprocessed_dir / 'time.csv'
        if not time_file.exists():
            raise FileNotFoundError(f"time.csv not found in {preprocessed_dir}")
        instance.time = np.loadtxt(time_file, delimiter=',', skiprows=0)

        # Load boundary conditions (optional for validation)
        bc_file = preprocessed_dir / 'bc.csv'
        if bc_file.exists():
            instance.bc = np.loadtxt(bc_file, delimiter=',', skiprows=0, dtype=int)
        else:
            instance.bc = None

        # Validate loaded data
        instance._validate()

        print(f"[OK] Loaded real data: {instance.n_nodes} nodes, "
              f"{instance.n_elements} elements, {instance.n_timesteps} timesteps")

        return instance

    # ========== Properties (read-only access) ==========

    @property
    def n_nodes(self) -> int:
        """Number of nodes in mesh."""
        return self.coord.shape[0]
    
    @property
    def n_elements(self) -> int:
        """Number of elements in mesh."""
        return self.conne.shape[0]
    
    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return len(self.time)
    
    @property
    def dt(self) -> np.ndarray:
        """
        Time step sizes.
        Returns array where dt[i] = time[i] - time[i-1], dt[0] = 0.
        """
        dt = np.zeros_like(self.time)
        dt[1:] = self.time[1:] - self.time[:-1]
        return dt
    
    def __repr__(self):
        return (f"ExperimentData({self.experiment_number}): "
                f"{self.n_nodes} nodes, {self.n_elements} elements, "
                f"{self.n_timesteps} timesteps")


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test data loading with experiment 998.
    Run from inverse_problem folder: python data.py
    """
    
    try:
        # Load experiment data
        exp_data = ExperimentData(999)
        
        print("\n" + "="*60)
        print(exp_data)
        print("="*60)
        print(f"Time range: {exp_data.time[0]:.2f} to {exp_data.time[-1]:.2f} seconds")
        print(f"Force range: {exp_data.F.min():.2f} to {exp_data.F.max():.2f} N")
        print(f"Displacement range: {exp_data.U.min():.6f} to {exp_data.U.max():.6f}")
        print("\nFirst 3 node coordinates:")
        print(exp_data.coord[:3, :3])
        print("\nFirst 3 connectivity entries:")
        print(exp_data.conne[:3, :])
        
        # ========== DATA VALIDATION & VISUALIZATION ==========
        # Uncomment this section when you need to diagnose data quality
        # Comment out when running the main pipeline for speed
        
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("DATA VALIDATION & DIAGNOSTICS")
        print("="*60)
        
        # 1. Force data structure analysis
        print(f"\n1. FORCE DATA ANALYSIS:")
        print(f"   Force shape: {exp_data.F.shape}")
        print(f"   Force components: {exp_data.F.shape[0]}")
        print(f"   Force timesteps: {exp_data.F.shape[1]}")
        
        if exp_data.F.shape[0] <= 10:  # Print all if few components
            for i in range(exp_data.F.shape[0]):
                f_mean = exp_data.F[i, :].mean()
                f_std = exp_data.F[i, :].std()
                f_min, f_max = exp_data.F[i, :].min(), exp_data.F[i, :].max()
                print(f"   F[{i}]: mean={f_mean:.2f}N, std={f_std:.2f}N, range=[{f_min:.2f}, {f_max:.2f}]N")
        
        # 2. Time evolution analysis
        print(f"\n2. TIME EVOLUTION ANALYSIS:")
        print(f"   Duration: {exp_data.time[-1] - exp_data.time[0]:.1f} seconds")
        print(f"   Average dt: {np.mean(np.diff(exp_data.time)):.3f} seconds")
        print(f"   dt range: [{np.diff(exp_data.time).min():.3f}, {np.diff(exp_data.time).max():.3f}] seconds")
        print(f"   Time uniformity: {'âœ“ Uniform' if np.allclose(np.diff(exp_data.time), np.diff(exp_data.time)[0]) else 'âŒ Non-uniform'}")
        
        # 3. Displacement field analysis
        print(f"\n3. DISPLACEMENT FIELD ANALYSIS:")
        U_x = exp_data.U[0::2, :]  # X displacements
        U_y = exp_data.U[1::2, :]  # Y displacements
        
        # Initial vs final displacements
        initial_disp_x = U_x[:, 0]
        final_disp_x = U_x[:, -1]
        total_disp_x = final_disp_x - initial_disp_x
        
        initial_disp_y = U_y[:, 0]
        final_disp_y = U_y[:, -1]
        total_disp_y = final_disp_y - initial_disp_y
        
        print(f"   X-displacement - Total range: [{total_disp_x.min():.4f}, {total_disp_x.max():.4f}] mm")
        print(f"   Y-displacement - Total range: [{total_disp_y.min():.4f}, {total_disp_y.max():.4f}] mm")
        print(f"   Predominant direction: {'Y' if abs(total_disp_y).mean() > abs(total_disp_x).mean() else 'X'}")
        
        # 4. Noise analysis
        print(f"\n4. MEASUREMENT NOISE ANALYSIS:")
        if exp_data.n_timesteps > 10:
            # Temporal derivatives as noise proxy
            dU_dt_x = np.diff(U_x, axis=1)
            dU_dt_y = np.diff(U_y, axis=1)
            
            noise_x = np.std(dU_dt_x)
            noise_y = np.std(dU_dt_y)
            signal_x = np.std(total_disp_x)
            signal_y = np.std(total_disp_y)
            
            snr_x = signal_x / noise_x if noise_x > 0 else np.inf
            snr_y = signal_y / noise_y if noise_y > 0 else np.inf
            
            print(f"   Noise estimate (temporal derivative std):")
            print(f"     X-direction: {noise_x:.6f} mm/step, SNR: {snr_x:.1f}")
            print(f"     Y-direction: {noise_y:.6f} mm/step, SNR: {snr_y:.1f}")
            print(f"   Data quality: {'âœ“ Good' if min(snr_x, snr_y) > 5 else 'âš  Noisy' if min(snr_x, snr_y) > 1 else 'âŒ Very noisy'}")
        
        # 5. Physical consistency check
        print(f"\n5. PHYSICAL CONSISTENCY CHECK:")
        if exp_data.F.shape[0] > 0:
            avg_force = np.mean(exp_data.F[0, -100:])  # Average force in last 100 steps
            avg_y_displacement = np.mean(total_disp_y)
            
            print(f"   Applied force (average): {avg_force:.2f} N")
            print(f"   Average Y-displacement: {avg_y_displacement:.4f} mm")
            
            if abs(avg_force) > 1 and abs(avg_y_displacement) > 1e-4:
                apparent_stiffness = abs(avg_force) / abs(avg_y_displacement)  # N/mm
                print(f"   Apparent stiffness: {apparent_stiffness:.1f} N/mm")
                print(f"   Stiffness assessment: {'âœ“ Reasonable' if 1e2 < apparent_stiffness < 1e6 else 'âš  Check units/scaling'}")
            else:
                print("   âš  Cannot compute stiffness - force or displacement too small")
        
        # 6. Create diagnostic plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Data Validation - Experiment {exp_data.experiment_number}', fontsize=16)
        
        # Plot 1: Force vs time
        axes[0,0].plot(exp_data.time, exp_data.F.T)
        axes[0,0].set_xlabel('Time [s]')
        axes[0,0].set_ylabel('Force [N]')
        axes[0,0].set_title('Force Evolution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Sample displacement histories
        sample_nodes = np.linspace(0, exp_data.n_nodes-1, 5).astype(int)
        for i, node in enumerate(sample_nodes):
            axes[0,1].plot(exp_data.time, U_y[node, :], label=f'Node {node}', alpha=0.7)
        axes[0,1].set_xlabel('Time [s]')
        axes[0,1].set_ylabel('Y-displacement [mm]')
        axes[0,1].set_title('Sample Y-Displacement Histories')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Final displacement field
        x_coords = exp_data.coord[:, 1]
        y_coords = exp_data.coord[:, 2]
        scatter = axes[0,2].scatter(x_coords, y_coords, c=total_disp_y, cmap='RdBu_r', s=20)
        axes[0,2].set_xlabel('X [mm]')
        axes[0,2].set_ylabel('Y [mm]')
        axes[0,2].set_title('Final Y-Displacement Field')
        axes[0,2].axis('equal')
        plt.colorbar(scatter, ax=axes[0,2], label='Y-displacement [mm]')
        
        # Plot 4: Time step distribution
        dt_values = np.diff(exp_data.time)
        axes[1,0].hist(dt_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Time step [s]')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Time Step Distribution')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Force-displacement correlation
        if exp_data.F.shape[0] > 0:
            avg_disp_y = np.mean(U_y, axis=0)  # Average Y-displacement over all nodes
            axes[1,1].scatter(exp_data.F[0, :], avg_disp_y, alpha=0.6, s=1)
            axes[1,1].set_xlabel('Force [N]')
            axes[1,1].set_ylabel('Average Y-displacement [mm]')
            axes[1,1].set_title('Force vs Displacement')
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Displacement noise analysis
        if exp_data.n_timesteps > 10:
            displacement_noise_x = np.std(dU_dt_x, axis=1)
            displacement_noise_y = np.std(dU_dt_y, axis=1)
            axes[1,2].scatter(x_coords, y_coords, c=displacement_noise_y, cmap='plasma', s=20)
            axes[1,2].set_xlabel('X [mm]')
            axes[1,2].set_ylabel('Y [mm]')
            axes[1,2].set_title('Y-Displacement Noise Distribution')
            axes[1,2].axis('equal')
            plt.colorbar(axes[1,2].collections[0], ax=axes[1,2], label='Noise [mm/step]')
        
        plt.tight_layout()
        plt.show()
        
        # ========== END VISUALIZATION SECTION ==========
        
        print("data.py working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()