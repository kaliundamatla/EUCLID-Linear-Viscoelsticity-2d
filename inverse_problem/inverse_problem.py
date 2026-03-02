"""
Main inverse problem orchestrator.
This is the user-facing interface that coordinates all components.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import time

from core.data import ExperimentData
from core.geometry import Mesh
from core.material import MaterialModel
from core.history import BetaComputer, HistoryData
from core.assembly import SystemAssembler
from core.boundary import BoundaryAssembler, BoundaryCondition, TopBottomForce
from core.solver import Solver, NNLSSolver, ParameterSet
from core.visualization import InverseProblemVisualizer
from core.clustering import ParameterClusterer


class InverseProblem:
    """
    Main class for viscoelastic parameter identification.

    Orchestrates the complete inverse problem workflow:
    1. Load experimental data
    2. Create finite element mesh
    3. Define material structure
    4. Compute history variables (beta coefficients)
    5. Assemble system matrices
    6. Solve for Prony parameters

    Usage:
        problem = InverseProblem(experiment_number=800)
        parameters = problem.run()
    """

    def __init__(self,
                 experiment_number: int,
                 data_path: Optional[Path] = None,
                 n_maxwell_shear: int = 150,
                 n_maxwell_bulk: int = 150,
                 tau_min: float = 1.0,
                 tau_max: float = 600.0,
                 element_type: str = "Triangle3Node",
                 boundary_condition: Optional[BoundaryCondition] = None,
                 solver: Optional[Solver] = None,
                 lambda_interior: float = 0.0,
                 lambda_boundary: float = 1.0):
        """
        Initialize inverse problem.

        Args:
            experiment_number: Experiment ID (e.g., 800, 900)
            data_path: Path to data folder (default: auto-detect)
            n_maxwell_shear: Number of deviatoric Maxwell elements (nG)
            n_maxwell_bulk: Number of volumetric Maxwell elements (nK)
            tau_min: Minimum relaxation time [seconds]
            tau_max: Maximum relaxation time [seconds]
            element_type: FE element type ("Triangle3Node" or "Triangle6Node")
            boundary_condition: BC strategy (default: TopBottomForce)
            solver: Optimization solver (default: NNLSSolver)
            lambda_interior: Weight for interior equations
            lambda_boundary: Weight for boundary equations
        """
        self.experiment_number = experiment_number

        # Configuration
        self.config = {
            'n_maxwell_shear': n_maxwell_shear,
            'n_maxwell_bulk': n_maxwell_bulk,
            'tau_min': tau_min,
            'tau_max': tau_max,
            'element_type': element_type,
            'lambda_interior': lambda_interior,
            'lambda_boundary': lambda_boundary
        }

        # Components (initialized in setup)
        self.exp_data: Optional[ExperimentData] = None
        self.mesh: Optional[Mesh] = None
        self.material: Optional[MaterialModel] = None
        self.history: Optional[HistoryData] = None
        self.system_assembler: Optional[SystemAssembler] = None
        self.boundary_assembler: Optional[BoundaryAssembler] = None
        self.parameters: Optional[ParameterSet] = None

        # System matrices
        self.A_exp: Optional[np.ndarray] = None
        self.R_exp: Optional[np.ndarray] = None

        # Strategy objects
        self.boundary_condition = boundary_condition or TopBottomForce()
        self.solver = solver or NNLSSolver()

        # Timing
        self.timings = {}

        print("="*70)
        print("EUCLID INVERSE PROBLEM - PARAMETER IDENTIFICATION")
        print("="*70)
        print(f"Experiment: {experiment_number}")
        print(f"Material model: nG={n_maxwell_shear}, nK={n_maxwell_bulk}")
        print(f"Relaxation times: tau in [{tau_min}, {tau_max}]s")
        print(f"Element type: {element_type}")
        print(f"Weights: lambda_interior={lambda_interior}, lambda_boundary={lambda_boundary}")
        print("="*70)

        # Initialize components
        self._setup(data_path)

    def _setup(self, data_path: Optional[Path]):
        """Initialize all components."""
        print("\n[1/3] SETUP")
        print("-" * 70)

        t_start = time.time()

        # Load data
        self.exp_data = ExperimentData(self.experiment_number, data_path)

        # Create mesh
        self.mesh = Mesh(
            self.exp_data.coord,
            self.exp_data.conne,
            element_type=self.config['element_type']
        )

        print("  Using raw data (no preprocessing)")

        # Create material model
        self.material = MaterialModel(
            n_maxwell_shear=self.config['n_maxwell_shear'],
            n_maxwell_bulk=self.config['n_maxwell_bulk'],
            tau_min=self.config['tau_min'],
            tau_max=self.config['tau_max']
        )

        self.timings['setup'] = time.time() - t_start
        print(f"Setup complete ({self.timings['setup']:.1f}s)")

    def compute_history(self):
        """Compute beta coefficients (history variables)."""
        print("\n[2/3] HISTORY COMPUTATION")
        print("-" * 70)

        t_start = time.time()

        beta_computer = BetaComputer(self.mesh, self.material)
        self.history = beta_computer.compute(self.exp_data)

        self.timings['history'] = time.time() - t_start
        print(f"History computation complete ({self.timings['history']:.1f}s)")

        return self.history

    def assemble_system(self):
        """Assemble system matrices A_exp and R_exp."""
        print("\n[3/3] SYSTEM ASSEMBLY")
        print("-" * 70)

        t_start = time.time()

        # Element matrices
        self.system_assembler = SystemAssembler(
            self.mesh,
            self.material,
            self.exp_data,
            self.history
        )
        self.system_assembler.assemble()

        # Boundary assembly
        self.boundary_assembler = BoundaryAssembler(
            self.mesh,
            self.system_assembler,
            self.exp_data,
            self.boundary_condition,
            lambda_interior=self.config['lambda_interior'],
            lambda_boundary=self.config['lambda_boundary']
        )
        self.A_exp, self.R_exp = self.boundary_assembler.assemble()

        self.timings['assembly'] = time.time() - t_start
        print(f"System assembly complete ({self.timings['assembly']:.1f}s)")

        return self.A_exp, self.R_exp

    def solve(self):
        """Solve for Prony parameters."""
        print("\n[4/4] OPTIMIZATION")
        print("-" * 70)

        if self.A_exp is None or self.R_exp is None:
            raise RuntimeError("System not assembled. Call assemble_system() first.")

        t_start = time.time()

        theta = self.solver.solve(self.A_exp, self.R_exp)
        self.parameters = ParameterSet(theta, self.material)

        self.timings['solve'] = time.time() - t_start
        print(f"Optimization complete ({self.timings['solve']:.1f}s)")

        return self.parameters

    def run(self, create_plots: bool = True, apply_clustering: bool = True, clustering_range: float = 0.3, output_path: Path = None) -> ParameterSet:
        """
        Run complete inverse problem pipeline.

        Args:
            create_plots: Generate visualization plots
            apply_clustering: Apply parameter clustering after NNLS
            clustering_range: Relative distance threshold for clustering (default 0.3)
            output_path: Path to save figures (if None, uses experiment_number only)

        Returns:
            ParameterSet with identified Prony parameters (clustered if requested)
        """
        print("\n" + "="*70)
        print("RUNNING INVERSE PROBLEM PIPELINE")
        print("="*70)

        # Run pipeline
        self.compute_history()
        self.assemble_system()
        parameters_raw = self.solve()

        # Store raw parameters
        self.parameters_raw = parameters_raw

        # Apply clustering if requested
        if apply_clustering:
            print("\n" + "="*70)
            print("PARAMETER CLUSTERING")
            print("="*70)
            clusterer = ParameterClusterer(clustering_range=clustering_range)
            parameters = clusterer.cluster(parameters_raw)
            self.parameters_clustered = parameters
        else:
            parameters = parameters_raw
            self.parameters_clustered = None

        # Visualize
        if create_plots:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
            self.visualize(parameters, output_path=output_path)

            # Add clustering comparison if clustering was applied
            if apply_clustering and self.parameters_clustered is not None:
                print("\n[9/9] Clustering Comparison Plot...")
                self.visualizer.plot_clustering_comparison(
                    self.parameters_raw,
                    self.parameters_clustered
                )

        # Store final parameters for saving
        self.parameters = parameters

        # Print summary
        self._print_summary(parameters)

        return parameters

    def _print_summary(self, parameters: ParameterSet):
        """Print a text summary of the identified material model."""
        print("\n" + "="*70)
        print("IDENTIFICATION SUMMARY")
        print("="*70)

        tau_G_nz, G_nz = parameters.get_nonzero_G()
        tau_K_nz, K_nz = parameters.get_nonzero_K()

        print(f"Experiment: {self.experiment_number}")
        print(f"nG (shear Maxwell terms): {self.material.nG}")
        print(f"nK (bulk  Maxwell terms): {self.material.nK}")
        print(f"G_inf: {parameters.G_inf:.3e} MPa")
        print(f"K_inf: {parameters.K_inf:.3e} MPa")
        print(f"Total G(0): {parameters.total_G:.3e} MPa")
        print(f"Total K(0): {parameters.total_K:.3e} MPa")
        print(f"Nonzero G terms: {len(G_nz)}")
        print(f"Nonzero K terms: {len(K_nz)}")

        metrics = self.solver.get_metrics() if hasattr(self.solver, "get_metrics") else {}
        print("\nSolver Metrics:")
        print(f"  residual_norm: {metrics.get('residual_norm', 'N/A')}")
        print(f"  mse:           {metrics.get('mse', 'N/A')}")
        print(f"  cost:          {metrics.get('cost', 'N/A')}")

        print("\nTimings [s]:")
        for k, v in self.timings.items():
            print(f"  {k:10s}: {v:.2f}")
        print(f"  total     : {sum(self.timings.values()):.2f}")

        print("="*70)

    def visualize(self, parameters: ParameterSet, output_path: Path = None):
        """
        Generate all standard visualization plots (Blocks 1-8).

        Args:
            parameters: ParameterSet to visualize
            output_path: Path to save figures (if None, uses experiment_number only)
        """
        self.parameters = parameters

        if output_path is None:
            output_dir = Path("./Postprocessing/final_outputs") / str(self.experiment_number)
        else:
            output_dir = Path(output_path)

        self.visualizer = InverseProblemVisualizer(self, output_dir)
        self.visualizer.plot_all()

    def save_results(self, output_path: Path):
        """
        Save identified parameters to file.

        Args:
            output_path: Directory path to save results (saves as results.npz)
        """
        if self.parameters is None:
            raise RuntimeError("No results to save. Run solve() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tau_G_nz, G_nz = self.parameters.get_nonzero_G()
        tau_K_nz, K_nz = self.parameters.get_nonzero_K()

        results = {
            'experiment_number': self.experiment_number,
            'theta': self.parameters.theta,
            'G_inf': self.parameters.G_inf,
            'G_params': self.parameters.G_params,
            'K_inf': self.parameters.K_inf,
            'K_params': self.parameters.K_params,
            'tau_G': self.parameters.tau_G,
            'tau_K': self.parameters.tau_K,
            'tau_G_nonzero': tau_G_nz,
            'G_nonzero': G_nz,
            'tau_K_nonzero': tau_K_nz,
            'K_nonzero': K_nz,
            'mse': self.solver.get_metrics().get('mse', 0),
            'residual_norm': self.solver.get_metrics().get('residual_norm', 0),
            'timings': self.timings
        }

        if hasattr(self, 'parameters_raw') and self.parameters_raw is not None:
            tau_G_raw_nz, G_raw_nz = self.parameters_raw.get_nonzero_G()
            tau_K_raw_nz, K_raw_nz = self.parameters_raw.get_nonzero_K()
            results['G_params_raw'] = self.parameters_raw.G_params
            results['K_params_raw'] = self.parameters_raw.K_params
            results['tau_G_raw_nonzero'] = tau_G_raw_nz
            results['G_raw_nonzero'] = G_raw_nz
            results['tau_K_raw_nonzero'] = tau_K_raw_nz
            results['K_raw_nonzero'] = K_raw_nz

        filename = output_path / "results.npz"
        np.savez(filename, **results)

        print(f"\nResults saved to: {filename}")
