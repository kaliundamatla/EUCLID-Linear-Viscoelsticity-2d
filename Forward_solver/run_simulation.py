"""
Simple launcher for run_full_simulation.py

Usage:
    cd Forward_solver
    python run_simulation.py

This is a convenience wrapper that imports and runs the main simulation script.
You can edit parameters directly in this file or in scripts/run_full_simulation.py
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main function
from Forward_solver.scripts.run_full_simulation import run_full_simulation

if __name__ == "__main__":
    """
    Configure and run simulation here.

    Example usage:

    # Simple geometry (experiments 800-802)
    run_full_simulation(
        experiment_id=800,
        width=20.0,
        height=50.0,
        nx=11,
        ny=26,
        dt=0.5,
        n_timesteps=1200,
        force_magnitude=1000.0
    )

    # Complex geometry (experiments 804-806) with mesh control
    run_full_simulation(
        experiment_id=804,
        use_complex_geometry=True,
        dt=1.0,
        n_timesteps=600,
        mesh_size_outer=2.0,    # Coarser mesh at outer boundary
        mesh_size_hole=0.5      # Finer mesh around holes
    )
    """

    # Default: Run experiment 800
    print("\n" + "="*70)
    print("RUNNING FORWARD SIMULATION")
    print("="*70)
    print("\nEdit this file (run_simulation.py) to change parameters")
    print("Or call run_full_simulation() with your desired parameters\n")

    # ============================================================================
    # EDIT THESE PARAMETERS AS NEEDED
    # ============================================================================

    experiment_id = 811        # Experiment number (800-802: simple, 804-806: complex)

    # ---- Geometry Parameters ----
    width = 24               # Domain width [mm]
    height = 65.3              # Domain height [mm]
    nx = 12                   # Nodes in x direction (for simple geometry)
    ny = 30                  # Nodes in y direction (for simple geometry)

    # ---- Time Parameters ----
    dt = 1.0                   # Time step [s]
    n_timesteps = 600         # Total number of timesteps

    # ---- Loading ----
    load = 50.0                # Distributed load [N/mm] (positive = tension)

    # ---- Complex Geometry Options (for experiments 804-806) ----
    use_complex_geometry = False   # False = simple structured mesh (for mesh convergence study)
    mesh_size_outer = None         # Element size at outer boundary [mm] (e.g., 2.0)
    mesh_size_hole = None          # Element size at hole boundary [mm] (e.g., 0.5)

    # ============================================================================

    # Run simulation
    run_full_simulation(
        experiment_id=experiment_id,
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        dt=dt,
        n_timesteps=n_timesteps,
        load=load,
        use_complex_geometry=use_complex_geometry,
        mesh_size_outer=mesh_size_outer,
        mesh_size_hole=mesh_size_hole
    )
