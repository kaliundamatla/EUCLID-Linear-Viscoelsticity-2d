"""
Geometry configuration for Experiment 820: Solid Rectangle (Control)

SYNTHETIC GEOMETRY STUDY - Control specimen
- Solid rectangular specimen (no holes)
- Same material, time parameters, and loading as 821/822
- Uses structured mesh for comparison baseline
"""

EXPERIMENT_CONFIG = {
    'id': 820,
    'name': 'Solid_Rectangle_Control',
    'description': 'Solid rectangle - geometry study control',

    'domain': {
        'width': 23.0,   # mm (same as 821 ellipsoid)
        'height': 59.0,  # mm (same as 821 ellipsoid)
    },

    # Note: This experiment uses structured mesh (simple geometry mode)
    # The mesh_params are included for consistency but not used
    'mesh_params': {
        'mesh_size_outer': 2.0,
        'mesh_size_hole': 2.0,
        'algorithm': 'delaunay'
    },

    'holes': [],  # No holes - solid specimen

    'simulation_params': {
        'dt': 1.0,           # Time step [s]
        'n_timesteps': 600,  # Number of timesteps
        'load': 50.0,        # Load magnitude [N/mm]
    },

    # Structured mesh parameters (for run_simulation.py)
    'structured_mesh': {
        'nx': 12,   # ~560 elements (matching mesh convergence study 812)
        'ny': 30,
    }
}
