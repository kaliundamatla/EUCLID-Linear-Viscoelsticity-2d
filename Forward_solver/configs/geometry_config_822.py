"""
Geometry configuration for Experiment 822: Three Circular Holes

SYNTHETIC GEOMETRY STUDY - Three holes specimen
- Three circular holes arranged vertically (small-large-small pattern)
- Same material, time parameters, and loading as 820/821
- Tests effect of multiple holes on parameter identification
"""

EXPERIMENT_CONFIG = {
    'id': 822,
    'name': 'Three_Holes_Study',
    'description': 'Three circular holes - geometry study',

    'domain': {
        'width': 22.83,   # mm
        'height': 59.62,  # mm
    },

    'mesh_params': {
        'mesh_size_outer': 2.0,      # Outer boundary element size [mm]
        'mesh_size_hole': 2.0,       # Hole boundary element size [mm]
        'algorithm': 'delaunay'      # Meshing algorithm
    },

    'holes': [
        {
            'type': 'circle',
            'center': (11.5, 41.7),  # Top hole (3/4 height)
            'radius': 2.5,           # mm (small)
            'name': 'top_small'
        },
        {
            'type': 'circle',
            'center': (11.5, 30.0),  # Middle hole (1/2 height)
            'radius': 5.0,           # mm (large)
            'name': 'middle_large'
        },
        {
            'type': 'circle',
            'center': (11.5, 18),    # Bottom hole (1/4 height)
            'radius': 2.5,           # mm (small)
            'name': 'bottom_small'
        }
    ],

    'simulation_params': {
        'dt': 1.0,           # Time step [s]
        'n_timesteps': 1200,  # Number of timesteps
        'load': 50.0,        # Load magnitude [N/mm]
    }
}
