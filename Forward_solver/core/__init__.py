"""
Core modules for Forward Solver.

This package contains all core functionality for the forward viscoelastic solver:
- Material models (viscoelastic Prony series)
- Mesh generation and handling
- Time integration algorithms
- FEM assembly routines
- Forward solver engine
- Synthetic data generation
- Geometry builders (simple and advanced)
- Mesh converters and importers
"""

from .material import (
    ViscoelasticMaterial,
    create_matlab_reference_material,
    create_simple_test_material
)
from .mesh import MeshGenerator, Node, MeshLoader
from .time_integration import ForwardTimeIntegrator, HistoryVariables
from .assembly import ForwardAssembler
from .solver import ForwardSolver, BoundaryConditions, LoadingProtocol
from .data_generation import SyntheticDataGenerator
from .geometry_builder import GeometryBuilder, load_config
from .geometry_advanced import AdvancedMeshGenerator
from .mesh_converter import MeshConverter

__all__ = [
    # Material models
    'ViscoelasticMaterial',
    'create_matlab_reference_material',
    'create_simple_test_material',

    # Mesh generation
    'MeshGenerator',
    'Node',
    'MeshLoader',

    # Time integration
    'ForwardTimeIntegrator',
    'HistoryVariables',

    # Assembly
    'ForwardAssembler',

    # Solver
    'ForwardSolver',
    'BoundaryConditions',
    'LoadingProtocol',

    # Data generation
    'SyntheticDataGenerator',

    # Geometry builders
    'GeometryBuilder',
    'load_config',
    'AdvancedMeshGenerator',

    # Converters
    'MeshConverter',
]
