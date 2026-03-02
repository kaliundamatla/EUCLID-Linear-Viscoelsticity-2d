"""
Core inverse problem modules.

This package contains all core functionality for the inverse viscoelastic solver:
- Data loading and validation
- Mesh generation and handling
- Material model (Prony series)
- History variable computation (beta coefficients)
- System assembly routines
- Boundary condition handling
- Solver engines (NNLS, LASSO)
- Parameter clustering
- Visualization
"""

from .data import ExperimentData
from .geometry import Mesh
from .material import MaterialModel
from .history import BetaComputer, HistoryData
from .assembly import SystemAssembler
from .boundary import BoundaryAssembler, BoundaryCondition, TopBottomForce, BottomForceBC
from .solver import Solver, NNLSSolver, ParameterSet
from .visualization import InverseProblemVisualizer
from .clustering import ParameterClusterer

__all__ = [
    # Data and mesh
    'ExperimentData',
    'Mesh',

    # Material model
    'MaterialModel',

    # History variables
    'BetaComputer',
    'HistoryData',

    # Assembly
    'SystemAssembler',

    # Boundary conditions
    'BoundaryAssembler',
    'BoundaryCondition',
    'TopBottomForce',
    'BottomForceBC',

    # Solvers
    'Solver',
    'NNLSSolver',
    'ParameterSet',

    # Visualization
    'InverseProblemVisualizer',

    # Clustering
    'ParameterClusterer',
]
