"""
Unified DIC Preprocessing Pipeline for EUCLID Experiments.

This module provides a unified preprocessing pipeline for all experiments (1-4),
handling both solid specimens and specimens with holes.

Pipeline Steps:
    Step 1: Load Data (from dic_data_analysis.py config)
    Step 2: Identify Permanent Nodes
    Step 3: Load/Generate Synthetic Mesh
    Step 4: Coordinate Alignment (center, scale, rotate)
    Step 5: Map DIC to Mesh (nearest neighbor)
    Step 6: Verify Mapping Quality
    Step 7: Remove Rigid Body Motion
    Step 8: Build Force Matrix
    Step 9: Export Data
    Step 10: Generate Diagnostic Plots

Output Format:
    - coord.csv: [id, x, y, verticalGDL, horizontalGDL]
    - conne.txt: [elem_id, n1, n2, n3]
    - U.csv: [2*nNodes × nTime] grouped format (all Ux, then all Uy)
    - F.csv: [nBoundaries × nTime] boundary forces in kN
    - time.csv: [nTime] time vector
    - bc.csv: [nNodes] boundary condition labels
    - metadata.txt: preprocessing summary

Boundary Condition Labels:
    0 = interior (free)
    1 = top (loaded, Neumann BC)
    2 = bottom (fixed, Dirichlet BC)
    3 = right
    4 = left
    5 = hole (free boundary, for specimens with holes)

Usage:
    python unified_preprocessor.py [experiment_number]

    Example: python unified_preprocessor.py 3

Author: EUCLID Project
Date: 2024
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Import experiment configurations from dic_data_analysis
from dic_data_analysis import EXPERIMENT_CONFIGS, ExperimentConfig, get_base_path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Extended configuration for preprocessing."""
    experiment_id: int
    base_config: ExperimentConfig

    # Paths (derived)
    experiment_path: Path = None
    dic_path: Path = None
    force_path: Path = None
    output_path: Path = None
    synthetic_mesh_path: Path = None

    # Geometry type
    geometry_type: str = "solid"  # "solid" or "with_holes"
    hole_type: str = None  # "ellipse", "three_circles", etc.

    # Synthetic mesh ID mapping
    synthetic_mesh_id: int = None

    # Processing parameters
    boundary_tolerance: float = 0.5  # mm, for boundary detection
    mapping_error_threshold: float = 0.5  # mm, max acceptable mapping error

    def __post_init__(self):
        """Derive paths from base configuration."""
        base_path = get_base_path()
        self.experiment_path = base_path / str(self.experiment_id)
        self.dic_path = self.experiment_path / self.base_config.dic_folder
        self.force_path = self.experiment_path / self.base_config.force_file

        # Output path (synthetic_data/90X)
        project_root = Path(__file__).parent.parent.parent
        self.output_path = project_root / "synthetic_data" / f"90{self.experiment_id}"

        # Determine geometry type and synthetic mesh based on experiment
        self._set_geometry_config()

    def _set_geometry_config(self):
        """Set geometry type and synthetic mesh ID based on experiment."""
        if self.experiment_id == 1:
            self.geometry_type = "solid"
            self.synthetic_mesh_id = 810  # Solid rectangular (matched to Exp 1)
        elif self.experiment_id == 2:
            self.geometry_type = "solid"
            self.synthetic_mesh_id = 811  # Same solid rectangular (TODO: create specific mesh if needed)
        elif self.experiment_id == 3:
            self.geometry_type = "with_holes"
            self.hole_type = "ellipse"
            self.synthetic_mesh_id = 821  # Ellipse hole (matched to Exp 3)
        elif self.experiment_id == 4:
            self.geometry_type = "with_holes"
            self.hole_type = "three_circles"
            self.synthetic_mesh_id = 822  # Three holes (TODO: verify mesh exists)

        # Set synthetic mesh path
        project_root = Path(__file__).parent.parent.parent
        self.synthetic_mesh_path = project_root / "synthetic_data" / str(self.synthetic_mesh_id)


@dataclass
class PreprocessingData:
    """Container for all preprocessing data."""
    # Raw data
    dic_times: np.ndarray = None
    force_data: np.ndarray = None
    dic_dataframes: List[pd.DataFrame] = field(default_factory=list)

    # Permanent nodes
    permanent_nodes: np.ndarray = None
    permanent_coords: np.ndarray = None  # [N x 2] reference coordinates

    # Mesh data
    mesh_coords: np.ndarray = None  # [M x 2]
    mesh_connectivity: np.ndarray = None  # [E x 3]
    vertical_dofs: np.ndarray = None
    horizontal_dofs: np.ndarray = None

    # Mapping
    mesh_to_dic_map: np.ndarray = None  # [M] mesh node -> DIC node index
    mapping_distances: np.ndarray = None  # [M] distance to nearest DIC node

    # Aligned coordinates
    dic_coords_aligned: np.ndarray = None  # [N x 2] aligned to mesh

    # Processed data
    U_raw: np.ndarray = None  # Raw displacements [2*M x T]
    U_corrected: np.ndarray = None  # After rigid body removal [2*M x T]
    F_matrix: np.ndarray = None  # Force matrix [nBoundaries x T]
    bc_labels: np.ndarray = None  # Boundary condition labels [M]

    # Quality metrics
    alignment_params: Dict = field(default_factory=dict)
    preprocessing_stats: Dict = field(default_factory=dict)


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def step1_load_data(config: PreprocessingConfig) -> PreprocessingData:
    """
    Step 1: Load DIC displacement data and machine force data.

    This step:
    1. Discovers available DIC files in the experiment folder
    2. Extracts timestamps from filenames
    3. Loads machine force data
    4. Interpolates force to DIC timestamps

    Args:
        config: Preprocessing configuration

    Returns:
        PreprocessingData with dic_times, force_data, dic_dataframes populated
    """
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)

    data = PreprocessingData()
    base_config = config.base_config

    # Verify paths exist
    if not config.dic_path.exists():
        raise FileNotFoundError(f"DIC folder not found: {config.dic_path}")
    if not config.force_path.exists():
        raise FileNotFoundError(f"Force file not found: {config.force_path}")

    print(f"\nExperiment {config.experiment_id}: {base_config.description}")
    print(f"DIC folder: {config.dic_path}")
    print(f"Force file: {config.force_path}")

    # -------------------------------------------------------------------------
    # 1.1 Discover DIC files and extract timestamps
    # -------------------------------------------------------------------------
    print(f"\n[1.1] Discovering DIC files...")

    dic_files = sorted(config.dic_path.glob("Flächenkomponente 1_*.csv"))

    if len(dic_files) == 0:
        raise FileNotFoundError(f"No DIC files found in {config.dic_path}")

    # Extract timestamps from filenames
    dic_times = []
    valid_files = []

    for f in dic_files:
        try:
            # Pattern: "Flächenkomponente 1_123.000 s.csv" -> 123.0
            time_str = f.stem.split('_')[1].split(' ')[0]
            dic_times.append(float(time_str))
            valid_files.append(f)
        except (IndexError, ValueError) as e:
            print(f"  [WARNING] Could not parse time from: {f.name}")
            continue

    data.dic_times = np.array(sorted(dic_times))

    print(f"  Found {len(data.dic_times)} DIC files")
    print(f"  Time range: [{data.dic_times.min():.1f}, {data.dic_times.max():.1f}] s")
    print(f"  Duration: {(data.dic_times.max() - data.dic_times.min())/60:.1f} minutes")

    # -------------------------------------------------------------------------
    # 1.2 Load machine force data
    # -------------------------------------------------------------------------
    print(f"\n[1.2] Loading machine force data...")

    try:
        machine_df = pd.read_csv(
            config.force_path,
            sep=';',
            skiprows=base_config.force_skip_rows,
            encoding='latin1',
            on_bad_lines='skip'
        )
    except Exception as e:
        raise IOError(f"Failed to read force file: {e}")

    # Extract time and force columns
    time_machine = pd.to_numeric(machine_df.iloc[:, base_config.time_column], errors='coerce').values
    force_machine = pd.to_numeric(machine_df.iloc[:, base_config.force_column], errors='coerce').values

    # Remove NaN values
    valid_mask = ~(np.isnan(time_machine) | np.isnan(force_machine))
    time_machine = time_machine[valid_mask]
    force_machine = force_machine[valid_mask]

    # Force is in kN, convert to N
    force_machine_N = force_machine * 1000.0

    print(f"  Machine data: {len(time_machine)} samples")
    print(f"  Time range: [{time_machine.min():.1f}, {time_machine.max():.1f}] s")
    print(f"  Force range: [{force_machine.min():.4f}, {force_machine.max():.4f}] kN")
    print(f"             = [{force_machine_N.min():.1f}, {force_machine_N.max():.1f}] N")
    print(f"  Force column: {base_config.force_column} (10kN sensor)")

    # -------------------------------------------------------------------------
    # 1.3 Interpolate force to DIC timestamps
    # -------------------------------------------------------------------------
    print(f"\n[1.3] Interpolating force to DIC timestamps...")

    data.force_data = np.interp(data.dic_times, time_machine, force_machine_N)

    print(f"  Interpolated force range: [{data.force_data.min():.1f}, {data.force_data.max():.1f}] N")

    # -------------------------------------------------------------------------
    # 1.4 Load DIC displacement files
    # -------------------------------------------------------------------------
    print(f"\n[1.4] Loading DIC displacement files...")

    data.dic_dataframes = []

    for i, t in enumerate(data.dic_times):
        # Build filename
        filename = f"Flächenkomponente 1_{t:.3f} s.csv"
        filepath = config.dic_path / filename

        if not filepath.exists():
            print(f"  [WARNING] Missing file: {filename}")
            continue

        # Read DIC file
        df = pd.read_csv(
            filepath,
            sep=base_config.dic_separator,
            skiprows=base_config.dic_skip_rows,
            encoding='utf-8'
        )
        df.columns = df.columns.str.strip()

        # Standardize column names
        df_standardized = pd.DataFrame({
            'id': df['id'],
            'x': df['x'],
            'y': df['y'],
            'displacement_x': df['displacement_x'],
            'displacement_y': df['displacement_y']
        })

        data.dic_dataframes.append(df_standardized)

        # Progress indicator
        if (i + 1) % 200 == 0 or i == len(data.dic_times) - 1:
            print(f"  Loaded {i + 1}/{len(data.dic_times)} files...")

    # -------------------------------------------------------------------------
    # 1.5 Summary statistics
    # -------------------------------------------------------------------------
    print(f"\n[1.5] Data Loading Summary")
    print("-" * 40)

    first_df = data.dic_dataframes[0]
    n_nodes_first = len(first_df)
    x_range = (first_df['x'].min(), first_df['x'].max())
    y_range = (first_df['y'].min(), first_df['y'].max())

    data.preprocessing_stats['n_dic_files'] = len(data.dic_dataframes)
    data.preprocessing_stats['n_nodes_first'] = n_nodes_first
    data.preprocessing_stats['time_range'] = (data.dic_times.min(), data.dic_times.max())
    data.preprocessing_stats['force_range'] = (data.force_data.min(), data.force_data.max())
    data.preprocessing_stats['x_range'] = x_range
    data.preprocessing_stats['y_range'] = y_range
    data.preprocessing_stats['specimen_width'] = x_range[1] - x_range[0]
    data.preprocessing_stats['specimen_height'] = y_range[1] - y_range[0]

    print(f"  DIC files loaded: {data.preprocessing_stats['n_dic_files']}")
    print(f"  Nodes at t=0: {n_nodes_first}")
    print(f"  Specimen dimensions: {data.preprocessing_stats['specimen_width']:.2f} x {data.preprocessing_stats['specimen_height']:.2f} mm")
    print(f"  X range: [{x_range[0]:.2f}, {x_range[1]:.2f}] mm")
    print(f"  Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}] mm")

    print("\n[OK] Step 1 completed successfully")

    return data


# =============================================================================
# STEP 2: IDENTIFY PERMANENT NODES
# =============================================================================

def step2_identify_permanent_nodes(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 2: Identify nodes present in ALL timesteps.

    DIC tracking can lose nodes over time due to:
    - Surface damage/speckle loss
    - Out-of-plane motion
    - Shadow/reflection changes

    This step finds the intersection of nodes across all timesteps
    to ensure consistent data for the inverse problem.

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 1

    Returns:
        PreprocessingData with permanent_nodes and permanent_coords populated
    """
    print("\n" + "="*70)
    print("STEP 2: IDENTIFY PERMANENT NODES")
    print("="*70)

    n_timesteps = len(data.dic_dataframes)

    # -------------------------------------------------------------------------
    # 2.1 Get initial node set from first timestep
    # -------------------------------------------------------------------------
    print(f"\n[2.1] Analyzing node sets across {n_timesteps} timesteps...")

    initial_nodes = set(data.dic_dataframes[0]['id'].values)
    n_initial = len(initial_nodes)

    print(f"  Initial nodes at t=0: {n_initial}")

    # -------------------------------------------------------------------------
    # 2.2 Find intersection across all timesteps
    # -------------------------------------------------------------------------
    print(f"\n[2.2] Finding nodes present in ALL timesteps...")

    permanent_set = initial_nodes.copy()
    nodes_lost_at = []  # Track when nodes are lost

    # Sample every 10th timestep for efficiency (then verify with remaining)
    sample_indices = list(range(0, n_timesteps, 10))
    if (n_timesteps - 1) not in sample_indices:
        sample_indices.append(n_timesteps - 1)

    for i in sample_indices:
        df = data.dic_dataframes[i]
        current_nodes = set(df['id'].values)
        prev_count = len(permanent_set)
        permanent_set &= current_nodes
        nodes_lost = prev_count - len(permanent_set)

        if nodes_lost > 0:
            nodes_lost_at.append((i, data.dic_times[i], nodes_lost))

        # Progress indicator
        if i % 100 == 0:
            print(f"    Checked t={data.dic_times[i]:.1f}s: {len(permanent_set)} permanent nodes")

    # Final verification with all timesteps
    print(f"\n  Verifying with all {n_timesteps} timesteps...")
    for i, df in enumerate(data.dic_dataframes):
        current_nodes = set(df['id'].values)
        permanent_set &= current_nodes

    data.permanent_nodes = np.array(sorted(permanent_set))
    n_permanent = len(data.permanent_nodes)

    # -------------------------------------------------------------------------
    # 2.3 Extract permanent node coordinates from reference frame (t=0)
    # -------------------------------------------------------------------------
    print(f"\n[2.3] Extracting permanent node coordinates...")

    df_ref = data.dic_dataframes[0]
    df_permanent = df_ref[df_ref['id'].isin(data.permanent_nodes)].copy()
    df_permanent = df_permanent.sort_values('id').reset_index(drop=True)

    data.permanent_coords = df_permanent[['x', 'y']].values

    # -------------------------------------------------------------------------
    # 2.4 Statistics and summary
    # -------------------------------------------------------------------------
    print(f"\n[2.4] Permanent Node Statistics")
    print("-" * 40)

    n_lost = n_initial - n_permanent
    retention_rate = (n_permanent / n_initial) * 100

    # Coordinate statistics
    x_perm = data.permanent_coords[:, 0]
    y_perm = data.permanent_coords[:, 1]

    data.preprocessing_stats['n_permanent_nodes'] = n_permanent
    data.preprocessing_stats['n_lost_nodes'] = n_lost
    data.preprocessing_stats['retention_rate'] = retention_rate
    data.preprocessing_stats['permanent_x_range'] = (x_perm.min(), x_perm.max())
    data.preprocessing_stats['permanent_y_range'] = (y_perm.min(), y_perm.max())

    print(f"  Initial nodes (t=0):    {n_initial}")
    print(f"  Permanent nodes:        {n_permanent}")
    print(f"  Nodes lost:             {n_lost}")
    print(f"  Retention rate:         {retention_rate:.1f}%")

    if nodes_lost_at:
        print(f"\n  Node losses occurred at {len(nodes_lost_at)} checkpoints:")
        for i, t, lost in nodes_lost_at[:5]:
            print(f"    t={t:.1f}s: {lost} nodes lost")
        if len(nodes_lost_at) > 5:
            print(f"    ... and {len(nodes_lost_at) - 5} more")
    else:
        print(f"\n  [OK] No node losses detected (all nodes tracked throughout)")

    print(f"\n  Permanent node coordinates:")
    print(f"    X range: [{x_perm.min():.2f}, {x_perm.max():.2f}] mm")
    print(f"    Y range: [{y_perm.min():.2f}, {y_perm.max():.2f}] mm")
    print(f"    Width: {x_perm.max() - x_perm.min():.2f} mm")
    print(f"    Height: {y_perm.max() - y_perm.min():.2f} mm")

    # Quality assessment
    if retention_rate < 80:
        print(f"\n  [WARNING] Low retention rate ({retention_rate:.1f}%). Check for:")
        print(f"    - Surface quality issues")
        print(f"    - Out-of-plane deformation")
        print(f"    - Illumination changes")
    elif retention_rate < 95:
        print(f"\n  [INFO] Moderate retention rate ({retention_rate:.1f}%). Acceptable for processing.")
    else:
        print(f"\n  [EXCELLENT] High retention rate ({retention_rate:.1f}%). Good data quality.")

    print("\n[OK] Step 2 completed successfully")

    return data


# =============================================================================
# STEP 3: LOAD SYNTHETIC MESH
# =============================================================================

def step3_load_synthetic_mesh(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 3: Load synthetic mesh from Forward Solver output.

    The synthetic mesh provides:
    - Consistent element quality (vs DIC point cloud)
    - Proper boundary node labeling
    - Hole boundary support

    Boundary labels (from Forward Solver):
    - verticalGDL: 0=interior, 1=top, 2=bottom
    - horizontalGDL: 0=interior, 3=right, 4=left, 5=hole

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 2

    Returns:
        PreprocessingData with mesh data populated
    """
    print("\n" + "="*70)
    print("STEP 3: LOAD SYNTHETIC MESH")
    print("="*70)

    mesh_path = config.synthetic_mesh_path

    # -------------------------------------------------------------------------
    # 3.1 Check if synthetic mesh exists
    # -------------------------------------------------------------------------
    print(f"\n[3.1] Checking for synthetic mesh...")
    print(f"  Synthetic mesh ID: {config.synthetic_mesh_id}")
    print(f"  Path: {mesh_path}")

    coord_file = mesh_path / "coord.csv"
    conne_file = mesh_path / "conne.txt"

    if not coord_file.exists():
        raise FileNotFoundError(f"Mesh coord.csv not found: {coord_file}")
    if not conne_file.exists():
        raise FileNotFoundError(f"Mesh conne.txt not found: {conne_file}")

    print(f"  [OK] Mesh files found")

    # -------------------------------------------------------------------------
    # 3.2 Load mesh coordinates
    # -------------------------------------------------------------------------
    print(f"\n[3.2] Loading mesh coordinates...")

    coord_df = pd.read_csv(coord_file)
    data.mesh_coords = coord_df[['x', 'y']].values
    n_mesh_nodes = len(data.mesh_coords)

    print(f"  Nodes: {n_mesh_nodes}")
    print(f"  X range: [{data.mesh_coords[:, 0].min():.2f}, {data.mesh_coords[:, 0].max():.2f}] mm")
    print(f"  Y range: [{data.mesh_coords[:, 1].min():.2f}, {data.mesh_coords[:, 1].max():.2f}] mm")

    # -------------------------------------------------------------------------
    # 3.3 Load mesh connectivity
    # -------------------------------------------------------------------------
    print(f"\n[3.3] Loading mesh connectivity...")

    conne_df = pd.read_csv(conne_file)

    # Handle different column naming conventions
    if 'n1' in conne_df.columns:
        data.mesh_connectivity = conne_df[['n1', 'n2', 'n3']].values
    elif 'node1' in conne_df.columns:
        data.mesh_connectivity = conne_df[['node1', 'node2', 'node3']].values
    else:
        # Assume columns 1, 2, 3 are the node indices (skip elem_id column)
        data.mesh_connectivity = conne_df.iloc[:, 1:4].values

    n_elements = len(data.mesh_connectivity)
    print(f"  Elements: {n_elements}")

    # -------------------------------------------------------------------------
    # 3.4 Load boundary labels
    # -------------------------------------------------------------------------
    print(f"\n[3.4] Loading boundary labels...")

    # Try different column naming conventions
    if 'verticalGDL' in coord_df.columns:
        data.vertical_dofs = coord_df['verticalGDL'].values
    elif 'verticalDoFs' in coord_df.columns:
        data.vertical_dofs = coord_df['verticalDoFs'].values
    else:
        print(f"  [INFO] No vertical labels in file, detecting from coordinates...")
        data.vertical_dofs = _detect_vertical_boundaries(data.mesh_coords, config.boundary_tolerance)

    if 'horizontalGDL' in coord_df.columns:
        data.horizontal_dofs = coord_df['horizontalGDL'].values
    elif 'horizontalDoFs' in coord_df.columns:
        data.horizontal_dofs = coord_df['horizontalDoFs'].values
    else:
        print(f"  [INFO] No horizontal labels in file, detecting from coordinates...")
        data.horizontal_dofs = _detect_horizontal_boundaries(data.mesh_coords, config.boundary_tolerance)

    # -------------------------------------------------------------------------
    # 3.5 Boundary statistics
    # -------------------------------------------------------------------------
    print(f"\n[3.5] Boundary Node Statistics")
    print("-" * 40)

    n_top = np.sum(data.vertical_dofs == 1)
    n_bottom = np.sum(data.vertical_dofs == 2)
    n_right = np.sum(data.horizontal_dofs == 3)
    n_left = np.sum(data.horizontal_dofs == 4)
    n_hole = np.sum(data.horizontal_dofs == 5)
    n_interior = n_mesh_nodes - n_top - n_bottom - n_right - n_left - n_hole

    # Handle overlapping boundary nodes (corners)
    n_interior = np.sum((data.vertical_dofs == 0) & (data.horizontal_dofs == 0))

    data.preprocessing_stats['n_mesh_nodes'] = n_mesh_nodes
    data.preprocessing_stats['n_mesh_elements'] = n_elements
    data.preprocessing_stats['n_top'] = n_top
    data.preprocessing_stats['n_bottom'] = n_bottom
    data.preprocessing_stats['n_left'] = n_left
    data.preprocessing_stats['n_right'] = n_right
    data.preprocessing_stats['n_hole'] = n_hole
    data.preprocessing_stats['n_interior'] = n_interior

    print(f"  Interior:       {n_interior}")
    print(f"  Top (loaded):   {n_top}")
    print(f"  Bottom (fixed): {n_bottom}")
    print(f"  Left:           {n_left}")
    print(f"  Right:          {n_right}")

    if config.geometry_type == "with_holes":
        print(f"  Hole boundary:  {n_hole}")
        if n_hole == 0:
            print(f"  [WARNING] Expected hole boundary but found 0 nodes!")

    # -------------------------------------------------------------------------
    # 3.6 Dimension comparison
    # -------------------------------------------------------------------------
    mesh_width = data.mesh_coords[:, 0].max() - data.mesh_coords[:, 0].min()
    mesh_height = data.mesh_coords[:, 1].max() - data.mesh_coords[:, 1].min()
    dic_width = data.preprocessing_stats['specimen_width']
    dic_height = data.preprocessing_stats['specimen_height']

    data.preprocessing_stats['mesh_width'] = mesh_width
    data.preprocessing_stats['mesh_height'] = mesh_height

    print(f"\n  Dimension comparison:")
    print(f"    Mesh: {mesh_width:.2f} x {mesh_height:.2f} mm")
    print(f"    DIC:  {dic_width:.2f} x {dic_height:.2f} mm")

    width_ratio = mesh_width / dic_width
    height_ratio = mesh_height / dic_height

    print(f"    Ratios: width={width_ratio:.3f}, height={height_ratio:.3f}")

    if abs(width_ratio - 1.0) > 0.1 or abs(height_ratio - 1.0) > 0.1:
        print(f"    [INFO] Scaling will be needed in Step 4")

    print("\n[OK] Step 3 completed successfully")

    return data


# =============================================================================
# STEP 4: COORDINATE ALIGNMENT
# =============================================================================

def step4_align_coordinates(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 4: Align DIC coordinates to synthetic mesh coordinate system.

    DIC coordinates are typically centered around the camera origin,
    while the synthetic mesh uses a corner-origin system (0,0) at bottom-left.

    Alignment procedure:
    1. Translate DIC to corner-origin (shift min to 0)
    2. Scale to match mesh dimensions (uniform scaling)

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 3

    Returns:
        PreprocessingData with dic_coords_aligned and alignment_params populated
    """
    print("\n" + "="*70)
    print("STEP 4: COORDINATE ALIGNMENT")
    print("="*70)

    dic_x = data.permanent_coords[:, 0].copy()
    dic_y = data.permanent_coords[:, 1].copy()

    mesh_x = data.mesh_coords[:, 0]
    mesh_y = data.mesh_coords[:, 1]

    # -------------------------------------------------------------------------
    # 4.1 Original DIC coordinate system
    # -------------------------------------------------------------------------
    print(f"\n[4.1] Original DIC coordinates")
    print(f"  X: [{dic_x.min():.2f}, {dic_x.max():.2f}] mm")
    print(f"  Y: [{dic_y.min():.2f}, {dic_y.max():.2f}] mm")

    # -------------------------------------------------------------------------
    # 4.2 Translate to corner-origin (bottom-left = 0,0)
    # -------------------------------------------------------------------------
    print(f"\n[4.2] Translating DIC to corner-origin...")

    dic_x_offset = dic_x.min()
    dic_y_offset = dic_y.min()

    dic_x_aligned = dic_x - dic_x_offset
    dic_y_aligned = dic_y - dic_y_offset

    print(f"  Translation: ({-dic_x_offset:.2f}, {-dic_y_offset:.2f}) mm")
    print(f"  After translation: X=[{dic_x_aligned.min():.2f}, {dic_x_aligned.max():.2f}], Y=[{dic_y_aligned.min():.2f}, {dic_y_aligned.max():.2f}]")

    # -------------------------------------------------------------------------
    # 4.3 Scale to match mesh dimensions
    # -------------------------------------------------------------------------
    print(f"\n[4.3] Scaling to match mesh dimensions...")

    dic_width = dic_x_aligned.max() - dic_x_aligned.min()
    dic_height = dic_y_aligned.max() - dic_y_aligned.min()
    mesh_width = mesh_x.max() - mesh_x.min()
    mesh_height = mesh_y.max() - mesh_y.min()

    scale_x = mesh_width / dic_width
    scale_y = mesh_height / dic_height
    scale_uniform = (scale_x + scale_y) / 2.0

    print(f"  Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")
    print(f"  Scale anisotropy: {abs(scale_x - scale_y) / scale_uniform * 100:.2f}%")

    if abs(scale_x - scale_y) / scale_uniform > 0.02:
        print(f"  [WARNING] Anisotropy > 2%, using uniform scale: {scale_uniform:.4f}")
    else:
        print(f"  [OK] Low anisotropy, using uniform scale: {scale_uniform:.4f}")

    dic_x_aligned *= scale_uniform
    dic_y_aligned *= scale_uniform

    # -------------------------------------------------------------------------
    # 4.4 Translate to match mesh origin offset
    # -------------------------------------------------------------------------
    mesh_x_min = mesh_x.min()
    mesh_y_min = mesh_y.min()

    if abs(mesh_x_min) > 0.01 or abs(mesh_y_min) > 0.01:
        print(f"\n[4.4] Adjusting to mesh origin offset ({mesh_x_min:.2f}, {mesh_y_min:.2f})...")
        dic_x_aligned += mesh_x_min
        dic_y_aligned += mesh_y_min

    # -------------------------------------------------------------------------
    # 4.5 Summary
    # -------------------------------------------------------------------------
    data.dic_coords_aligned = np.column_stack([dic_x_aligned, dic_y_aligned])

    data.alignment_params = {
        'dic_offset': np.array([dic_x_offset, dic_y_offset]),
        'scale_x': scale_x,
        'scale_y': scale_y,
        'scale_uniform': scale_uniform,
        'mesh_origin': np.array([mesh_x_min, mesh_y_min]),
    }

    print(f"\n[4.5] Alignment Summary")
    print("-" * 40)
    print(f"  Aligned DIC: X=[{dic_x_aligned.min():.2f}, {dic_x_aligned.max():.2f}], Y=[{dic_y_aligned.min():.2f}, {dic_y_aligned.max():.2f}]")
    print(f"  Mesh:        X=[{mesh_x.min():.2f}, {mesh_x.max():.2f}], Y=[{mesh_y.min():.2f}, {mesh_y.max():.2f}]")

    # Compute overlap statistics
    x_overlap = min(dic_x_aligned.max(), mesh_x.max()) - max(dic_x_aligned.min(), mesh_x.min())
    y_overlap = min(dic_y_aligned.max(), mesh_y.max()) - max(dic_y_aligned.min(), mesh_y.min())

    print(f"  Overlap: X={x_overlap:.2f} mm ({x_overlap/mesh_width*100:.1f}%), Y={y_overlap:.2f} mm ({y_overlap/mesh_height*100:.1f}%)")

    print("\n[OK] Step 4 completed successfully")

    return data


# =============================================================================
# STEP 5: MAP DIC TO MESH (NEAREST NEIGHBOR)
# =============================================================================

def step5_map_dic_to_mesh(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 5: Map each mesh node to its nearest DIC node.

    For each synthetic mesh node, find the nearest aligned DIC node.
    The DIC displacement at that node will be assigned to the mesh node.

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 4

    Returns:
        PreprocessingData with mesh_to_dic_map and mapping_distances populated
    """
    print("\n" + "="*70)
    print("STEP 5: MAP DIC TO MESH (NEAREST NEIGHBOR)")
    print("="*70)

    n_mesh = len(data.mesh_coords)
    n_dic = len(data.dic_coords_aligned)

    print(f"\n  Mapping {n_mesh} mesh nodes to {n_dic} DIC nodes...")

    data.mesh_to_dic_map = np.zeros(n_mesh, dtype=int)
    data.mapping_distances = np.zeros(n_mesh)

    for i in range(n_mesh):
        distances = np.linalg.norm(data.dic_coords_aligned - data.mesh_coords[i], axis=1)
        nearest = np.argmin(distances)
        data.mesh_to_dic_map[i] = nearest
        data.mapping_distances[i] = distances[nearest]

        if (i + 1) % 200 == 0:
            print(f"    Mapped {i + 1}/{n_mesh} nodes...")

    # Statistics
    print(f"\n  Mapping Statistics")
    print("-" * 40)
    print(f"  Mean distance:   {data.mapping_distances.mean():.3f} mm")
    print(f"  Median distance: {np.median(data.mapping_distances):.3f} mm")
    print(f"  Max distance:    {data.mapping_distances.max():.3f} mm")
    print(f"  Min distance:    {data.mapping_distances.min():.3f} mm")
    print(f"  Std deviation:   {data.mapping_distances.std():.3f} mm")

    # Separate interior vs boundary mapping errors
    is_boundary = (data.vertical_dofs != 0) | (data.horizontal_dofs != 0)
    is_interior = ~is_boundary

    interior_dist = data.mapping_distances[is_interior]
    boundary_dist = data.mapping_distances[is_boundary]

    print(f"\n  Interior Nodes ({is_interior.sum()}):")
    print(f"    Mean: {interior_dist.mean():.3f} mm, Max: {interior_dist.max():.3f} mm")
    print(f"\n  Boundary Nodes ({is_boundary.sum()}):")
    print(f"    Mean: {boundary_dist.mean():.3f} mm, Max: {boundary_dist.max():.3f} mm")

    # Store in stats for metadata
    data.preprocessing_stats['mapping_interior_mean'] = interior_dist.mean()
    data.preprocessing_stats['mapping_interior_max'] = interior_dist.max()
    data.preprocessing_stats['mapping_boundary_mean'] = boundary_dist.mean()
    data.preprocessing_stats['mapping_boundary_max'] = boundary_dist.max()

    # Quality check
    threshold = config.mapping_error_threshold
    n_large = np.sum(data.mapping_distances > threshold)
    n_large_interior = np.sum(interior_dist > threshold)
    n_large_boundary = np.sum(boundary_dist > threshold)

    if n_large > 0:
        print(f"\n  [WARNING] {n_large} nodes have mapping error > {threshold} mm")
        print(f"    Interior: {n_large_interior}, Boundary: {n_large_boundary}")
        worst_idx = np.argmax(data.mapping_distances)
        print(f"    Worst node: mesh[{worst_idx}] at ({data.mesh_coords[worst_idx, 0]:.2f}, {data.mesh_coords[worst_idx, 1]:.2f})")
        if n_large_interior == 0:
            print(f"    [OK] All interior nodes within threshold - boundary errors due to DIC coverage limits")
    else:
        print(f"\n  [OK] All mapping errors < {threshold} mm")

    # Check for duplicate mappings
    unique_dic = len(np.unique(data.mesh_to_dic_map))
    print(f"\n  Unique DIC nodes used: {unique_dic}/{n_dic}")

    if unique_dic < n_mesh:
        print(f"  [INFO] {n_mesh - unique_dic} mesh nodes share DIC nodes (expected if mesh is coarser)")

    print("\n[OK] Step 5 completed successfully")

    return data


# =============================================================================
# STEP 6: EXTRACT DISPLACEMENTS & REMOVE RIGID BODY MOTION
# =============================================================================

def step6_extract_displacements(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 6: Extract displacements from DIC data for mapped mesh nodes.

    Builds the displacement matrix U[2*nNodes x nTimesteps] in grouped format:
    - Rows 0..N-1: X displacements for all nodes
    - Rows N..2N-1: Y displacements for all nodes

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 5

    Returns:
        PreprocessingData with U_raw populated
    """
    print("\n" + "="*70)
    print("STEP 6: EXTRACT DISPLACEMENTS")
    print("="*70)

    n_mesh = len(data.mesh_coords)
    n_timesteps = len(data.dic_dataframes)
    n_dic_permanent = len(data.permanent_nodes)

    print(f"\n  Building displacement matrix: [{2*n_mesh} x {n_timesteps}]")

    # -------------------------------------------------------------------------
    # 6.1 Load all DIC displacements for permanent nodes
    # -------------------------------------------------------------------------
    print(f"\n[6.1] Loading DIC displacements for permanent nodes...")

    # Pre-allocate: [2*n_dic_permanent x n_timesteps]
    U_dic = np.zeros((2 * n_dic_permanent, n_timesteps))

    for t_idx, df in enumerate(data.dic_dataframes):
        # Filter to permanent nodes and sort by id
        df_perm = df[df['id'].isin(data.permanent_nodes)].copy()
        df_perm = df_perm.sort_values('id').reset_index(drop=True)

        # Store in grouped format
        U_dic[:n_dic_permanent, t_idx] = df_perm['displacement_x'].values
        U_dic[n_dic_permanent:, t_idx] = df_perm['displacement_y'].values

        if (t_idx + 1) % 200 == 0 or t_idx == n_timesteps - 1:
            print(f"    Loaded {t_idx + 1}/{n_timesteps} timesteps...")

    print(f"  DIC displacement matrix: {U_dic.shape}")
    print(f"  Displacement range X: [{U_dic[:n_dic_permanent, :].min():.4f}, {U_dic[:n_dic_permanent, :].max():.4f}] mm")
    print(f"  Displacement range Y: [{U_dic[n_dic_permanent:, :].min():.4f}, {U_dic[n_dic_permanent:, :].max():.4f}] mm")

    # -------------------------------------------------------------------------
    # 6.2 Map DIC displacements to mesh nodes
    # -------------------------------------------------------------------------
    print(f"\n[6.2] Mapping displacements to mesh nodes...")

    data.U_raw = np.zeros((2 * n_mesh, n_timesteps))

    for i_mesh in range(n_mesh):
        dic_idx = data.mesh_to_dic_map[i_mesh]
        data.U_raw[i_mesh, :] = U_dic[dic_idx, :]  # X displacement
        data.U_raw[n_mesh + i_mesh, :] = U_dic[n_dic_permanent + dic_idx, :]  # Y displacement

    print(f"  Mesh displacement matrix: {data.U_raw.shape}")
    print(f"  Displacement range X: [{data.U_raw[:n_mesh, :].min():.4f}, {data.U_raw[:n_mesh, :].max():.4f}] mm")
    print(f"  Displacement range Y: [{data.U_raw[n_mesh:, :].min():.4f}, {data.U_raw[n_mesh:, :].max():.4f}] mm")

    print("\n[OK] Step 6 completed successfully")

    return data


# =============================================================================
# STEP 7: REMOVE RIGID BODY MOTION
# =============================================================================

def step7_remove_rigid_body_motion(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 7: Remove rigid body translation using a fixed reference node.

    Uses the last bottom boundary node as reference (matching MATLAB convention).
    Subtracts the reference node displacement from all nodes at each timestep.

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 6

    Returns:
        PreprocessingData with U_corrected populated
    """
    print("\n" + "="*70)
    print("STEP 7: REMOVE RIGID BODY MOTION")
    print("="*70)

    n_mesh = len(data.mesh_coords)
    n_timesteps = data.U_raw.shape[1]

    # -------------------------------------------------------------------------
    # 7.1 Select reference node (last bottom node)
    # -------------------------------------------------------------------------
    bottom_nodes = np.where(data.vertical_dofs == 2)[0]
    reference_node = bottom_nodes[-1]

    print(f"\n[7.1] Reference node: {reference_node}")
    print(f"  Position: ({data.mesh_coords[reference_node, 0]:.2f}, {data.mesh_coords[reference_node, 1]:.2f}) mm")

    # Reference displacement at t=0
    ref_ux_0 = data.U_raw[reference_node, 0]
    ref_uy_0 = data.U_raw[n_mesh + reference_node, 0]
    print(f"  Reference displacement at t=0: ({ref_ux_0:.6f}, {ref_uy_0:.6f}) mm")

    # -------------------------------------------------------------------------
    # 7.2 Subtract reference displacement
    # -------------------------------------------------------------------------
    print(f"\n[7.2] Subtracting rigid body translation...")

    data.U_corrected = data.U_raw.copy()

    for t in range(n_timesteps):
        ref_ux = data.U_raw[reference_node, t]
        ref_uy = data.U_raw[n_mesh + reference_node, t]

        data.U_corrected[:n_mesh, t] -= ref_ux
        data.U_corrected[n_mesh:, t] -= ref_uy

    # -------------------------------------------------------------------------
    # 7.3 Summary
    # -------------------------------------------------------------------------
    print(f"\n[7.3] Corrected Displacement Summary")
    print("-" * 40)
    print(f"  X range: [{data.U_corrected[:n_mesh, :].min():.4f}, {data.U_corrected[:n_mesh, :].max():.4f}] mm")
    print(f"  Y range: [{data.U_corrected[n_mesh:, :].min():.4f}, {data.U_corrected[n_mesh:, :].max():.4f}] mm")

    # Verify reference node is zero
    ref_check_x = data.U_corrected[reference_node, :].max()
    ref_check_y = data.U_corrected[n_mesh + reference_node, :].max()
    print(f"  Reference node max residual: ({ref_check_x:.2e}, {ref_check_y:.2e}) mm")

    if ref_check_x > 1e-10 or ref_check_y > 1e-10:
        print(f"  [WARNING] Reference node not exactly zero!")
    else:
        print(f"  [OK] Reference node zeroed successfully")

    data.preprocessing_stats['reference_node'] = reference_node

    print("\n[OK] Step 7 completed successfully")

    return data


# =============================================================================
# STEP 8: BUILD FORCE MATRIX
# =============================================================================

def step8_build_force_matrix(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 8: Build force matrix for inverse problem.

    Format:
    - Solid specimens: [4 x nTime] → top_y, bottom_y, right_x, left_x
    - Hole specimens:  [5 x nTime] → top_y, bottom_y, right_x, left_x, hole

    Forces are TOTAL force per boundary (in N).
    Exported in kN for inverse problem compatibility.

    Args:
        config: Preprocessing configuration
        data: PreprocessingData from Step 7

    Returns:
        PreprocessingData with F_matrix populated
    """
    print("\n" + "="*70)
    print("STEP 8: BUILD FORCE MATRIX")
    print("="*70)

    n_timesteps = len(data.dic_times)
    has_hole = config.geometry_type == "with_holes"
    n_boundaries = 5 if has_hole else 4

    data.F_matrix = np.zeros((n_boundaries, n_timesteps))

    # Uniaxial tension: force applied on top, reaction on bottom
    data.F_matrix[0, :] = data.force_data       # Top Y: applied load (N)
    data.F_matrix[1, :] = -data.force_data      # Bottom Y: reaction (N)
    # F_matrix[2, :] = 0                        # Right X: no load
    # F_matrix[3, :] = 0                        # Left X: no load
    if has_hole:
        pass  # F_matrix[4, :] = 0              # Hole: free boundary

    print(f"\n  Force matrix shape: {data.F_matrix.shape}")
    print(f"  Boundaries: {n_boundaries} ({'with hole' if has_hole else 'solid'})")
    print(f"  Top (applied):  [{data.F_matrix[0, :].min():.1f}, {data.F_matrix[0, :].max():.1f}] N")
    print(f"  Bottom (react): [{data.F_matrix[1, :].min():.1f}, {data.F_matrix[1, :].max():.1f}] N")

    print("\n[OK] Step 8 completed successfully")

    return data


# =============================================================================
# STEP 9: EXPORT DATA
# =============================================================================

def step9_export_data(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 9: Export preprocessed data for inverse problem.

    Output files:
    - coord.csv: [id, x, y, verticalGDL, horizontalGDL]
    - conne.txt: copied from synthetic mesh
    - U.csv: [2*nNodes x nTime] grouped format, no header
    - F.csv: [nBoundaries x nTime] in kN, no header
    - time.csv: [nTime] vector, no header
    - bc.csv: [nNodes] labels, no header
    - metadata.txt: preprocessing summary
    """
    print("\n" + "="*70)
    print("STEP 9: EXPORT DATA")
    print("="*70)

    import shutil

    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    n_mesh = len(data.mesh_coords)
    n_timesteps = len(data.dic_times)

    print(f"\n  Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 9.1 coord.csv - Copy from synthetic mesh (has boundary labels)
    # -------------------------------------------------------------------------
    src_coord = config.synthetic_mesh_path / "coord.csv"
    shutil.copy2(src_coord, output_dir / "coord.csv")
    print(f"  [OK] coord.csv ({n_mesh} nodes)")

    # -------------------------------------------------------------------------
    # 9.2 conne.txt - Copy from synthetic mesh
    # -------------------------------------------------------------------------
    src_conne = config.synthetic_mesh_path / "conne.txt"
    shutil.copy2(src_conne, output_dir / "conne.txt")
    print(f"  [OK] conne.txt ({data.preprocessing_stats['n_mesh_elements']} elements)")

    # -------------------------------------------------------------------------
    # 9.3 U.csv - Displacement matrix (grouped format, no header)
    # -------------------------------------------------------------------------
    np.savetxt(output_dir / "U.csv", data.U_corrected, delimiter=',', fmt='%.6e')
    print(f"  [OK] U.csv ({data.U_corrected.shape[0]} x {data.U_corrected.shape[1]})")

    # -------------------------------------------------------------------------
    # 9.4 F.csv - Force matrix in kN (no header)
    # -------------------------------------------------------------------------
    F_kN = data.F_matrix / 1000.0
    np.savetxt(output_dir / "F.csv", F_kN, delimiter=',', fmt='%.6e')
    print(f"  [OK] F.csv ({F_kN.shape[0]} x {F_kN.shape[1]}) in kN")

    # -------------------------------------------------------------------------
    # 9.5 time.csv
    # -------------------------------------------------------------------------
    np.savetxt(output_dir / "time.csv", data.dic_times, delimiter=',', fmt='%.3f')
    print(f"  [OK] time.csv ({n_timesteps} timesteps)")

    # -------------------------------------------------------------------------
    # 9.6 bc.csv - Boundary condition labels
    # -------------------------------------------------------------------------
    # Build combined BC: 0=free, 1=top, 2=bottom, 3=right, 4=left, 5=hole
    bc = np.zeros(n_mesh, dtype=int)
    bc[data.vertical_dofs == 1] = 1   # top
    bc[data.vertical_dofs == 2] = 2   # bottom
    bc[data.horizontal_dofs == 3] = 3  # right
    bc[data.horizontal_dofs == 4] = 4  # left
    bc[data.horizontal_dofs == 5] = 5  # hole

    data.bc_labels = bc
    np.savetxt(output_dir / "bc.csv", bc, delimiter=',', fmt='%d')
    print(f"  [OK] bc.csv ({n_mesh} nodes)")

    # -------------------------------------------------------------------------
    # 9.7 metadata.txt
    # -------------------------------------------------------------------------
    stats = data.preprocessing_stats
    with open(output_dir / "metadata.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"PREPROCESSING METADATA - EXPERIMENT {config.experiment_id}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Description: {config.base_config.description}\n")
        f.write(f"Geometry: {config.geometry_type}\n")
        f.write(f"Synthetic mesh ID: {config.synthetic_mesh_id}\n\n")

        f.write("DIC Data:\n")
        f.write(f"  Files: {stats.get('n_dic_files', '?')}\n")
        f.write(f"  Permanent nodes: {stats.get('n_permanent_nodes', '?')}\n")
        f.write(f"  Retention: {stats.get('retention_rate', '?'):.1f}%\n")
        f.write(f"  DIC dimensions: {stats.get('specimen_width', 0):.2f} x {stats.get('specimen_height', 0):.2f} mm\n\n")

        f.write("Mesh:\n")
        f.write(f"  Nodes: {stats.get('n_mesh_nodes', '?')}\n")
        f.write(f"  Elements: {stats.get('n_mesh_elements', '?')}\n")
        f.write(f"  Mesh dimensions: {stats.get('mesh_width', 0):.2f} x {stats.get('mesh_height', 0):.2f} mm\n\n")

        f.write("Boundaries:\n")
        f.write(f"  Top: {stats.get('n_top', 0)}\n")
        f.write(f"  Bottom: {stats.get('n_bottom', 0)}\n")
        f.write(f"  Left: {stats.get('n_left', 0)}\n")
        f.write(f"  Right: {stats.get('n_right', 0)}\n")
        f.write(f"  Hole: {stats.get('n_hole', 0)}\n\n")

        f.write("Alignment:\n")
        ap = data.alignment_params
        f.write(f"  Scale: {ap.get('scale_uniform', 0):.4f}\n")
        f.write(f"  DIC offset: ({ap.get('dic_offset', [0,0])[0]:.2f}, {ap.get('dic_offset', [0,0])[1]:.2f})\n\n")

        f.write("Mapping:\n")
        f.write(f"  Overall - Mean: {data.mapping_distances.mean():.3f} mm, Max: {data.mapping_distances.max():.3f} mm\n")
        f.write(f"  Interior - Mean: {stats.get('mapping_interior_mean', 0):.3f} mm, Max: {stats.get('mapping_interior_max', 0):.3f} mm\n")
        f.write(f"  Boundary - Mean: {stats.get('mapping_boundary_mean', 0):.3f} mm, Max: {stats.get('mapping_boundary_max', 0):.3f} mm\n\n")

        f.write("Force:\n")
        f.write(f"  Sensor: 10kN (Col {config.base_config.force_column})\n")
        f.write(f"  Range: [{data.force_data.min():.1f}, {data.force_data.max():.1f}] N\n\n")

        f.write("Reference node: " + str(stats.get('reference_node', '?')) + "\n")

    print(f"  [OK] metadata.txt")

    # -------------------------------------------------------------------------
    # 9.8 mapping.csv - DIC-to-mesh mapping for debugging
    # -------------------------------------------------------------------------
    pd.DataFrame({
        'mesh_node': np.arange(n_mesh),
        'dic_node_idx': data.mesh_to_dic_map,
        'distance_mm': data.mapping_distances
    }).to_csv(output_dir / "mapping.csv", index=False)
    print(f"  [OK] mapping.csv")

    print(f"\n  All files exported to: {output_dir}")

    print("\n[OK] Step 9 completed successfully")

    return data


# =============================================================================
# STEP 10: GENERATE DIAGNOSTIC PLOTS
# =============================================================================

def step10_generate_diagnostic_plots(config: PreprocessingConfig, data: PreprocessingData) -> PreprocessingData:
    """
    Step 10: Generate diagnostic plots for quality assessment.

    Generates 4 plots:
        1. DIC point cloud with boundaries
        2. Synthetic mesh with boundary labels
        3. DIC-Mesh mapping quality
        4. Force vs time and displacement summary
    """
    print("\n" + "="*70)
    print("STEP 10: GENERATE DIAGNOSTIC PLOTS")
    print("="*70)

    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    n_mesh = len(data.mesh_coords)

    # -------------------------------------------------------------------------
    # Plot 1: DIC Point Cloud
    # -------------------------------------------------------------------------
    print("\n  [Plot 1] DIC point cloud...")
    fig, ax = plt.subplots(figsize=(10, 12))

    dic_x = data.dic_coords_aligned[:, 0]
    dic_y = data.dic_coords_aligned[:, 1]
    ax.scatter(dic_x, dic_y, c='steelblue', s=2, alpha=0.6, label='DIC Points')
    ax.axhline(dic_y.min(), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Bottom')
    ax.axhline(dic_y.max(), color='green', linestyle='--', linewidth=1, alpha=0.5, label='Top')
    ax.axvline(dic_x.min(), color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Left')
    ax.axvline(dic_x.max(), color='purple', linestyle='--', linewidth=1, alpha=0.5, label='Right')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'DIC Point Cloud (Aligned)\n{len(dic_x)} Permanent Nodes', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'vis_1_dic_point_cloud.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: vis_1_dic_point_cloud.png")

    # -------------------------------------------------------------------------
    # Plot 2: Synthetic Mesh with Boundary Labels
    # -------------------------------------------------------------------------
    print("  [Plot 2] Synthetic mesh with boundaries...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    triangles_0idx = data.mesh_connectivity - 1
    triang = mtri.Triangulation(data.mesh_coords[:, 0], data.mesh_coords[:, 1], triangles_0idx)

    # Vertical boundaries
    ax1.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
    interior_v = data.vertical_dofs == 0
    top = data.vertical_dofs == 1
    bottom = data.vertical_dofs == 2
    ax1.scatter(data.mesh_coords[interior_v, 0], data.mesh_coords[interior_v, 1], c='lightgray', s=5, label='Interior', zorder=2)
    ax1.scatter(data.mesh_coords[top, 0], data.mesh_coords[top, 1], c='green', s=30, label=f'Top ({top.sum()})', zorder=3, marker='^')
    ax1.scatter(data.mesh_coords[bottom, 0], data.mesh_coords[bottom, 1], c='red', s=30, label=f'Bottom ({bottom.sum()})', zorder=3, marker='v')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('Vertical Boundaries', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Horizontal boundaries
    ax2.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
    interior_h = data.horizontal_dofs == 0
    left = data.horizontal_dofs == 4
    right = data.horizontal_dofs == 3
    hole = data.horizontal_dofs == 5
    ax2.scatter(data.mesh_coords[interior_h, 0], data.mesh_coords[interior_h, 1], c='lightgray', s=5, label='Interior', zorder=2)
    ax2.scatter(data.mesh_coords[left, 0], data.mesh_coords[left, 1], c='orange', s=30, label=f'Left ({left.sum()})', zorder=3, marker='<')
    ax2.scatter(data.mesh_coords[right, 0], data.mesh_coords[right, 1], c='purple', s=30, label=f'Right ({right.sum()})', zorder=3, marker='>')
    if hole.any():
        ax2.scatter(data.mesh_coords[hole, 0], data.mesh_coords[hole, 1], c='darkred', s=40, label=f'Hole ({hole.sum()})', zorder=4, marker='o', edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title('Horizontal Boundaries & Hole', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'vis_2_synthetic_mesh.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: vis_2_synthetic_mesh.png")

    # -------------------------------------------------------------------------
    # Plot 3: DIC-Mesh Mapping Quality
    # -------------------------------------------------------------------------
    print("  [Plot 3] Mapping quality...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Overlay
    ax1.scatter(dic_x, dic_y, c='lightblue', s=3, alpha=0.4, label='DIC Nodes')
    ax1.scatter(data.mesh_coords[:, 0], data.mesh_coords[:, 1], c='red', s=20, alpha=0.7, marker='x', label='Mesh Nodes', linewidths=1.5)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title(f'DIC-Mesh Overlay\nDIC: {len(dic_x)}, Mesh: {n_mesh}', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Mapping error spatial distribution
    sc = ax2.scatter(data.mesh_coords[:, 0], data.mesh_coords[:, 1], c=data.mapping_distances, s=30,
                     cmap='RdYlGn_r', vmin=0, vmax=max(data.mapping_distances.max(), 0.5), edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Mapping Distance [mm]')
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title(f'Mapping Quality\nMean: {data.mapping_distances.mean():.3f} mm, Max: {data.mapping_distances.max():.3f} mm', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'vis_3_mapping_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: vis_3_mapping_quality.png")

    # -------------------------------------------------------------------------
    # Plot 4: Force vs Time and Displacement Summary
    # -------------------------------------------------------------------------
    print("  [Plot 4] Force & displacement summary...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4a: Force vs time
    ax = axes[0, 0]
    F_kN = data.F_matrix / 1000.0
    ax.plot(data.dic_times, F_kN[0, :], 'b-', linewidth=1, label='Top (loaded)')
    if F_kN.shape[0] > 2:
        ax.plot(data.dic_times, F_kN[1, :], 'r--', linewidth=1, label='Bottom')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [kN]')
    ax.set_title('Force vs Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4b: Max displacement vs time
    ax = axes[0, 1]
    U = data.U_corrected
    ux = U[:n_mesh, :]
    uy = U[n_mesh:, :]
    ax.plot(data.dic_times, np.max(np.abs(ux), axis=0), 'b-', linewidth=1, label='max |Ux|')
    ax.plot(data.dic_times, np.max(np.abs(uy), axis=0), 'r-', linewidth=1, label='max |Uy|')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [mm]')
    ax.set_title('Max Displacement vs Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4c: Displacement field at last timestep (Uy)
    ax = axes[1, 0]
    sc = ax.scatter(data.mesh_coords[:, 0], data.mesh_coords[:, 1], c=uy[:, -1], s=15, cmap='coolwarm')
    plt.colorbar(sc, ax=ax, label='Uy [mm]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'Uy at t={data.dic_times[-1]:.1f}s', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 4d: Displacement field at last timestep (Ux)
    ax = axes[1, 1]
    sc = ax.scatter(data.mesh_coords[:, 0], data.mesh_coords[:, 1], c=ux[:, -1], s=15, cmap='coolwarm')
    plt.colorbar(sc, ax=ax, label='Ux [mm]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'Ux at t={data.dic_times[-1]:.1f}s', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Experiment {config.experiment_id}: {config.base_config.description}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'vis_4_force_displacement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: vis_4_force_displacement.png")

    print("\n[OK] Step 10 completed - all plots saved")

    return data


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_vertical_boundaries(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Detect vertical boundary nodes (top/bottom) from coordinates."""
    y = coords[:, 1]
    y_min, y_max = y.min(), y.max()

    vertical_dofs = np.zeros(len(coords), dtype=int)
    vertical_dofs[np.abs(y - y_max) < tolerance] = 1  # top
    vertical_dofs[np.abs(y - y_min) < tolerance] = 2  # bottom

    return vertical_dofs


def _detect_horizontal_boundaries(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Detect horizontal boundary nodes (left/right) from coordinates."""
    x = coords[:, 0]
    x_min, x_max = x.min(), x.max()

    horizontal_dofs = np.zeros(len(coords), dtype=int)
    horizontal_dofs[np.abs(x - x_max) < tolerance] = 3  # right
    horizontal_dofs[np.abs(x - x_min) < tolerance] = 4  # left

    return horizontal_dofs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def preprocess_experiment(experiment_id: int, verbose: bool = True) -> Tuple[PreprocessingConfig, PreprocessingData]:
    """
    Run the complete preprocessing pipeline for an experiment.

    Args:
        experiment_id: Experiment number (1, 2, 3, or 4)
        verbose: Whether to print detailed output

    Returns:
        Tuple of (config, data)
    """
    print("="*70)
    print(f"UNIFIED PREPROCESSING PIPELINE - EXPERIMENT {experiment_id}")
    print("="*70)

    # Validate experiment ID
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Invalid experiment ID: {experiment_id}. Available: {list(EXPERIMENT_CONFIGS.keys())}")

    # Create configuration
    base_config = EXPERIMENT_CONFIGS[experiment_id]
    config = PreprocessingConfig(
        experiment_id=experiment_id,
        base_config=base_config
    )

    print(f"\nConfiguration:")
    print(f"  Experiment: {config.experiment_id}")
    print(f"  Description: {base_config.description}")
    print(f"  Geometry type: {config.geometry_type}")
    print(f"  Synthetic mesh ID: {config.synthetic_mesh_id}")
    print(f"  Output path: {config.output_path}")

    # Step 1: Load Data
    data = step1_load_data(config)

    # Step 2: Identify Permanent Nodes
    data = step2_identify_permanent_nodes(config, data)

    # Step 3: Load Synthetic Mesh
    data = step3_load_synthetic_mesh(config, data)

    # Step 4: Coordinate Alignment
    data = step4_align_coordinates(config, data)

    # Step 5: Map DIC to Mesh
    data = step5_map_dic_to_mesh(config, data)

    # Step 6: Extract Displacements
    data = step6_extract_displacements(config, data)

    # Step 7: Remove Rigid Body Motion
    data = step7_remove_rigid_body_motion(config, data)

    # Step 8: Build Force Matrix
    data = step8_build_force_matrix(config, data)

    # Step 9: Export Data
    data = step9_export_data(config, data)

    # Step 10: Diagnostic Plots
    data = step10_generate_diagnostic_plots(config, data)

    print("\n" + "="*70)
    print(f"PREPROCESSING COMPLETE - Experiment {experiment_id}")
    print(f"Output: {config.output_path}")
    print("="*70)

    return config, data


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            exp_id = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid argument: {sys.argv[1]}")
            print("Usage: python unified_preprocessor.py [experiment_number]")
            sys.exit(1)
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("  UNIFIED PREPROCESSING PIPELINE")
        print("="*60)
        print("\n  Available experiments:")
        for exp_id, config in EXPERIMENT_CONFIGS.items():
            print(f"    {exp_id}: {config.description}")

        user_input = input("\n  Enter experiment number (1-4): ").strip()
        try:
            exp_id = int(user_input)
        except ValueError:
            print(f"[ERROR] Invalid input: {user_input}")
            sys.exit(1)

    # Run preprocessing
    try:
        config, data = preprocess_experiment(exp_id)
        print("\n[SUCCESS] Preprocessing completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
