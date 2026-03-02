"""
Geometry builder using pygmsh for complex geometries.

This module creates meshes for experiments 804-806 using pygmsh (OpenCASCADE kernel)
and exports them in a format compatible with the forward solver.

Key features:
- True ellipses (not circular approximations)
- Boolean operations for complex shapes
- Smooth hole boundaries
- Configurable mesh refinement
"""

import pygmsh
import meshio
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys


class GeometryBuilder:
    """
    Build complex geometries using pygmsh (OpenCASCADE kernel).

    Supports:
    - Circular holes
    - Elliptical holes
    - Multiple holes
    - Configurable mesh refinement
    """

    def __init__(self, config: Dict):
        """
        Initialize geometry builder with configuration.

        Args:
            config: Configuration dictionary with domain, holes, mesh_params
                   Expected structure:
                   {
                       'id': int,
                       'name': str,
                       'description': str,
                       'domain': {'width': float, 'height': float},
                       'holes': [{'type': str, 'center': tuple, ...}],
                       'mesh_params': {'mesh_size_outer': float, ...}
                   }
        """
        self.config = config
        self.domain = config['domain']
        self.holes = config.get('holes', [])
        self.mesh_params = config.get('mesh_params', {})

        print(f"="*70)
        print(f"GEOMETRY BUILDER - Experiment {config['id']}")
        print(f"="*70)
        print(f"Name: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Domain: {self.domain['width']:.1f}mm × {self.domain['height']:.1f}mm")
        print(f"Holes: {len(self.holes)}")

    def build(self) -> meshio.Mesh:
        """
        Build geometry and generate mesh using pygmsh.

        Returns:
            meshio.Mesh object containing nodes and elements

        Raises:
            RuntimeError: If mesh generation fails
        """
        print(f"\nBuilding geometry with pygmsh (OpenCASCADE)...")

        try:
            with pygmsh.occ.Geometry() as geom:
                # Set mesh size parameters
                mesh_size_hole = self.mesh_params.get('mesh_size_hole', 0.3)
                mesh_size_outer = self.mesh_params.get('mesh_size_outer', 2.0)

                # Use the smaller of the two as global max to ensure hole is refined
                geom.characteristic_length_min = mesh_size_hole
                geom.characteristic_length_max = min(mesh_size_outer, mesh_size_hole * 3)

                # Create outer rectangle
                width = self.domain['width']
                height = self.domain['height']

                print(f"  Creating rectangle: {width:.1f} × {height:.1f} mm")
                rectangle = geom.add_rectangle([0.0, 0.0, 0.0], width, height)

                # Create holes
                hole_objects = []
                for i, hole_config in enumerate(self.holes, 1):
                    hole_obj = self._create_hole(geom, hole_config, i)
                    hole_objects.append(hole_obj)

                # Boolean difference: subtract holes from rectangle
                if hole_objects:
                    print(f"\n  Performing boolean difference (subtracting {len(hole_objects)} holes)...")
                    specimen = geom.boolean_difference(rectangle, hole_objects)
                else:
                    specimen = rectangle

                # Generate mesh
                print(f"  Generating 2D mesh...")
                mesh = geom.generate_mesh(dim=2)

            print(f"  [OK] Mesh generated:")
            print(f"      Nodes: {len(mesh.points)}")

            # Count triangular elements
            n_triangles = 0
            if 'triangle' in mesh.cells_dict:
                n_triangles = len(mesh.cells_dict['triangle'])

            print(f"      Elements: {n_triangles}")

            return mesh

        except Exception as e:
            print(f"  [ERROR] Mesh generation failed: {e}")
            raise RuntimeError(f"Failed to build geometry: {e}")

    def _create_hole(self, geom, hole_config: Dict, hole_num: int):
        """
        Create a single hole (circle or ellipse).

        Args:
            geom: pygmsh geometry object
            hole_config: Hole configuration dictionary
            hole_num: Hole number (for logging)

        Returns:
            Hole geometry object

        Raises:
            ValueError: If hole type is unknown
        """
        hole_type = hole_config['type']
        center = hole_config['center']

        if hole_type == 'circle':
            radius = hole_config['radius']
            print(f"  Hole {hole_num}: Circle at ({center[0]:.1f}, {center[1]:.1f}), r={radius:.1f}mm")

            hole = geom.add_disk(
                [center[0], center[1], 0.0],
                radius
            )

        elif hole_type == 'ellipse':
            semi_major = hole_config['semi_major']
            semi_minor = hole_config['semi_minor']
            rotation = hole_config.get('rotation', 0.0)

            print(f"  Hole {hole_num}: Ellipse at ({center[0]:.1f}, {center[1]:.1f}), "
                  f"axes={semi_major:.1f}×{semi_minor:.1f}mm, rot={rotation:.1f}°")

            # Create circle then scale to ellipse
            hole = geom.add_disk(
                [center[0], center[1], 0.0],
                semi_major
            )

            # Scale to create ellipse
            geom.dilate(hole, [center[0], center[1], 0.0],
                       [1.0, semi_minor/semi_major, 1.0])

            # Rotate if needed
            if rotation != 0.0:
                angle_rad = np.radians(rotation)
                geom.rotate(hole, [center[0], center[1], 0.0],
                           angle_rad, [0, 0, 1])

        else:
            raise ValueError(f"Unknown hole type: {hole_type}")

        return hole

    def save_mesh(self, mesh: meshio.Mesh, output_dir: Path):
        """
        Save mesh to various formats for visualization.

        Args:
            mesh: meshio mesh object
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exp_id = self.config['id']

        print(f"\nSaving mesh files to: {output_dir}")

        # Save VTK (for ParaView visualization)
        vtk_file = output_dir / f"geometry_exp{exp_id}.vtk"
        try:
            meshio.write(vtk_file, mesh)
            print(f"  Saved: {vtk_file.name}")
        except Exception as e:
            print(f"  [WARNING] Could not save VTK: {e}")

        # Save MSH (Gmsh format v2.2 for compatibility)
        msh_file = output_dir / f"geometry_exp{exp_id}.msh"
        try:
            meshio.write(msh_file, mesh, file_format="gmsh22")
            print(f"  Saved: {msh_file.name}")
        except Exception as e:
            print(f"  [WARNING] Could not save MSH: {e}")


def load_config(experiment_id: int) -> Dict:
    """
    Load configuration for a given experiment.

    Args:
        experiment_id: Experiment number (804, 805, 806)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If configuration not found
    """
    try:
        # Dynamic import based on experiment ID
        config_module_name = f'Forward_solver.configs.geometry_config_{experiment_id}'
        config_module = __import__(config_module_name, fromlist=['EXPERIMENT_CONFIG'])
        return config_module.EXPERIMENT_CONFIG
    except ImportError as e:
        raise ValueError(f"No configuration found for experiment {experiment_id}: {e}")


# ========== Testing Code ==========
if __name__ == "__main__":
    """Test geometry builder with experiment 804"""

    import argparse

    parser = argparse.ArgumentParser(description='Build geometry for complex experiments')
    parser.add_argument('--experiment', type=int, default=804,
                       help='Experiment number (804, 805, 806)')
    parser.add_argument('--output', type=str, default='./synthetic_data',
                       help='Output directory')

    args = parser.parse_args()

    try:
        print("="*70)
        print("GEOMETRY BUILDER TEST")
        print("="*70)

        # Load configuration
        print(f"\n[1/3] Loading configuration for experiment {args.experiment}...")
        config = load_config(args.experiment)

        # Build geometry
        print(f"\n[2/3] Building geometry...")
        builder = GeometryBuilder(config)
        mesh = builder.build()

        # Save mesh
        print(f"\n[3/3] Saving mesh files...")
        output_dir = Path(args.output) / str(args.experiment)
        builder.save_mesh(mesh, output_dir)

        print("\n" + "="*70)
        print("GEOMETRY BUILDING COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {output_dir.absolute()}")
        print(f"\nYou can visualize the mesh:")
        print(f"  paraview {output_dir}/geometry_exp{args.experiment}.vtk")

    except Exception as e:
        print(f"\n[ERROR] Failed to build geometry: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
