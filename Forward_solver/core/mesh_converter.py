"""
Convert meshio mesh to EUCLID format (coord.csv, conne.txt).

This module bridges pygmsh-generated meshes with the forward solver by:
1. Identifying boundary nodes from mesh topology
2. Classifying boundaries (top, bottom, left, right, holes)
3. Applying EUCLID boundary labels (verticalGDL, horizontalGDL)
4. Exporting to coord.csv and conne.txt format
"""

import numpy as np
import meshio
from pathlib import Path
from typing import Tuple, Dict
from collections import defaultdict


class MeshConverter:
    """
    Convert meshio mesh to EUCLID forward solver format.

    EUCLID Format:
    - coord.csv: [id, x, y, verticalGDL, horizontalGDL]
      - verticalGDL: 0=interior, 1=top, 2=bottom
      - horizontalGDL: 0=interior, 3=right, 4=left, 5=hole
    - conne.txt: [elem_id, n1, n2, n3] (1-indexed)
    """

    def __init__(self, mesh: meshio.Mesh, domain_width: float, domain_height: float):
        """
        Initialize converter.

        Args:
            mesh: meshio mesh object from pygmsh
            domain_width: Domain width [mm]
            domain_height: Domain height [mm]
        """
        self.mesh = mesh
        self.width = domain_width
        self.height = domain_height
        self.points = mesh.points[:, :2]  # Only x, y coordinates (ignore z)
        self.triangles = mesh.cells_dict.get('triangle', np.array([]))
        self.n_nodes = len(self.points)

        print(f"\n{'='*70}")
        print(f"MESH CONVERTER: meshio -> EUCLID format")
        print(f"{'='*70}")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  Elements: {len(self.triangles)}")

    def convert(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert mesh to coord.csv and conne.txt format.

        Returns:
            (coord, conne): EUCLID format arrays
                coord: (n_nodes, 5) array [id, x, y, verticalGDL, horizontalGDL]
                conne: (n_elements, 3) array [n1, n2, n3] (0-indexed)
        """
        # Step 1: Identify boundaries
        boundaries = self._identify_boundaries()

        # Step 2: Create coord array [id, x, y, verticalGDL, horizontalGDL]
        coord = self._create_coord_array(boundaries)

        # Step 3: Create conne array (0-indexed)
        conne = self.triangles

        print(f"\n  [OK] Conversion complete")

        return coord, conne

    def _identify_boundaries(self, tolerance=1e-6) -> Dict:
        """
        Identify boundary nodes from mesh topology.

        Uses edge detection: boundary edges appear in exactly one triangle.

        Args:
            tolerance: Geometric tolerance for outer boundary detection [mm]

        Returns:
            Dictionary with boundary classifications:
            {
                'top': [node_ids],
                'bottom': [node_ids],
                'left': [node_ids],
                'right': [node_ids],
                'holes': [node_ids]
            }
        """
        print(f"\n  Identifying boundary nodes...")

        boundaries = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
            'holes': []
        }

        # Find boundary edges (edges that belong to only one triangle)
        edge_count = defaultdict(int)

        for tri in self.triangles:
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_count[edge] += 1

        # Boundary edges appear exactly once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        # Get all boundary nodes
        boundary_node_set = set()
        for edge in boundary_edges:
            boundary_node_set.add(edge[0])
            boundary_node_set.add(edge[1])

        print(f"      Total boundary nodes: {len(boundary_node_set)}")

        # Classify outer boundary nodes by position
        for node_id in boundary_node_set:
            x, y = self.points[node_id]

            is_outer = False

            # Check outer boundaries
            if abs(y - self.height) < tolerance:  # Top
                boundaries['top'].append(node_id)
                is_outer = True
            if abs(y - 0.0) < tolerance:  # Bottom
                boundaries['bottom'].append(node_id)
                is_outer = True
            if abs(x - 0.0) < tolerance:  # Left
                boundaries['left'].append(node_id)
                is_outer = True
            if abs(x - self.width) < tolerance:  # Right
                boundaries['right'].append(node_id)
                is_outer = True

            # If not on outer boundary, must be on hole boundary
            if not is_outer:
                boundaries['holes'].append(node_id)

        print(f"      Top: {len(boundaries['top'])} nodes")
        print(f"      Bottom: {len(boundaries['bottom'])} nodes")
        print(f"      Left: {len(boundaries['left'])} nodes")
        print(f"      Right: {len(boundaries['right'])} nodes")
        print(f"      Holes: {len(boundaries['holes'])} nodes")

        return boundaries

    def _create_coord_array(self, boundaries: Dict) -> np.ndarray:
        """
        Create coord array with boundary labels.

        Format: [id, x, y, verticalGDL, horizontalGDL]

        Boundary Labels:
        - verticalGDL: 0=interior, 1=top, 2=bottom
        - horizontalGDL: 0=interior, 3=right, 4=left, 5=hole

        Args:
            boundaries: Boundary classification dictionary

        Returns:
            coord array (n_nodes, 5)
        """
        print(f"\n  Creating coord array with boundary labels...")

        coord = np.zeros((self.n_nodes, 5))

        # Column 0: Node ID (1-indexed for EUCLID compatibility)
        coord[:, 0] = np.arange(1, self.n_nodes + 1)

        # Columns 1-2: x, y coordinates
        coord[:, 1:3] = self.points

        # Column 3: verticalGDL
        for node_id in boundaries['bottom']:
            coord[node_id, 3] = 2
        for node_id in boundaries['top']:
            coord[node_id, 3] = 1

        # Column 4: horizontalGDL
        for node_id in boundaries['left']:
            coord[node_id, 4] = 4
        for node_id in boundaries['right']:
            coord[node_id, 4] = 3

        # Hole boundaries (all get label 5 for now)
        # For multiple holes, could cluster by proximity
        for node_id in boundaries['holes']:
            coord[node_id, 4] = 5

        print(f"      Applied boundary labels")

        return coord

    def save(self, coord: np.ndarray, conne: np.ndarray, output_dir: Path):
        """
        Save coord.csv and conne.txt files.

        Args:
            coord: Coordinate array (n_nodes, 5)
            conne: Connectivity array (n_elements, 3) [0-indexed]
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Saving EUCLID format files to: {output_dir}")

        # Save coord.csv
        coord_file = output_dir / "coord.csv"
        np.savetxt(coord_file, coord, delimiter=',',
                   fmt='%d,%.6f,%.6f,%d,%d',
                   header='id,x,y,verticalGDL,horizontalGDL',
                   comments='')
        print(f"      coord.csv: {len(coord)} nodes")

        # Save conne.txt (convert to 1-indexed)
        conne_file = output_dir / "conne.txt"
        conne_1indexed = np.column_stack([
            np.arange(1, len(conne) + 1),
            conne + 1  # Convert to 1-indexed
        ])
        np.savetxt(conne_file, conne_1indexed, delimiter=',',
                   fmt='%d,%d,%d,%d',
                   header='elem_id,n1,n2,n3',
                   comments='')
        print(f"      conne.txt: {len(conne)} elements")


# ========== Testing Code ==========
if __name__ == "__main__":
    """Test mesh converter with generated meshes"""

    import sys
    from pathlib import Path

    print("="*70)
    print("MESH CONVERTER TEST")
    print("="*70)

    # Test with experiment 804
    exp_id = 804
    mesh_file = Path(f"test_output/{exp_id}/geometry_exp{exp_id}.vtk")

    if not mesh_file.exists():
        print(f"\n[ERROR] Mesh file not found: {mesh_file}")
        print("Run geometry_builder.py first to generate meshes.")
        sys.exit(1)

    try:
        # Load mesh
        print(f"\n[1/3] Loading mesh from: {mesh_file}")
        mesh = meshio.read(mesh_file)
        print(f"  Loaded: {len(mesh.points)} nodes, {len(mesh.cells_dict.get('triangle', []))} elements")

        # Convert
        print(f"\n[2/3] Converting to EUCLID format...")
        converter = MeshConverter(mesh, domain_width=20.0, domain_height=60.0)
        coord, conne = converter.convert()

        # Verify
        print(f"\n  Verification:")
        print(f"    coord shape: {coord.shape} (expected: (n_nodes, 5))")
        print(f"    conne shape: {conne.shape} (expected: (n_elements, 3))")
        print(f"    Node IDs: {coord[:, 0].min():.0f} to {coord[:, 0].max():.0f}")
        print(f"    Boundary labels present:")
        print(f"      verticalGDL: {np.unique(coord[:, 3].astype(int))}")
        print(f"      horizontalGDL: {np.unique(coord[:, 4].astype(int))}")

        # Save
        print(f"\n[3/3] Saving files...")
        output_dir = Path(f"test_output/{exp_id}")
        converter.save(coord, conne, output_dir)

        print("\n" + "="*70)
        print("MESH CONVERSION COMPLETE")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  {output_dir}/coord.csv")
        print(f"  {output_dir}/conne.txt")

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
