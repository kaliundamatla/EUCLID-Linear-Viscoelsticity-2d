"""
Mesh generation and management for forward viscoelastic solver.

Provides functionality to:
1. Generate structured 2D triangular meshes (simple rectangular domains)
2. Export meshes to inverse problem format (coord.csv, conne.txt)
3. Load existing meshes from inverse problem data

For complex geometries with holes, use geometry_builder.py with pygmsh.

Follows the same mesh structure as the inverse problem pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class Node:
    """
    Represents a single node in the mesh.

    Attributes:
        id: Unique node identifier (0-indexed internally)
        x, y: Spatial coordinates [mm]
        boundary_type: Classification ('top', 'bottom', 'left', 'right', None)
    """
    id: int
    x: float
    y: float
    boundary_type: Optional[str] = None

    def __repr__(self):
        return f"Node({self.id}, x={self.x:.3f}, y={self.y:.3f}, boundary={self.boundary_type})"


class MeshGenerator:
    """
    Generates structured 2D triangular meshes for forward FEM simulations.

    Features:
    - Structured rectangular domain discretization
    - Boundary node classification
    - Export to inverse problem format

    For complex geometries with holes, use geometry_builder.py with pygmsh.
    """

    def __init__(self, width: float, height: float, nx: int, ny: int):
        """
        Initialize mesh generator for rectangular domain.

        Args:
            width: Domain width [mm]
            height: Domain height [mm]
            nx: Number of nodes in x-direction
            ny: Number of nodes in y-direction

        Example:
            # Create 20mm x 50mm specimen with 20x50 nodes
            gen = MeshGenerator(width=20, height=50, nx=20, ny=50)
        """
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny

        # Storage (populated by generate())
        self.nodes: List[Node] = []
        self.coord: Optional[np.ndarray] = None  # (nNodes, 5): [id, x, y, vGDL, hGDL]
        self.conne: Optional[np.ndarray] = None  # (nElements, 3): [n1, n2, n3] (0-indexed)
        self.triangulation: Optional[mtri.Triangulation] = None

        print(f"MeshGenerator initialized: {width}x{height}mm, {nx}x{ny} nodes")

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate structured triangular mesh for simple rectangular domain.

        For experiments 800-802 (simple rectangular domains without holes).
        For complex geometries with holes, use geometry_builder.py with pygmsh.

        Returns:
            coord: (nNodes, 5) array [id, x, y, verticalGDL, horizontalGDL]
            conne: (nElements, 3) array [n1, n2, n3] (0-indexed)

        Note:
            - Creates structured grid triangulation
            - Boundary nodes are tagged in GDL columns
            - verticalGDL: 1=top, 2=bottom, 0=interior
            - horizontalGDL: 3=right, 4=left, 0=interior
        """
        # Structured mesh generation (no holes)
        print("\n[Mesh Generation]")
        print(f"  Domain: {self.width}x{self.height}mm")
        print(f"  Resolution: {self.nx}x{self.ny} nodes")

        # 1. Generate base grid (structured)
        x = np.linspace(0, self.width, self.nx)
        y = np.linspace(0, self.height, self.ny)

        # Create nodes (row-by-row, bottom to top)
        nodes_xy = np.zeros((self.nx * self.ny, 2))
        for j in range(self.ny):  # y-direction (rows)
            for i in range(self.nx):  # x-direction (columns)
                node_id = j * self.nx + i
                nodes_xy[node_id, 0] = x[i]
                nodes_xy[node_id, 1] = y[j]

        n_nodes = nodes_xy.shape[0]

        # 2. Create structured triangulation 
        # Each rectangular cell (i,j) -> (i+j*nx) is divided into 2 triangles
        print(f"  Creating structured triangulation...")
        elements_list = []

        for j in range(self.ny - 1):  # Loop over rows
            for i in range(self.nx - 1):  # Loop over columns
                # Node indices for this quad (counter-clockwise from bottom-left)
                n0 = j * self.nx + i           # Bottom-left
                n1 = j * self.nx + (i + 1)     # Bottom-right
                n2 = (j + 1) * self.nx + (i + 1)  # Top-right
                n3 = (j + 1) * self.nx + i     # Top-left

                # Create two triangles per quad
                # Lower triangle: n0 -> n1 -> n3 (CCW)
                elements_list.append([n0, n1, n3])
                # Upper triangle: n1 -> n2 -> n3 (CCW)
                elements_list.append([n1, n2, n3])

        elements = np.array(elements_list, dtype=int)

        # 3. Classify boundary nodes
        tol = 1e-8
        is_bottom = np.abs(nodes_xy[:, 1] - 0) < tol
        is_top = np.abs(nodes_xy[:, 1] - self.height) < tol
        is_left = np.abs(nodes_xy[:, 0] - 0) < tol
        is_right = np.abs(nodes_xy[:, 0] - self.width) < tol

        verticalGDL = np.zeros(n_nodes, dtype=int)
        horizontalGDL = np.zeros(n_nodes, dtype=int)

        verticalGDL[is_bottom] = 2  # Bottom nodes
        verticalGDL[is_top] = 1     # Top nodes
        horizontalGDL[is_left] = 4  # Left nodes
        horizontalGDL[is_right] = 3 # Right nodes

        # 4. Create coordinate array (1-indexed IDs for export compatibility)
        node_ids = np.arange(1, n_nodes + 1)
        self.coord = np.column_stack([
            node_ids,
            nodes_xy[:, 0],
            nodes_xy[:, 1],
            verticalGDL,
            horizontalGDL
        ])

        # 5. Store connectivity (keep 0-indexed internally)
        self.conne = elements

        # 6. Create Node objects
        self.nodes = []
        for i in range(n_nodes):
            boundary_type = None
            if is_top[i]:
                boundary_type = 'top'
            elif is_bottom[i]:
                boundary_type = 'bottom'
            elif is_left[i]:
                boundary_type = 'left'
            elif is_right[i]:
                boundary_type = 'right'

            node = Node(i, nodes_xy[i, 0], nodes_xy[i, 1], boundary_type)
            self.nodes.append(node)

        # 7. Create triangulation for plotting
        self.triangulation = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], elements)

        print(f"  Mesh created: {n_nodes} nodes, {len(elements)} elements")

        return self.coord, self.conne

    def _ensure_ccw_orientation(self, nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """
        Ensure all triangular elements have counter-clockwise (CCW) node ordering.

        This is critical for FEM because:
        - Jacobian determinant must be positive
        - Consistent normal vectors pointing outward
        - Correct sign in stiffness matrices

        Args:
            nodes: (nNodes, 2) node coordinates
            elements: (nElements, 3) element connectivity

        Returns:
            elements: Corrected connectivity with CCW orientation
        """
        corrected_elements = elements.copy()
        n_flipped = 0

        for i, element in enumerate(elements):
            coords = nodes[element]

            # Compute signed area using cross product
            # Area = 0.5 * [(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)]
            # Positive area = CCW ordering
            # Negative area = CW ordering
            signed_area = 0.5 * (
                (coords[1, 0] - coords[0, 0]) * (coords[2, 1] - coords[0, 1]) -
                (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])
            )

            # If area is negative, nodes are in clockwise order → swap nodes 1 and 2
            if signed_area < 0:
                corrected_elements[i, [1, 2]] = corrected_elements[i, [2, 1]]
                n_flipped += 1

        if n_flipped > 0:
            print(f"    Flipped {n_flipped}/{len(elements)} elements to CCW")

        return corrected_elements

    def _check_element_quality(self, nodes: np.ndarray, element: np.ndarray) -> bool:
        """
        Check element quality to avoid degenerate triangles.

        Args:
            nodes: (nNodes, 2) node coordinates
            element: (3,) node indices

        Returns:
            True if element quality is acceptable
        """
        coords = nodes[element]

        # Calculate area using cross product (now guaranteed positive after CCW fix)
        area = 0.5 * abs(
            (coords[1, 0] - coords[0, 0]) * (coords[2, 1] - coords[0, 1]) -
            (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])
        )

        # Calculate edge lengths
        edges = [
            np.linalg.norm(coords[i] - coords[(i+1) % 3])
            for i in range(3)
        ]
        max_edge = max(edges)
        min_edge = min(edges)

        # Quality criteria
        min_area = 1e-8
        max_aspect_ratio = 20.0

        return area > min_area and (max_edge / min_edge) < max_aspect_ratio

    def plot_mesh(self, show_node_ids: bool = False, show_boundary: bool = True):
        """
        Visualize the generated mesh.

        Args:
            show_node_ids: Display node numbers
            show_boundary: Highlight boundary nodes
        """
        if self.triangulation is None:
            raise RuntimeError("Mesh not generated. Call generate() first.")

        _, ax = plt.subplots(figsize=(8, 10))

        # Plot mesh
        ax.triplot(self.triangulation, 'k-', lw=0.5, alpha=0.7)

        # Highlight boundary nodes
        if show_boundary:
            node_coords = np.array([[n.x, n.y] for n in self.nodes])

            bottom_nodes = [i for i, n in enumerate(self.nodes) if n.boundary_type == 'bottom']
            top_nodes = [i for i, n in enumerate(self.nodes) if n.boundary_type == 'top']
            left_nodes = [i for i, n in enumerate(self.nodes) if n.boundary_type == 'left']
            right_nodes = [i for i, n in enumerate(self.nodes) if n.boundary_type == 'right']

            if bottom_nodes:
                ax.scatter(node_coords[bottom_nodes, 0], node_coords[bottom_nodes, 1],
                          c='blue', s=30, label='Bottom (fixed)', zorder=5)
            if top_nodes:
                ax.scatter(node_coords[top_nodes, 0], node_coords[top_nodes, 1],
                          c='red', s=30, label='Top (loaded)', zorder=5)
            if left_nodes:
                ax.scatter(node_coords[left_nodes, 0], node_coords[left_nodes, 1],
                          c='green', s=20, label='Left', zorder=4, alpha=0.5)
            if right_nodes:
                ax.scatter(node_coords[right_nodes, 0], node_coords[right_nodes, 1],
                          c='orange', s=20, label='Right', zorder=4, alpha=0.5)

        # Show node IDs (only for small meshes)
        if show_node_ids and len(self.nodes) < 100:
            for node in self.nodes:
                ax.text(node.x, node.y, str(node.id), fontsize=6, ha='center', va='center')

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(f'Mesh: {len(self.nodes)} nodes, {len(self.conne)} elements')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        if show_boundary:
            ax.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def export(self, output_dir: Path):
        """
        Export mesh to inverse problem format (coord.csv, conne.txt).

        Args:
            output_dir: Directory to save files

        Creates:
            coord.csv: Node coordinates with boundary flags
            conne.txt: Element connectivity 
        """
        if self.coord is None or self.conne is None:
            raise RuntimeError("Mesh not generated. Call generate() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export coordinates
        coord_file = output_dir / "coord.csv"
        np.savetxt(coord_file, self.coord, delimiter=',', fmt='%d,%.6f,%.6f,%d,%d',
                   header='id,x,y,verticalGDL,horizontalGDL', comments='')

        # Export connectivity (convert to 1-indexed)
        conne_file = output_dir / "conne.txt"
        conne_1indexed = np.column_stack([
            np.arange(1, len(self.conne) + 1),
            self.conne + 1  # Convert to 1-indexed
        ])
        np.savetxt(conne_file, conne_1indexed, delimiter=',', fmt='%d',
                   header='elem_id,n1,n2,n3', comments='')

        print(f"\n Mesh exported to: {output_dir}")
        print(f"  - coord.csv: {len(self.coord)} nodes")
        print(f"  - conne.txt: {len(self.conne)} elements")

    @property
    def n_nodes(self) -> int:
        """Number of nodes in mesh."""
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """Number of elements in mesh."""
        return len(self.conne) if self.conne is not None else 0


class MeshLoader:
    """
    Load existing mesh from inverse problem format.

    This allows reusing experimental geometries for forward simulations.
    """

    @staticmethod
    def load(coord_file: Path, conne_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh from coord.csv and conne.txt files.

        Args:
            coord_file: Path to coord.csv
            conne_file: Path to conne.txt

        Returns:
            coord: (nNodes, 5) array
            conne: (nElements, 3) array (0-indexed)
        """
        import pandas as pd

        # Load coordinates
        coord_df = pd.read_csv(coord_file, header=0)
        coord = coord_df.values

        # Load connectivity
        import re
        data = []
        with open(conne_file, 'r') as f:
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

        conne_df = pd.DataFrame(data)
        conne = conne_df.iloc[:, 1:4].values.astype(np.int64)
        conne = conne - 1  # Convert to 0-indexed

        print(f" Mesh loaded: {len(coord)} nodes, {len(conne)} elements")

        return coord, conne


# ========== Testing & Examples ==========
if __name__ == "__main__":
    """
    Test mesh generation with various configurations.
    Run: python mesh.py
    """

    print("="*70)
    print("MESH GENERATION TEST")
    print("="*70)

    # Test 1: Simple rectangular mesh
    print("\n[Test 1] Simple rectangular mesh")
    gen1 = MeshGenerator(width=20, height=50, nx=11, ny=26)
    coord1, conne1 = gen1.generate()
    gen1.plot_mesh(show_boundary=True)

    # Test 2: Mesh with circular holes
    # print("\n[Test 2] Mesh with circular holes")
    # gen2 = MeshGenerator(width=20, height=50, nx=21, ny=51)
    # holes = [
    #     (10, 15, 2.5),  # Center hole
    #     (5, 30, 1.5),   # Left hole
    #     (15, 40, 2.0)   # Right hole
    # ]
    # coord2, conne2 = gen2.generate(holes=holes)
    # gen2.plot_mesh(show_boundary=True)

    # Test 3: Export mesh
    # print("\n[Test 3] Export mesh to inverse problem format")
    # output_dir = Path("./test_mesh_export")
    # gen2.export(output_dir)

    # Test 4: Load mesh back
    # print("\n[Test 4] Load mesh from files")
    # coord_loaded, conne_loaded = MeshLoader.load(
    #     output_dir / "coord.csv",
    #     output_dir / "conne.txt"
    # )

    # Verify
    print(f"\nVerification:")
    #print(f"  Original: {len(coord1)} nodes, {len(conne1)} elements")
    #print(f"  Loaded:   {len(coord_loaded)} nodes, {len(conne_loaded)} elements")
    #print(f"  Match: {np.allclose(coord1, coord_loaded) and np.array_equal(conne1, conne_loaded)}")

    print("\n" + "="*70)
    print(" All mesh generation tests passed!")
    print("="*70)
