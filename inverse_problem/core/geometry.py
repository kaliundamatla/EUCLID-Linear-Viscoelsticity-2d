"""
Geometry handling for EUCLID inverse problem.
Defines mesh structure, nodes, and finite elements.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class Node:
    """
    Represents a single node in the mesh.
    """
    
    def __init__(self, node_id: int, x: float, y: float, boundary_type: Optional[str] = None, hole_label: Optional[int] = None):
        """
        Create a node.

        Args:
            node_id: Unique node identifier (0-indexed)
            x, y: Spatial coordinates
            boundary_type: Optional boundary classification ('top', 'bottom', 'left', 'right', 'hole')
            hole_label: Optional hole identifier (5, 7, 9, ...) for nodes on hole boundaries
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.boundary_type = boundary_type
        self.hole_label = hole_label
    
    def __repr__(self):
        return f"Node({self.id}, x={self.x:.3f}, y={self.y:.3f}, boundary={self.boundary_type})"


class Element(ABC):
    """
    Abstract base class for finite elements.
    All element types must implement compute_geometry().
    """
    
    def __init__(self, element_id: int, nodes: List[Node]):
        """
        Create an element.
        
        Args:
            element_id: Unique element identifier
            nodes: List of Node objects that define this element
        """
        self.id = element_id
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes  # 2D: each node has x and y displacement
        
        # Geometry quantities (computed by subclass)
        self.detJ: Optional[float] = None        # Jacobian determinant
        self.Be: Optional[np.ndarray] = None     # Strain-displacement matrix
        self.Bd: Optional[np.ndarray] = None     # Deviatoric projection of Be
        self.b: Optional[np.ndarray] = None      # Volumetric projection vector
        
        # Compute geometry (calls subclass implementation)
        self._compute_geometry()
    
    @abstractmethod
    def _compute_geometry(self):
        """
        Compute element geometry matrices.
        Must be implemented by subclass.
        """
        pass
    
    def get_nodal_coords(self) -> np.ndarray:
        """
        Returns nodal coordinates as (n_nodes, 2) array.
        
        Returns:
            Array of [x, y] coordinates for each node
        """
        return np.array([[node.x, node.y] for node in self.nodes])
    
    def get_global_dofs(self) -> np.ndarray:
        """
        Returns global DOF indices for this element.
        
        For 2D: [node0_x, node0_y, node1_x, node1_y, ...]
        
        Returns:
            Array of global DOF indices
        """
        dofs = []
        for node in self.nodes:
            dofs.extend([2 * node.id, 2 * node.id + 1])
        return np.array(dofs, dtype=np.int64)
    
    def __repr__(self):
        node_ids = [n.id for n in self.nodes]
        return f"{self.__class__.__name__}({self.id}, nodes={node_ids})"


class Triangle3Node(Element):
    """
    Linear 3-node triangular element for plane stress.
    
    Extracted from trail_inv.py Block 5.
    Uses constant strain (CST) formulation.
    """
    
    def _compute_geometry(self):
        """
        Compute Be, Bd, b, detJ for 3-node triangle.
        
        This is the EXACT logic from trail_inv.py lines ~250-320.
        """
        if self.n_nodes != 3:
            raise ValueError(f"Triangle3Node requires 3 nodes, got {self.n_nodes}")
        
        # Get nodal coordinates
        coords = self.get_nodal_coords()
        X = coords[:, 0]  # x-coordinates of 3 nodes
        Y = coords[:, 1]  # y-coordinates of 3 nodes
        
        # Shape function derivatives in natural coordinates (ξ, η)
        # For 3-node triangle at centroid (1/3, 1/3)
        N_xi = np.array([-1, 1, 0])
        N_eta = np.array([-1, 0, 1])
        
        # Jacobian matrix: J = [∂x/∂ξ  ∂y/∂ξ]
        #                      [∂x/∂η  ∂y/∂η]
        Jmat = np.zeros((2, 2))
        Jmat[0, 0] = N_xi @ X
        Jmat[0, 1] = N_xi @ Y
        Jmat[1, 0] = N_eta @ X
        Jmat[1, 1] = N_eta @ Y
        
        # Jacobian determinant
        self.detJ = np.linalg.det(Jmat)
        
        if self.detJ <= 0:
            raise ValueError(
                f"Element {self.id}: Negative or zero Jacobian determinant ({self.detJ:.6e}). "
                f"Check node ordering (should be counter-clockwise)."
            )
        
        # Inverse Jacobian
        Jm1 = (1.0 / self.detJ) * np.array([
            [Jmat[1, 1], -Jmat[0, 1]],
            [-Jmat[1, 0], Jmat[0, 0]]
        ])
        
        # Shape function gradients in global coordinates
        # ∂N/∂x = J⁻¹ @ [∂N/∂ξ, ∂N/∂η]ᵀ
        N_grads = []
        for i in range(3):
            N_grad = Jm1 @ np.array([N_xi[i], N_eta[i]])
            N_grads.append(N_grad)
        
        # Strain-displacement matrix Be (3 × 6 for plane stress)
        # ε = [εxx, εyy, γxy]ᵀ = Be @ u_element
        self.Be = np.zeros((3, 6))
        for i in range(3):
            N_x, N_y = N_grads[i]
            self.Be[0, 2*i] = N_x       # εxx: ∂u/∂x
            self.Be[1, 2*i+1] = N_y     # εyy: ∂v/∂y
            self.Be[2, 2*i] = N_y       # γxy: ∂u/∂y + ∂v/∂x
            self.Be[2, 2*i+1] = N_x
        
        # Projection matrices for plane stress
        m = np.array([1, 1, 0]).reshape(-1, 1)  # Trace vector
        Idev = np.eye(3) - 0.5 * (m @ m.T)      # Deviatoric projector
        
        # Deviatoric strain-displacement matrix
        self.Bd = Idev @ self.Be
        
        # Volumetric projection vector (b = mᵀ @ Be)
        self.b = (m.T @ self.Be).flatten()


class Triangle6Node(Element):
    """
    Quadratic 6-node triangular element (future implementation).
    
    Placeholder for future enhancement.
    Mid-side nodes provide quadratic interpolation.
    """
    
    def _compute_geometry(self):
        """
        TODO: Implement 6-node triangle geometry.
        
        Will use:
        - Quadratic shape functions
        - 3-point Gauss integration
        - Mid-side nodes for better accuracy
        """
        raise NotImplementedError(
            "Triangle6Node not yet implemented. "
            "Use Triangle3Node for now."
        )


class Mesh:
    """
    Manages the finite element mesh (nodes and elements).
    """
    
    def __init__(self, coord: np.ndarray, conne: np.ndarray, element_type: str = "Triangle3Node"):
        """
        Create mesh from coordinate and connectivity arrays.
        
        Args:
            coord: Node coordinates array (nNodes, 3+) [id, x, y, boundary_flags...]
            conne: Connectivity array (nElements, 3) [node_id0, node_id1, node_id2] (0-indexed!)
            element_type: Type of element to use ("Triangle3Node" or "Triangle6Node")
        """
        self.coord = coord
        self.conne = conne
        self.element_type = element_type
        
        self.nodes: List[Node] = []
        self.elements: List[Element] = []
        
        # Create nodes and elements
        self._create_nodes()
        self._create_elements()
        
        print(f"✓ Mesh created: {self.n_nodes} nodes, {self.n_elements} elements ({element_type})")
    
    def _create_nodes(self):
        """Create Node objects from coordinate array."""
        n_nodes = self.coord.shape[0]

        for i in range(n_nodes):
            node_id = int(self.coord[i, 0]) - 1  # Convert to 0-indexed
            x = self.coord[i, 1]
            y = self.coord[i, 2]

            # Determine boundary type from coord columns 3-4 if present
            boundary_type = None
            hole_label = None

            if self.coord.shape[1] > 3:
                if self.coord[i, 3] == 1:
                    boundary_type = 'top'
                elif self.coord[i, 3] == 2:
                    boundary_type = 'bottom'
                elif self.coord.shape[1] > 4:
                    if self.coord[i, 4] == 3:
                        boundary_type = 'right'
                    elif self.coord[i, 4] == 4:
                        boundary_type = 'left'
                    # Detect hole boundaries (labels 5, 7, 9, ...)
                    elif self.coord[i, 4] >= 5 and int(self.coord[i, 4]) % 2 == 1:
                        boundary_type = 'hole'
                        hole_label = int(self.coord[i, 4])

            node = Node(node_id, x, y, boundary_type, hole_label)
            self.nodes.append(node)
    
    def _create_elements(self):
        """Create Element objects from connectivity array."""
        n_elements = self.conne.shape[0]
        
        # Select element class based on type
        if self.element_type == "Triangle3Node":
            ElementClass = Triangle3Node
        elif self.element_type == "Triangle6Node":
            ElementClass = Triangle6Node
        else:
            raise ValueError(f"Unknown element type: {self.element_type}")
        
        for e in range(n_elements):
            node_ids = self.conne[e, :].astype(np.int64)
            element_nodes = [self.nodes[nid] for nid in node_ids]
            
            try:
                element = ElementClass(e, element_nodes)
                self.elements.append(element)
            except ValueError as err:
                print(f"Warning: Failed to create element {e}: {err}")
                # Continue with other elements
        
        if len(self.elements) < n_elements:
            print(f"Warning: Only {len(self.elements)}/{n_elements} elements created successfully")
    
    def get_boundary_nodes(self, boundary_type: str) -> List[Node]:
        """
        Get all nodes on a specific boundary.

        Args:
            boundary_type: 'top', 'bottom', 'left', 'right', or 'hole'

        Returns:
            List of Node objects on that boundary
        """
        return [node for node in self.nodes if node.boundary_type == boundary_type]

    def get_hole_boundary_nodes(self, hole_label: Optional[int] = None) -> List[Node]:
        """
        Get nodes on hole boundaries.

        Args:
            hole_label: Specific hole label (5, 7, 9, ...) or None for all holes

        Returns:
            List of Node objects on hole boundary/boundaries
        """
        if hole_label is None:
            return [node for node in self.nodes if node.boundary_type == 'hole']
        else:
            return [node for node in self.nodes if node.boundary_type == 'hole' and node.hole_label == hole_label]

    def get_unique_hole_labels(self) -> List[int]:
        """
        Get all unique hole labels present in the mesh.

        Returns:
            Sorted list of hole labels (e.g., [5, 7, 9])
        """
        labels = set()
        for node in self.nodes:
            if node.boundary_type == 'hole' and node.hole_label is not None:
                labels.add(node.hole_label)
        return sorted(labels)
    
    def get_boundary_dofs(self, boundary_type: str, direction: Optional[str] = None) -> np.ndarray:
        """
        Get DOF indices for nodes on a boundary.
        
        Args:
            boundary_type: 'top', 'bottom', 'left', 'right'
            direction: 'x', 'y', or None (both directions)
        
        Returns:
            Array of global DOF indices
        """
        boundary_nodes = self.get_boundary_nodes(boundary_type)
        dofs = []
        
        for node in boundary_nodes:
            if direction is None or direction == 'x':
                dofs.append(2 * node.id)
            if direction is None or direction == 'y':
                dofs.append(2 * node.id + 1)
        
        return np.array(dofs, dtype=np.int64)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in mesh."""
        return len(self.nodes)
    
    @property
    def n_elements(self) -> int:
        """Number of elements in mesh."""
        return len(self.elements)
    
    def __repr__(self):
        return f"Mesh: {self.n_nodes} nodes, {self.n_elements} {self.element_type} elements"


# ========== Testing Code ==========
if __name__ == "__main__":
    """
    Test mesh creation with experiment 713 data.
    Run: python geometry.py
    """
    from data import ExperimentData
    
    try:
        # Load data
        exp_data = ExperimentData(998)
        
        # Create mesh
        mesh = Mesh(exp_data.coord, exp_data.conne)
        
        print("\n" + "="*60)
        print("MESH TEST")
        print("="*60)
        print(mesh)
        
        # Test element 0
        elem0 = mesh.elements[0]
        print(f"\nElement 0:")
        print(f"  Nodes: {[n.id for n in elem0.nodes]}")
        print(f"  detJ: {elem0.detJ:.6f}")
        print(f"  Be shape: {elem0.Be.shape}")
        print(f"  Bd shape: {elem0.Bd.shape}")
        print(f"  b shape: {elem0.b.shape}")
        
        # Test boundary identification
        top_nodes = mesh.get_boundary_nodes('top')
        bottom_nodes = mesh.get_boundary_nodes('bottom')
        print(f"\nBoundary nodes:")
        print(f"  Top: {len(top_nodes)} nodes")
        print(f"  Bottom: {len(bottom_nodes)} nodes")
        
        if len(top_nodes) > 0:
            print(f"  First top node: {top_nodes[0]}")
        
        # Test DOF extraction
        top_dofs = mesh.get_boundary_dofs('top', direction='y')
        print(f"\nTop Y DOFs: {len(top_dofs)} DOFs")
        print(f"  First 5: {top_dofs[:5]}")
        
        print("\n✓ geometry.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()