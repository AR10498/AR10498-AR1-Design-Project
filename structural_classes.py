import string
import math 

# ==========================
#   CLASSES
# ==========================
class Node:
    """
    Node object with coordinates ()
    """
    def __init__(self, id, coords, label=None, condition="Pinned"):
        """
        Initialises with default properties:
        F_x = 0.0 # Point load in x
        F_y = 0.0 # Point load in y
        R_x = 0.0 # Reaction in x
        R_y = 0.0 # Reaction in y
        """
        self.id = id
        self.coords = list(coords)
        nodelabels = [c * i for i in range(1, 4) for c in string.ascii_uppercase]
        self.label = label or nodelabels[id]
        self.condition = condition
        self.unknowns = 0
        self.F_x = 0.0
        self.F_y = 0.0
        self.R_x = 0.0
        self.R_y = 0.0        
        self.connectivity=[] # undirected connections


class Member:
    def __init__(self, id, node_start, node_end):
        self.id = id
        self.node_start = node_start
        self.node_end = node_end
        self.length = self.compute_member_length()
        self.rotation = 0
        self.name = "-".join(sorted([node_start.label, node_end.label]))
        self.material = None
        # Standardized property keys
        self.properties = {}
        # forces
        # Uniformly distributed load per member (global axes)
        self.F_x = 0.0 # Fx = horizontal UDL
        self.F_y = 0.0 # Fy = vertical UDL (positive down)        
        self.force = None  # will store solved force
        self.moment = {}

    # --- Member lengths ---
    def compute_member_length(self):
        dx = self.node_end.coords[0] - self.node_start.coords[0]
        dy = self.node_end.coords[1] - self.node_start.coords[1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            L = 0.0
        return L

    def vector_from(self, node):
        if node == self.node_start:
            other = self.node_end
        elif node == self.node_end:
            other = self.node_start
        else:
            return None
        dx = other.coords[0] - node.coords[0]
        dy = other.coords[1] - node.coords[1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            return None
        return (dx, dy)

        