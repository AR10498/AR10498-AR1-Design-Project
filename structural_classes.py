import string
import math 

# ==========================
#   CLASSES
# ==========================
class Node:
    """
    Node object with coordinates (x, y, z)
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
    def __init__(self, id, nodes, startnode, endnode):
        self.id = id
        self.nodes = nodes
        self.startnode = startnode
        self.endnode = endnode
        #self.length = self.compute_member_length()
        self.compute_length()
        self.rotation = 0
        self.name = "-".join(sorted([startnode, endnode]))
        self.material = None
        # Standardized property keys
        self.properties = {}
        # forces
        # Uniformly distributed load per member (global axes)
        self.F_x = 0.0 # Fx = horizontal UDL
        self.F_y = 0.0 # Fy = vertical UDL (positive down)        
        self.force = 0.0  # will store solved force
        self.moments = {}
        self.stress = 0.0
        self.utilisation = 0.0

    def node_start(self):
        return self.nodes[self.startnode]

    def node_end(self):
        return self.nodes[self.endnode]

    # --- Member lengths ---
    def compute_length(self):
        dx = self.node_end().coords[0] - self.node_start().coords[0]
        dy = self.node_end().coords[1] - self.node_start().coords[1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            L = 0.0
        self.length = L
        return L

    def vector_from(self, node):
        if node == self.node_start():
            other = self.node_end()
        elif node == self.node_end():
            other = self.node_start()
        else:
            return None
        dx = other.coords[0] - node.coords[0]
        dy = other.coords[1] - node.coords[1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            return None
        return (dx, dy)

        