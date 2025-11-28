import itertools
import math



# --- Member lengths ---
def compute_member_lengths(members):
    lengths = {}
    for m in members.values():
        dx = m.node_end.coords[0] - m.node_start.coords[0]
        dy = m.node_end.coords[1] - m.node_start.coords[1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            L = 0.0
        lengths[m.name] = L
    return lengths


def compute_node_angles(nodes, members):
    node_angles = {}
    for n in nodes.values():
        connected = [m for m in members.values() if n in (m.node_start, m.node_end)]
        if len(connected) < 2:
            continue
        node_angles[n.label] = []
        for m1, m2 in itertools.combinations(connected, 2):
            v1 = m1.vector_from(n)
            v2 = m2.vector_from(n)
            if not v1 or not v2:
                continue
            norm1 = math.hypot(*v1)
            norm2 = math.hypot(*v2)
            if norm1 * norm2 == 0:
                continue
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_theta = max(min(dot / (norm1*norm2), 1), -1)
            angle_deg = math.degrees(math.acos(cos_theta))
            node_angles[n.label].append((m1.name, m2.name, angle_deg, v1, v2))
    return node_angles
