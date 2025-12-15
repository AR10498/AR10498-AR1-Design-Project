import math
import numpy as np
from computation_tools import compute_member_lengths

# ==========================
#   STRUCTURAL ANALYSIS - Reactions
# ==========================
def reset_reactions_and_loads(nodes):
    """
    Clear only the stored reaction data on nodes.
    Does NOT touch applied loads (Fx, Fy, Fx_total, Fy_total, etc.)
    """
    for n in nodes.values():
        n.Rx = 0.0
        n.Ry = 0.0

def compute_support_reactions(nodes, verbose=True):
    """
    Compute 2D support reactions satisfying ΣFx=0, ΣFy=0, ΣM=0.
    Solves moment equilibrium first.
    Uses moment convention: M = x*Fy + y*Fx
    Respects support types:
        Reaction_Fixed    -> Rx & Ry
        Reaction_Pinned   -> Rx & Ry
        Reaction_Roller_H -> Ry only
        Reaction_Roller_V -> Rx only
    """
    # --- Reset previous reactions and total loads ---
    reset_reactions_and_loads(nodes)
    
    # --- Step 1: Identify supports ---
    supports = {label: n for label, n in nodes.items()
                if getattr(n, "condition", "").startswith("Reaction")}
    reactions = {label: {"Rx": 0.0, "Ry": 0.0} for label in supports}

    if len(supports) < 1:
        print("! No supports found. Returning zeros.")
        return reactions

    # --- Step 2: Restrained directions ---
    def restrained(cond):
        if cond == "Reaction_Fixed":      return True, True
        if cond == "Reaction_Pinned":     return True, True
        if cond == "Reaction_Roller_H":   return False, True
        if cond == "Reaction_Roller_V":   return True, False
        return False, False  # internal joints

    # --- Step 3: Sum applied forces ---
    total_Fx = sum(getattr(n, "F_xtotal", getattr(n, "F_x", 0.0)) for n in nodes.values())
    total_Fy = sum(getattr(n, "F_ytotal", getattr(n, "F_y", 0.0)) for n in nodes.values())

    # --- Step 4: Reference point for moments ---
    labels = list(supports.keys())
    ref_label = labels[0]
    x_ref, y_ref = supports[ref_label].coords[:2]

    # --- Step 5: Total applied moment (xFy + yFx) ---
    M_loads = 0.0
    for n in nodes.values():
        Fx = n.F_xtotal
        Fy = n.F_ytotal
        x, y = n.coords[:2]
        dx, dy = x - x_ref, y - y_ref
        Mz = dx * Fy + dy * Fx
        M_loads += Mz

    # --- Step 6: Build unknowns only for restrained directions ---
    unknowns = []
    label_to_vars = {}
    for label, s in supports.items():
        Rx_rest, Ry_rest = restrained(s.condition)
        label_to_vars[label] = {"Rx": None, "Ry": None}
        if Rx_rest:
            label_to_vars[label]["Rx"] = len(unknowns)
            unknowns.append(f"{label}_Rx")
        if Ry_rest:
            label_to_vars[label]["Ry"] = len(unknowns)
            unknowns.append(f"{label}_Ry")

    n_unknowns = len(unknowns)
    if n_unknowns == 0:
        print("! No restrained directions. Returning zeros.")
        return reactions

    # --- Step 7: Assemble equilibrium equations ---
    A = np.zeros((3, n_unknowns))
    b = np.zeros(3)

    # ΣFx = 0
    for label, vars in label_to_vars.items():
        if vars["Rx"] is not None:
            A[0, vars["Rx"]] = 1.0
    b[0] = -total_Fx

    # ΣFy = 0
    for label, vars in label_to_vars.items():
        if vars["Ry"] is not None:
            A[1, vars["Ry"]] = 1.0
    b[1] = -total_Fy

    # ΣM = 0 about reference
    for label, vars in label_to_vars.items():
        x, y = supports[label].coords[:2]
        dx, dy = x - x_ref, y - y_ref
        if vars["Rx"] is not None:
            A[2, vars["Rx"]] = dy
        if vars["Ry"] is not None:
            A[2, vars["Ry"]] = dx
    b[2] = -M_loads

    # --- Step 8: Solve ---
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    for i, name in enumerate(unknowns):
        label, comp = name.split("_")
        reactions[label][comp] = sol[i]

    # --- Step 9: Equilibrium check ---
    Fx_check = total_Fx + sum(r["Rx"] for r in reactions.values())
    Fy_check = total_Fy + sum(r["Ry"] for r in reactions.values())
    M_check = M_loads + sum(
        (supports[l].coords[0] - x_ref) * r["Ry"] +
        (supports[l].coords[1] - y_ref) * r["Rx"]
        for l, r in reactions.items()
    )

    if verbose:
        print("\n--- Support Reactions ---")
        for label, r in reactions.items():
            print(f"{label}: Rx={r['Rx']:+.3f}, Ry={r['Ry']:+.3f}")
        print("\n--- Equilibrium Check ---")
        print(f"ΣFx = {Fx_check:+.3e}")
        print(f"ΣFy = {Fy_check:+.3e}")
        print(f"ΣM  = {M_check:+.3e} (about {ref_label})")

    for label,comp in reactions.items():
        nodes[label].R_x = comp['Rx']
        nodes[label].R_y = comp['Ry']

    return reactions




# ======================
# STRUCTURAL ANALYSIS - Calculate Axial Forces
# ======================
def solve_truss_joint_iteration(nodes, members, tol_length=1e-12, max_iter=100):
    """
    Solve truss member forces using joint iteration method.
    Accounts for:
      - Point loads at nodes (Fx, Fy)
      - Equivalent nodal loads from member UDLs (Fx_total, Fy_total)
      - Support reactions
    Returns dict: {member_name: force_value}
    """
    member_forces = {mname: None for mname in members}

    # Map node labels to connected members
    node_connections = {n.label: [] for n in nodes.values()}
    for m in members.values():
        node_connections[m.node_start().label].append(m)
        node_connections[m.node_end().label].append(m)

    # Total forces at nodes = point loads + UDLs + reactions
    node_forces = {}
    for n in nodes.values():
        Fx = n.F_xtotal + n.R_x
        Fy = n.F_ytotal + n.R_y
        node_forces[n.label] = np.array([Fx, Fy])

    for iteration in range(max_iter):
        progress = False
        for n_label, connected_members in node_connections.items():
            node = next(nd for nd in nodes.values() if nd.label == n_label)
            unknowns = [m for m in connected_members if member_forces[m.name] is None]
            if not unknowns:
                continue
            if len(unknowns) > 2:
                continue  # skip complex nodes for now

            # Helper: unit direction vector from node along member
            def dir_cos(m):
                other = m.node_end() if m.node_start().label == n_label else m.node_start()
                dx = other.coords[0] - node.coords[0]
                dy = other.coords[1] - node.coords[1]
                L = math.hypot(dx, dy)
                if L < tol_length:
                    return None
                return dx / L, dy / L

            # --- One unknown member ---
            if len(unknowns) == 1:
                m = unknowns[0]
                d = dir_cos(m)
                if not d:
                    continue
                dx, dy = d
                sum_fx = sum_fy = 0.0
                for km in connected_members:
                    Fk = member_forces[km.name]
                    if Fk is None:
                        continue
                    dkm = dir_cos(km)
                    if not dkm:
                        continue
                    sum_fx += Fk * dkm[0]
                    sum_fy += Fk * dkm[1]

                # Solve along member direction
                if abs(dx) > tol_length:
                    F = (node_forces[n_label][0] - sum_fx) / dx
                else:
                    F = (node_forces[n_label][1] - sum_fy) / dy
                member_forces[m.name] = F
                progress = True

            # --- Two unknown members ---
            elif len(unknowns) == 2:
                m1, m2 = unknowns
                d1 = dir_cos(m1)
                d2 = dir_cos(m2)
                if not d1 or not d2:
                    continue
                dx1, dy1 = d1
                dx2, dy2 = d2
                A = np.array([[dx1, dx2],
                              [dy1, dy2]])
                sum_fx = sum_fy = 0.0
                for km in connected_members:
                    Fk = member_forces[km.name]
                    if Fk is None:
                        continue
                    dkm = dir_cos(km)
                    sum_fx += Fk * dkm[0]
                    sum_fy += Fk * dkm[1]

                b = node_forces[n_label] - np.array([sum_fx, sum_fy])
                if abs(np.linalg.det(A)) < tol_length:
                    continue  # skip singular cases
                F1, F2 = np.linalg.solve(A, b)
                member_forces[m1.name] = F1
                member_forces[m2.name] = F2

                progress = True

        if not progress:
            break

    for mname,mforce in member_forces.items():
        members[mname].force = mforce
    
    return member_forces

# ======================
# STRUCTURAL ANALYSIS - Calculate Bending Moment
# ======================
def compute_member_bending_moments(members):
    """
    Compute bending moments along members using UDLs and node conditions.
    - members: list of Member objects

    Returns dict {member_name: {'M_start', 'M_end', 'M_max', 'M_min', 'points'}}
    Sagging is positive and plotted perpendicular to the load direction.
    """
    moments = {}
    lengths = compute_member_lengths(members)

    for m in members.values():
        L = lengths[m.name]
        Fx_udl = m.F_x
        Fy_udl = m.F_y

        # Determine end moments based on node condition
        if "Fixed" in getattr(m.node_start, 'condition', ''):
            M_start = (Fy_udl*L**2/12 if abs(Fy_udl) > 1e-12 else Fx_udl*L**2/12)
        else:
            M_start = 0.0

        if "Fixed" in getattr(m.node_end, 'condition', ''):
            M_end = (Fy_udl*L**2/12 if abs(Fy_udl) > 1e-12 else Fx_udl*L**2/12)
        else:
            M_end = 0.0

        # Maximum moment at midspan for simply supported member with UDL
        mid_M = (Fy_udl*L**2/8 if abs(Fy_udl) > 1e-12 else Fx_udl*L**2/8)

        # Store min/max moments
        M_values = [M_start, M_end, mid_M]
        M_max = max(M_values)
        M_min = min(M_values)

        # Smooth quadratic moment curve along member
        n_points = 20
        xs_frac = np.linspace(0, 1, n_points)
        M_curve = []
        for frac in xs_frac:
            # superposition: linear interpolation of end moments + parabolic UDL effect
            if abs(Fy_udl) > 1e-12:
                Mx = M_start*(1-frac) + M_end*frac + Fy_udl*L**2*frac*(1-frac)/2
            elif abs(Fx_udl) > 1e-12:
                Mx = M_start*(1-frac) + M_end*frac + Fx_udl*L**2*frac*(1-frac)/2
            else:
                Mx = 0.0
            M_curve.append(Mx)

        momentvalues = {
            'M_start': M_start,
            'M_end': M_end,
            'M_max': M_max,
            'M_min': M_min,
            'points': M_curve,
            'Fx_udl': Fx_udl,
            'Fy_udl': Fy_udl
        }
        moments[m.name] = momentvalues
        m.moments = momentvalues
        

    return moments
