import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def visualize_truss(
    nodes,
    members,
    stresses=None,
    utilisations=None,
    max_stress=None,
    max_util=None,
    textscale=1.0,
    reactions=None,
    plot_loads=False,
    forces=None,
    moments=None,
    show_stresses=False,
    show_utilisations=False,
    show_moments=False,
    show_forces=False,
    show_reaction_labels=True,
    show_node_labels=False,
    show_member_labels=False,
    show_force_labels=False,
    show_moment_labels=False,
    show_stress_labels=False,
    show_material_labels=False,
    show_load_labels=False,
    show_colorbar=False,
    signed_stresses=False,
    moment_scale=0.5
):
    """
    Unified visualization for truss/frame structure.
    Can display stresses, utilisations, or both.
    - show_stresses=True → stress color map
    - show_utilisations=True → rainbow utilization overlay
    """

    # Node symbol mapping
    mapping = {
        "Pinned": ("o", "b", "none"),           # marker, edgecolor, facecolor
        "Fixed": ("s", "b", "none"),
        "Reaction_Fixed": ("s", "k", "k"),
        "Reaction_Pinned": ("o", "k", "k"),
        "Reaction_Roller_H": ("^", "k", "k"),
        "Reaction_Roller_V": ("v", "k", "k")
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True)

    xs = [n.coords[0] for n in nodes.values()]
    ys = [n.coords[1] for n in nodes.values()]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    arrow_len = 0.1 * span
    head_width = 0.25 * arrow_len
    head_length = 0.25 * arrow_len
    text_offset = 0.03 * span / len(nodes)

    # --- Stress colormap setup ---

        # --- Ensure stresses/utilisations are dict-like ---
    if stresses is not None and not isinstance(stresses, dict):
        try:
            stresses = {m.name: s for m, s in zip(members.values(), stresses)}
        except Exception:
            stresses = {}

    if utilisations is not None and not isinstance(utilisations, dict):
        try:
            utilisations = {m.name: u for m, u in zip(members.values(), utilisations)}
        except Exception:
            utilisations = {}
    
    stress_cmap, stress_norm = None, None
    if show_stresses and stresses:
        #vmax = max(stresses.values())
        vmax = max(abs(max(stresses.values())), abs(min(stresses.values())))
        vmin = -vmax #
        #vmin = min(stresses.values())
        stress_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        stress_cmap = mcolors.LinearSegmentedColormap.from_list(
            "stressmap", [(0.0, "darkred"), (0.5, "lightgrey"), (1.0, "darkblue")]
        )

    # --- Utilisation colormap setup ---
    util_cmap, util_norm = None, None
    util_colors = {}
    if show_utilisations and utilisations:
        if max_util is None:
            max_util = max(utilisations.values())
        util_cap = max(1.5, max_util)
        util_norm = mcolors.Normalize(vmin=0, vmax=util_cap)
        util_cmap = plt.get_cmap("rainbow")
        util_colors = {mname: util_cmap(util_norm(min(u, util_cap))) for mname, u in zip(members, [utilisations[mname] for mname in members])}

    # --- Draw members and labels ---
    for m in members.values():
        x1, y1 = m.node_start.coords[:2]
        x2, y2 = m.node_end.coords[:2]

        # Base color for member
        col = "grey"
        if forces and not show_stresses:
            F = forces.get(m.name, 0.0)
            if F > 1e-6: col="blue"
            elif F < -1e-6: col="red"
       
        # Stress coloring
        if show_stresses and stresses and stress_cmap:
            sigma = stresses.get(m.name, 0.0)
            col = stress_cmap(stress_norm(sigma))

        # Utilisation overlay
        if show_utilisations and utilisations and m.name in util_colors:
            col = util_colors[m.name]


        # Draw straight member
        ax.plot([x1,x2],[y1,y2], color=col, lw=2)

        # --- Member labels ---
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        label = ""
        if show_member_labels: label += f"{m.name}"
        if show_material_labels and hasattr(m,"material") and m.material is not None:
            label += f"\nMat: {m.material}"
        if show_force_labels and forces and m.name in forces:
            F_val = forces[m.name]
            if abs(F_val) > 1e-6:
                label += f"\nF = {forces[m.name]:.1f}"
        if show_moment_labels and moments and m.name in moments:
            M_val = moments[m.name].get('M_max',0.0)
            if abs(M_val) > 1e-6:            
                label += f"\nM = {moments[m.name].get('M_max',0.0):.1f}"
        if show_stress_labels and stresses and m.name in stresses:
            sigma_val = stresses[m.name]
            if abs(sigma_val) > 1e-6:
                label += f"\nσ = {stresses[m.name]:.1f} MPa"
        if label:
            ax.text(mid_x, mid_y + text_offset,
                    label, rotation=30 ,fontsize=9*textscale, ha="center", va="center", color="purple")

    # --- Draw bending moment curves separately ---
    if show_moments and moments:
        for m in members.values():
            if m.name not in moments: continue
            M_curve = np.array(moments[m.name]['points'])
            sx = np.sign(moments[m.name]['Fx_udl']) #positive to right
            sy = -np.sign(moments[m.name]['Fy_udl']) #positive down
            if np.allclose(M_curve,0.0): continue

            x1, y1 = m.node_start.coords[:2]
            x2, y2 = m.node_end.coords[:2]
            dx, dy = abs(x2-x1), abs(y2-y1)
            L = math.hypot(dx,dy)
            if L < 1e-9: continue

            n_points = len(M_curve)
            xs_curve = np.linspace(x1,x2,n_points)
            ys_curve = np.linspace(y1,y2,n_points)
            nx, ny = dy/L, dx/L
            sagging = M_curve*moment_scale
            plot_xs = xs_curve + sx*nx*sagging
            plot_ys = ys_curve + sy*ny*sagging
            ax.plot(plot_xs, plot_ys, color="orange", lw=2, zorder=3)

    # --- Plot nodes ---
    for n in nodes.values():
        marker, edge, face = mapping.get(n.condition,("o","b","none"))
        ax.plot(n.coords[0], n.coords[1], marker=marker, markeredgecolor=edge, markerfacecolor=face, markersize=8)
        if show_node_labels:
            ax.text(n.coords[0]+0.02, n.coords[1]+0.02, n.label,
                    fontsize=10*textscale, weight="bold")

    # --- Plot node loads ---
    if plot_loads:
        for n in nodes.values():
            Fx = n.F_x
            Fy = n.F_y
            if Fx != 0:
                start_x = n.coords[0]-np.sign(Fx)*arrow_len
                ax.arrow(start_x,n.coords[1], np.sign(Fx)*arrow_len,0,
                         fc='red', ec='red', head_width=head_width, head_length=head_length,
                         length_includes_head=True)
                if show_load_labels:
                    ax.text(start_x-np.sign(Fx)*text_offset,n.coords[1],
                            f"Fx={Fx:+.1f}kN", color='red', fontsize=8*textscale, ha='center')
            if Fy != 0:
                dir_y = np.sign(-Fy)
                start_y = n.coords[1]-dir_y*arrow_len
                ax.arrow(n.coords[0], start_y, 0, dir_y*arrow_len,
                         fc='blue', ec='blue', head_width=head_width, head_length=head_length,
                         length_includes_head=True)
                if show_load_labels:
                    ax.text(n.coords[0], start_y-dir_y*text_offset,
                            f"Fy={Fy:+.1f}kN", color='red', fontsize=8*textscale, ha='center')

    # --- Plot Member UDLs (Global Orientation) ---
    if plot_loads:
        for m in members.values():    
            Fx_udl = m.F_x 
            Fy_udl = m.F_y
    
            if abs(Fx_udl) < 1e-6 and abs(Fy_udl) < 1e-6:
                continue
    
            # Member geometry
            x1, y1 = m.node_start.coords[:2]
            x2, y2 = m.node_end.coords[:2]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1e-9:
                continue
    
            # GLOBAL load direction
            qx, qy = Fx_udl, -Fy_udl  # +Fy downward
            qmag = math.hypot(qx, qy)
            if qmag == 0:
                continue
            qx, qy = qx / qmag, qy / qmag
    
            # Arrow parameters
            offset = 0.15 * arrow_len
            tail_length = 0.4 * arrow_len
            n_arrows = max(5, int(L / (0.3 * span)))
            tail_points = []
    
            for i in range(n_arrows + 1):
                frac = i / n_arrows
                px = x1 + frac * (x2 - x1)
                py = y1 + frac * (y2 - y1)
    
                # Offset in global direction
                tail_x = px - qx * (offset + tail_length)
                tail_y = py - qy * (offset + tail_length)
    
                ax.arrow(
                    tail_x,
                    tail_y,
                    qx * tail_length,
                    qy * tail_length,
                    fc='orange',
                    ec='orange',
                    head_width=head_width * 0.5,
                    head_length=head_length * 0.5,
                    length_includes_head=True,
                    zorder=2
                )
    
                tail_points.append((tail_x, tail_y))
    
            # Connect all arrow tails with a line
            tail_xs, tail_ys = zip(*tail_points)
            ax.plot(tail_xs, tail_ys, color='orange', lw=1, zorder=1)
    
            # --- Place label at the end of the center arrow tail ---
            mid_i = len(tail_points) // 2
            tail_x, tail_y = tail_points[mid_i]
    
            # Move label opposite to load direction (outside tail)
            label_offset = max(text_offset * arrow_len, 0.1)
            label_x = tail_x - qx * label_offset
            label_y = tail_y - qy * label_offset
    
            ax.text(
                label_x, label_y,
                f"{qmag:.1f} kN/m",
                color='orange',
                fontsize=8 * textscale,
                weight='bold',
                ha='center',
                va='center'
            )

    # --- Plot reactions ---
    if reactions:
        for label, R in reactions.items():
            if label not in nodes: 
                continue
            n = nodes[label]
            Rx, Ry, Mz = R.get('Rx', 0.0), R.get('Ry', 0.0), R.get('Mz', 0.0)
    
            # Horizontal reaction arrow
            if Rx != 0:
                dx = -Rx/abs(Rx)*arrow_len
                ax.arrow(n.coords[0]+dx, n.coords[1], -dx, 0,
                         fc='black', ec='black', head_width=head_width, head_length=head_length,
                         length_includes_head=True)
                if show_reaction_labels:
                    ax.text(n.coords[0]+dx-np.sign(dx)*text_offset, n.coords[1],
                            f"Rₓ={Rx:.1f} kN", color='black', fontsize=8*textscale, ha='center')
    
            # Vertical reaction arrow: always drawn below node
            vert_offset = arrow_len
            if Ry > 0:
                ax.arrow(n.coords[0], n.coords[1], 0, -vert_offset, fc='black', ec='black',
                         head_width=head_width, head_length=head_length, length_includes_head=True)
                if show_reaction_labels:
                    ax.text(n.coords[0], n.coords[1]-vert_offset-text_offset,
                            f"Rᵧ={Ry:.1f} kN", color='black', fontsize=8*textscale, ha='center', va='top')
            elif Ry < 0:
                ax.arrow(n.coords[0], n.coords[1]-vert_offset, 0, vert_offset, fc='black', ec='black',
                         head_width=head_width, head_length=head_length, length_includes_head=True)
                if show_reaction_labels:
                    ax.text(n.coords[0], n.coords[1]-vert_offset-text_offset,
                            f"Rᵧ={Ry:.1f} kN", color='black', fontsize=8*textscale, ha='center', va='top')
    
            # Moment reaction (Mz) at fixed supports
            if Mz != 0:
                # Draw a curved arrow representing moment
                r = 0.05 * max(ax.get_xlim()[1]-ax.get_xlim()[0], ax.get_ylim()[1]-ax.get_ylim()[0])
                theta = np.linspace(0, np.sign(Mz)*np.pi*0.8, 20)
                x_circle = n.coords[0] + r*np.cos(theta)
                y_circle = n.coords[1] + r*np.sin(theta)
                ax.plot(x_circle, y_circle, color='purple', lw=2)
                # Optional arrowhead at end
                ax.arrow(x_circle[-2], y_circle[-2],
                         x_circle[-1]-x_circle[-2], y_circle[-1]-y_circle[-2],
                         shape='full', lw=0, length_includes_head=True,
                         head_width=0.02*r, head_length=0.02*r, fc='purple', ec='purple')
                if show_reaction_labels:
                    ax.text(n.coords[0]+r, n.coords[1]+r,
                            f"M={Mz:.1f} kNm", color='purple', fontsize=8*textscale, ha='left', va='bottom')
    

    # --- Colorbars ---
    if show_colorbar:
        if show_stresses and stress_cmap:
            sm = cm.ScalarMappable(norm=stress_norm, cmap=stress_cmap)
            cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
            cbar.set_label("Member Stress (MPa)\nBlue=Tension, Red=Compression", fontsize=9 * textscale)

        if show_utilisations and util_cmap:
            sm2 = cm.ScalarMappable(norm=util_norm, cmap=util_cmap)
            cbar2 = plt.colorbar(sm2, ax=ax, orientation="vertical", shrink=0.8)
            cbar2.set_label("Utilisation ratio (σ / allowable)", fontsize=9 * textscale)

    ax.set_title("Truss Visualization", fontsize=11 * textscale)
    plt.show()