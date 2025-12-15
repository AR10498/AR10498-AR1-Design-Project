from structural_classes import Node, Member
import ezdxf
from shapely.geometry import LineString
import itertools 
import numpy as np
from scipy.spatial import KDTree

import pandas as pd

#############################
#   GEOMETRY EXTRACTION
#############################
def extract_structure_from_dxf(file_path, units='mm', tol=1e-3):
    """
    Extracts nodes and members from a DXF, merges nodes within a tolerance, 
    and optionally visualizes merged/conflicted nodes.
    """
    MM_TO_M = 0.001  # millimeters to meters

    # Read DXF 
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = []

    # Extract line geometry 
    for entity in msp.query("LINE"):
        start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
        end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
        lines.append((start, end))

    # Unit conversion 
    factor = MM_TO_M if units.lower() == 'mm' else 1.0
    lines_m = [((s[0]*factor, s[1]*factor, s[2]*factor),
                (e[0]*factor, e[1]*factor, e[2]*factor)) for s, e in lines]

    # Find intersections 
    shapely_lines = [LineString([(s[0], s[1]), (e[0], e[1])]) for s, e in lines_m]
    intersections = set()
    for line1, line2 in itertools.combinations(shapely_lines, 2):
        inter = line1.intersection(line2)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            intersections.add((inter.x, inter.y, 0))
        elif inter.geom_type == "MultiPoint":
            for p in inter.geoms:
                intersections.add((p.x, p.y, 0))

    # Add endpoints 
    for start, end in lines_m:
        intersections.add(start)
        intersections.add(end)

    # Convert to numpy array
    pts = np.array(list(intersections))

    # Merge close nodes 
    tree = KDTree(pts[:, :2])
    visited = np.zeros(len(pts), dtype=bool)
    merged_nodes = []
    conflict_groups = []

    for i, p in enumerate(pts):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(p[:2], tol)
        visited[idxs] = True
        group_pts = pts[idxs]
        merged_pt = np.mean(group_pts, axis=0)
        merged_nodes.append(merged_pt)
        if len(idxs) > 1:
            conflict_groups.append(group_pts)

    # Create Node objects 
    #print(np.array(merged_nodes))
    nodes_tmp = {i: Node(i, tuple(coord)) for i, coord in enumerate(merged_nodes)}

    # Reindex nodes by label 
    nodes = {n.label: n for n in nodes_tmp.values()}

    # Create Members 
    members = {}
    for i, (start, end) in enumerate(lines_m):
        # Match start & end to nearest merged nodes
        start_node = min(nodes_tmp.values(), key=lambda n: np.linalg.norm(np.array(n.coords[:2]) - np.array(start[:2])))
        end_node = min(nodes_tmp.values(), key=lambda n: np.linalg.norm(np.array(n.coords[:2]) - np.array(end[:2])))
        startlabel = start_node.label
        endlabel = end_node.label
        m = Member(i, nodes, startlabel, endlabel)
        members[m.name] = m

    # Build connectivity
    for m in members.values():
        n1 = m.node_start().label
        n2 = m.node_end().label
        nodes[n1].connectivity.append(n2)
        nodes[n2].connectivity.append(n1)  # undirected connection

    return nodes, members

##############################
#   MATERIAL HANDLING (robust)
##############################
def read_materials_from_excel(file_path, print_contents=True):
    """
    Reads Excel and returns a normalized DataFrame keyed by Material name.
    Recognizes common column headers and maps them to: Material, E, Fy, Density.
    """
    df = pd.read_excel(file_path, skiprows=1)
    if df.empty or df.shape[1] < 1:
        raise ValueError("Excel file must have at least one column for material labels.")
    
    # normalize column names: 
    normalized = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "label" in cl:
            normalized[c] = "Label"
        #!!! potential issue if any other column heading starts with the letter e or E !!!
        elif "e" == cl:
            normalized[c] = "E"
        elif "fc" == cl:
            normalized[c] = "fc"
        elif "ft" == cl:
            normalized[c] = "ft"
        elif "density" == cl or "rho" == cl:
            normalized[c] = "Density"
        else:
            # ignore unknown columns but keep them in case needed
            normalized[c] = c
    
    df = df.rename(columns=normalized)
    
    # require Material column
    if "Label" not in df.columns:
        raise ValueError("Excel file must contain a 'Label' column (header contains 'Label').")
    
    # Ensure the standard columns exist
    for std in ("E", "fc", "ft", "Density"):
        if std not in df.columns:
            df[std] = 0.0
    
    # Coerce numeric columns to floats, fill NaN with 0
    for std in ("E", "fc", "ft", "Density"):
        df[std] = pd.to_numeric(df[std], errors="coerce").fillna(0.0).astype(float)
    
    # Trim material names and convert to string
    df["Label"] = df["Label"].astype(str).str.strip()
    
    # transform to enable simple lookup
    df.set_index('Label', inplace=True)
    df = df.transpose()
    if print_contents:
        print("\n<= Imported Materials (normalized):\n")
        print(df)

    return df 


############
#   EXPORT
############
def export_to_dxf(nodes, members, filename="output.dxf", units='mm'):
    M_TO_MM = 1000.0  # millimeters to meters

    factor = M_TO_MM if units.lower() == 'mm' else 1.0
    
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for m in members.values():
        x1,y1 = m.node_start().coords[:2]
        x2,y2 = m.node_end().coords[:2]
        msp.add_line([x1*factor, y1*factor], [x2*factor, y2*factor])
    doc.saveas(filename)
    print(f"=> DXF exported to {filename}")
