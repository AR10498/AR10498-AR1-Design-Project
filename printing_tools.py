import pandas as pd

# ==========================
#   DATA TABLE PRINTING
# ==========================

def member_dataframe(members):
    colnames = 'Member', 'Material', 'E', 'fc', 'ft', 'Density'
    df=pd.DataFrame(columns=colnames)
    for i,m in enumerate(members.values()):
        df._set_value(index=i, col='Member', value=m.name)
        df._set_value(index=i, col='Material', value=m.material)
        props = m.properties
        for p in props:
            df._set_value(index=i, col=p, value=props[p])
    df.set_index("Member", inplace=True)
    title="    Assigned Member Material Properties"
    print(f'{title}\n', '-'*len(title))
    print(df)
    return df


def comparison_data(original_angles, new_angles, original_lengths, new_lengths, print_output=True):
    cols = 'Node', 'Member Pair', 'Member', 'Old Angle (°)', 'New Angle (°)', 'Old Length', 'New Length'
    df = pd.DataFrame(columns=cols)
    for i,(node_label,angles) in enumerate(new_angles.items()):
        for j, (m1, m2, new_ang, v1, v2) in enumerate(angles):
            old_ang = next((ang for a_m1, a_m2, ang, _, _ in original_angles.get(node_label, []) if a_m1 == m1 and a_m2 == m2), None)
            
            vals1 = node_label, f'{m1}--{m2}', f'{m1}', old_ang, new_ang, original_lengths[m1], new_lengths[m1]
            for c, val in zip(cols, vals1):
                df._set_value(index=len(angles)*2*i+j, col=c, value=val)
            
            vals2 = node_label, f'{m1}--{m2}', f'{m2}', old_ang, new_ang, original_lengths[m2], new_lengths[m2]
            for c, val in zip(cols, vals2):
                df._set_value(index=len(angles)*2*i+j+1, col=c, value=val)            

    df.set_index("Member Pair", inplace=True)
    if print_output:
        header = "\n  Comparison Table: Old vs New Lengths & Angles"
        print(header)
        print("-" * len(header))
        print(df)
        
    return df