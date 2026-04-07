def residue_letter(n):
    return chr(64 + n)

def format_side_atom(prefix, n, index):
    return f"{prefix}{residue_letter(n)}{index:02d}"

def generate_polymer_coordinates(input_list):
    coordinates = []
    for i, (carbon, side_chain) in enumerate(input_list):
        n = i + 1
        coordinates.append((f"C{residue_letter(n)}", (4 * n, (-1)**n * 1, 0)))
        if side_chain.startswith("E"):
            m = int(side_chain[1:])
            for j in range(1, m + 1):
                if n % 2 == 1:
                    coordinates.append((format_side_atom("E", n, j), (4 * n + (-1)**j * 1, -4 * j, 0)))
                else:
                    coordinates.append((format_side_atom("E", n, j), (4 * n + (-1)**j * 1, 4 * j, 0)))
        elif side_chain.startswith("S"):
            g = int(side_chain[1:])
            for k in range(1, g + 1):
                if n % 2 == 1:
                    base_y = -4 * k
                    shift = -2
                else:
                    base_y = 4 * k
                    shift = 2
                coordinates.append((format_side_atom("S", n, k), (4 * n + (-1)**k * 1, base_y, 0)))
                coordinates.append((format_side_atom("X", n, k), (4 * n, base_y, (-1)**k * 4)))
                coordinates.append((format_side_atom("Y", n, k), (4 * n, base_y + shift, (-1)**k * 8.33)))
                coordinates.append((format_side_atom("Z", n, k), (4 * n, base_y - shift, (-1)**k * 8.33)))
        elif side_chain.startswith("Q"):
            g = int(side_chain[1:])
            for k in range(1, g + 1):
                if n % 2 == 1:
                    base_y = -4 * k
                else:
                    base_y = 4 * k
                coordinates.append((format_side_atom("P", n, k), (4 * n+(-1)**k*1, base_y, 0)))
                coordinates.append((format_side_atom("L", n, k), (4 * n, base_y, (-1)**k * 4)))
                coordinates.append((format_side_atom("M", n, k), (4 * n, base_y, (-1)**k * 8)))
                coordinates.append((format_side_atom("N", n, k), (4 * n, base_y, (-1)**k * 12)))
    return coordinates

def save_pdb_format(coordinates, filename="output.pdb"):
    pdb_lines = []
    atom_counter = 1
    pdb_lines.append("REMARK original generated coordinate pdb file")
    for atom, coord in coordinates:
        x, y, z = coord
        atom_name = atom
        res_name = "LIG"
        res_seq_numb = 1
        chain_id = "SEG1"
        element_symbol = atom[0]
        pdb_line = f"ATOM  {atom_counter:>5} {atom_name:<4} {res_name:<3}  {res_seq_numb:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {chain_id} {element_symbol}"
        pdb_lines.append(pdb_line)
        atom_counter += 1
    with open(filename, "w") as file:
        for line in pdb_lines:
            file.write(line + "\n")

def generate_bonds(n, label):
    bond_lines = []
    if label.startswith("E"):
        j = int(label[1:])
        if j != 0:
            bonds = [f"C{residue_letter(n)} {format_side_atom('E', n, 1)}"]
            for k in range(1, j):
                bonds.append(f"{format_side_atom('E', n, k)} {format_side_atom('E', n, k+1)}")
            bond_lines.extend(split_bonds(bonds))
    elif label.startswith("S"):
        j = int(label[1:])
        if j != 0:
            bond_line1 = [f"C{residue_letter(n)} {format_side_atom('S', n, 1)}"]
            for k in range(1, j):
                bond_line1.append(f"{format_side_atom('S', n, k)} {format_side_atom('S', n, k+1)}")
            bond_lines.extend(split_bonds(bond_line1))
            bond_line2 = [f"{format_side_atom('S', n, k)} {format_side_atom('X', n, k)}" for k in range(1, j + 1)]
            bond_lines.extend(split_bonds(bond_line2))
            bond_line3 = []
            for k in range(1, j + 1):
                bond_line3.append(f"{format_side_atom('X', n, k)} {format_side_atom('Y', n, k)}")
                bond_line3.append(f"{format_side_atom('X', n, k)} {format_side_atom('Z', n, k)}")
                bond_line3.append(f"{format_side_atom('Y', n, k)} {format_side_atom('Z', n, k)}")
            bond_lines.extend(split_bonds(bond_line3))
    elif label.startswith("Q"):
        j = int(label[1:])
        if j != 0:
            bond_line1 = [f"C{residue_letter(n)} {format_side_atom('P', n, 1)}"]
            for k in range(1, j):
                bond_line1.append(f"{format_side_atom('P', n, k)} {format_side_atom('P', n, k+1)}")
            bond_lines.extend(split_bonds(bond_line1))
            bond_line2 = [f"{format_side_atom('P', n, k)} {format_side_atom('L', n, k)}" for k in range(1, j + 1)]
            bond_lines.extend(split_bonds(bond_line2))
            bond_line3 = [f"{format_side_atom('L', n, k)} {format_side_atom('M', n, k)}" for k in range(1, j + 1)]
            bond_lines.extend(split_bonds(bond_line3))
            bond_line4 = [f"{format_side_atom('M', n, k)} {format_side_atom('N', n, k)}" for k in range(1, j + 1)]
            bond_lines.extend(split_bonds(bond_line4))
    return bond_lines

def split_bonds(bonds):
    max_pairs_per_line = 10
    return ["BOND  " + "  ".join(bonds[i:i + max_pairs_per_line]) for i in range(0, len(bonds), max_pairs_per_line)]

def process_bond_list(input_list):
    all_bond_lines = []
    for n, label in enumerate(input_list, start=1):
        bond_lines = generate_bonds(n, label[1])
        all_bond_lines.extend(bond_lines)
    backbone_connections = [f"C{residue_letter(k+1)} C{residue_letter(k+2)}" for k in range(len(input_list) - 1)]
    all_bond_lines.extend(split_bonds(backbone_connections))
    return all_bond_lines

def save_topology1_format(coordinates, input_list, filename="topology"):
    pdb_lines = []
    pdb_lines.append("MASS    1    C1        72")
    pdb_lines.append("MASS    1    EO        45")
    pdb_lines.append("MASS    1    SCY       45")
    pdb_lines.append("MASS    1    P4        72")
    pdb_lines.append("MASS    1    BP4       72")
    pdb_lines.append("MASS    1    Qa        72")
    pdb_lines.append("MASS    1    Q0        72")
    pdb_lines.append("MASS    1    C4        72")
    pdb_lines.append("MASS    1    Na        72")
    pdb_lines.append("")
    pdb_lines.append("RESI BEN         0")
    pdb_lines.append("GROUP")
    pdb_lines.append("ATOM B1    SCY   0")
    pdb_lines.append("ATOM B2    SCY   0")
    pdb_lines.append("ATOM B3    SCY   0")
    pdb_lines.append("")
    pdb_lines.append("BOND B1 B2   B2 B3  B1 B3")
    pdb_lines.append("")
    pdb_lines.append("RESI HOH         0")
    pdb_lines.append("ATOM    W   P4    0.00")
    pdb_lines.append("")
    pdb_lines.append("RESI AF          0")
    pdb_lines.append("ATOM    WAF   BP4    0.00")
    pdb_lines.append("")
    pdb_lines.append("RESI LIG          0")
    pdb_lines.append("GROUP")
    for atom, coord in coordinates:
        atom_name = atom[0]
        element_symbol = (
            'SCY' if atom_name in ['X', 'S', 'Y', 'Z'] else
            'EO' if atom_name == 'E' else
            'C4' if atom_name == 'P' else
            'Na' if atom_name == 'L' else
            'Qa' if atom_name == 'M' else
            'Q0' if atom_name == 'N' else
            'C1' if atom_name == 'C' else 'UNK'
        )
        element_charge = (
            '0' if atom_name in ['X', 'S', 'Y', 'Z'] else
            '0' if atom_name == 'E' else
            '0' if atom_name == 'P' else
            '0' if atom_name == 'L' else
            '-1' if atom_name == 'M' else
            '1' if atom_name == 'N' else
            '0' if atom_name == 'C' else 'UNK'
        )
        pdb_line = f"ATOM    {atom:<4}    {element_symbol}    {element_charge}"
        pdb_lines.append(pdb_line)
    bond_lines = process_bond_list(input_list)
    pdb_lines.extend(bond_lines)
    with open(filename, "w") as file:
        for line in pdb_lines:
            file.write(line + "\n")

input_list = [(1, "Q2"), (2, "E4"), (3, "S3")]

coordinates = generate_polymer_coordinates(input_list)
save_pdb_format(coordinates)
save_topology1_format(coordinates, input_list)