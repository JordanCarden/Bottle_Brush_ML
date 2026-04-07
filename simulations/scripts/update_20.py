def update_pdb_coordinates_in_place(pdb_file, coord_file):
    with open(coord_file, "r") as coord:
        new_coords = [line.strip().split() for line in coord.readlines()]

    with open(pdb_file, "r") as pdb:
        lines = pdb.readlines()

    updated_lines = []
    coord_index = 0

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            x, y, z = new_coords[coord_index]
            updated_line = (
                line[:30]
                + f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                + line[54:]
            )
            updated_lines.append(updated_line)
            coord_index += 1
        else:
            updated_lines.append(line)

    with open(pdb_file, "w") as pdb:
        pdb.writelines(updated_lines)

    print(f"PDB file '{pdb_file}' has been updated with new coordinates.")


update_pdb_coordinates_in_place(
    pdb_file="20_polymers.pdb", coord_file="20_coordinate.txt"
)
