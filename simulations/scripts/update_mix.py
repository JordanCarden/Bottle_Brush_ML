import re

pdb_file = "output.pdb"
mix_file = "mix.txt"
extract_script = "extract.sh"


max_atom = 0
try:
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                try:
                    atom_number = int(line.split()[1])
                    max_atom = max(max_atom, atom_number)
                except (IndexError, ValueError):
                    continue
except FileNotFoundError:
    print(f"Error: File {pdb_file} not found.")
    exit(1)


try:
    with open(mix_file, "r") as file:
        lines = file.readlines()

    with open(mix_file, "w") as file:
        previous_line = ""
        for line in lines:

            if previous_line.strip() == "end atoms" and re.match(
                r"^\s*atoms\s+\d+", line
            ):
                file.write(f"atoms {max_atom}\n")
            else:
                file.write(line)
            previous_line = line
except FileNotFoundError:
    print(f"Error: File {mix_file} not found.")
    exit(1)
except PermissionError:
    print(f"Error: Insufficient permissions to write to {mix_file}.")
    exit(1)


try:
    with open(extract_script, "r") as file:
        lines = file.readlines()

    with open(extract_script, "w") as file:
        for line in lines:
            if line.strip().startswith("start1=$((28"):
                file.write(f"start1=$(({max_atom} + 5))\n")
            elif line.strip().startswith("start2=$((28"):
                file.write(f"start2=$(({max_atom} + 5))\n")
            else:
                file.write(line)
except FileNotFoundError:
    print(f"Error: File {extract_script} not found.")
    exit(1)
except PermissionError:
    print(f"Error: Insufficient permissions to write to {extract_script}.")
    exit(1)
