import sys
import ast
import random


def generate_random_input_list():
    chain_types = ["E", "S"]
    sidechain_length_range = list(range(0, 10))
    backbone_length = random.randint(10, 15)
    input_list = []
    for i in range(1, backbone_length + 1):
        chain_type = random.choice(chain_types)
        sidechain_length = random.choice(sidechain_length_range)
        input_list.append((i, f"{chain_type}{sidechain_length}"))
    return input_list


def update_modifybond_script(file_path, new_input_list):
    with open(file_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.strip().startswith("input_list ="):
            updated_lines.append(f"input_list = {new_input_list}\n")
        else:
            updated_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(updated_lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            input_list = ast.literal_eval(sys.argv[1])
        except Exception as e:
            print("Invalid Input List. Error:", e)
            sys.exit(1)
    else:
        input_list = generate_random_input_list()

    script_path = "modifybond.py"
    update_modifybond_script(script_path, input_list)
