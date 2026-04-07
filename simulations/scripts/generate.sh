#!/bin/bash
# shellcheck source=./config.sh

source "$(dirname "$0")/config.sh"

BASE_DIR="$LOCAL_PROJECT_DIR"
SCRIPT_DIR="$BASE_DIR/scripts"
COPY_DIR="$BASE_DIR/copy"
NAMD_DIR="$BASE_DIR/NAMD_2.14_Linux-x86_64-multicore"
POLYMERS_DIR="$BASE_DIR/polymers"
NAME_FILE="$SCRIPT_DIR/names.txt"

create_polymer() (
    local sequence="$1"

    local max_number=0
    for dir in "$POLYMERS_DIR"/polymerJ*; do
        [ -d "$dir" ] || continue
        local basename
        basename=$(basename "$dir")
        if [[ $basename =~ ^polymerJ([0-9]+)$ ]]; then
            local number=${BASH_REMATCH[1]}
            (( number > max_number )) && max_number=$number
        fi
    done
    local next_number=$((max_number + 1))
    local POLYMER_NAME="polymerJ${next_number}"
    local POLYMER_DIR="$POLYMERS_DIR/$POLYMER_NAME"

    mkdir -p "$POLYMER_DIR"
    cp -r "$COPY_DIR"/* "$POLYMER_DIR"
    cd "$POLYMER_DIR/copy" || exit
    if [ -n "$sequence" ]; then
        python3 "$SCRIPT_DIR/update_modifybond.py" "$sequence"
    else
        python3 "$SCRIPT_DIR/update_modifybond.py"
    fi
    python3 modifybond.py
    cp output.pdb topology ../relax
    cd ../relax || exit
    cp relaxcopy/* .
    "$NAMD_DIR/psfgen" relaxmonomer.pgn
    "$NAMD_DIR/namd2" relax.conf
    vmd -dispdev text -e "$SCRIPT_DIR/save_relax.tcl"
    cp relax.pdb system.dcd ../
    cd .. || exit
    cp copy/* .
    python3 "$SCRIPT_DIR/update_mix.py"
    packmol < mix.txt
    # shellcheck source=extract.sh
    source extract.sh
    "$NAMD_DIR/psfgen" newmonomer.pgn
    vmd -dispdev text -e "$SCRIPT_DIR/save_single_polymer.tcl"
    cp single_polymer.pdb topology 20sim
    cd 20sim || exit
    cp 20copy/* .
    python3 20_polymers.py
    packmol < pack.txt
    python3 "$SCRIPT_DIR/update_20.py"
    packmol < 20_mix.txt
    # shellcheck source=20_extract.sh
    source 20_extract.sh
    "$NAMD_DIR/psfgen" 20_newmonomer.pgn
    echo "$POLYMER_NAME" >> "$NAME_FILE"
)

main() {
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 (number of polymers) or $0 (polymer sequences .txt)"
        exit 1
    fi

    : > "$NAME_FILE"

    if [[ $1 =~ ^[0-9]+$ ]]; then
        local number=$1
        for ((i = 0; i < number; i++)); do
            create_polymer
        done
    else
        local sequences=$1
        if [ ! -f "$sequences" ]; then
            echo "$sequences NOT FOUND"
            exit 1
        fi

        mapfile -t sequences_array < "$sequences"
        for sequence in "${sequences_array[@]}"; do
            create_polymer "$sequence"
        done
    fi
}

main "$@"
