#!/bin/bash
# shellcheck source=./config.sh

source "$(dirname "$0")/config.sh"

SCRIPT_DIR="$LOCAL_PROJECT_DIR/scripts"
NAME_FILE="$SCRIPT_DIR/names.txt"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit.sh"
NAMD_DIR="$HPC_PROJECT_DIR/NAMD_2.14_Linux-x86_64-multicore-CUDA"
POLYMERS_DIR="$HPC_PROJECT_DIR/polymers"

generate_script() {
    mapfile -t polymer_names < "$NAME_FILE"
    local simulations=${#polymer_names[@]}

    case "$HPC" in
        mike)
            local tasks=4
            local cores=16
            local gres="gpu:4"
            local partition="gpu"
            local account="hpc_hpcsuvo02"
            ;;
        loni)
            if [ "$simulations" -eq 4 ]; then
                local tasks=4
                local cores=16
                local gres="gpu:4"
                local partition="gpu4"
                local account="loni_pdrug"
            elif [ "$simulations" -eq 2 ]; then
                local tasks=2
                local cores=32
                local gres="gpu:2"
                local partition="gpu2"
                local account="loni_pdrug"
            else
                exit 1
            fi
            ;;
        *)
            exit 1
            ;;
    esac

    cat <<EOF_INNER >"$SUBMIT_SCRIPT"
#!/bin/bash
#SBATCH -N 1
#SBATCH -n $tasks
#SBATCH --cpus-per-task=$cores
#SBATCH --gres=$gres
#SBATCH -t 72:00:00
#SBATCH -p $partition
#SBATCH -A $account
#SBATCH -o /dev/null

module load cuda
EOF_INNER

    for poly_name in "${polymer_names[@]}"; do
        cat <<EOF_INNER >>"$SUBMIT_SCRIPT"
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=$cores \
    "$NAMD_DIR/namd2" \
    +p$cores "$POLYMERS_DIR/${poly_name}/20sim/20_waterbox.conf" \
    > "$POLYMERS_DIR/${poly_name}/20sim/output.log" &
EOF_INNER
    done

    echo -e "\nwait" >>"$SUBMIT_SCRIPT"
}

submit() {
    case "$HPC" in
        mike)
            ssh "${USER}@mike.hpc.lsu.edu" "sbatch $HPC_PROJECT_DIR/scripts/submit.sh"
            ;;
        loni)
            ssh "${USER}@qbd.loni.org" "sbatch $HPC_PROJECT_DIR/scripts/submit.sh"
            ;;
    esac
}

main() {
    if [ ! -f "$NAME_FILE" ]; then
        exit 0
    fi
    generate_script
    submit
    rm "$NAME_FILE"
}

main "$@"
