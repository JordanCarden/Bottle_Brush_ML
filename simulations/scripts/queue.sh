#!/bin/bash
# shellcheck source=./config.sh

source "$(dirname "$0")/config.sh"

check_gpu() {
    local job_count
    job_count=$(ssh "${USER}@mike.hpc.lsu.edu" 'squeue -u '"${USER}"' -h | grep gpu | wc -l')
    if [ "$job_count" -lt 4 ]; then
        return 0
    else
        return 1
    fi
}

check_gpu4() {
    local job_count
    job_count=$(ssh "${USER}@qbd.loni.org" 'squeue -u '"${USER}"' -h | grep gpu4 | wc -l')
    if [ "$job_count" -lt 4 ]; then
        return 0
    else
        return 1
    fi
}

check_gpu2() {
    local job_count
    job_count=$(ssh "${USER}@qbd.loni.org" 'squeue -u '"${USER}"' -h | grep gpu2 | wc -l')
    if [ "$job_count" -lt 4 ]; then
        return 0
    else
        return 1
    fi
}

main() {
    case "$HPC" in
        mike)
            if check_gpu; then
                echo 4
            else
                echo 0
            fi
            ;;
        loni)
            if check_gpu4; then
                echo 4
            elif check_gpu2; then
                echo 2
            else
                echo 0
            fi
            ;;
        *)
            echo 0
            ;;
    esac
}

main "$@"
