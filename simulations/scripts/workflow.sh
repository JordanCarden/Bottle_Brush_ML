#!/bin/bash
set -e
# shellcheck source=./config.sh

script_dir="$(dirname "$0")"
source "${script_dir}/config.sh"

main() {
    local simulations
    simulations=$("${script_dir}/queue.sh" | tail -n 1)

    if [ "$simulations" -gt 0 ]; then
        "${script_dir}/generate.sh" "$simulations"
        "${script_dir}/run.sh"
    else
        echo "QUEUE FULL"
    fi
}

main "$@"
