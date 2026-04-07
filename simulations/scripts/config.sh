#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${LOCAL_PROJECT_DIR}")"

USER="${SIMULATION_USER:-jcarde7}"
HPC="${SIMULATION_HPC:-loni}"
HPC_PROJECT_DIR="${SIMULATION_HPC_PROJECT_DIR:-/work/${USER}/${PROJECT_NAME}}"
