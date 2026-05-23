#!/usr/bin/env bash
# Shared venv bootstrap for the index_*.sh wrappers.
# 
# This script:
# - Creates `genai-env/` inside data_indexing/ if it doesn't already exist
# - Activates it
# - Installs requirements.txt
# - Installs any extra packages passed as arguments (e.g. azure-storage-blob, boto3)
#
# Override the venv location with VENV_DIR=<path> before sourcing.

VENV_DIR="${VENV_DIR:-genai-env}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in '$VENV_DIR' ..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt ..."
pip install -q --disable-pip-version-check -r requirements.txt

if [[ $# -gt 0 ]]; then
    echo "Installing extra dependencies: $* ..."
    pip install -q --disable-pip-version-check "$@"
fi
