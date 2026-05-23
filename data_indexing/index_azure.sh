#!/usr/bin/env bash
# Index files from an Azure Blob Storage container into Redis.
#
# Usage:
#   ./index_azure.sh <container_name> [index_name]
#
# Defaults: index_name=test_index_1
# Requires AZURE_STORAGE_CONNECTION_STRING in the project-root .env.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <container_name> [index_name]" >&2
    exit 1
fi

CONTAINER_NAME="$1"
INDEX_NAME="${2:-test_index_1}"

cd "$(dirname "$0")"

ROOT_ENV="$(cd .. && pwd)/.env"
if [[ ! -f "$ROOT_ENV" ]]; then
    echo "ERROR: project-root .env not found at $ROOT_ENV" >&2
    exit 1
fi

if ! grep -E '^OPENAI_API_KEY=.+' "$ROOT_ENV" >/dev/null; then
    echo "ERROR: OPENAI_API_KEY is not set in $ROOT_ENV" >&2
    exit 1
fi

if ! grep -E '^AZURE_STORAGE_CONNECTION_STRING=.+' "$ROOT_ENV" >/dev/null; then
    echo "ERROR: AZURE_STORAGE_CONNECTION_STRING is not set in $ROOT_ENV" >&2
    exit 1
fi

# shellcheck disable=SC1091
source ./_setup_env.sh azure-storage-blob

echo "Indexing Azure Blob container '$CONTAINER_NAME' into index '$INDEX_NAME' ..."
python3 data_indexing.py azure "$CONTAINER_NAME" "$INDEX_NAME"
