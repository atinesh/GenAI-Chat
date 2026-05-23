#!/usr/bin/env bash
# Index files from the local `corpus/` directory into Redis.
#
# Usage:
#   ./index_local.sh [corpus_dir] [index_name]
#
# Defaults: corpus_dir=corpus, index_name=test_index_1

set -euo pipefail

CORPUS_DIR="${1:-corpus}"
INDEX_NAME="${2:-test_index_1}"

cd "$(dirname "$0")"

ROOT_ENV="$(cd .. && pwd)/.env"
if [[ ! -f "$ROOT_ENV" ]]; then
    echo "ERROR: project-root .env not found at $ROOT_ENV" >&2
    exit 1
fi

if [[ ! -d "$CORPUS_DIR" ]]; then
    echo "ERROR: corpus directory not found: $CORPUS_DIR" >&2
    echo "Put files to index into data_indexing/corpus/ (or pass a custom path)." >&2
    exit 1
fi

if ! grep -E '^OPENAI_API_KEY=.+' "$ROOT_ENV" >/dev/null; then
    echo "ERROR: OPENAI_API_KEY is not set in $ROOT_ENV" >&2
    exit 1
fi

# shellcheck disable=SC1091
source ./_setup_env.sh

echo "Indexing local files from '$CORPUS_DIR' into index '$INDEX_NAME' ..."
python3 data_indexing.py local "$CORPUS_DIR" "$INDEX_NAME"
