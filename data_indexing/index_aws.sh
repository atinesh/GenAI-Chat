#!/usr/bin/env bash
# Index files from an AWS S3 bucket into Redis.
#
# Usage:
#   ./index_aws.sh <bucket_name> [index_name]
#
# Defaults: index_name=test_index_1
# Requires AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION in
# the project-root .env (or any standard boto3 credential source).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <bucket_name> [index_name]" >&2
    exit 1
fi

BUCKET_NAME="$1"
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

if ! grep -E '^AWS_ACCESS_KEY_ID=.+' "$ROOT_ENV" >/dev/null; then
    echo "WARNING: AWS_ACCESS_KEY_ID not in $ROOT_ENV — relying on default boto3 credential chain." >&2
fi

# shellcheck disable=SC1091
source ./_setup_env.sh boto3

echo "Indexing AWS S3 bucket '$BUCKET_NAME' into index '$INDEX_NAME' ..."
python3 data_indexing.py aws "$BUCKET_NAME" "$INDEX_NAME"
