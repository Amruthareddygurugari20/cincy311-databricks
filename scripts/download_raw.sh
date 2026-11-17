#!/usr/bin/env bash
# Land the raw Cincinnati 311 CSV in the Unity Catalog Volume.
# Serverless compute in this workspace has no outbound internet, so the file is
# downloaded locally and uploaded with the Databricks CLI. Re-run safe: --overwrite.
#
# Usage: ./scripts/download_raw.sh [databricks-profile]
set -euo pipefail

PROFILE="${1:-DEFAULT}"
URL="https://data.cincinnati-oh.gov/api/views/gcej-gmiw/rows.csv?accessType=DOWNLOAD"
TARGET="dbfs:/Volumes/workspace/cincy311/raw/cincinnati_311_service_requests.csv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -sS -L --fail -o "$TMP/raw.csv" "$URL"
ls -lh "$TMP/raw.csv"
databricks fs cp "$TMP/raw.csv" "$TARGET" --overwrite -p "$PROFILE"
databricks fs ls dbfs:/Volumes/workspace/cincy311/raw/ -l -p "$PROFILE"
