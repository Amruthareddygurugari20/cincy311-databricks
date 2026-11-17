#!/usr/bin/env bash
# Create (or update) the daily bronze -> silver -> gold job from jobs/cincy311_daily_pipeline.json.
# Notebook paths are filled in for the logged-in user, so no personal path is stored in the repo.
# Re-run safe: if a job with the same name exists, it is reset to this definition instead of duplicated.
#
# Usage: ./scripts/create_job.sh [databricks-profile]
set -euo pipefail

PROFILE="${1:-DEFAULT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ME="$(databricks current-user me -p "$PROFILE" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')"
SPEC="$(sed "s#__WORKSPACE_DIR__#/Users/${ME}/cincy311#g" "$ROOT/jobs/cincy311_daily_pipeline.json")"
NAME="$(python3 -c 'import sys,json;print(json.load(sys.stdin)["name"])' <<<"$SPEC")"

JOB_ID="$(databricks jobs list --name "$NAME" -p "$PROFILE" -o json \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get("jobs",[]); print(d[0]["job_id"] if d else "")')"

if [[ -n "$JOB_ID" ]]; then
  databricks jobs reset -p "$PROFILE" --json "{\"job_id\": $JOB_ID, \"new_settings\": $SPEC}"
  echo "Updated job $JOB_ID ($NAME)"
else
  JOB_ID="$(databricks jobs create -p "$PROFILE" --json "$SPEC" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')"
  echo "Created job $JOB_ID ($NAME)"
fi
