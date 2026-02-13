#!/usr/bin/env bash
set -e

# If you store your service account JSON in an env var (recommended)
if [[ -n "${GCP_SA_JSON}" ]]; then
  echo "${GCP_SA_JSON}" > /tmp/gcp-sa.json
  export GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcp-sa.json"
fi

exec streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT}"
