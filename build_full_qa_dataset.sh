#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build full QA dataset pipeline for all three tracks:
1) view+filter
2) view-only
3) filter-only

Pipeline steps:
- (optional) regenerate access_control.json from schema.json
- build qa_config for each track
- build pq_pairs for each track
- naturalize each track with DSPy + LM Studio

Usage:
  experiments/build_full_qa_dataset.sh --model <lmstudio_model_id> [options]

Options:
  --model <id>                 Required. LM Studio loaded model id.
  --base-dir <path>            Dataset base dir (default: data/P3T2Q_benchmark/v0)
  --db <db_id>                 Optional DB filter. Repeatable.
  --api-base <url>             LM Studio OpenAI endpoint (default: http://127.0.0.1:1234/v1)
  --api-key <key>              API key placeholder (default: local)
  --temperature <float>        Naturalization temperature (default: 0.1)
  --max-tokens <int>           Naturalization max tokens (default: 180)
  --limit <int>                Optional naturalization record limit per file.
  --ensure-lmstudio-sdk        Run lmstudio SDK preflight check.
  --skip-access-control        Skip access_control regeneration step.
  -h, --help                   Show this help.

Examples:
  experiments/build_full_qa_dataset.sh --model "qwen/qwen3-4b-2507"

  experiments/build_full_qa_dataset.sh \
    --model "qwen/qwen3-4b-2507" \
    --db financial \
    --ensure-lmstudio-sdk
EOF
}

MODEL_ID=""
BASE_DIR="data/P3T2Q_benchmark/v0"
API_BASE="http://127.0.0.1:1234/v1"
API_KEY="local"
TEMPERATURE="0.1"
MAX_TOKENS="180"
NAT_LIMIT=""
ENSURE_LMSTUDIO_SDK="0"
SKIP_ACCESS_CONTROL="0"
DB_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="${2:-}"
      shift 2
      ;;
    --db)
      DB_ARGS+=("--db" "${2:-}")
      shift 2
      ;;
    --api-base)
      API_BASE="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="${2:-}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --limit)
      NAT_LIMIT="${2:-}"
      shift 2
      ;;
    --ensure-lmstudio-sdk)
      ENSURE_LMSTUDIO_SDK="1"
      shift
      ;;
    --skip-access-control)
      SKIP_ACCESS_CONTROL="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_ID}" ]]; then
  echo "Error: --model is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Error: base dir not found: ${BASE_DIR}" >&2
  exit 1
fi

COMMON_DB_ARGS=()
if [[ ${#DB_ARGS[@]} -gt 0 ]]; then
  COMMON_DB_ARGS=("${DB_ARGS[@]}")
fi

NAT_ARGS=(
  --model "${MODEL_ID}"
  --api-base "${API_BASE}"
  --api-key "${API_KEY}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ -n "${NAT_LIMIT}" ]]; then
  NAT_ARGS+=(--limit "${NAT_LIMIT}")
fi

if [[ "${ENSURE_LMSTUDIO_SDK}" == "1" ]]; then
  NAT_ARGS+=(--ensure-lmstudio-sdk)
fi

echo "==> Base dir: ${BASE_DIR}"
if [[ ${#COMMON_DB_ARGS[@]} -gt 0 ]]; then
  echo "==> DB filter: ${COMMON_DB_ARGS[*]}"
else
  echo "==> DB filter: (all DBs)"
fi

if [[ "${SKIP_ACCESS_CONTROL}" == "0" ]]; then
  echo "==> Regenerating access_control.json using modular builder"
  python3 experiments/build_access_control_policy.py \
    --base-dir "${BASE_DIR}"
else
  echo "==> Skipping access_control regeneration"
fi

echo "==> Building view+filter qa_config"
python3 experiments/build_view_and_filter_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view+filter pq_pairs"
python3 experiments/build_view_and_filter_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_view_and_filter.json \
  --output-name pq_pairs_view_and_filter.json \
  --combined-output pq_pairs_view_and_filter_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view-only qa_config"
python3 experiments/build_view_only_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view-only pq_pairs"
python3 experiments/build_view_only_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_view_only.json \
  --output-name pq_pairs_view_only.json \
  --combined-output pq_pairs_view_only_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building filter-only qa_config"
python3 experiments/build_filter_only_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building filter-only pq_pairs"
python3 experiments/build_filter_only_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_filter_only.json \
  --output-name pq_pairs_filter_only.json \
  --combined-output pq_pairs_filter_only_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Naturalizing view+filter pairs"
python3 experiments/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_view_and_filter.json \
  --output-name qa_pairs_view_and_filter_naturalized.json \
  --combined-output qa_pairs_view_and_filter_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

echo "==> Naturalizing view-only pairs"
python3 experiments/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_view_only.json \
  --output-name qa_pairs_view_only_naturalized.json \
  --combined-output qa_pairs_view_only_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

echo "==> Naturalizing filter-only pairs"
python3 experiments/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_filter_only.json \
  --output-name qa_pairs_filter_only_naturalized.json \
  --combined-output qa_pairs_filter_only_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

echo "==> Full QA dataset build complete."
