#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build full QA dataset pipeline for all three tracks:
1) view+filter
2) view-only
3) filter-only

Pipeline steps:
- (optional) bootstrap v2 DB folders from an existing benchmark version
- (optional) regenerate access_control.json from schema.json
- build qa_config for each track
- build pq_pairs for each track
- naturalize each track with DSPy + LM Studio
- consolidate naturalized QA into per-DB datasets (qa.json + bird_qa.json)

Usage:
  ./build_full_qa_dataset.sh --model <lmstudio_model_id> [options]

Options:
  --model <id>                 Required. LM Studio loaded model id.
  --base-dir <path>            Target dataset base dir (default: data/P3T2Q_benchmark/v2)
  --source-base-dir <path>     Source benchmark dir for bootstrap (default: data/P3T2Q_benchmark/v1)
  --no-bootstrap               Do not copy DB assets from source-base-dir.
  --db <db_id>                 Optional DB filter. Repeatable.
  --api-base <url>             LM Studio OpenAI endpoint (default: http://127.0.0.1:1234/v1)
  --api-key <key>              API key placeholder (default: local)
  --temperature <float>        Naturalization temperature (default: 0.1)
  --max-tokens <int>           Naturalization max tokens (default: 180)
  --limit <int>                Optional naturalization record limit per file.
  --ensure-lmstudio-sdk        Run lmstudio SDK preflight check.
  --skip-access-control        Skip access_control regeneration step.
  --skip-consolidation         Skip qa.json / bird_qa.json consolidation step.
  -h, --help                   Show this help.

Examples:
  ./build_full_qa_dataset.sh --model "qwen/qwen3-4b-2507"

  ./build_full_qa_dataset.sh \
    --model "qwen/qwen3-4b-2507" \
    --db financial \
    --ensure-lmstudio-sdk
EOF
}

MODEL_ID=""
BASE_DIR="data/P3T2Q_benchmark/v2"
SOURCE_BASE_DIR="data/P3T2Q_benchmark/v1"
BOOTSTRAP_FROM_SOURCE="1"
API_BASE="http://127.0.0.1:1234/v1"
API_KEY="local"
TEMPERATURE="0.1"
MAX_TOKENS="180"
NAT_LIMIT=""
ENSURE_LMSTUDIO_SDK="0"
SKIP_ACCESS_CONTROL="0"
SKIP_CONSOLIDATION="0"
DB_ARGS=()
DB_IDS=()

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
    --source-base-dir)
      SOURCE_BASE_DIR="${2:-}"
      shift 2
      ;;
    --no-bootstrap)
      BOOTSTRAP_FROM_SOURCE="0"
      shift
      ;;
    --db)
      DB_ARGS+=("--db" "${2:-}")
      DB_IDS+=("${2:-}")
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
    --skip-consolidation)
      SKIP_CONSOLIDATION="1"
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
if [[ -d "${SCRIPT_DIR}/p3t2q_benchmark_building" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

copy_if_missing() {
  local src_path="$1"
  local dst_path="$2"
  if [[ ! -e "${src_path}" ]]; then
    return
  fi
  if [[ -e "${dst_path}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${dst_path}")"
  cp -a "${src_path}" "${dst_path}"
}

bootstrap_db_assets_if_needed() {
  mkdir -p "${BASE_DIR}"

  if [[ "${BOOTSTRAP_FROM_SOURCE}" != "1" ]]; then
    return
  fi

  if [[ ! -d "${SOURCE_BASE_DIR}" ]]; then
    echo "Error: source base dir not found for bootstrap: ${SOURCE_BASE_DIR}" >&2
    exit 1
  fi

  local selected_db_ids=()
  if [[ ${#DB_IDS[@]} -gt 0 ]]; then
    selected_db_ids=("${DB_IDS[@]}")
  else
    while IFS= read -r db_id; do
      if [[ -f "${SOURCE_BASE_DIR}/${db_id}/schema.json" ]]; then
        selected_db_ids+=("${db_id}")
      fi
    done < <(find "${SOURCE_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  fi

  if [[ ${#selected_db_ids[@]} -eq 0 ]]; then
    echo "Error: no DB folders found in source base dir: ${SOURCE_BASE_DIR}" >&2
    exit 1
  fi

  for db_id in "${selected_db_ids[@]}"; do
    local src_db_dir="${SOURCE_BASE_DIR}/${db_id}"
    local dst_db_dir="${BASE_DIR}/${db_id}"

    if [[ ! -f "${src_db_dir}/schema.json" ]]; then
      echo "[warn] skipping bootstrap for '${db_id}': source schema.json not found"
      continue
    fi

    mkdir -p "${dst_db_dir}"
    copy_if_missing "${src_db_dir}/schema.json" "${dst_db_dir}/schema.json"
    copy_if_missing "${src_db_dir}/access_control.json" "${dst_db_dir}/access_control.json"
    copy_if_missing "${src_db_dir}/${db_id}.sqlite" "${dst_db_dir}/${db_id}.sqlite"
    copy_if_missing "${src_db_dir}/database_description" "${dst_db_dir}/database_description"
    mkdir -p "${dst_db_dir}/qa_configs"
  done
}

bootstrap_db_assets_if_needed

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
echo "==> Source base dir: ${SOURCE_BASE_DIR}"
echo "==> Bootstrap from source: ${BOOTSTRAP_FROM_SOURCE}"
if [[ ${#COMMON_DB_ARGS[@]} -gt 0 ]]; then
  echo "==> DB filter: ${COMMON_DB_ARGS[*]}"
else
  echo "==> DB filter: (all DBs)"
fi

if [[ "${SKIP_ACCESS_CONTROL}" == "0" ]]; then
  echo "==> Regenerating access_control.json using modular builder"
  python3 p3t2q_benchmark_building/build_access_control_policy.py \
    --base-dir "${BASE_DIR}"
else
  echo "==> Skipping access_control regeneration"
fi

echo "==> Building view+filter qa_config"
python3 p3t2q_benchmark_building/build_view_and_filter_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view+filter pq_pairs"
python3 p3t2q_benchmark_building/build_view_and_filter_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_view_and_filter.json \
  --output-name pq_pairs_view_and_filter.json \
  --combined-output pq_pairs_view_and_filter_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view-only qa_config"
python3 p3t2q_benchmark_building/build_view_only_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building view-only pq_pairs"
python3 p3t2q_benchmark_building/build_view_only_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_view_only.json \
  --output-name pq_pairs_view_only.json \
  --combined-output pq_pairs_view_only_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building filter-only qa_config"
python3 p3t2q_benchmark_building/build_filter_only_qa_config.py \
  --base-dir "${BASE_DIR}" \
  "${COMMON_DB_ARGS[@]}"

echo "==> Building filter-only pq_pairs"
python3 p3t2q_benchmark_building/build_filter_only_qa_pairs.py \
  --base-dir "${BASE_DIR}" \
  --qa-config-name qa_config_filter_only.json \
  --output-name pq_pairs_filter_only.json \
  --combined-output pq_pairs_filter_only_all.json \
  "${COMMON_DB_ARGS[@]}"

echo "==> Naturalizing view+filter pairs"
python3 p3t2q_benchmark_building/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_view_and_filter.json \
  --output-name qa_pairs_view_and_filter_naturalized.json \
  --combined-output qa_pairs_view_and_filter_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

echo "==> Naturalizing view-only pairs"
python3 p3t2q_benchmark_building/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_view_only.json \
  --output-name qa_pairs_view_only_naturalized.json \
  --combined-output qa_pairs_view_only_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

echo "==> Naturalizing filter-only pairs"
python3 p3t2q_benchmark_building/naturalize_qa_pairs_dspy.py \
  --base-dir "${BASE_DIR}" \
  --input-name pq_pairs_filter_only.json \
  --output-name qa_pairs_filter_only_naturalized.json \
  --combined-output qa_pairs_filter_only_naturalized_all.json \
  "${COMMON_DB_ARGS[@]}" \
  "${NAT_ARGS[@]}"

if [[ "${SKIP_CONSOLIDATION}" == "0" ]]; then
  echo "==> Building per-DB consolidated QA dataset (qa.json)"
  python3 p3t2q_benchmark_building/build_bird_style_dataset.py \
    --base-dir "${BASE_DIR}" \
    --per-db-output-name qa.json \
    "${COMMON_DB_ARGS[@]}"

  echo "==> Building per-DB compatibility dataset (bird_qa.json)"
  python3 p3t2q_benchmark_building/build_bird_style_dataset.py \
    --base-dir "${BASE_DIR}" \
    --per-db-output-name bird_qa.json \
    "${COMMON_DB_ARGS[@]}"
else
  echo "==> Skipping consolidation step"
fi

echo "==> Full QA dataset build complete."
