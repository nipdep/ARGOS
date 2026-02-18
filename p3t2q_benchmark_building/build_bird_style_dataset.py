#!/usr/bin/env python3
"""
Build BIRD-style JSON datasets from naturalized QA-pair files.

Input files are discovered under:
  <base_dir>/<db_id>/qa_configs/qa_pairs*_naturalized.json

Output shape per row:
{
  "question_id": 1,
  "db_id": "financial",
  "question": "...",
  "role": "public",
  "evidence": "",
  "intent_preserving_expected_query": "...",
  "privacy_preserving_expected_query": "...",
  "question_type": "filter only",
  "evidence_added": false,
  "metadata": {
    "queryable": true,
    "allowed_view_columns": [...],
    "allowed_process_columns": [...]
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate naturalized QA pairs into BIRD-style JSON files."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v1",
        help="Benchmark root directory containing per-db folders.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional single output JSON path. "
            "Default: write one consolidated file per DB under <base_dir>/<db_id>/bird_qa.json."
        ),
    )
    parser.add_argument(
        "--per-db-output-name",
        default="qa.json",
        help="Per-DB output filename when --output is not provided.",
    )
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        help="Optional db_id filter. Repeat for multiple DBs.",
    )
    parser.add_argument(
        "--skip-qa-pairs-naturalized",
        action="store_true",
        help="Skip files named exactly qa_pairs_naturalized.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def infer_question_type(file_name: str) -> str:
    if "view_and_filter" in file_name:
        return "view and filter"
    if "view_only" in file_name:
        return "view only"
    if "filter_only" in file_name:
        return "filter only"
    if file_name == "qa_pairs_naturalized.json":
        return "general"
    return "unknown"


def infer_db_id_from_path(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass

    if path.parent.name == "qa_configs":
        return path.parent.parent.name
    return path.parent.name


def discover_input_files(
    base_dir: Path,
    db_filters: set[str],
    skip_qa_pairs_naturalized: bool,
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in ["*/qa_configs/qa_pairs*_naturalized.json", "*/qa_pairs*_naturalized.json"]:
        for path in sorted(base_dir.glob(pattern)):
            if path in seen:
                continue
            files.append(path)
            seen.add(path)

    selected: list[Path] = []
    for path in files:
        db_id = infer_db_id_from_path(path=path, base_dir=base_dir)
        if db_filters and db_id not in db_filters:
            continue
        if skip_qa_pairs_naturalized and path.name == "qa_pairs_naturalized.json":
            continue
        selected.append(path)
    return selected


def build_rows_from_file(path: Path, base_dir: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return []

    db_id = payload.get("db_id") or infer_db_id_from_path(path=path, base_dir=base_dir)
    question_type = infer_question_type(path.name)
    source_config = payload.get("source_config", "")

    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        question = record.get("question_naturalized") or record.get("question")
        base_query = record.get("base_query")
        role_expected = record.get("role_expected", {})
        if not isinstance(role_expected, dict):
            continue

        for role, expected in role_expected.items():
            if not isinstance(expected, dict):
                continue

            privacy_query = expected.get("expected_query")
            queryable = bool(expected.get("is_usable", privacy_query is not None))

            rows.append(
                {
                    "db_id": db_id,
                    "question": question,
                    "role": role,
                    "evidence": "",
                    "intent_preserving_expected_query": base_query,
                    "privacy_preserving_expected_query": privacy_query,
                    "question_type": question_type,
                    "evidence_added": False,
                    "metadata": {
                        "queryable": queryable,
                        "allowed_view_columns": expected.get("used_projection_columns", []),
                        "allowed_process_columns": expected.get("used_process_columns", []),
                        "template_id": record.get("template_id"),
                        "mode": record.get("mode"),
                        "source_file": str(path.relative_to(base_dir)),
                        "source_config": f"{db_id}/{source_config}" if source_config else "",
                        "access_control_config": f"{db_id}/access_control.json",
                        "reason_unusable": expected.get("reason_unusable"),
                    },
                }
            )

    return rows


def assign_question_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with_ids: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        enriched = {"question_id": index, **row}
        with_ids.append(enriched)
    return with_ids


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")

    db_filters = set(args.db or [])
    input_files = discover_input_files(
        base_dir=base_dir,
        db_filters=db_filters,
        skip_qa_pairs_naturalized=args.skip_qa_pairs_naturalized,
    )
    if not input_files:
        raise ValueError(
            "no input files found with pattern '*/qa_configs/qa_pairs*_naturalized.json'"
        )

    rows_by_db: dict[str, list[dict[str, Any]]] = {}
    per_file_counts: dict[str, int] = {}
    for path in input_files:
        rows = build_rows_from_file(path=path, base_dir=base_dir)
        db_id = infer_db_id_from_path(path=path, base_dir=base_dir)
        rows_by_db.setdefault(db_id, []).extend(rows)
        per_file_counts[str(path.relative_to(base_dir))] = len(rows)

    if not rows_by_db:
        raise ValueError("no rows were produced from discovered QA files")

    summary: dict[str, Any] = {
        "input_files": len(input_files),
        "per_file_counts": per_file_counts,
    }
    if args.output:
        output_path = Path(args.output)
        combined_rows: list[dict[str, Any]] = []
        for db_id in sorted(rows_by_db):
            combined_rows.extend(rows_by_db[db_id])
        result = assign_question_ids(combined_rows)
        write_json(output_path, result)
        summary.update(
            {
                "output_mode": "single_file",
                "output_rows": len(result),
                "output_path": str(output_path),
            }
        )
    else:
        per_db_outputs: dict[str, dict[str, Any]] = {}
        total_rows = 0
        for db_id in sorted(rows_by_db):
            result = assign_question_ids(rows_by_db[db_id])
            output_path = base_dir / db_id / args.per_db_output_name
            write_json(output_path, result)
            per_db_outputs[db_id] = {
                "output_rows": len(result),
                "output_path": str(output_path.relative_to(base_dir)),
            }
            total_rows += len(result)
        summary.update(
            {
                "output_mode": "per_db",
                "output_rows": total_rows,
                "db_count": len(per_db_outputs),
                "per_db_outputs": per_db_outputs,
            }
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
