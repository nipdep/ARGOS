#!/usr/bin/env python3
"""
Naturalize structured QA questions using DSPy with a local LM Studio model.

Input per DB:
- pq_pairs.json (from deterministic pair builder)

Output per DB:
- qa_pairs.json (or custom output name)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naturalize question text in pq/qa pair files via DSPy."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Base directory containing per-DB folders.",
    )
    parser.add_argument(
        "--db",
        action="append",
        help="Target DB id(s). Repeat to select multiple DBs. Defaults to all DB folders.",
    )
    parser.add_argument(
        "--input-name",
        default="pq_pairs.json",
        help="Input file name under each DB folder.",
    )
    parser.add_argument(
        "--output-name",
        default="qa_pairs.json",
        help="Output file name under each DB folder.",
    )
    parser.add_argument(
        "--combined-output",
        default="qa_pairs_all.json",
        help="Combined output file under base-dir. Set empty string to skip.",
    )
    parser.add_argument(
        "--target-field",
        default="question_naturalized",
        help="Field name to store naturalized question text.",
    )
    parser.add_argument(
        "--replace-question",
        action="store_true",
        help="Replace record['question'] with naturalized text. Original is stored as question_original.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='LM Studio loaded model id (example: "qwen/qwen3-4b-2507").',
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234/v1",
        help="LM Studio OpenAI-compatible base URL for DSPy LM client.",
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="API key for local endpoint (LM Studio accepts placeholder values).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for rewrite calls.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160,
        help="Max generated tokens per rewrite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N records per file (debug helper).",
    )
    parser.add_argument(
        "--ensure-lmstudio-sdk",
        action="store_true",
        help=(
            "Optional preflight: use lmstudio Python SDK to verify the model is reachable "
            "before DSPy rewriting starts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run rewriting pipeline without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def iter_db_dirs(base_dir: Path, selected_db_ids: set[str] | None, input_name: str) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if selected_db_ids and p.name not in selected_db_ids:
            continue
        if (p / input_name).exists():
            dirs.append(p)
    return dirs


def ensure_lmstudio_model_reachable(model_id: str) -> None:
    try:
        import lmstudio as lms  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "lmstudio SDK is not installed. Run: pip install lmstudio"
        ) from exc

    # LM Studio SDK convenience API. Creating the model handle is a lightweight
    # preflight that validates SDK connectivity + model routing.
    _ = lms.llm(model_id)


def build_dspy_rewriter(
    *,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
):
    try:
        import dspy  # type: ignore
    except Exception as exc:
        raise RuntimeError("dspy is not installed. Run: pip install dspy") from exc

    class NaturalizeFilterQuestion(dspy.Signature):
        """
        Rewrite a benchmark-style question into natural, user-like language.
        Preserve intent and constraints exactly.
        The output must be unambiguous and self-contained.
        Include every detail required to reproduce the expected SQL behavior specially with filtering conditions, grouping, and aggregation.
        For those look into the base SQL query and explicitly call out any operations or constraints that are present in the SQL but not clearly specified in the original question.
        Keep one sentence.
        Explicitly preserve operation details when present:
        - DISTINCT usage
        - ORDER BY column and direction
        - LIMIT value
        - filtering/grouping/aggregation cues from the source question
        Avoid vague terms like "appropriate", "relevant", or "some records".
        Do not add new columns, values, filters, or operations.
        """

        mode = dspy.InputField(desc="SQL operation mode: where, group_by, aggregate")
        original_question = dspy.InputField(desc="Original synthetic question text")
        base_query = dspy.InputField(desc="Reference SQL query")
        query_operations = dspy.InputField(
            desc="Optional structured ops metadata such as DISTINCT/ORDER BY/LIMIT"
        )
        projection_columns = dspy.InputField(desc="Columns returned in SELECT context")
        process_columns = dspy.InputField(desc="Columns used in filtering/grouping/aggregation")
        natural_question = dspy.OutputField(
            desc="Naturalized one-sentence question with full non-ambiguous constraints"
        )

    class Naturalizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(NaturalizeFilterQuestion)

        def forward(
            self,
            *,
            mode: str,
            original_question: str,
            base_query: str,
            query_operations: str,
            projection_columns: str,
            process_columns: str,
        ):
            return self.predict(
                mode=mode,
                original_question=original_question,
                base_query=base_query,
                query_operations=query_operations,
                projection_columns=projection_columns,
                process_columns=process_columns,
            )

    # DSPy with local LM Studio via OpenAI-compatible endpoint.
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    return Naturalizer()


def normalize_text(text: str) -> str:
    out = " ".join(text.strip().split())
    if out.startswith('"') and out.endswith('"') and len(out) > 1:
        out = out[1:-1].strip()
    return out


def stringify_columns(columns: Any, limit: int = 8) -> str:
    if not isinstance(columns, list):
        return ""
    cols = [str(c) for c in columns[:limit]]
    if len(columns) > limit:
        cols.append("...")
    return ", ".join(cols)


def naturalize_records(
    payload: dict[str, Any],
    rewriter: Any,
    *,
    target_field: str,
    replace_question: bool,
    limit: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Expected top-level key 'records' as a list.")

    changed = 0
    failed = 0
    total = 0

    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break
        if not isinstance(record, dict):
            failed += 1
            continue

        total += 1
        original_question = str(record.get("question", "")).strip()
        mode = str(record.get("mode", "")).strip()
        base_query = str(record.get("base_query", "")).strip()
        query_operations = json.dumps(record.get("ops", {}), ensure_ascii=True, sort_keys=True)
        projection_cols = stringify_columns(record.get("base_used_projection_columns", []))
        process_cols = stringify_columns(record.get("base_used_process_columns", []))

        if not original_question or not base_query:
            record[target_field] = original_question
            record["naturalization_error"] = "missing_question_or_base_query"
            failed += 1
            continue

        try:
            pred = rewriter(
                mode=mode,
                original_question=original_question,
                base_query=base_query,
                query_operations=query_operations,
                projection_columns=projection_cols,
                process_columns=process_cols,
            )
            naturalized = normalize_text(str(pred.natural_question))
            if not naturalized:
                naturalized = original_question

            if replace_question:
                if "question_original" not in record:
                    record["question_original"] = original_question
                record["question"] = naturalized
            record[target_field] = naturalized
            changed += 1

        except Exception as exc:  # noqa: BLE001
            record[target_field] = original_question
            record["naturalization_error"] = f"{type(exc).__name__}: {exc}"
            failed += 1

    stats = {
        "processed": total,
        "changed": changed,
        "failed": failed,
    }
    return payload, stats


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    if args.ensure_lmstudio_sdk:
        ensure_lmstudio_model_reachable(args.model)

    rewriter = build_dspy_rewriter(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    selected = set(args.db) if args.db else None
    db_dirs = iter_db_dirs(base_dir, selected, args.input_name)
    if not db_dirs:
        raise ValueError(
            f"No DB folders found with input file '{args.input_name}'."
        )

    combined: dict[str, Any] = {
        "base_dir": str(base_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "model": args.model,
        "api_base": args.api_base,
        "items": [],
    }

    for db_dir in db_dirs:
        src_path = db_dir / args.input_name
        payload = load_json(src_path)
        db_id = str(payload.get("db_id") or db_dir.name)

        updated_payload, stats = naturalize_records(
            payload=payload,
            rewriter=rewriter,
            target_field=args.target_field,
            replace_question=args.replace_question,
            limit=args.limit,
        )

        updated_payload["naturalization"] = {
            "framework": "dspy",
            "model": args.model,
            "api_base": args.api_base,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "target_field": args.target_field,
            "replace_question": args.replace_question,
            "processed_records": stats["processed"],
            "failed_records": stats["failed"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        print(
            f"[{db_id}] processed={stats['processed']} changed={stats['changed']} "
            f"failed={stats['failed']}"
        )

        if not args.dry_run:
            out_path = db_dir / args.output_name
            save_json(out_path, updated_payload)

        combined["items"].append(updated_payload)

    if not args.dry_run and args.combined_output:
        combined_path = base_dir / args.combined_output
        save_json(combined_path, combined)
        print(f"combined output -> {combined_path}")


if __name__ == "__main__":
    main()
