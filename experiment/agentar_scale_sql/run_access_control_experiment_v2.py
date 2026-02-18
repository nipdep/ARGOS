#!/usr/bin/env python3
"""
Run Agentar ScaleSQL text2sql experiments across access-control methods and
evaluate with P3T2Q metrics.

Modes:
1) base
2) dbms_access_control
3) argos_access_control
4) prompt_filtered_access_control
5) prompt_filtered_dbms_access_control
6) prompt_filtered_argos_access_control
7) view_filtered_access_control
8) view_filtered_dbms_access_control
9) view_filtered_argos_access_control
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[2]
SCALESQL_ROOT = ROOT / "experiment" / "agentar_scale_sql"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCALESQL_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALESQL_ROOT))

from scalesql.llms.llm import create_openai_llm
from scalesql.modules.keyword_extraction import KeywordExtractor
from scalesql.modules.dbms_access_control import (
    SQLiteAccessController,
    run_dbms_access_control_case,
)
from scalesql.modules.argos_access_control import (
    ArgosAccessController,
    run_argos_access_control_case,
)
from scalesql.modules.prompt_filtered_access_control import (
    apply_prompt_filtering_to_evidence,
    build_prompt_filtered_context,
)
from scalesql.modules.view_filtered_access_control import (
    build_filtered_column_descriptions,
    build_filtered_light_schema,
    build_view_filtered_context,
    load_column_description_map,
)

try:
    from scalesql.modules.retrieve import DatabaseCellRetrieval, skeleton_retrieve

    HAS_RETRIEVAL_MODULE = True
    RETRIEVAL_IMPORT_ERROR = ""
except Exception as exc:
    DatabaseCellRetrieval = None  # type: ignore[assignment]
    skeleton_retrieve = None  # type: ignore[assignment]
    HAS_RETRIEVAL_MODULE = False
    RETRIEVAL_IMPORT_ERROR = str(exc)

from p3t2q_benchmark_building.evaluate_qa_pipeline import (
    GTAnnotation,
    build_role_access_annotation,
    evaluate,
    extract_sql_candidate,
)
from src.operators.astObject import SqlglotOperator

ACCESS_CONTROL_MODES = [
    "base",
    "dbms_access_control",
    "argos_access_control",
    "prompt_filtered_access_control",
    "prompt_filtered_dbms_access_control",
    "prompt_filtered_argos_access_control",
    "view_filtered_access_control",
    "view_filtered_dbms_access_control",
    "view_filtered_argos_access_control",
]
MODE_TO_GENERATION_STRATEGY = {
    "base": "base",
    "dbms_access_control": "base",
    "argos_access_control": "base",
    "prompt_filtered_access_control": "prompt_filtered",
    "prompt_filtered_dbms_access_control": "prompt_filtered",
    "prompt_filtered_argos_access_control": "prompt_filtered",
    "view_filtered_access_control": "view_filtered",
    "view_filtered_dbms_access_control": "view_filtered",
    "view_filtered_argos_access_control": "view_filtered",
}
MODE_TO_POST_LAYER = {
    "base": "none",
    "dbms_access_control": "dbms",
    "argos_access_control": "argos",
    "prompt_filtered_access_control": "none",
    "prompt_filtered_dbms_access_control": "dbms",
    "prompt_filtered_argos_access_control": "argos",
    "view_filtered_access_control": "none",
    "view_filtered_dbms_access_control": "dbms",
    "view_filtered_argos_access_control": "argos",
}
DEFAULT_PER_DB_DATASET_FILENAME = "qa.json"
LEGACY_PER_DB_DATASET_FILENAME = "text_query_all.json"
REQUIRED_DATASET_COLUMNS = [
    "db_id",
    "question",
    "role",
    "intent_preserving_expected_query",
    "privacy_preserving_expected_query",
]
QUESTION_TYPE_CANONICAL_ORDER = [
    "filter only",
    "view only",
    "filter and view",
]


def resolve_access_control_modes(requested_modes: List[str] | None) -> List[str]:
    if not requested_modes:
        return list(ACCESS_CONTROL_MODES)
    requested_set = {str(mode).strip() for mode in requested_modes if str(mode).strip()}
    invalid = sorted(mode for mode in requested_set if mode not in ACCESS_CONTROL_MODES)
    if invalid:
        raise ValueError(
            f"invalid access control mode(s): {invalid}. "
            f"Allowed: {ACCESS_CONTROL_MODES}"
        )
    return [mode for mode in ACCESS_CONTROL_MODES if mode in requested_set]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agentar ScaleSQL experiments for multiple access-control methods."
    )
    parser.add_argument(
        "--db-root",
        default="data/P3T2Q_benchmark/v2",
        help=(
            "Benchmark root directory containing per-db folders. "
            "Default dataset loading reads <db-root>/<db_id>/bird_qa.json."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="",
        help=(
            "Optional explicit dataset JSON path. "
            "If omitted, load per-db datasets from <db-root>/<db_id>/bird_qa.json "
            "(fallback: text_query_all.json)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiment/outputs",
        help="Output directory for per-sample and aggregate results.",
    )
    parser.add_argument(
        "--output-prefix",
        default="agenta_scalesql",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        help="Optional db_id filter. Repeat for multiple DBs.",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Optional role filter. Repeat for multiple roles.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index (after db/role filtering).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End row index (exclusive). Use -1 for all remaining rows.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help=(
            "Refresh aggregate outputs (summary + question-type breakdown) every N processed samples. "
            "Set to 0 or a negative value to disable periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help=(
            "Update running outputs (raw outputs + per-sample eval + run_meta) every N processed samples. "
            "Uses the same output files (no extra checkpoint files). "
            "Set to 0 or a negative value to disable."
        ),
    )
    parser.add_argument(
        "--resume-meta",
        default="",
        help=(
            "Path to an existing *_run_meta.json file. "
            "When provided, the run resumes in-place and updates the same output files."
        ),
    )
    parser.add_argument(
        "--save-argos-failures",
        action="store_true",
        help=(
            "Write a separate JSON file containing only ARGOS-applied rows "
            "that fail at least one evaluation measure."
        ),
    )
    parser.add_argument(
        "--argos-failures-path",
        default="",
        help=(
            "Optional explicit output path for ARGOS failure rows JSON. "
            "If omitted and --save-argos-failures is set, a default file name is used."
        ),
    )
    parser.add_argument(
        "--access-control-mode",
        action="append",
        default=[],
        choices=ACCESS_CONTROL_MODES,
        help=(
            "Access-control mode to run. Repeat this flag to select a subset. "
            "Default: run all modes."
        ),
    )
    parser.add_argument(
        "--save-argos-db-abox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-db ARGOS ABOX files for this run (default: disabled).",
    )
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-20b",
        help="LLM model name.",
    )
    parser.add_argument(
        "--api-base",
        default="http://spark-6d47:1234/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request timeout for LLM calls.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Maximum retries for LLM calls.",
    )
    parser.add_argument(
        "--sql-temperature",
        type=float,
        default=0.2,
        help="Temperature for SQL generation model.",
    )
    parser.add_argument(
        "--keyword-temperature",
        type=float,
        default=0.0,
        help="Temperature for keyword extraction model.",
    )
    parser.add_argument(
        "--schema-sample-rows",
        type=int,
        default=3,
        help="Number of sample rows per column when building light schema.",
    )
    parser.add_argument(
        "--skeleton-chroma-path",
        default="/tmp/ScaleSQL/chroma/bird_train_skeleton",
        help="Chroma path for skeleton retrieval.",
    )
    parser.add_argument(
        "--skeleton-collection-name",
        default="bird_train_skeleton",
        help="Collection name for skeleton retrieval.",
    )
    parser.add_argument(
        "--cell-chroma-path",
        default="/tmp/ScaleSQL/chroma/bird_test",
        help="Chroma path for DB cell retrieval.",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Disable retrieval even if module and indexes are available.",
    )
    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for DB cell retrieval.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Top-k for retrieval.",
    )
    parser.add_argument(
        "--db-query-timeout-seconds",
        type=float,
        default=5.0,
        help="Timeout for SQLite query execution checks (DBMS + evaluation).",
    )
    parser.add_argument(
        "--argos-ontology-path",
        default="data/ontology_file/argos_v2.0.rdf",
        help="Ontology path used by ARGOS access-control refinement.",
    )
    parser.add_argument(
        "--max-generated-sql-chars",
        type=int,
        default=10000,
        help="Hard cap for generated SQL length before evaluation.",
    )
    return parser.parse_args()


class ScaleSQLAccessControlRunner:
    KEYWORD_SCHEMA_FALLBACK_CHARS = 7000
    KEYWORD_EVIDENCE_FALLBACK_CHARS = 1500
    SQL_SCHEMA_FALLBACK_CHARS = 7000
    SQL_COLUMNS_DESC_FALLBACK_CHARS = 2500
    SQL_EVIDENCE_FALLBACK_CHARS = 2000

    def __init__(self, args: argparse.Namespace, run_timestamp: str = ""):
        self.args = args
        self.db_root = Path(args.db_root)
        self.skeleton_chroma_path = Path(args.skeleton_chroma_path)
        self.cell_chroma_path = Path(args.cell_chroma_path)
        self.selected_modes = resolve_access_control_modes(getattr(args, "access_control_mode", []))
        self.save_argos_db_abox = bool(getattr(args, "save_argos_db_abox", True))
        resolved_run_timestamp = (
            str(run_timestamp).strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        self.argos_db_abox_dir: Path | None = None
        if self.save_argos_db_abox:
            self.argos_db_abox_dir = (
                Path(args.output_dir) / f"{args.output_prefix}_{resolved_run_timestamp}_db_abox"
            )
            self.argos_db_abox_dir.mkdir(parents=True, exist_ok=True)

        self.keyword_llm = create_openai_llm(
            model=args.model_name,
            base_url=args.api_base,
            api_key=args.api_key,
            temperature=args.keyword_temperature,
            max_retries=args.llm_max_retries,
            request_timeout_seconds=args.llm_timeout_seconds,
        )
        self.sql_generation_llm = create_openai_llm(
            model=args.model_name,
            base_url=args.api_base,
            api_key=args.api_key,
            temperature=args.sql_temperature,
            max_retries=args.llm_max_retries,
            request_timeout_seconds=args.llm_timeout_seconds,
        )
        self.keyword_extractor = KeywordExtractor(self.keyword_llm)

        self._base_context_cache: Dict[str, Dict[str, Any]] = {}
        self._view_context_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._prompt_context_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._dbms_controller_cache: Dict[str, SQLiteAccessController] = {}
        self._argos_controller_cache: Dict[str, ArgosAccessController] = {}

    @staticmethod
    def _extract_model_text(response: Any) -> str:
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    @staticmethod
    def _is_query_syntax_valid(sql_query: str | None) -> bool:
        candidate = extract_sql_candidate(sql_query)
        if not candidate:
            return False
        parsed = SqlglotOperator(candidate)
        return parsed.ast is not None

    def _clip_sql(self, sql_text: str) -> tuple[str, bool]:
        max_chars = int(self.args.max_generated_sql_chars)
        if max_chars <= 0:
            return sql_text, False
        if len(sql_text) <= max_chars:
            return sql_text, False
        return sql_text[:max_chars].rstrip(), True

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    @staticmethod
    def _is_context_overflow_error(error_text: str) -> bool:
        if not error_text:
            return False
        lowered = error_text.lower()
        return (
            "context the overflows" in lowered
            or "context length of only" in lowered
            or "maximum context length" in lowered
            or "prompt is too long" in lowered
            or "too many tokens" in lowered
        )

    def run_keyword_extraction(self, question: str, evidence: str, schema: str) -> Dict[str, Any]:
        payload = {
            "Database Schema": schema,
            "Question": question,
            "Evidence": evidence or "",
        }
        try:
            return self.keyword_extractor.invoke(payload)
        except Exception as exc:
            base_error = str(exc)
            if self._is_context_overflow_error(base_error):
                compact_payload = {
                    "Database Schema": self._truncate_text(
                        payload["Database Schema"],
                        self.KEYWORD_SCHEMA_FALLBACK_CHARS,
                    ),
                    "Question": payload["Question"],
                    "Evidence": self._truncate_text(
                        payload["Evidence"],
                        self.KEYWORD_EVIDENCE_FALLBACK_CHARS,
                    ),
                }
                try:
                    result = self.keyword_extractor.invoke(compact_payload)
                    if isinstance(result, dict):
                        result["warning"] = "keyword_context_overflow_retried_with_compact_payload"
                    return result
                except Exception as retry_exc:
                    return {
                        "database_literals": [],
                        "question_skeleton": "",
                        "error": f"{base_error} | compact-retry failed: {retry_exc}",
                    }
            return {"database_literals": [], "question_skeleton": "", "error": base_error}

    def try_retrieve_similar_skeletons(self, question_skeleton: str) -> List[str]:
        if self.args.disable_retrieval:
            return []
        if not HAS_RETRIEVAL_MODULE:
            return []
        if not question_skeleton or not question_skeleton.strip():
            return []
        if not self.skeleton_chroma_path.exists():
            return []

        try:
            return skeleton_retrieve(
                skeleton_client_path=str(self.skeleton_chroma_path),
                skeleton_collection_name=self.args.skeleton_collection_name,
                question_skeleton=question_skeleton,
                k=self.args.retrieval_k,
            )
        except Exception:
            return []

    def try_retrieve_db_cells(self, db_id: str, database_literals: List[str]) -> List[Dict[str, Any]]:
        if self.args.disable_retrieval:
            return []
        if not HAS_RETRIEVAL_MODULE:
            return []
        if not database_literals:
            return []
        if not self.cell_chroma_path.exists():
            return []

        try:
            retriever = DatabaseCellRetrieval(
                database_literals=database_literals,
                search_client=str(self.cell_chroma_path),
                collection_name=db_id,
            )
            return retriever.retrieve(
                threshold=self.args.retrieval_threshold,
                k=self.args.retrieval_k,
            )
        except Exception:
            return []

    @staticmethod
    def _format_similar_examples(similar_examples: List[str]) -> str:
        if not similar_examples:
            return "(none)"
        return "\n\n".join(similar_examples[:5])

    @staticmethod
    def _format_matched_contents(matched_contents: List[Dict[str, Any]]) -> str:
        if not matched_contents:
            return "(none)"
        lines = []
        for item in matched_contents[:30]:
            table = item.get("table", "")
            column = item.get("column", "")
            content = item.get("content", "")
            lines.append(f"- {table}.{column}: {content}")
        return "\n".join(lines)

    def build_sql_generation_prompt(
        self,
        *,
        question: str,
        evidence: str,
        schema: str,
        columns_descriptions: str,
        db_id: str,
        question_skeleton: str,
        database_literals: List[str],
        similar_examples: List[str],
        matched_contents: List[Dict[str, Any]],
    ) -> str:
        literals_text = ", ".join(map(str, database_literals)) if database_literals else "(none)"
        return f"""You are a senior SQLite text-to-SQL engineer.
Return exactly one executable SQL query for the question.
Do not include explanations.

Database ID: {db_id}

Database Schema:
{schema}

Column Descriptions:
{columns_descriptions}

Question:
{question}

Evidence:
{evidence or "(none)"}

Extracted Question Skeleton:
{question_skeleton or "(none)"}

Extracted Database Literals:
{literals_text}

Similar Solved Examples:
{self._format_similar_examples(similar_examples)}

Matched Database Cell Values:
{self._format_matched_contents(matched_contents)}

Output SQL only."""

    def run_scalesql_generator(
        self,
        *,
        question: str,
        evidence: str,
        db_id: str,
        schema: str,
        columns_descriptions: str,
    ) -> Dict[str, Any]:
        keyword_result = self.run_keyword_extraction(
            question=question,
            evidence=evidence,
            schema=schema,
        )
        database_literals = keyword_result.get("database_literals", []) or []
        question_skeleton = keyword_result.get("question_skeleton", "") or ""

        similar_examples = self.try_retrieve_similar_skeletons(question_skeleton=question_skeleton)
        matched_contents = self.try_retrieve_db_cells(
            db_id=db_id,
            database_literals=database_literals,
        )

        generation_prompt = self.build_sql_generation_prompt(
            question=question,
            evidence=evidence,
            schema=schema,
            columns_descriptions=columns_descriptions,
            db_id=db_id,
            question_skeleton=question_skeleton,
            database_literals=database_literals,
            similar_examples=similar_examples,
            matched_contents=matched_contents,
        )

        try:
            try:
                response = self.sql_generation_llm.invoke([HumanMessage(content=generation_prompt)])
            except Exception:
                response = self.sql_generation_llm.invoke(generation_prompt)
            raw_text = self._extract_model_text(response)
            final_query, was_clipped = self._clip_sql(extract_sql_candidate(raw_text) or "")
            error_text = ""
            if was_clipped:
                error_text = (
                    f"Generated SQL clipped to {self.args.max_generated_sql_chars} chars "
                    f"for safety."
                )
        except Exception as exc:
            raw_text = ""
            final_query = ""
            error_text = str(exc)
            if self._is_context_overflow_error(error_text):
                compact_prompt = self.build_sql_generation_prompt(
                    question=question,
                    evidence=self._truncate_text(evidence, self.SQL_EVIDENCE_FALLBACK_CHARS),
                    schema=self._truncate_text(schema, self.SQL_SCHEMA_FALLBACK_CHARS),
                    columns_descriptions=self._truncate_text(
                        columns_descriptions,
                        self.SQL_COLUMNS_DESC_FALLBACK_CHARS,
                    ),
                    db_id=db_id,
                    question_skeleton=question_skeleton,
                    database_literals=database_literals,
                    similar_examples=[],
                    matched_contents=[],
                )
                try:
                    try:
                        retry_response = self.sql_generation_llm.invoke(
                            [HumanMessage(content=compact_prompt)]
                        )
                    except Exception:
                        retry_response = self.sql_generation_llm.invoke(compact_prompt)
                    raw_text = self._extract_model_text(retry_response)
                    final_query, was_clipped = self._clip_sql(extract_sql_candidate(raw_text) or "")
                    generation_prompt = compact_prompt
                    if was_clipped:
                        error_text = (
                            "sql_generation_context_overflow_retried_with_compact_prompt; "
                            f"generated SQL clipped to {self.args.max_generated_sql_chars} chars for safety."
                        )
                    else:
                        error_text = (
                            "sql_generation_context_overflow_retried_with_compact_prompt; "
                            f"original_error={error_text}"
                        )
                except Exception as retry_exc:
                    raw_text = ""
                    final_query = ""
                    error_text = (
                        f"{error_text} | compact-prompt retry failed: {retry_exc}"
                    )

        return {
            "final_query": final_query,
            "raw_response_text": raw_text,
            "error": error_text,
            "keyword_extraction": keyword_result,
            "similar_examples": similar_examples,
            "matched_contents": matched_contents,
            "generation_prompt": generation_prompt,
        }

    def get_base_context(self, db_id: str) -> Dict[str, Any]:
        if db_id in self._base_context_cache:
            return self._base_context_cache[db_id]

        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        desc_dir = self.db_root / db_id / "database_description"
        col_desc_map = load_column_description_map(str(desc_dir))

        schema = build_filtered_light_schema(
            db_path=str(db_path),
            denied_tables=set(),
            denied_columns=set(),
            sample_rows=self.args.schema_sample_rows,
            column_descriptions=col_desc_map,
        )
        columns_descriptions = build_filtered_column_descriptions(
            database_description_dir=str(desc_dir),
            denied_tables=set(),
            denied_columns=set(),
        )

        context = {
            "db_path": str(db_path),
            "database_description_dir": str(desc_dir),
            "schema": schema,
            "columns_descriptions": columns_descriptions,
        }
        self._base_context_cache[db_id] = context
        return context

    def get_view_context(self, db_id: str, role: str) -> Dict[str, Any]:
        key = (db_id, role)
        if key not in self._view_context_cache:
            self._view_context_cache[key] = build_view_filtered_context(
                benchmark_root=str(self.db_root),
                db_id=db_id,
                role=role,
                sample_rows=self.args.schema_sample_rows,
            )
        return self._view_context_cache[key]

    def get_prompt_context(self, db_id: str, role: str) -> Dict[str, Any]:
        key = (db_id, role)
        if key not in self._prompt_context_cache:
            self._prompt_context_cache[key] = build_prompt_filtered_context(
                benchmark_root=str(self.db_root),
                db_id=db_id,
                role=role,
                sample_rows=self.args.schema_sample_rows,
            )
        return self._prompt_context_cache[key]

    def get_dbms_controller(self, db_id: str) -> SQLiteAccessController:
        if db_id not in self._dbms_controller_cache:
            self._dbms_controller_cache[db_id] = SQLiteAccessController.from_benchmark_db(
                benchmark_root=str(self.db_root),
                db_id=db_id,
                query_timeout_seconds=self.args.db_query_timeout_seconds,
            )
        return self._dbms_controller_cache[db_id]

    def get_argos_controller(self, db_id: str) -> ArgosAccessController:
        if db_id not in self._argos_controller_cache:
            ontology_path = Path(self.args.argos_ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = ROOT / ontology_path
            db_abox_output_path = (
                self.argos_db_abox_dir / f"{db_id}_abox.rdf"
                if self.argos_db_abox_dir is not None
                else None
            )
            self._argos_controller_cache[db_id] = ArgosAccessController.from_benchmark_db(
                benchmark_root=str(self.db_root),
                db_id=db_id,
                ontology_path=str(ontology_path),
                save_db_abox=self.save_argos_db_abox,
                db_abox_output_path=str(db_abox_output_path) if db_abox_output_path is not None else None,
            )
        return self._argos_controller_cache[db_id]

    def run_all_modes_for_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        question = sample["question"]
        evidence = str(sample.get("evidence", "") or "")
        db_id = sample["db_id"]
        role = sample["role"]
        mode_outputs: List[Dict[str, Any]] = []
        selected_modes = set(self.selected_modes)
        required_strategies = {
            MODE_TO_GENERATION_STRATEGY[mode]
            for mode in selected_modes
            if mode in MODE_TO_GENERATION_STRATEGY
        }
        generation_results: Dict[str, Dict[str, Any]] = {}

        if "base" in required_strategies:
            base_ctx = self.get_base_context(db_id)
            base_result = self.run_scalesql_generator(
                question=question,
                evidence=evidence,
                db_id=db_id,
                schema=base_ctx["schema"],
                columns_descriptions=base_ctx["columns_descriptions"],
            )
            base_result["source_generation_strategy"] = "base"
            generation_results["base"] = base_result

        if "prompt_filtered" in required_strategies:
            prompt_ctx = self.get_prompt_context(db_id, role)
            prompt_evidence = apply_prompt_filtering_to_evidence(
                evidence=evidence,
                restriction_hint=prompt_ctx.get("restriction_hint", ""),
            )
            prompt_result = self.run_scalesql_generator(
                question=question,
                evidence=prompt_evidence,
                db_id=db_id,
                schema=prompt_ctx["schema"],
                columns_descriptions=prompt_ctx["columns_descriptions"],
            )
            prompt_result["source_generation_strategy"] = "prompt_filtered"
            prompt_result["restriction_hint"] = prompt_ctx.get("restriction_hint", "")
            prompt_result["restricted_tables"] = sorted(prompt_ctx.get("restricted_tables", []))
            prompt_result["restricted_columns"] = sorted(prompt_ctx.get("restricted_columns", []))
            generation_results["prompt_filtered"] = prompt_result

        if "view_filtered" in required_strategies:
            view_ctx = self.get_view_context(db_id, role)
            view_result = self.run_scalesql_generator(
                question=question,
                evidence=evidence,
                db_id=db_id,
                schema=view_ctx["schema"],
                columns_descriptions=view_ctx["columns_descriptions"],
            )
            view_result["source_generation_strategy"] = "view_filtered"
            view_result["denied_tables"] = sorted(view_ctx.get("denied_tables", []))
            view_result["denied_columns"] = sorted(view_ctx.get("denied_columns", []))
            generation_results["view_filtered"] = view_result

        for mode in ACCESS_CONTROL_MODES:
            if mode not in selected_modes:
                continue

            strategy = MODE_TO_GENERATION_STRATEGY[mode]
            post_layer = MODE_TO_POST_LAYER[mode]
            source_result = generation_results.get(strategy, {})
            source_query = str(source_result.get("final_query", "") or "")
            source_query_syntax_valid = self._is_query_syntax_valid(source_query)

            if post_layer == "none":
                mode_result = dict(source_result)
                mode_result["access_control_mode"] = mode
                mode_outputs.append(mode_result)
                continue

            if post_layer == "dbms":
                controller = self.get_dbms_controller(db_id)
                post_result = run_dbms_access_control_case(
                    controller=controller,
                    role=role,
                    sql_query=source_query,
                )
                mode_result = dict(source_result)
                mode_result.update(post_result)
                mode_result["access_control_mode"] = mode
                mode_result["source_query_before_dbms"] = source_query
                mode_outputs.append(mode_result)
                continue

            if post_layer == "argos":
                if not source_query_syntax_valid:
                    skip_reason = f"Skipped ARGOS: {strategy} query is syntactically invalid"
                    post_result = {
                        "final_query": "",
                        "answer_metadata": {
                            "query": source_query,
                            "refined_query": "",
                            "invalid_refined_query": "",
                            "argos_access_control_status": "skipped_invalid_source_query",
                            "argos_access_control_error": skip_reason,
                            "role": role,
                            "db_id": db_id,
                            "active_policies": [],
                            "active_rules": [],
                            "table_status": {},
                            "column_status": {},
                        },
                        "error": skip_reason,
                        "skip_evaluation": True,
                        "skip_evaluation_reason": f"{strategy}_query_syntax_invalid",
                    }
                else:
                    argos_controller = self.get_argos_controller(db_id)
                    post_result = run_argos_access_control_case(
                        controller=argos_controller,
                        role=role,
                        sql_query=source_query,
                    )
                mode_result = dict(source_result)
                mode_result.update(post_result)
                mode_result["access_control_mode"] = mode
                mode_result["source_query_before_argos"] = source_query
                mode_outputs.append(mode_result)
                continue

        return mode_outputs

    def evaluate_mode_output(
        self,
        sample: Dict[str, Any],
        mode_output: Dict[str, Any],
        sample_position: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        generation_prompt = mode_output.get("generation_prompt")
        prompt_char_size = (
            len(generation_prompt) if isinstance(generation_prompt, str) else None
        )
        metadata = sample.get("metadata", {}) or {}
        gt_annotation = GTAnnotation(
            expected_query=sample.get("privacy_preserving_expected_query"),
            expected_projection_columns=set(metadata.get("allowed_view_columns", [])),
            expected_process_columns=set(metadata.get("allowed_process_columns", [])),
            expected_is_usable=bool(
                metadata.get("queryable", sample.get("privacy_preserving_expected_query") is not None)
            ),
            base_query=sample.get("intent_preserving_expected_query"),
        )
        role_access_annotation = build_role_access_annotation(
            db_root=self.db_root,
            db_id=sample["db_id"],
            role=sample.get("role", "public"),
            access_control_config=metadata.get("access_control_config"),
        )

        eval_row, contrib = evaluate(
            question=sample.get("question"),
            role=sample.get("role", "public"),
            gt_annotation=gt_annotation,
            role_access_annotation=role_access_annotation,
            generated_query=mode_output.get("final_query", ""),
            db_id=sample["db_id"],
            db_root=self.db_root,
            sample_id=f"{sample.get('question_id')}::{mode_output.get('access_control_mode')}",
            template_id=metadata.get("template_id", ""),
            mode=metadata.get("mode", ""),
            row_index=int(sample.get("question_id", 0)),
            query_timeout_seconds=self.args.db_query_timeout_seconds,
            prompt_char_size=prompt_char_size,
        )

        eval_row.update(
            {
                "question_id": sample.get("question_id"),
                "sample_position": int(sample_position),
                "access_control_mode": mode_output.get("access_control_mode"),
                "question_type": normalize_question_type(sample.get("question_type")),
                "intent_preserving_expected_query": sample.get("intent_preserving_expected_query"),
                "privacy_preserving_expected_query": sample.get("privacy_preserving_expected_query"),
                "generator_raw_text": mode_output.get("raw_response_text", ""),
                "generator_error": mode_output.get("error", ""),
                "generation_prompt_char_size": (
                    int(prompt_char_size) if prompt_char_size is not None else 0
                ),
                "keyword_extraction": mode_output.get("keyword_extraction", {}),
                "similar_examples_count": len(mode_output.get("similar_examples", [])),
                "matched_contents_count": len(mode_output.get("matched_contents", [])),
                "dbms_answer_metadata": (
                    mode_output.get("answer_metadata", {})
                    if str(mode_output.get("access_control_mode", "")).strip().endswith(
                        "dbms_access_control"
                    )
                    else {}
                ),
                "argos_answer_metadata": (
                    mode_output.get("answer_metadata", {})
                    if str(mode_output.get("access_control_mode", "")).strip().endswith(
                        "argos_access_control"
                    )
                    else {}
                ),
                "source_query_before_dbms": mode_output.get("source_query_before_dbms", ""),
                "source_query_before_argos": mode_output.get("source_query_before_argos", ""),
            }
        )
        contrib.update(
            {
                "question_id": sample.get("question_id"),
                "sample_position": int(sample_position),
                "access_control_mode": mode_output.get("access_control_mode"),
            }
        )
        return eval_row, contrib

    def close(self) -> None:
        for controller in self._dbms_controller_cache.values():
            try:
                controller.close()
            except Exception:
                pass
        self._dbms_controller_cache.clear()
        for controller in self._argos_controller_cache.values():
            try:
                controller.close()
            except Exception:
                pass
        self._argos_controller_cache.clear()


def _aggregate_group_metrics(group: pd.DataFrame) -> Dict[str, Any]:
    policy_total = group["policy_total_cols"].sum()
    policy_binary_total = group["policy_viol_binary_total"].sum()
    direct_total = group["direct_total_cols"].sum()
    answer_total = group["answer_total"].sum()
    exec_total = group["exec_total"].sum()
    syntax_total = group["syntax_total"].sum()
    expected_query_match_total = group["expected_query_match_total"].sum()
    intent_total = group["intent_total"].sum()
    prompt_char_size_total = group["prompt_char_size_total"].sum()

    return {
        "sample_count": int(len(group)),
        "policy_violation_rate": (
            group["policy_viol_count"].sum() / policy_total if policy_total else 0.0
        ),
        "policy_violation_binary_rate": (
            group["policy_viol_binary_count"].sum() / policy_binary_total
            if policy_binary_total
            else 0.0
        ),
        "policy_violation_binary_scored_count": int(policy_binary_total),
        "direct_disclosure_rate": (
            group["direct_viol_count"].sum() / direct_total if direct_total else 0.0
        ),
        "answer_agreement": (
            group["answer_match"].sum() / answer_total if answer_total else 0.0
        ),
        "answer_agreement_scored_count": int(answer_total),
        "execution_success_rate": (
            group["exec_success"].sum() / exec_total if exec_total else 0.0
        ),
        "execution_success_scored_count": int(exec_total),
        "syntactic_correctness_rate": (
            group["syntax_success"].sum() / syntax_total if syntax_total else 0.0
        ),
        "syntactic_correctness_scored_count": int(syntax_total),
        "expected_query_matching_ratio": (
            group["expected_query_match_sum"].sum() / expected_query_match_total
            if expected_query_match_total
            else 0.0
        ),
        "expected_query_matching_ratio_scored_count": int(expected_query_match_total),
        "intent_preservation": (
            group["intent_sum"].sum() / intent_total if intent_total else 0.0
        ),
        "intent_preservation_scored_count": int(intent_total),
        "prompt_char_size_mean": (
            group["prompt_char_size_sum"].sum() / prompt_char_size_total
            if prompt_char_size_total
            else 0.0
        ),
        "prompt_char_size_scored_count": int(prompt_char_size_total),
    }


def _build_base_relative_question_ids(base_group: pd.DataFrame) -> set[Any]:
    base_relative_question_ids: set[Any] = set()
    if base_group is None or base_group.empty:
        return base_relative_question_ids

    # Relative cohort: questions where base mode has an exact APTED-based
    # intent-preservation match (score == 1.0) and syntax passed.
    epsilon = 1e-9
    base_relative_df = base_group[
        (base_group["intent_total"] > 0)
        & (base_group["intent_sum"] >= (1.0 - epsilon))
        & (base_group["syntax_total"] > 0)
        & (base_group["syntax_success"] > 0)
    ]
    for question_id in base_relative_df["question_id"].tolist():
        if question_id is None:
            continue
        if isinstance(question_id, float) and pd.isna(question_id):
            continue
        base_relative_question_ids.add(question_id)
    return base_relative_question_ids


def aggregate_mode_scores(contrib_df: pd.DataFrame) -> pd.DataFrame:
    mode_groups = {
        str(mode): group.copy()
        for mode, group in contrib_df.groupby("access_control_mode", dropna=False)
    }
    rows: List[Dict[str, Any]] = []

    base_group = mode_groups.get("base")
    base_relative_question_ids = _build_base_relative_question_ids(base_group)

    base_relative_question_count = int(len(base_relative_question_ids))

    for mode in sorted(mode_groups.keys()):
        group = mode_groups[mode]
        overall_metrics = _aggregate_group_metrics(group)
        subset_group = (
            group[group["question_id"].isin(base_relative_question_ids)]
            if base_relative_question_ids
            else group.iloc[0:0]
        )
        relative_metrics = _aggregate_group_metrics(subset_group)

        row: Dict[str, Any] = {
            "access_control_mode": mode,
            **overall_metrics,
            "relative_base_correct_question_count": base_relative_question_count,
            "relative_base_intent_exact_match_question_count": base_relative_question_count,
            "relative_sample_count": int(relative_metrics["sample_count"]),
            "relative_policy_violation_rate": relative_metrics["policy_violation_rate"],
            "relative_policy_violation_binary_rate": relative_metrics[
                "policy_violation_binary_rate"
            ],
            "relative_policy_violation_binary_scored_count": int(
                relative_metrics["policy_violation_binary_scored_count"]
            ),
            "relative_direct_disclosure_rate": relative_metrics["direct_disclosure_rate"],
            "relative_answer_agreement": relative_metrics["answer_agreement"],
            "relative_answer_agreement_scored_count": int(
                relative_metrics["answer_agreement_scored_count"]
            ),
            "relative_execution_success_rate": relative_metrics["execution_success_rate"],
            "relative_execution_success_scored_count": int(
                relative_metrics["execution_success_scored_count"]
            ),
            "relative_syntactic_correctness_rate": relative_metrics["syntactic_correctness_rate"],
            "relative_syntactic_correctness_scored_count": int(
                relative_metrics["syntactic_correctness_scored_count"]
            ),
            "relative_expected_query_matching_ratio": relative_metrics[
                "expected_query_matching_ratio"
            ],
            "relative_expected_query_matching_ratio_scored_count": int(
                relative_metrics["expected_query_matching_ratio_scored_count"]
            ),
            "relative_intent_preservation": relative_metrics["intent_preservation"],
            "relative_intent_preservation_scored_count": int(
                relative_metrics["intent_preservation_scored_count"]
            ),
            "relative_prompt_char_size_mean": relative_metrics["prompt_char_size_mean"],
            "relative_prompt_char_size_scored_count": int(
                relative_metrics["prompt_char_size_scored_count"]
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("access_control_mode").reset_index(drop=True)


def build_question_type_breakdown(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    working_df = eval_df.copy().reset_index(drop=True)
    if "question_type" not in working_df.columns:
        working_df["question_type"] = "unspecified"
    working_df["question_type"] = working_df["question_type"].apply(normalize_question_type)

    contrib_rows = [contrib_from_eval_row(row) for row in working_df.to_dict(orient="records")]
    contrib_df = pd.DataFrame(contrib_rows)
    if contrib_df.empty:
        return pd.DataFrame()
    contrib_df["question_type"] = working_df["question_type"]

    mode_qtype_groups = {
        (str(mode), str(question_type)): group.copy()
        for (mode, question_type), group in contrib_df.groupby(
            ["access_control_mode", "question_type"], dropna=False
        )
    }
    base_by_qtype: Dict[str, set[Any]] = {}
    base_qtype_groups = contrib_df[contrib_df["access_control_mode"] == "base"].groupby(
        "question_type", dropna=False
    )
    for question_type, base_group in base_qtype_groups:
        base_by_qtype[str(question_type)] = _build_base_relative_question_ids(base_group.copy())

    rows: List[Dict[str, Any]] = []
    for (mode, question_type), group in mode_qtype_groups.items():
        overall_metrics = _aggregate_group_metrics(group)
        base_relative_question_ids = base_by_qtype.get(question_type, set())
        subset_group = (
            group[group["question_id"].isin(base_relative_question_ids)]
            if base_relative_question_ids
            else group.iloc[0:0]
        )
        relative_metrics = _aggregate_group_metrics(subset_group)

        row: Dict[str, Any] = {
            "access_control_mode": mode,
            "question_type": question_type,
            **overall_metrics,
            # Backward-compatible aliases for existing notebooks/plots.
            "mean_policy_violation_rate": overall_metrics["policy_violation_rate"],
            "mean_policy_violation_binary": overall_metrics["policy_violation_binary_rate"],
            "mean_direct_disclosure_rate": overall_metrics["direct_disclosure_rate"],
            "mean_execution_success": overall_metrics["execution_success_rate"],
            "mean_syntactic_correctness": overall_metrics["syntactic_correctness_rate"],
            "mean_prompt_char_size": overall_metrics["prompt_char_size_mean"],
            "relative_base_correct_question_count": int(len(base_relative_question_ids)),
            "relative_base_intent_exact_match_question_count": int(len(base_relative_question_ids)),
            "relative_sample_count": int(relative_metrics["sample_count"]),
            "relative_policy_violation_rate": relative_metrics["policy_violation_rate"],
            "relative_policy_violation_binary_rate": relative_metrics[
                "policy_violation_binary_rate"
            ],
            "relative_policy_violation_binary_scored_count": int(
                relative_metrics["policy_violation_binary_scored_count"]
            ),
            "relative_direct_disclosure_rate": relative_metrics["direct_disclosure_rate"],
            "relative_answer_agreement": relative_metrics["answer_agreement"],
            "relative_answer_agreement_scored_count": int(
                relative_metrics["answer_agreement_scored_count"]
            ),
            "relative_execution_success_rate": relative_metrics["execution_success_rate"],
            "relative_execution_success_scored_count": int(
                relative_metrics["execution_success_scored_count"]
            ),
            "relative_syntactic_correctness_rate": relative_metrics["syntactic_correctness_rate"],
            "relative_syntactic_correctness_scored_count": int(
                relative_metrics["syntactic_correctness_scored_count"]
            ),
            "relative_expected_query_matching_ratio": relative_metrics[
                "expected_query_matching_ratio"
            ],
            "relative_expected_query_matching_ratio_scored_count": int(
                relative_metrics["expected_query_matching_ratio_scored_count"]
            ),
            "relative_intent_preservation": relative_metrics["intent_preservation"],
            "relative_intent_preservation_scored_count": int(
                relative_metrics["intent_preservation_scored_count"]
            ),
            "relative_prompt_char_size_mean": relative_metrics["prompt_char_size_mean"],
            "relative_prompt_char_size_scored_count": int(
                relative_metrics["prompt_char_size_scored_count"]
            ),
        }
        rows.append(row)

    breakdown_df = pd.DataFrame(rows)
    if breakdown_df.empty:
        return breakdown_df
    breakdown_df["_question_type_sort"] = breakdown_df["question_type"].apply(
        lambda x: _question_type_sort_key(x)[0]
    )
    return (
        breakdown_df.sort_values(
            ["_question_type_sort", "question_type", "access_control_mode"]
        )
        .drop(columns=["_question_type_sort"])
        .reset_index(drop=True)
    )


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, float) and pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_binary(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return 1
        if normalized in {"0", "false", "no", "n", ""}:
            return 0
    return 1 if _safe_float(value) != 0 else 0


def normalize_question_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unspecified"

    normalized = raw.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return "unspecified"

    has_filter = "filter" in normalized
    has_view = "view" in normalized
    if has_filter and has_view:
        return "filter and view"
    if has_filter:
        return "filter only"
    if has_view or "projection" in normalized or "select only" in normalized:
        return "view only"
    return normalized


def _question_type_sort_key(value: Any) -> tuple[int, str]:
    normalized = normalize_question_type(value)
    if normalized in QUESTION_TYPE_CANONICAL_ORDER:
        return (QUESTION_TYPE_CANONICAL_ORDER.index(normalized), normalized)
    if normalized == "unspecified":
        return (len(QUESTION_TYPE_CANONICAL_ORDER) + 1, normalized)
    return (len(QUESTION_TYPE_CANONICAL_ORDER), normalized)


def contrib_from_eval_row(eval_row: Dict[str, Any]) -> Dict[str, Any]:
    execution_scored = _safe_binary(eval_row.get("execution_scored", 1))
    syntax_scored = _safe_binary(eval_row.get("syntax_scored", 1))
    policy_violation_binary = _safe_binary(
        eval_row.get(
            "policy_violation_binary",
            1 if _safe_int(eval_row.get("policy_violation_count", 0)) > 0 else 0,
        )
    )
    expected_query_ratio = _safe_float(
        eval_row.get(
            "expected_query_matching_ratio_match_count",
            eval_row.get("minimal_distortion_match_count", 0),
        )
    )
    expected_query_ratio_scored = _safe_float(
        eval_row.get(
            "expected_query_matching_ratio_scored",
            eval_row.get("minimal_distortion_scored", 0),
        )
    )
    prompt_char_size_scored = _safe_binary(eval_row.get("prompt_char_size_scored", 0))
    prompt_char_size_value = _safe_int(eval_row.get("prompt_char_size", 0))
    return {
        "policy_viol_count": _safe_int(eval_row.get("policy_violation_count", 0)),
        "policy_total_cols": _safe_int(eval_row.get("policy_total_columns", 0)),
        "policy_viol_binary_count": policy_violation_binary,
        "policy_viol_binary_total": 1,
        "direct_viol_count": _safe_int(eval_row.get("direct_disclosure_count", 0)),
        "direct_total_cols": _safe_int(eval_row.get("direct_total_columns", 0)),
        "answer_match": _safe_binary(eval_row.get("answer_agreement", 0)),
        "answer_total": _safe_binary(eval_row.get("answer_scored", 0)),
        "exec_success": _safe_binary(eval_row.get("execution_success", 0)) if execution_scored else 0,
        "exec_total": execution_scored,
        "syntax_success": _safe_binary(eval_row.get("syntactic_correct", 0)) if syntax_scored else 0,
        "syntax_total": syntax_scored,
        "expected_query_match_sum": expected_query_ratio if expected_query_ratio_scored else 0.0,
        "expected_query_match_total": expected_query_ratio_scored,
        "intent_sum": _safe_float(eval_row.get("intent_preservation_match_count", 0)),
        "intent_total": _safe_float(eval_row.get("intent_preservation_scored", 0)),
        "prompt_char_size_sum": prompt_char_size_value if prompt_char_size_scored else 0,
        "prompt_char_size_total": prompt_char_size_scored,
        "question_id": eval_row.get("question_id"),
        "access_control_mode": eval_row.get("access_control_mode"),
    }


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _normalize_filters(values: List[str] | None) -> List[str]:
    if not values:
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def _resolve_per_db_dataset_path(db_root: Path, db_id: str) -> Path:
    db_dir = db_root / db_id
    candidates = [
        db_dir / DEFAULT_PER_DB_DATASET_FILENAME,
        db_dir / LEGACY_PER_DB_DATASET_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no dataset file found for db '{db_id}'. "
        f"Expected one of: {candidates[0]} or {candidates[1]}"
    )


def _load_experiment_dataset_file(path: Path, db_id_hint: str = "") -> pd.DataFrame:
    raw_df = pd.read_json(path)
    if not isinstance(raw_df, pd.DataFrame):
        raw_df = pd.DataFrame(raw_df)
    dataset_df = raw_df.copy()

    if dataset_df.empty:
        if db_id_hint:
            dataset_df["db_id"] = []
        return dataset_df

    if "db_id" not in dataset_df.columns:
        if not db_id_hint:
            raise ValueError(f"dataset missing required column 'db_id': {path}")
        dataset_df["db_id"] = db_id_hint

    if "question_id" not in dataset_df.columns:
        dataset_df["question_id"] = list(range(1, len(dataset_df) + 1))
    if "evidence" not in dataset_df.columns:
        dataset_df["evidence"] = ""
    if "question_type" not in dataset_df.columns:
        dataset_df["question_type"] = ""
    if "metadata" not in dataset_df.columns:
        dataset_df["metadata"] = [{} for _ in range(len(dataset_df))]

    missing_columns = [col for col in REQUIRED_DATASET_COLUMNS if col not in dataset_df.columns]
    if missing_columns:
        raise ValueError(
            f"dataset {path} is missing required columns: {missing_columns}. "
            "Expected a consolidated P3T2Q access-control dataset."
        )
    return dataset_df


def load_experiment_dataset(
    *,
    db_root: Path,
    dataset: str,
    db_filters: List[str],
) -> tuple[pd.DataFrame, str, List[str]]:
    if dataset.strip():
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {dataset_path}")
        dataset_df = _load_experiment_dataset_file(dataset_path)
        return dataset_df.reset_index(drop=True), str(dataset_path), [str(dataset_path)]

    if not db_root.exists():
        raise FileNotFoundError(f"db root not found: {db_root}")

    requested_db_ids = _normalize_filters(db_filters)
    if requested_db_ids:
        db_ids = requested_db_ids
    else:
        db_ids = sorted(
            entry.name
            for entry in db_root.iterdir()
            if entry.is_dir() and (entry / "schema.json").exists()
        )

    dataframes: List[pd.DataFrame] = []
    dataset_files: List[str] = []
    missing_requested: List[str] = []

    for db_id in db_ids:
        try:
            dataset_path = _resolve_per_db_dataset_path(db_root=db_root, db_id=db_id)
        except FileNotFoundError:
            if requested_db_ids:
                missing_requested.append(db_id)
            continue
        db_df = _load_experiment_dataset_file(dataset_path, db_id_hint=db_id)
        if db_df.empty:
            continue
        dataframes.append(db_df)
        dataset_files.append(str(dataset_path))

    if missing_requested:
        raise FileNotFoundError(
            "dataset files not found for requested db(s): "
            + ", ".join(sorted(missing_requested))
        )
    if not dataframes:
        raise ValueError(
            "no dataset rows loaded. "
            "Expected per-db files like <db_root>/<db_id>/bird_qa.json."
        )

    dataset_df = pd.concat(dataframes, ignore_index=True)
    source_label = f"per-db ({len(dataset_files)} files)"
    return dataset_df.reset_index(drop=True), source_label, dataset_files


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(text)
    tmp_path.replace(path)


def _write_records_df(path: Path, df: pd.DataFrame) -> None:
    json_text = df.to_json(orient="records", force_ascii=False, indent=2)
    _atomic_write_text(path, f"{json_text}\n")


def _default_argos_failures_path(eval_path: Path) -> Path:
    name = eval_path.name
    if name.endswith("_per_sample_eval.json"):
        return eval_path.with_name(name.replace("_per_sample_eval.json", "_argos_failures.json"))
    return eval_path.with_name(f"{eval_path.stem}_argos_failures.json")


def _argos_failure_reasons(eval_row: Dict[str, Any]) -> List[str]:
    if not str(eval_row.get("access_control_mode", "")).strip().endswith("argos_access_control"):
        return []

    policy_violation_binary = _safe_binary(
        eval_row.get(
            "policy_violation_binary",
            1 if _safe_int(eval_row.get("policy_violation_count", 0)) > 0 else 0,
        )
    )
    if policy_violation_binary == 1:
        return ["policy_violation"]
    return []


def _build_argos_failure_rows(evaluation_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failure_rows: List[Dict[str, Any]] = []
    for row in evaluation_rows:
        reasons = _argos_failure_reasons(row)
        if not reasons:
            continue
        enriched_row = dict(row)
        enriched_row["argos_failure_reasons"] = reasons
        failure_rows.append(enriched_row)
    return failure_rows


def persist_run_outputs(
    *,
    eval_path: Path,
    summary_path: Path,
    breakdown_path: Path,
    raw_path: Path,
    meta_path: Path,
    evaluation_rows: List[Dict[str, Any]],
    contrib_rows: List[Dict[str, Any]],
    all_mode_outputs: List[Dict[str, Any]],
    meta_template: Dict[str, Any],
    processed_samples: int,
    total_samples: int,
    checkpoint_every: int,
    progress_every: int,
    checkpoint_reason: str,
    is_final: bool,
    include_aggregates: bool,
    argos_failures_path: Path | None = None,
    save_argos_failures: bool = False,
    run_error: str = "",
) -> pd.DataFrame:
    eval_df = pd.DataFrame(evaluation_rows)
    contrib_df = pd.DataFrame(contrib_rows)
    outputs_df = pd.DataFrame(all_mode_outputs)
    summary_df = pd.DataFrame()
    argos_failure_rows: List[Dict[str, Any]] = []

    _write_records_df(eval_path, eval_df)
    _write_records_df(raw_path, outputs_df)
    if save_argos_failures and argos_failures_path is not None:
        argos_failure_rows = _build_argos_failure_rows(evaluation_rows)
        _write_records_df(argos_failures_path, pd.DataFrame(argos_failure_rows))
    if include_aggregates:
        summary_df = aggregate_mode_scores(contrib_df) if not contrib_df.empty else pd.DataFrame()
        breakdown_df = build_question_type_breakdown(eval_df)
        _write_records_df(summary_path, summary_df)
        _write_records_df(breakdown_path, breakdown_df)

    meta_obj = dict(meta_template)
    meta_obj.update(
        {
            "processed_samples": int(processed_samples),
            "total_samples": int(total_samples),
            "checkpoint_every": int(checkpoint_every),
            "progress_every": int(progress_every),
            "last_checkpoint_reason": checkpoint_reason,
            "aggregates_refreshed": bool(include_aggregates),
            "is_final": bool(is_final),
            "run_error": run_error,
            "written_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "argos_failure_count": int(len(argos_failure_rows)) if save_argos_failures else 0,
            "argos_failure_output_enabled": bool(save_argos_failures and argos_failures_path is not None),
        }
    )
    _atomic_write_text(meta_path, f"{json.dumps(meta_obj, indent=2, ensure_ascii=False)}\n")
    return summary_df


def main() -> None:
    args = parse_args()
    args.access_control_mode = resolve_access_control_modes(args.access_control_mode)
    dataset_df, dataset_source, dataset_files = load_experiment_dataset(
        db_root=Path(args.db_root),
        dataset=str(args.dataset or ""),
        db_filters=list(args.db or []),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.db:
        dataset_df = dataset_df[dataset_df["db_id"].isin(set(args.db))].copy()
    if args.role:
        dataset_df = dataset_df[dataset_df["role"].isin(set(args.role))].copy()
    dataset_df = dataset_df.reset_index(drop=True)

    start_index = max(0, int(args.start_index))
    end_index = len(dataset_df) if args.end_index < 0 else min(int(args.end_index), len(dataset_df))
    exp_df = dataset_df.iloc[start_index:end_index].reset_index(drop=True)

    resume_meta_obj: Dict[str, Any] = {}
    is_resuming = bool(args.resume_meta)
    argos_failures_path: Path | None = None
    if is_resuming:
        meta_path = Path(args.resume_meta)
        if not meta_path.exists():
            raise FileNotFoundError(f"resume meta not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        if not isinstance(loaded_meta, dict):
            raise ValueError(f"resume meta is invalid: {meta_path}")
        resume_meta_obj = loaded_meta
        output_files = resume_meta_obj.get("output_files", {}) or {}
        eval_path_str = str(output_files.get("per_sample_eval", "")).strip()
        summary_path_str = str(output_files.get("mode_summary", "")).strip()
        breakdown_path_str = str(output_files.get("question_type_breakdown", "")).strip()
        raw_path_str = str(output_files.get("raw_outputs", "")).strip()
        argos_failures_path_str = str(output_files.get("argos_failures", "")).strip()
        if not eval_path_str or not summary_path_str or not breakdown_path_str or not raw_path_str:
            raise ValueError(f"resume meta output file paths are invalid: {meta_path}")
        eval_path = Path(eval_path_str)
        summary_path = Path(summary_path_str)
        breakdown_path = Path(breakdown_path_str)
        raw_path = Path(raw_path_str)
        if args.argos_failures_path.strip():
            argos_failures_path = Path(args.argos_failures_path)
        elif argos_failures_path_str:
            argos_failures_path = Path(argos_failures_path_str)
        elif args.save_argos_failures:
            argos_failures_path = _default_argos_failures_path(eval_path)
        timestamp = str(
            resume_meta_obj.get("timestamp_utc")
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end_label = "all" if args.end_index < 0 else str(end_index)
        prefix = f"{args.output_prefix}_{timestamp}_{start_index}_{end_label}"
        eval_path = output_dir / f"{prefix}_per_sample_eval.json"
        summary_path = output_dir / f"{prefix}_mode_summary.json"
        breakdown_path = output_dir / f"{prefix}_mode_question_type_breakdown.json"
        raw_path = output_dir / f"{prefix}_raw_outputs.json"
        meta_path = output_dir / f"{prefix}_run_meta.json"
        if args.argos_failures_path.strip():
            argos_failures_path = Path(args.argos_failures_path)
        elif args.save_argos_failures:
            argos_failures_path = output_dir / f"{prefix}_argos_failures.json"

    save_argos_failures = argos_failures_path is not None

    for path in [eval_path, summary_path, breakdown_path, raw_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if argos_failures_path is not None:
        argos_failures_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] dataset_source={dataset_source}")
    if not args.dataset:
        print(f"[info] dataset_files_loaded={len(dataset_files)}")
    print(f"[info] filtered_rows={len(dataset_df)} | experiment_rows={len(exp_df)}")
    print(f"[info] db_filter={args.db or '(all)'} role_filter={args.role or '(all)'}")
    print(f"[info] access_control_modes={args.access_control_mode}")
    print(f"[info] save_argos_db_abox={args.save_argos_db_abox}")
    print(f"[info] retrieval_module_available={HAS_RETRIEVAL_MODULE}")
    print(f"[info] checkpoint_every={args.checkpoint_every}")
    print(f"[info] progress_every={args.progress_every}")
    print(f"[info] resuming={is_resuming}")
    if is_resuming:
        prev_processed = _safe_int(resume_meta_obj.get("processed_samples", 0))
        print(f"[info] resume_meta={meta_path} previous_processed={prev_processed}")
        prev_args = resume_meta_obj.get("args", {}) or {}
        for key in [
            "db_root",
            "dataset",
            "db",
            "role",
            "start_index",
            "end_index",
            "access_control_mode",
            "save_argos_db_abox",
        ]:
            if key in prev_args and prev_args.get(key) != getattr(args, key, None):
                print(
                    f"[warn] resume arg mismatch for {key}: "
                    f"previous={prev_args.get(key)} current={getattr(args, key, None)}"
                )
    print("[info] output files:")
    print(f"  - {eval_path}")
    print(f"  - {summary_path}")
    print(f"  - {breakdown_path}")
    print(f"  - {raw_path}")
    if argos_failures_path is not None:
        print(f"  - {argos_failures_path}")
    print(f"  - {meta_path}")
    if not HAS_RETRIEVAL_MODULE:
        print(f"[warn] retrieval disabled due to import error: {RETRIEVAL_IMPORT_ERROR}")

    runner = ScaleSQLAccessControlRunner(args, run_timestamp=timestamp)
    if runner.argos_db_abox_dir is not None:
        print(f"[info] argos_db_abox_dir={runner.argos_db_abox_dir}")
    else:
        print("[info] argos_db_abox_dir=(disabled)")
    all_mode_outputs: List[Dict[str, Any]] = read_json_records(raw_path) if is_resuming else []
    evaluation_rows: List[Dict[str, Any]] = read_json_records(eval_path) if is_resuming else []
    contrib_rows: List[Dict[str, Any]] = [contrib_from_eval_row(row) for row in evaluation_rows]
    processed_samples = 0
    if is_resuming:
        meta_processed = _safe_int(resume_meta_obj.get("processed_samples", 0))
        sample_positions = [
            _safe_int(row.get("sample_position"))
            for row in evaluation_rows
            if row.get("sample_position") is not None
        ]
        inferred_processed = (max(sample_positions) + 1) if sample_positions else 0
        if not sample_positions and evaluation_rows:
            unique_question_ids = {
                row.get("question_id") for row in evaluation_rows if row.get("question_id") is not None
            }
            inferred_processed = len(unique_question_ids)
        processed_samples = max(meta_processed, inferred_processed)
    processed_samples = min(processed_samples, len(exp_df))
    if is_resuming:
        print(f"[info] resume_start_position={processed_samples}/{len(exp_df)}")
    run_error = ""

    meta_template = {
        "timestamp_utc": timestamp,
        "args": vars(args),
        "dataset_path": str(dataset_source),
        "dataset_files": dataset_files,
        "filtered_rows": int(len(dataset_df)),
        "experiment_rows": int(len(exp_df)),
        "retrieval_module_available": HAS_RETRIEVAL_MODULE,
        "retrieval_import_error": RETRIEVAL_IMPORT_ERROR,
        "output_files": {
            "per_sample_eval": str(eval_path),
            "mode_summary": str(summary_path),
            "question_type_breakdown": str(breakdown_path),
            "raw_outputs": str(raw_path),
            "argos_failures": str(argos_failures_path) if argos_failures_path is not None else "",
            "argos_db_abox_dir": str(runner.argos_db_abox_dir) if runner.argos_db_abox_dir else "",
        },
        "resumed_from_meta": str(meta_path) if is_resuming else "",
    }
    summary_df = persist_run_outputs(
        eval_path=eval_path,
        summary_path=summary_path,
        breakdown_path=breakdown_path,
        raw_path=raw_path,
        meta_path=meta_path,
        evaluation_rows=evaluation_rows,
        contrib_rows=contrib_rows,
        all_mode_outputs=all_mode_outputs,
        meta_template=meta_template,
        processed_samples=processed_samples,
        total_samples=len(exp_df),
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        checkpoint_reason="resume_start" if is_resuming else "start",
        is_final=False,
        include_aggregates=True,
        argos_failures_path=argos_failures_path,
        save_argos_failures=save_argos_failures,
    )

    try:
        for sample_position in range(processed_samples, len(exp_df)):
            sample = exp_df.iloc[sample_position].to_dict()
            print(
                f"[run] {sample_position + 1}/{len(exp_df)} "
                f"question_id={sample.get('question_id')} "
                f"db={sample.get('db_id')} role={sample.get('role')}"
            )
            mode_outputs = runner.run_all_modes_for_sample(sample)
            for mode_output in mode_outputs:
                mode_name = mode_output.get("access_control_mode", "")
                mode_error = str(mode_output.get("error", "") or "").strip()
                generation_prompt = mode_output.get("generation_prompt")
                generation_prompt_char_size = (
                    len(generation_prompt) if isinstance(generation_prompt, str) else 0
                )
                if mode_error:
                    print(
                        f"[warn] question_id={sample.get('question_id')} "
                        f"mode={mode_name} error={mode_error}"
                    )
                if mode_name.endswith("argos_access_control"):
                    argos_meta = mode_output.get("answer_metadata", {}) or {}
                    if isinstance(argos_meta, dict):
                        argos_error = str(argos_meta.get("argos_access_control_error", "") or "").strip()
                        if argos_error:
                            print(
                                f"[warn] question_id={sample.get('question_id')} "
                                f"mode={mode_name} argos_error={argos_error}"
                            )
                all_mode_outputs.append(
                    {
                        "question_id": sample.get("question_id"),
                        "sample_position": int(sample_position),
                        "db_id": sample.get("db_id"),
                        "role": sample.get("role"),
                        "access_control_mode": mode_output.get("access_control_mode"),
                        "final_query": mode_output.get("final_query", ""),
                        "source_query_before_dbms": mode_output.get("source_query_before_dbms", ""),
                        "source_query_before_argos": mode_output.get("source_query_before_argos", ""),
                        "error": mode_output.get("error", ""),
                        "generation_prompt_char_size": int(generation_prompt_char_size),
                        "skip_evaluation": bool(mode_output.get("skip_evaluation", False)),
                        "skip_evaluation_reason": mode_output.get("skip_evaluation_reason", ""),
                    }
                )
                if mode_output.get("skip_evaluation", False):
                    continue
                eval_row, contrib = runner.evaluate_mode_output(
                    sample,
                    mode_output,
                    sample_position=sample_position,
                )
                evaluation_rows.append(eval_row)
                contrib_rows.append(contrib)
            processed_samples = sample_position + 1

            should_checkpoint = (
                args.checkpoint_every > 0 and processed_samples % args.checkpoint_every == 0
            )
            should_progress = args.progress_every > 0 and processed_samples % args.progress_every == 0

            if should_checkpoint:
                summary_df = persist_run_outputs(
                    eval_path=eval_path,
                    summary_path=summary_path,
                    breakdown_path=breakdown_path,
                    raw_path=raw_path,
                    meta_path=meta_path,
                    evaluation_rows=evaluation_rows,
                    contrib_rows=contrib_rows,
                    all_mode_outputs=all_mode_outputs,
                    meta_template=meta_template,
                    processed_samples=processed_samples,
                    total_samples=len(exp_df),
                    checkpoint_every=args.checkpoint_every,
                    progress_every=args.progress_every,
                    checkpoint_reason=f"checkpoint_{processed_samples}",
                    is_final=False,
                    include_aggregates=True,
                    argos_failures_path=argos_failures_path,
                    save_argos_failures=save_argos_failures,
                )
                print(f"[checkpoint] saved progress at {processed_samples}/{len(exp_df)} samples")
            elif should_progress:
                persist_run_outputs(
                    eval_path=eval_path,
                    summary_path=summary_path,
                    breakdown_path=breakdown_path,
                    raw_path=raw_path,
                    meta_path=meta_path,
                    evaluation_rows=evaluation_rows,
                    contrib_rows=contrib_rows,
                    all_mode_outputs=all_mode_outputs,
                    meta_template=meta_template,
                    processed_samples=processed_samples,
                    total_samples=len(exp_df),
                    checkpoint_every=args.checkpoint_every,
                    progress_every=args.progress_every,
                    checkpoint_reason=f"progress_{processed_samples}",
                    is_final=False,
                    include_aggregates=False,
                    argos_failures_path=argos_failures_path,
                    save_argos_failures=save_argos_failures,
                )
    except Exception as exc:
        run_error = str(exc)
        persist_run_outputs(
            eval_path=eval_path,
            summary_path=summary_path,
            breakdown_path=breakdown_path,
            raw_path=raw_path,
            meta_path=meta_path,
            evaluation_rows=evaluation_rows,
            contrib_rows=contrib_rows,
            all_mode_outputs=all_mode_outputs,
            meta_template=meta_template,
            processed_samples=processed_samples,
            total_samples=len(exp_df),
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            checkpoint_reason="error",
            is_final=False,
            include_aggregates=True,
            argos_failures_path=argos_failures_path,
            save_argos_failures=save_argos_failures,
            run_error=run_error,
        )
        print(f"[error] run interrupted after {processed_samples}/{len(exp_df)} samples: {run_error}")
        raise
    finally:
        runner.close()
    summary_df = persist_run_outputs(
        eval_path=eval_path,
        summary_path=summary_path,
        breakdown_path=breakdown_path,
        raw_path=raw_path,
        meta_path=meta_path,
        evaluation_rows=evaluation_rows,
        contrib_rows=contrib_rows,
        all_mode_outputs=all_mode_outputs,
        meta_template=meta_template,
        processed_samples=processed_samples,
        total_samples=len(exp_df),
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        checkpoint_reason="final",
        is_final=True,
        include_aggregates=True,
        argos_failures_path=argos_failures_path,
        save_argos_failures=save_argos_failures,
    )

    print("[done] outputs written:")
    print(f"  - {eval_path}")
    print(f"  - {summary_path}")
    print(f"  - {breakdown_path}")
    print(f"  - {raw_path}")
    if argos_failures_path is not None:
        print(f"  - {argos_failures_path}")
    if runner.argos_db_abox_dir is not None:
        print(f"  - {runner.argos_db_abox_dir}")
    print(f"  - {meta_path}")

    if not summary_df.empty:
        print("\n[summary]")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
