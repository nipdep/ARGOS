#!/usr/bin/env python3
"""
Run Agentar ScaleSQL access-control evaluation using llama_cpp Python library
(direct local inference, no OpenAI-compatible HTTP server).

This script keeps the same output schema as run_access_control_experiment.py:
1) *_per_sample_eval.json
2) *_mode_summary.json
3) *_mode_question_type_breakdown.json
4) *_raw_outputs.json
5) *_run_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCALESQL_ROOT = ROOT / "experiment" / "agentar_scale_sql"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCALESQL_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALESQL_ROOT))

from run_access_control_experiment import (  # noqa: E402
    ACCESS_CONTROL_MODES,
    HAS_RETRIEVAL_MODULE,
    RETRIEVAL_IMPORT_ERROR,
    _default_argos_failures_path,
    _safe_int,
    contrib_from_eval_row,
    load_experiment_dataset,
    persist_run_outputs,
    read_json_records,
    resolve_access_control_modes,
)
from scalesql.modules.dbms_access_control import (  # noqa: E402
    SQLiteAccessController,
    run_dbms_access_control_case,
)
from scalesql.modules.argos_access_control import (  # noqa: E402
    ArgosAccessController,
    run_argos_access_control_case,
)
from scalesql.modules.prompt_filtered_access_control import (  # noqa: E402
    apply_prompt_filtering_to_evidence,
    build_prompt_filtered_context,
)
from scalesql.modules.view_filtered_access_control import (  # noqa: E402
    build_filtered_column_descriptions,
    build_filtered_light_schema,
    build_view_filtered_context,
    load_column_description_map,
)

try:
    from scalesql.modules.retrieve import DatabaseCellRetrieval, skeleton_retrieve  # noqa: E402
except Exception:
    DatabaseCellRetrieval = None  # type: ignore[assignment]
    skeleton_retrieve = None  # type: ignore[assignment]

from p3t2q_benchmark_building.evaluate_qa_pipeline import (  # noqa: E402
    GTAnnotation,
    build_role_access_annotation,
    evaluate,
    extract_sql_candidate,
)
from src.operators.astObject import SqlglotOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Agentar ScaleSQL runner using direct llama_cpp Python library calls."
        )
    )
    parser.add_argument(
        "--db-root",
        default="data/P3T2Q_benchmark/v1",
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
        default="agenta_llamacpp_batch",
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
            "Write a separate JSON file containing only argos_access_control rows "
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
        default=True,
        help="Persist per-db ARGOS ABOX files for this run (default: enabled).",
    )

    parser.add_argument(
        "--sql-model-path",
        required=True,
        help=(
            "Path to GGUF model file (or directory containing GGUF files) "
            "used for SQL generation (llama_cpp.Llama)."
        ),
    )
    parser.add_argument(
        "--keyword-model-path",
        default="",
        help=(
            "Optional GGUF model file (or directory) for keyword extraction. "
            "Defaults to --sql-model-path."
        ),
    )
    parser.add_argument(
        "--llama-chat-format",
        default="",
        help="Optional llama_cpp chat_format override (e.g., chatml, llama-2).",
    )
    parser.add_argument(
        "--llama-n-ctx",
        type=int,
        default=8192,
        help="Context size for llama_cpp model.",
    )
    parser.add_argument(
        "--llama-n-batch",
        type=int,
        default=512,
        help="Token batch size for llama_cpp model.",
    )
    parser.add_argument(
        "--llama-n-gpu-layers",
        type=int,
        default=-1,
        help="Number of model layers to offload to GPU (-1 for all possible).",
    )
    parser.add_argument(
        "--llama-n-threads",
        type=int,
        default=8,
        help="CPU threads for llama_cpp sampling/tokenization.",
    )
    parser.add_argument(
        "--llama-max-tokens",
        type=int,
        default=512,
        help="Max generation tokens for SQL generation calls.",
    )
    parser.add_argument(
        "--keyword-max-tokens",
        type=int,
        default=256,
        help="Max generation tokens for keyword extraction calls.",
    )
    parser.add_argument(
        "--sql-temperature",
        type=float,
        default=0.2,
        help="Temperature for SQL generation.",
    )
    parser.add_argument(
        "--keyword-temperature",
        type=float,
        default=0.0,
        help="Temperature for keyword extraction.",
    )
    parser.add_argument(
        "--llama-top-p",
        type=float,
        default=0.95,
        help="Top-p for llama_cpp generation.",
    )
    parser.add_argument(
        "--llama-repeat-penalty",
        type=float,
        default=1.05,
        help="Repeat penalty for llama_cpp generation.",
    )
    parser.add_argument(
        "--llama-verbose",
        action="store_true",
        help="Enable llama_cpp verbose logs.",
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

    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=8,
        help="Samples processed before the next batch-level persistence pass.",
    )
    return parser.parse_args()


class LlamaCppAccessControlRunner:
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

        try:
            import llama_cpp  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "llama_cpp is not installed. Install with: pip install llama-cpp-python"
            ) from exc

        self._llama_cpp = llama_cpp
        sql_model_input = Path(args.sql_model_path).expanduser()
        keyword_model_input = (
            Path(args.keyword_model_path).expanduser()
            if args.keyword_model_path
            else sql_model_input
        )
        sql_model_path = self._resolve_gguf_model_path(sql_model_input, label="SQL")
        keyword_model_path = self._resolve_gguf_model_path(
            keyword_model_input,
            label="keyword",
        )
        print(f"[info] sql_model_file={sql_model_path}")
        print(f"[info] keyword_model_file={keyword_model_path}")

        self.sql_llm = self._build_llama_model(sql_model_path)
        if keyword_model_path.resolve() == sql_model_path.resolve():
            self.keyword_llm = self.sql_llm
        else:
            self.keyword_llm = self._build_llama_model(keyword_model_path)

        self._base_context_cache: Dict[str, Dict[str, Any]] = {}
        self._view_context_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._prompt_context_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._dbms_controller_cache: Dict[str, SQLiteAccessController] = {}
        self._argos_controller_cache: Dict[str, ArgosAccessController] = {}

    def _build_llama_model(self, model_path: Path):
        llama_kwargs: Dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": int(self.args.llama_n_ctx),
            "n_batch": int(self.args.llama_n_batch),
            "n_gpu_layers": int(self.args.llama_n_gpu_layers),
            "n_threads": int(self.args.llama_n_threads),
            "verbose": bool(self.args.llama_verbose),
        }
        chat_format = str(self.args.llama_chat_format).strip()
        if chat_format:
            llama_kwargs["chat_format"] = chat_format
        try:
            return self._llama_cpp.Llama(**llama_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load llama.cpp model from '{model_path}'. "
                "Use a valid GGUF file path (or a directory containing one). "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _iter_gguf_files(root: Path) -> List[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root] if root.suffix.lower() == ".gguf" else []
        candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".gguf"]
        return sorted(candidates)

    @staticmethod
    def _select_best_gguf(candidates: List[Path]) -> Path:
        def rank(path: Path) -> tuple[int, int]:
            name = path.name.lower()
            # Prefer higher-precision quantizations when multiple candidates exist.
            quant_rank = 0
            if "q8_0" in name:
                quant_rank = 6
            elif "q6" in name:
                quant_rank = 5
            elif "q5" in name:
                quant_rank = 4
            elif "q4" in name:
                quant_rank = 3
            elif "q3" in name:
                quant_rank = 2
            elif "q2" in name:
                quant_rank = 1
            try:
                size_rank = int(path.stat().st_size)
            except OSError:
                size_rank = 0
            return quant_rank, size_rank

        return max(candidates, key=rank)

    def _resolve_gguf_model_path(self, raw_path: Path, *, label: str) -> Path:
        if not raw_path.exists():
            raise FileNotFoundError(f"{label} model path not found: {raw_path}")

        if raw_path.is_file():
            if raw_path.suffix.lower() != ".gguf":
                raise ValueError(
                    f"{label} model path must point to a .gguf file or a directory containing .gguf files: {raw_path}"
                )
            return raw_path

        candidates = self._iter_gguf_files(raw_path)
        if not candidates:
            raise FileNotFoundError(f"No .gguf files found under {label} model directory: {raw_path}")
        if len(candidates) == 1:
            return candidates[0]

        chosen = self._select_best_gguf(candidates)
        print(
            f"[warn] Multiple GGUF files found for {label} model. "
            f"Using '{chosen.name}'."
        )
        return chosen

    def _chat(self, *, llm, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=float(self.args.llama_top_p),
            repeat_penalty=float(self.args.llama_repeat_penalty),
        )
        if not isinstance(response, dict):
            return str(response)
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        raw = text.strip()
        if not raw:
            return {}
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass

        matches = re.findall(r"\{[\s\S]*?\}", raw)
        for candidate in matches:
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                continue
        return {}

    def _extract_literals_fallback(self, question: str, evidence: str) -> List[str]:
        text = f"{question} {evidence}"
        quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
        literals = [a or b for a, b in quoted]
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        literals.extend(numbers)
        out: List[str] = []
        seen = set()
        for lit in literals:
            value = str(lit).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= 20:
                break
        return out

    def _clip_sql(self, sql_text: str) -> tuple[str, bool]:
        max_chars = int(self.args.max_generated_sql_chars)
        if max_chars <= 0:
            return sql_text, False
        if len(sql_text) <= max_chars:
            return sql_text, False
        return sql_text[:max_chars].rstrip(), True

    @staticmethod
    def _is_query_syntax_valid(sql_query: str | None) -> bool:
        candidate = extract_sql_candidate(sql_query)
        if not candidate:
            return False
        parsed = SqlglotOperator(candidate)
        return parsed.ast is not None

    def run_keyword_extraction(self, question: str, evidence: str, schema: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a text-to-SQL analysis assistant. "
            "Return only valid JSON with keys: database_literals (array of strings), "
            "question_skeleton (string)."
        )
        user_prompt = (
            "Identify literals from question/evidence and produce a generalized question skeleton.\n\n"
            f"Database Schema:\n{schema}\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence or '(none)'}\n\n"
            "Rules:\n"
            "- database_literals should contain concrete values likely used in filters.\n"
            "- question_skeleton should keep intent words and replace specific values/columns with placeholders.\n"
            "- Output JSON only."
        )
        try:
            raw = self._chat(
                llm=self.keyword_llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=int(self.args.keyword_max_tokens),
                temperature=float(self.args.keyword_temperature),
            )
            parsed = self._extract_json_object(raw)
            literals = parsed.get("database_literals")
            if not isinstance(literals, list):
                literals = self._extract_literals_fallback(question, evidence)
            cleaned_literals = []
            seen = set()
            for lit in literals:
                value = str(lit).strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned_literals.append(value)
                if len(cleaned_literals) >= 20:
                    break

            skeleton = parsed.get("question_skeleton")
            if not isinstance(skeleton, str) or not skeleton.strip():
                skeleton = question

            return {
                "database_literals": cleaned_literals,
                "question_skeleton": skeleton.strip(),
                "raw_response": raw,
            }
        except Exception as exc:
            return {
                "database_literals": self._extract_literals_fallback(question, evidence),
                "question_skeleton": question,
                "error": str(exc),
            }

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
            raw_text = self._chat(
                llm=self.sql_llm,
                system_prompt="You are a precise SQL generator.",
                user_prompt=generation_prompt,
                max_tokens=int(self.args.llama_max_tokens),
                temperature=float(self.args.sql_temperature),
            )
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
        base_result: Dict[str, Any] | None = None
        base_final_query = ""
        base_query_syntax_valid = False

        if {"base", "dbms_access_control", "argos_access_control"} & selected_modes:
            base_ctx = self.get_base_context(db_id)
            base_result = self.run_scalesql_generator(
                question=question,
                evidence=evidence,
                db_id=db_id,
                schema=base_ctx["schema"],
                columns_descriptions=base_ctx["columns_descriptions"],
            )
            base_final_query = (base_result or {}).get("final_query", "")
            base_query_syntax_valid = self._is_query_syntax_valid(base_final_query)
            if "base" in selected_modes:
                base_result["access_control_mode"] = "base"
                mode_outputs.append(base_result)

        if "dbms_access_control" in selected_modes:
            controller = self.get_dbms_controller(db_id)
            dbms_result = run_dbms_access_control_case(
                controller=controller,
                role=role,
                sql_query=base_final_query,
            )
            dbms_result["access_control_mode"] = "dbms_access_control"
            dbms_result["source_query_before_dbms"] = base_final_query
            mode_outputs.append(dbms_result)

        if "argos_access_control" in selected_modes:
            if not base_query_syntax_valid:
                skip_reason = "Skipped ARGOS: base query is syntactically invalid"
                argos_result = {
                    "final_query": "",
                    "answer_metadata": {
                        "query": base_final_query,
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
                    "skip_evaluation_reason": "base_query_syntax_invalid",
                }
            else:
                argos_controller = self.get_argos_controller(db_id)
                argos_result = run_argos_access_control_case(
                    controller=argos_controller,
                    role=role,
                    sql_query=base_final_query,
                )
            argos_result["access_control_mode"] = "argos_access_control"
            argos_result["source_query_before_argos"] = base_final_query
            mode_outputs.append(argos_result)

        if "view_filtered_access_control" in selected_modes:
            view_ctx = self.get_view_context(db_id, role)
            view_result = self.run_scalesql_generator(
                question=question,
                evidence=evidence,
                db_id=db_id,
                schema=view_ctx["schema"],
                columns_descriptions=view_ctx["columns_descriptions"],
            )
            view_result["access_control_mode"] = "view_filtered_access_control"
            view_result["denied_tables"] = sorted(view_ctx.get("denied_tables", []))
            view_result["denied_columns"] = sorted(view_ctx.get("denied_columns", []))
            mode_outputs.append(view_result)

        if "prompt_filtered_access_control" in selected_modes:
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
            prompt_result["access_control_mode"] = "prompt_filtered_access_control"
            prompt_result["restriction_hint"] = prompt_ctx.get("restriction_hint", "")
            prompt_result["restricted_tables"] = sorted(prompt_ctx.get("restricted_tables", []))
            prompt_result["restricted_columns"] = sorted(prompt_ctx.get("restricted_columns", []))
            mode_outputs.append(prompt_result)

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
                "question_type": sample.get("question_type"),
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
                    if mode_output.get("access_control_mode") == "dbms_access_control"
                    else {}
                ),
                "argos_answer_metadata": (
                    mode_output.get("answer_metadata", {})
                    if mode_output.get("access_control_mode") == "argos_access_control"
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

        for llm in [self.keyword_llm, self.sql_llm]:
            close_fn = getattr(llm, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def _resolve_output_paths(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    start_index: int,
    end_index: int,
) -> tuple[Dict[str, Any], Dict[str, Path], str]:
    resume_meta_obj: Dict[str, Any] = {}
    is_resuming = bool(args.resume_meta)
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
        output_paths = {
            "eval": Path(eval_path_str),
            "summary": Path(summary_path_str),
            "breakdown": Path(breakdown_path_str),
            "raw": Path(raw_path_str),
            "meta": meta_path,
        }
        if args.argos_failures_path.strip():
            output_paths["argos_failures"] = Path(args.argos_failures_path)
        elif argos_failures_path_str:
            output_paths["argos_failures"] = Path(argos_failures_path_str)
        elif args.save_argos_failures:
            output_paths["argos_failures"] = _default_argos_failures_path(output_paths["eval"])
        timestamp = str(
            resume_meta_obj.get("timestamp_utc")
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end_label = "all" if args.end_index < 0 else str(end_index)
        prefix = f"{args.output_prefix}_{timestamp}_{start_index}_{end_label}"
        output_paths = {
            "eval": output_dir / f"{prefix}_per_sample_eval.json",
            "summary": output_dir / f"{prefix}_mode_summary.json",
            "breakdown": output_dir / f"{prefix}_mode_question_type_breakdown.json",
            "raw": output_dir / f"{prefix}_raw_outputs.json",
            "meta": output_dir / f"{prefix}_run_meta.json",
        }
        if args.argos_failures_path.strip():
            output_paths["argos_failures"] = Path(args.argos_failures_path)
        elif args.save_argos_failures:
            output_paths["argos_failures"] = output_dir / f"{prefix}_argos_failures.json"

    for key in ["eval", "summary", "breakdown", "raw", "meta", "argos_failures"]:
        if key not in output_paths:
            continue
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
    return resume_meta_obj, output_paths, timestamp


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

    resume_meta_obj, output_paths, timestamp = _resolve_output_paths(
        args=args,
        output_dir=output_dir,
        start_index=start_index,
        end_index=end_index,
    )
    is_resuming = bool(args.resume_meta)
    argos_failures_path = output_paths.get("argos_failures")
    save_argos_failures = argos_failures_path is not None

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
    print(f"[info] sample_batch_size={args.sample_batch_size}")
    print(f"[info] resuming={is_resuming}")
    if is_resuming:
        prev_processed = _safe_int(resume_meta_obj.get("processed_samples", 0))
        print(f"[info] resume_meta={output_paths['meta']} previous_processed={prev_processed}")
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
    print(f"  - {output_paths['eval']}")
    print(f"  - {output_paths['summary']}")
    print(f"  - {output_paths['breakdown']}")
    print(f"  - {output_paths['raw']}")
    if argos_failures_path is not None:
        print(f"  - {argos_failures_path}")
    print(f"  - {output_paths['meta']}")
    if not HAS_RETRIEVAL_MODULE:
        print(f"[warn] retrieval disabled due to import error: {RETRIEVAL_IMPORT_ERROR}")

    all_mode_outputs: List[Dict[str, Any]] = read_json_records(output_paths["raw"]) if is_resuming else []
    evaluation_rows: List[Dict[str, Any]] = read_json_records(output_paths["eval"]) if is_resuming else []
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
    runner = LlamaCppAccessControlRunner(args, run_timestamp=timestamp)
    if runner.argos_db_abox_dir is not None:
        print(f"[info] argos_db_abox_dir={runner.argos_db_abox_dir}")
    else:
        print("[info] argos_db_abox_dir=(disabled)")
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
            "per_sample_eval": str(output_paths["eval"]),
            "mode_summary": str(output_paths["summary"]),
            "question_type_breakdown": str(output_paths["breakdown"]),
            "raw_outputs": str(output_paths["raw"]),
            "argos_failures": str(argos_failures_path) if argos_failures_path is not None else "",
            "argos_db_abox_dir": str(runner.argos_db_abox_dir) if runner.argos_db_abox_dir else "",
        },
        "resumed_from_meta": str(output_paths["meta"]) if is_resuming else "",
        "runner_kind": "llamacpp_python_batch",
    }

    summary_df = persist_run_outputs(
        eval_path=output_paths["eval"],
        summary_path=output_paths["summary"],
        breakdown_path=output_paths["breakdown"],
        raw_path=output_paths["raw"],
        meta_path=output_paths["meta"],
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
        sample_batch_size = max(1, int(args.sample_batch_size))
        for batch_start in range(processed_samples, len(exp_df), sample_batch_size):
            batch_end = min(batch_start + sample_batch_size, len(exp_df))
            print(f"[batch] positions={batch_start}..{batch_end - 1}")

            for sample_position in range(batch_start, batch_end):
                sample = exp_df.iloc[sample_position].to_dict()
                print(
                    f"[run] {sample_position + 1}/{len(exp_df)} "
                    f"question_id={sample.get('question_id')} "
                    f"db={sample.get('db_id')} role={sample.get('role')}"
                )

                mode_outputs = runner.run_all_modes_for_sample(sample)
                for mode_output in mode_outputs:
                    generation_prompt = mode_output.get("generation_prompt")
                    generation_prompt_char_size = (
                        len(generation_prompt) if isinstance(generation_prompt, str) else 0
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
                        eval_path=output_paths["eval"],
                        summary_path=output_paths["summary"],
                        breakdown_path=output_paths["breakdown"],
                        raw_path=output_paths["raw"],
                        meta_path=output_paths["meta"],
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
                        eval_path=output_paths["eval"],
                        summary_path=output_paths["summary"],
                        breakdown_path=output_paths["breakdown"],
                        raw_path=output_paths["raw"],
                        meta_path=output_paths["meta"],
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
            eval_path=output_paths["eval"],
            summary_path=output_paths["summary"],
            breakdown_path=output_paths["breakdown"],
            raw_path=output_paths["raw"],
            meta_path=output_paths["meta"],
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
        eval_path=output_paths["eval"],
        summary_path=output_paths["summary"],
        breakdown_path=output_paths["breakdown"],
        raw_path=output_paths["raw"],
        meta_path=output_paths["meta"],
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
    print(f"  - {output_paths['eval']}")
    print(f"  - {output_paths['summary']}")
    print(f"  - {output_paths['breakdown']}")
    print(f"  - {output_paths['raw']}")
    if argos_failures_path is not None:
        print(f"  - {argos_failures_path}")
    if runner.argos_db_abox_dir is not None:
        print(f"  - {runner.argos_db_abox_dir}")
    print(f"  - {output_paths['meta']}")
    if not summary_df.empty:
        print("\n[summary]")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
