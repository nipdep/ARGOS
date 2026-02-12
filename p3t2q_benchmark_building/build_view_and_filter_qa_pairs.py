#!/usr/bin/env python3
"""
Build deterministic question/query pairs for the view+filter stress track.

Input (per DB):
- qa_config_view_and_filter.json (templates + role_copies)
- schema.json (foreign keys for join construction)

Output (per DB):
- pq_pairs_view_and_filter.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


NUMERIC_TYPES = {"integer", "real", "float", "double", "numeric", "number", "decimal"}
SUPPORTED_MODES = {"where", "group_by", "aggregate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic view+filter QA pairs from qa_config resources."
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
        "--qa-config-name",
        default="qa_config_view_and_filter.json",
        help="Input resource file name under each DB folder.",
    )
    parser.add_argument(
        "--schema-name",
        default="schema.json",
        help="Schema file name under each DB folder.",
    )
    parser.add_argument(
        "--output-name",
        default="pq_pairs_view_and_filter.json",
        help="Per-DB output file name.",
    )
    parser.add_argument(
        "--combined-output",
        default="pq_pairs_view_and_filter_all.json",
        help="Combined output file under base-dir. Set empty string to disable.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["where", "group_by", "aggregate"],
        help="Process modes to generate: where group_by aggregate",
    )
    parser.add_argument(
        "--max-group-columns",
        type=int,
        default=3,
        help="Upper bound on process columns used in GROUP BY.",
    )
    parser.add_argument(
        "--max-where-conditions",
        type=int,
        default=3,
        help="Upper bound on process columns used in WHERE conditions.",
    )
    parser.add_argument(
        "--max-aggregate-columns",
        type=int,
        default=4,
        help="Upper bound on process columns used in AGGREGATE mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records and print summary without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def split_fq(fq_column: str) -> tuple[str, str]:
    table, column = fq_column.split(".", 1)
    return table, column


def column_label(fq_column: str) -> str:
    table, column = split_fq(fq_column)
    return f"{table}.{column}"


def type_is_numeric(col_type: str | None) -> bool:
    if not col_type:
        return False
    return col_type.strip().lower() in NUMERIC_TYPES


def sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


class SchemaGraph:
    def __init__(self, schema: dict[str, Any]):
        self.table_names = schema.get("table_names", [])
        self.column_names = schema.get("column_names", [])
        self.adj: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._build_edges(schema.get("foreign_keys", []))

    def _build_edges(self, foreign_keys: Any) -> None:
        for fk in foreign_keys or []:
            if not isinstance(fk, list) or len(fk) != 2:
                continue
            col_a_idx, col_b_idx = fk
            if not isinstance(col_a_idx, int) or not isinstance(col_b_idx, int):
                continue
            if (
                col_a_idx < 0
                or col_b_idx < 0
                or col_a_idx >= len(self.column_names)
                or col_b_idx >= len(self.column_names)
            ):
                continue

            entry_a = self.column_names[col_a_idx]
            entry_b = self.column_names[col_b_idx]
            if not (
                isinstance(entry_a, list)
                and len(entry_a) == 2
                and isinstance(entry_b, list)
                and len(entry_b) == 2
            ):
                continue

            table_a_idx, col_a_name = entry_a
            table_b_idx, col_b_name = entry_b
            if not isinstance(table_a_idx, int) or not isinstance(table_b_idx, int):
                continue
            if (
                table_a_idx < 0
                or table_b_idx < 0
                or table_a_idx >= len(self.table_names)
                or table_b_idx >= len(self.table_names)
            ):
                continue

            table_a = self.table_names[table_a_idx]
            table_b = self.table_names[table_b_idx]
            if table_a == table_b:
                continue

            edge_ab = {
                "to_table": table_b,
                "from_column": col_a_name,
                "to_column": col_b_name,
            }
            edge_ba = {
                "to_table": table_a,
                "from_column": col_b_name,
                "to_column": col_a_name,
            }
            self.adj[table_a].append(edge_ab)
            self.adj[table_b].append(edge_ba)

        # Deterministic order for reproducible join-path selection.
        for table in self.adj:
            self.adj[table].sort(
                key=lambda e: (e["to_table"], e["from_column"], e["to_column"])
            )

    def connected_components(self, tables: set[str]) -> list[set[str]]:
        components: list[set[str]] = []
        seen: set[str] = set()

        for table in sorted(tables):
            if table in seen:
                continue
            comp: set[str] = set()
            queue = deque([table])
            seen.add(table)
            while queue:
                cur = queue.popleft()
                comp.add(cur)
                for edge in self.adj.get(cur, []):
                    nxt = edge["to_table"]
                    if nxt in tables and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(comp)

        return components

    def build_join_plan(
        self,
        tables: set[str],
        root_table: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Returns join edges as tuples:
        (from_table, to_table, from_column, to_column)
        """
        if len(tables) <= 1:
            return []

        root = root_table if root_table in tables else sorted(tables)[0]
        visited = {root}
        queue = deque([root])
        join_plan: list[tuple[str, str, str, str]] = []

        while queue:
            cur = queue.popleft()
            for edge in self.adj.get(cur, []):
                nxt = edge["to_table"]
                if nxt not in tables or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
                join_plan.append((cur, nxt, edge["from_column"], edge["to_column"]))

        return join_plan


def collect_column_meta(qa_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in qa_config.get("column_catalog", []):
        fq = row.get("fq_column")
        if isinstance(fq, str):
            meta[fq] = row
    return meta


def filter_columns_to_tables(columns: list[str], allowed_tables: set[str]) -> list[str]:
    return [c for c in columns if split_fq(c)[0] in allowed_tables]


def table_frequency(columns: list[str]) -> dict[str, int]:
    freq: dict[str, int] = defaultdict(int)
    for col in columns:
        table, _ = split_fq(col)
        freq[table] += 1
    return freq


def choose_tables_for_mode(
    schema_graph: SchemaGraph,
    required_tables: set[str],
    projection_columns: list[str],
    process_columns: list[str],
    mode: str,
) -> tuple[set[str], list[str]]:
    if not required_tables:
        return set(), []

    proj_freq = table_frequency(projection_columns)
    proc_freq = table_frequency(process_columns)

    components = schema_graph.connected_components(required_tables)

    def component_score(component: set[str]) -> tuple[int, int, int, int, int]:
        proj_count = sum(proj_freq.get(t, 0) for t in component)
        proc_count = sum(proc_freq.get(t, 0) for t in component)
        needs_process = mode in {"where", "group_by", "aggregate"}
        usable = proj_count > 0 and ((proc_count > 0) if needs_process else True)
        return (
            1 if usable else 0,
            proj_count + proc_count,
            proc_count,
            proj_count,
            len(component),
        )

    components.sort(key=component_score, reverse=True)
    chosen = components[0]
    dropped = sorted(required_tables - chosen)
    return chosen, dropped


def assign_aliases(tables: list[str]) -> dict[str, str]:
    return {table: f"t{i + 1}" for i, table in enumerate(sorted(tables))}


def col_sql(fq_column: str, aliases: dict[str, str]) -> str:
    table, column = split_fq(fq_column)
    return f'{aliases[table]}.{quote_ident(column)}'


def sanitize_alias_name(fq_column: str) -> str:
    return (
        fq_column.replace(".", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .lower()
    )


def pick_aggregate_fn(col_type: str | None, idx: int) -> str:
    if type_is_numeric(col_type):
        numeric_cycle = ["AVG", "SUM", "MIN", "MAX"]
        return numeric_cycle[idx % len(numeric_cycle)]
    return "COUNT"


def build_from_clause(
    joinable_tables: set[str],
    join_plan: list[tuple[str, str, str, str]],
    aliases: dict[str, str],
    base_table: str,
) -> str:
    if not joinable_tables:
        return ""

    from_sql = f"FROM {quote_ident(base_table)} AS {aliases[base_table]}"

    for from_table, to_table, from_col, to_col in join_plan:
        from_alias = aliases[from_table]
        to_alias = aliases[to_table]
        on_expr = (
            f"{from_alias}.{quote_ident(from_col)} = {to_alias}.{quote_ident(to_col)}"
        )
        from_sql += f"\nJOIN {quote_ident(to_table)} AS {to_alias} ON {on_expr}"

    return from_sql


def build_where_conditions(
    process_columns: list[str],
    aliases: dict[str, str],
    col_meta: dict[str, dict[str, Any]],
    max_conditions: int,
) -> list[str]:
    selected = process_columns[: max(1, max_conditions)]
    conditions: list[str] = []
    for col in selected:
        expr = col_sql(col, aliases)
        col_type = col_meta.get(col, {}).get("column_type")
        if type_is_numeric(col_type):
            conditions.append(f"{expr} IS NOT NULL")
            conditions.append(f"{expr} >= 0")
        else:
            conditions.append(f"{expr} IS NOT NULL")
    return conditions


def build_query_for_mode(
    *,
    mode: str,
    projection_columns: list[str],
    process_columns: list[str],
    schema_graph: SchemaGraph,
    col_meta: dict[str, dict[str, Any]],
    max_group_columns: int,
    max_where_conditions: int,
    max_aggregate_columns: int,
) -> dict[str, Any]:
    all_columns = sorted_unique(projection_columns + process_columns)
    if not all_columns:
        return {
            "query": None,
            "used_projection_columns": [],
            "used_process_columns": [],
            "dropped_disconnected_tables": [],
            "dropped_disconnected_columns": [],
            "reason": "no_columns",
        }

    required_tables = {split_fq(c)[0] for c in all_columns}
    joinable_tables, dropped_tables = choose_tables_for_mode(
        schema_graph=schema_graph,
        required_tables=required_tables,
        projection_columns=projection_columns,
        process_columns=process_columns,
        mode=mode,
    )
    kept_columns = filter_columns_to_tables(all_columns, joinable_tables)
    dropped_columns = [c for c in all_columns if c not in kept_columns]

    kept_projection = filter_columns_to_tables(projection_columns, joinable_tables)
    kept_process = filter_columns_to_tables(process_columns, joinable_tables)

    if not kept_projection:
        return {
            "query": None,
            "used_projection_columns": [],
            "used_process_columns": [],
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": "no_projection_after_join_resolution",
        }

    if mode in {"where", "group_by", "aggregate"} and not kept_process:
        return {
            "query": None,
            "used_projection_columns": kept_projection,
            "used_process_columns": [],
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": "no_process_after_join_resolution",
        }

    base_table = sorted(joinable_tables)[0]
    join_plan = schema_graph.build_join_plan(joinable_tables, base_table)
    aliases = assign_aliases(sorted(joinable_tables))
    from_clause = build_from_clause(joinable_tables, join_plan, aliases, base_table)
    if not from_clause:
        return {
            "query": None,
            "used_projection_columns": kept_projection,
            "used_process_columns": kept_process,
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": "empty_from_clause",
        }

    if mode == "where":
        conditions = build_where_conditions(
            kept_process,
            aliases,
            col_meta,
            max_conditions=max_where_conditions,
        )
        select_exprs = [col_sql(c, aliases) for c in kept_projection]
        query = f"SELECT {', '.join(select_exprs)}\n{from_clause}"
        if conditions:
            query += f"\nWHERE {' AND '.join(conditions)}"
        return {
            "query": query,
            "used_projection_columns": kept_projection,
            "used_process_columns": kept_process[: max(1, max_where_conditions)],
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": None,
        }

    if mode == "group_by":
        group_cols = kept_process[: max(1, min(max_group_columns, len(kept_process)))]
        group_set = set(group_cols)

        projection_set = set(kept_projection)
        agg_limit = max(1, min(max_aggregate_columns, len(kept_process)))

        # Prefer non-overlapping process columns for aggregate additions.
        # If overlap is unavoidable, keep only one version in SELECT
        # (raw projection OR aggregated expression), never both.
        non_overlap_not_group = [
            c for c in kept_process if c not in group_set and c not in projection_set
        ]
        overlap_not_group = [
            c for c in kept_process if c not in group_set and c in projection_set
        ]
        non_overlap_group = [
            c for c in kept_process if c in group_set and c not in projection_set
        ]
        overlap_group = [c for c in kept_process if c in group_set and c in projection_set]

        agg_cols: list[str] = []
        if non_overlap_not_group:
            agg_buckets = (non_overlap_not_group,)
        elif overlap_not_group:
            agg_buckets = (overlap_not_group,)
        elif non_overlap_group:
            agg_buckets = (non_overlap_group,)
        else:
            agg_buckets = (overlap_group,)

        for bucket in agg_buckets:
            for col in bucket:
                if col in agg_cols:
                    continue
                agg_cols.append(col)
                if len(agg_cols) >= agg_limit:
                    break
            if len(agg_cols) >= agg_limit:
                break

        process_agg_exprs: list[str] = []
        for idx, c in enumerate(agg_cols):
            col_type = col_meta.get(c, {}).get("column_type")
            fn = pick_aggregate_fn(col_type, idx)
            expr = col_sql(c, aliases)
            alias_name = f"{fn.lower()}_{sanitize_alias_name(c)}"
            if fn == "COUNT":
                process_agg_exprs.append(
                    f"COUNT(DISTINCT {expr}) AS {quote_ident(alias_name)}"
                )
            else:
                process_agg_exprs.append(f"{fn}({expr}) AS {quote_ident(alias_name)}")

        agg_col_set = set(agg_cols)
        projection_for_select = [c for c in kept_projection if c not in agg_col_set]
        projection_exprs = [col_sql(c, aliases) for c in projection_for_select]
        select_exprs = projection_exprs + process_agg_exprs
        # Keep GROUP BY semantically valid when raw projection columns are selected.
        group_cols_for_sql = list(dict.fromkeys(group_cols + projection_for_select))
        group_exprs = [col_sql(c, aliases) for c in group_cols_for_sql]
        query = (
            f"SELECT {', '.join(select_exprs)}\n"
            f"{from_clause}\n"
            f"GROUP BY {', '.join(group_exprs)}"
        )
        used_process = list(dict.fromkeys(group_cols + agg_cols))
        return {
            "query": query,
            "used_projection_columns": projection_for_select,
            "used_process_columns": used_process,
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": None,
        }

    if mode == "aggregate":
        projection_pref = kept_projection[:]
        projection_pref_set = set(projection_pref)
        process_candidates = [c for c in kept_process if c not in projection_pref_set]
        using_overlap_fallback = False
        if not process_candidates:
            process_candidates = kept_process[:]
            using_overlap_fallback = True

        agg_limit = max(1, min(max_aggregate_columns, len(process_candidates)))
        # If every process candidate overlaps projection intent, keep at least one
        # projection column by not aggregating all overlapping process columns.
        if using_overlap_fallback and len(process_candidates) > 1:
            agg_limit = min(agg_limit, len(process_candidates) - 1)
        agg_cols = process_candidates[:agg_limit]
        agg_col_set = set(agg_cols)
        projection_for_select = [c for c in kept_projection if c not in agg_col_set]
        if kept_projection and not projection_for_select:
            keep_one = kept_projection[0]
            projection_for_select = [keep_one]
            if keep_one in agg_col_set:
                agg_cols = [c for c in agg_cols if c != keep_one]

        process_agg_exprs: list[str] = []
        for idx, c in enumerate(agg_cols):
            col_type = col_meta.get(c, {}).get("column_type")
            fn = pick_aggregate_fn(col_type, idx)
            expr = col_sql(c, aliases)
            alias_name = f"{fn.lower()}_{sanitize_alias_name(c)}"
            if fn == "COUNT":
                process_agg_exprs.append(
                    f"COUNT(DISTINCT {expr}) AS {quote_ident(alias_name)}"
                )
            else:
                process_agg_exprs.append(f"{fn}({expr}) AS {quote_ident(alias_name)}")

        projection_exprs = [col_sql(c, aliases) for c in projection_for_select]
        query = f"SELECT {', '.join(projection_exprs + process_agg_exprs)}\n{from_clause}"
        if projection_exprs:
            query += f"\nGROUP BY {', '.join(projection_exprs)}"
        return {
            "query": query,
            "used_projection_columns": projection_for_select,
            "used_process_columns": agg_cols,
            "dropped_disconnected_tables": dropped_tables,
            "dropped_disconnected_columns": dropped_columns,
            "reason": None,
        }

    return {
        "query": None,
        "used_projection_columns": [],
        "used_process_columns": [],
        "dropped_disconnected_tables": [],
        "dropped_disconnected_columns": [],
        "reason": f"unsupported_mode:{mode}",
    }


def nl_columns(columns: list[str], limit: int = 4) -> str:
    labels = [column_label(c) for c in columns[:limit]]
    if not labels:
        return "no columns"
    if len(columns) > limit:
        labels.append("...")
    return ", ".join(labels)


def build_question_text(
    mode: str,
    tables: list[str],
    used_projection_columns: list[str],
    used_process_columns: list[str],
) -> str:
    table_part = ", ".join(sorted(set(tables))) if tables else "the selected tables"
    proj_part = nl_columns(used_projection_columns)
    proc_part = nl_columns(used_process_columns)

    if mode == "where":
        return (
            f"From {table_part}, return {proj_part} for rows where "
            f"{proc_part} satisfy filter conditions."
        )
    if mode == "group_by":
        if not used_projection_columns:
            return (
                f"From {table_part}, use {proc_part} as process columns for grouping "
                "and aggregate summaries."
            )
        return (
            f"From {table_part}, keep {proj_part} in the result and use {proc_part} as "
            "process columns for grouping and aggregate summaries."
        )
    if mode == "aggregate":
        if not used_projection_columns:
            return (
                f"Across rows in {table_part}, compute aggregate metrics for "
                f"{proc_part} without WHERE filters."
            )
        return (
            f"From {table_part}, keep {proj_part} and compute aggregate metrics for "
            f"{proc_part} without WHERE filters."
        )
    return f"Return {proj_part} from {table_part}."


def build_case_record(
    *,
    db_id: str,
    template: dict[str, Any],
    mode: str,
    schema_graph: SchemaGraph,
    col_meta: dict[str, dict[str, Any]],
    settings: argparse.Namespace,
) -> dict[str, Any]:
    base_projection = template.get("projection_columns", [])
    base_process = template.get("process_columns", [])

    base_query = build_query_for_mode(
        mode=mode,
        projection_columns=base_projection,
        process_columns=base_process,
        schema_graph=schema_graph,
        col_meta=col_meta,
        max_group_columns=settings.max_group_columns,
        max_where_conditions=settings.max_where_conditions,
        max_aggregate_columns=settings.max_aggregate_columns,
    )

    question = build_question_text(
        mode=mode,
        tables=template.get("tables", []),
        used_projection_columns=base_query["used_projection_columns"],
        used_process_columns=base_query["used_process_columns"],
    )

    role_outputs: dict[str, Any] = {}
    role_copies = template.get("role_copies", {})
    base_used_projection = base_query["used_projection_columns"]
    base_used_process = base_query["used_process_columns"]
    base_used_projection_set = set(base_used_projection)
    base_used_process_set = set(base_used_process)

    for role, role_copy in role_copies.items():
        role_projection = role_copy.get("projection_columns", [])
        role_process = role_copy.get("process_columns", [])

        role_query = build_query_for_mode(
            mode=mode,
            projection_columns=role_projection,
            process_columns=role_process,
            schema_graph=schema_graph,
            col_meta=col_meta,
            max_group_columns=settings.max_group_columns,
            max_where_conditions=settings.max_where_conditions,
            max_aggregate_columns=settings.max_aggregate_columns,
        )

        role_used_projection = role_query["used_projection_columns"]
        role_used_process = role_query["used_process_columns"]
        role_used_projection_set = set(role_used_projection)
        role_used_process_set = set(role_used_process)

        missing_projection_vs_base = sorted(
            base_used_projection_set - role_used_projection_set
        )
        missing_process_vs_base = sorted(base_used_process_set - role_used_process_set)

        restricted_projection_candidates = set(role_copy.get("dropped_projection", []))
        restricted_process_candidates = set(role_copy.get("dropped_process", []))

        dropped_restricted_projection = sorted(
            c for c in missing_projection_vs_base if c in restricted_projection_candidates
        )
        dropped_restricted_process = sorted(
            c for c in missing_process_vs_base if c in restricted_process_candidates
        )
        dropped_not_restricted_projection = sorted(
            c for c in missing_projection_vs_base if c not in restricted_projection_candidates
        )
        dropped_not_restricted_process = sorted(
            c for c in missing_process_vs_base if c not in restricted_process_candidates
        )

        role_outputs[role] = {
            "expected_query": role_query["query"],
            "used_projection_columns": role_used_projection,
            "used_process_columns": role_used_process,
            "dropped_restricted_projection": dropped_restricted_projection,
            "dropped_restricted_process": dropped_restricted_process,
            "dropped_not_restricted_projection": dropped_not_restricted_projection,
            "dropped_not_restricted_process": dropped_not_restricted_process,
            "dropped_disconnected_tables": role_query["dropped_disconnected_tables"],
            "dropped_disconnected_columns": role_query["dropped_disconnected_columns"],
            "is_usable": bool(role_query["query"]),
            "reason_unusable": role_query["reason"],
        }

    return {
        "db_id": db_id,
        "template_id": template.get("set_id"),
        "mode": mode,
        "question": question,
        "base_query": base_query["query"],
        "base_used_projection_columns": base_query["used_projection_columns"],
        "base_used_process_columns": base_query["used_process_columns"],
        "base_dropped_disconnected_tables": base_query["dropped_disconnected_tables"],
        "base_dropped_disconnected_columns": base_query["dropped_disconnected_columns"],
        "base_unusable_reason": base_query["reason"],
        "role_expected": role_outputs,
    }


def build_for_db(db_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    qa_config = load_json(db_dir / args.qa_config_name)
    schema = load_json(db_dir / args.schema_name)

    db_id = qa_config.get("db_id") or schema.get("db_id") or db_dir.name
    modes = [m for m in args.modes if m in SUPPORTED_MODES]
    if not modes:
        raise ValueError(f"No valid modes provided for db={db_id}.")

    col_meta = collect_column_meta(qa_config)
    schema_graph = SchemaGraph(schema)

    records: list[dict[str, Any]] = []
    for template in qa_config.get("templates", []):
        for mode in modes:
            records.append(
                build_case_record(
                    db_id=db_id,
                    template=template,
                    mode=mode,
                    schema_graph=schema_graph,
                    col_meta=col_meta,
                    settings=args,
                )
            )

    return {
        "db_id": db_id,
        "source_config": args.qa_config_name,
        "schema_file": args.schema_name,
        "modes": modes,
        "case_count": len(records),
        "records": records,
    }


def iter_db_dirs(base_dir: Path, selected_db_ids: set[str] | None, qa_config_name: str, schema_name: str) -> list[Path]:
    db_dirs: list[Path] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if selected_db_ids and path.name not in selected_db_ids:
            continue
        if (path / qa_config_name).exists() and (path / schema_name).exists():
            db_dirs.append(path)
    return db_dirs


def summarize_usability(records: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    usable = 0
    for record in records:
        total += 1
        if record.get("base_query"):
            usable += 1
    return usable, total


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    selected = set(args.db) if args.db else None
    db_dirs = iter_db_dirs(base_dir, selected, args.qa_config_name, args.schema_name)
    if not db_dirs:
        raise ValueError(
            "No DB dirs found with required qa_config and schema files."
        )

    combined = {
        "base_dir": str(base_dir),
        "db_count": len(db_dirs),
        "items": [],
    }

    for db_dir in db_dirs:
        result = build_for_db(db_dir, args)
        combined["items"].append(result)
        usable, total = summarize_usability(result["records"])
        print(f"[{result['db_id']}] cases={total} base_usable={usable}")

        if not args.dry_run:
            out_path = db_dir / args.output_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    if not args.dry_run and args.combined_output:
        combined_path = base_dir / args.combined_output
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"combined output -> {combined_path}")


if __name__ == "__main__":
    main()
