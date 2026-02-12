#!/usr/bin/env python3
"""
Build view-only stress QA pairs from qa_config_view_only.

This track intentionally focuses on view leakage attempts:
- projection includes restricted columns
- no WHERE conditions
- no aggregation
- process-like ops are limited to DISTINCT / ORDER BY / LIMIT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build view-only stress pq_pairs from qa_config_view_only."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Base dir containing per-DB folders.",
    )
    parser.add_argument(
        "--db",
        action="append",
        help="Target DB id(s). Repeat to select multiple DBs. Defaults to all.",
    )
    parser.add_argument(
        "--qa-config-name",
        default="qa_config_view_only.json",
        help="Input config name per DB.",
    )
    parser.add_argument(
        "--output-name",
        default="pq_pairs_view_only.json",
        help="Output pairs name per DB.",
    )
    parser.add_argument(
        "--combined-output",
        default="pq_pairs_view_only_all.json",
        help="Combined output under base-dir. Set empty string to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write outputs.",
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


def col_sql(fq_column: str, alias: str = "t1") -> str:
    table, column = split_fq(fq_column)
    _ = table
    return f"{alias}.{quote_ident(column)}"


def build_query(
    *,
    table: str,
    projection_columns: list[str],
    distinct: bool,
    order_by_column: str,
    sort_direction: str,
    limit_value: int,
) -> str:
    select_list = ", ".join(col_sql(c) for c in projection_columns)
    distinct_kw = "DISTINCT " if distinct else ""
    order_expr = col_sql(order_by_column)
    direction = "DESC" if str(sort_direction).upper() == "DESC" else "ASC"

    query = (
        f"SELECT {distinct_kw}{select_list}\n"
        f"FROM {quote_ident(table)} AS t1\n"
        f"ORDER BY {order_expr} {direction}\n"
        f"LIMIT {int(limit_value)}"
    )
    return query


def natural_question(
    *,
    table: str,
    projection_columns: list[str],
    distinct: bool,
    order_by_column: str,
    sort_direction: str,
    limit_value: int,
) -> str:
    cols = ", ".join(projection_columns[:4])
    if len(projection_columns) > 4:
        cols += ", ..."
    distinct_phrase = "distinct " if distinct else ""
    direction_text = "descending" if str(sort_direction).upper() == "DESC" else "ascending"
    return (
        f"Show {distinct_phrase}{cols} from {table}, sorted by {order_by_column} "
        f"in {direction_text} order, limited to {int(limit_value)} rows."
    )


def build_record(
    db_id: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    tables = template.get("tables", [])
    if not tables:
        table = split_fq(template["projection_columns"][0])[0]
    else:
        table = tables[0]

    projection_columns = list(template.get("projection_columns", []))
    ops = template.get("ops", {})
    distinct = bool(ops.get("distinct", False))
    order_by_items = ops.get("order_by", [])
    if not order_by_items:
        raise ValueError(f"template {template.get('set_id')} missing ops.order_by")
    order_by_column = order_by_items[0]["column"]
    sort_direction = order_by_items[0].get("direction", "ASC")
    limit_value = int(ops.get("limit", 10))

    base_query = build_query(
        table=table,
        projection_columns=projection_columns,
        distinct=distinct,
        order_by_column=order_by_column,
        sort_direction=sort_direction,
        limit_value=limit_value,
    )
    base_process_columns = [order_by_column]
    question = natural_question(
        table=table,
        projection_columns=projection_columns,
        distinct=distinct,
        order_by_column=order_by_column,
        sort_direction=sort_direction,
        limit_value=limit_value,
    )

    role_expected: dict[str, Any] = {}
    for role, role_copy in template.get("role_copies", {}).items():
        role_projection = list(role_copy.get("projection_columns", []))
        role_order_by = role_copy.get("order_by_column")

        if not role_projection:
            role_expected[role] = {
                "expected_query": None,
                "used_projection_columns": [],
                "used_process_columns": [],
                "dropped_restricted_projection": sorted(
                    set(projection_columns) - set(role_projection)
                ),
                "dropped_restricted_process": [],
                "dropped_not_restricted_projection": [],
                "dropped_not_restricted_process": [],
                "dropped_disconnected_tables": [],
                "dropped_disconnected_columns": [],
                "is_usable": False,
                "reason_unusable": "no_projection_after_restriction",
            }
            continue

        if not role_order_by:
            role_expected[role] = {
                "expected_query": None,
                "used_projection_columns": role_projection,
                "used_process_columns": [],
                "dropped_restricted_projection": sorted(
                    set(projection_columns) - set(role_projection)
                ),
                "dropped_restricted_process": [order_by_column],
                "dropped_not_restricted_projection": [],
                "dropped_not_restricted_process": [],
                "dropped_disconnected_tables": [],
                "dropped_disconnected_columns": [],
                "is_usable": False,
                "reason_unusable": "no_order_by_after_restriction",
            }
            continue

        role_query = build_query(
            table=table,
            projection_columns=role_projection,
            distinct=distinct,
            order_by_column=role_order_by,
            sort_direction=sort_direction,
            limit_value=limit_value,
        )

        missing_projection = sorted(set(projection_columns) - set(role_projection))
        dropped_restricted_projection = sorted(role_copy.get("dropped_projection", []))
        missing_order_by = (
            [order_by_column]
            if order_by_column not in [role_order_by]
            else []
        )

        role_expected[role] = {
            "expected_query": role_query,
            "used_projection_columns": role_projection,
            "used_process_columns": [role_order_by],
            "dropped_restricted_projection": [
                c for c in missing_projection if c in set(dropped_restricted_projection)
            ],
            "dropped_restricted_process": [
                c for c in missing_order_by if c in set(role_copy.get("dropped_order_by", []))
            ],
            "dropped_not_restricted_projection": [
                c for c in missing_projection if c not in set(dropped_restricted_projection)
            ],
            "dropped_not_restricted_process": [
                c for c in missing_order_by if c not in set(role_copy.get("dropped_order_by", []))
            ],
            "dropped_disconnected_tables": [],
            "dropped_disconnected_columns": [],
            "is_usable": True,
            "reason_unusable": None,
        }

    return {
        "db_id": db_id,
        "template_id": template.get("set_id"),
        "mode": "view_ops",
        "question": question,
        "base_query": base_query,
        "base_used_projection_columns": projection_columns,
        "base_used_process_columns": base_process_columns,
        "base_dropped_disconnected_tables": [],
        "base_dropped_disconnected_columns": [],
        "base_unusable_reason": None,
        "role_expected": role_expected,
        "ops": {
            "distinct": distinct,
            "order_by": [{"column": order_by_column, "direction": sort_direction}],
            "limit": limit_value,
        },
    }


def build_for_db(db_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_json(db_dir / args.qa_config_name)
    db_id = cfg.get("db_id") or db_dir.name
    records = [build_record(db_id, t) for t in cfg.get("templates", [])]

    return {
        "db_id": db_id,
        "source_config": args.qa_config_name,
        "modes": ["view_ops"],
        "case_count": len(records),
        "records": records,
    }


def iter_db_dirs(base_dir: Path, selected_db_ids: set[str] | None, config_name: str) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if selected_db_ids and p.name not in selected_db_ids:
            continue
        if (p / config_name).exists():
            dirs.append(p)
    return dirs


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    selected = set(args.db) if args.db else None
    db_dirs = iter_db_dirs(base_dir, selected, args.qa_config_name)
    if not db_dirs:
        raise ValueError(f"No DB folders found with {args.qa_config_name}")

    combined = {
        "base_dir": str(base_dir),
        "db_count": len(db_dirs),
        "items": [],
    }

    for db_dir in db_dirs:
        result = build_for_db(db_dir, args)
        combined["items"].append(result)
        print(f"[{result['db_id']}] cases={result['case_count']}")

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
