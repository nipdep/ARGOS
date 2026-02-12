#!/usr/bin/env python3
"""
Build true filter-only stress-test qa_config from access_control + schema.

Design:
- Projection columns are always S0 columns from S0 tables (view-safe baseline).
- Process columns are where/group/aggregate stress columns and may be restricted.
- Role copies apply:
  - view restrictions to projection
  - process restrictions to process columns
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROLES = ["admin", "analyst", "staff", "public"]


@dataclass(frozen=True)
class ColumnEntry:
    table: str
    column: str
    fq_column: str
    column_type: str | None
    column_category: str
    table_category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build qa_config for true filter-only stress tests."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Base dir containing one folder per DB.",
    )
    parser.add_argument(
        "--db",
        action="append",
        help="Target DB id(s). Repeat to select multiple DBs. Defaults to all.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=DEFAULT_ROLES,
        help="Role order to materialize role copies.",
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=20,
        help="Template count per DB.",
    )
    parser.add_argument(
        "--projection-size",
        type=int,
        default=4,
        help="Projection columns per template (from S0 columns in S0 tables).",
    )
    parser.add_argument(
        "--process-size",
        type=int,
        default=4,
        help="Process columns per template.",
    )
    parser.add_argument(
        "--output-name",
        default="qa_config_filter_only.json",
        help="Per-DB output file name.",
    )
    parser.add_argument(
        "--combined-output",
        default="qa_config_filter_only_all.json",
        help="Combined output file name under base-dir. Set empty string to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=77,
        help="Deterministic seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build only and print summary without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_classification_maps(
    access_control: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    classification = (
        access_control.get("classifiction")
        or access_control.get("classification")
        or {}
    )
    table_raw = classification.get("table", {})
    column_raw = classification.get("column", {})

    if not column_raw and "column_category_dict" in access_control:
        column_raw = access_control["column_category_dict"]
    if not table_raw and "table_category_dict" in access_control:
        table_raw = access_control["table_category_dict"]

    table_map: dict[str, str] = {}
    for cat, tables in table_raw.items():
        for table in tables:
            table_map[table] = cat

    column_map: dict[str, str] = {}
    for cat, cols in column_raw.items():
        for col in cols:
            column_map[col] = cat

    return table_map, column_map


def ordered_roles(access_control: dict[str, Any], requested_roles: list[str]) -> list[str]:
    discovered = set(requested_roles)

    explicit_roles = access_control.get("roles", [])
    if isinstance(explicit_roles, list):
        discovered.update(explicit_roles)

    for policy in access_control.get("policies", []):
        discovered.update(policy.get("roles", []))

    ordered: list[str] = []
    for role in requested_roles:
        if role in discovered:
            ordered.append(role)
    for role in sorted(discovered):
        if role not in ordered:
            ordered.append(role)
    return ordered


def denied_categories(
    access_control: dict[str, Any],
    role: str,
    *,
    scope: str,
    level: str,
) -> set[str]:
    denied: set[str] = set()
    for policy in access_control.get("policies", []):
        if policy.get("effect") != "deny":
            continue
        if policy.get("scope") != scope:
            continue
        if policy.get("level") != level:
            continue
        action = policy.get("action")
        if action and action != "read":
            continue
        if role in policy.get("roles", []):
            denied.update(policy.get("categories", []))

    legacy_key = f"{scope}_restriction"
    legacy = access_control.get(legacy_key, {})
    if isinstance(legacy, dict):
        for cat, denied_roles in legacy.items():
            if isinstance(denied_roles, list) and role in denied_roles:
                denied.add(cat)

    return denied


def build_column_entries(
    schema: dict[str, Any],
    table_category_map: dict[str, str],
    column_category_map: dict[str, str],
) -> list[ColumnEntry]:
    table_names = schema.get("table_names", [])
    column_names = schema.get("column_names", [])
    column_types = schema.get("column_types", [])

    entries: list[ColumnEntry] = []
    for idx, item in enumerate(column_names):
        if not isinstance(item, list) or len(item) != 2:
            continue
        table_idx, column_name = item
        if not isinstance(table_idx, int) or table_idx < 0:
            continue
        if table_idx >= len(table_names):
            continue
        if not isinstance(column_name, str) or column_name == "*":
            continue

        table_name = table_names[table_idx]
        entries.append(
            ColumnEntry(
                table=table_name,
                column=column_name,
                fq_column=f"{table_name}.{column_name}",
                column_type=column_types[idx] if idx < len(column_types) else None,
                column_category=column_category_map.get(column_name, "S0"),
                table_category=table_category_map.get(table_name, "S0"),
            )
        )
    return entries


def build_entry_maps(entries: list[ColumnEntry]) -> tuple[dict[str, ColumnEntry], dict[str, list[str]]]:
    by_fq = {e.fq_column: e for e in entries}
    by_table: dict[str, list[str]] = {}
    for e in entries:
        by_table.setdefault(e.table, []).append(e.fq_column)
    for t in by_table:
        by_table[t] = sorted(by_table[t])
    return by_fq, by_table


def build_role_pools(
    entries: list[ColumnEntry],
    access_control: dict[str, Any],
    roles: list[str],
) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for role in roles:
        denied_view_col = denied_categories(
            access_control, role, scope="view", level="column"
        )
        denied_view_table = denied_categories(
            access_control, role, scope="view", level="table"
        )
        denied_proc_col = denied_categories(
            access_control, role, scope="process", level="column"
        )
        denied_proc_table = denied_categories(
            access_control, role, scope="process", level="table"
        )

        view_allowed = {
            e.fq_column
            for e in entries
            if e.column_category not in denied_view_col
            and e.table_category not in denied_view_table
        }
        process_allowed = {
            e.fq_column
            for e in entries
            if e.column_category not in denied_proc_col
            and e.table_category not in denied_proc_table
        }

        pools[role] = {
            "view_allowed_columns": sorted(view_allowed),
            "process_allowed_columns": sorted(process_allowed),
            "view_denied_columns": sorted({e.fq_column for e in entries} - view_allowed),
            "process_denied_columns": sorted(
                {e.fq_column for e in entries} - process_allowed
            ),
            "denied_categories": {
                "view": {
                    "column": sorted(denied_view_col),
                    "table": sorted(denied_view_table),
                },
                "process": {
                    "column": sorted(denied_proc_col),
                    "table": sorted(denied_proc_table),
                },
            },
        }
    return pools


def make_rng(seed: int, db_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{db_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_global_projection_pool(
    *,
    entries: list[ColumnEntry],
    roles: list[str],
    role_pools: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Projection candidates by table: S0 col + S0 table + view-allowed for every role."""
    global_view_allowed: set[str] | None = None
    for role in roles:
        allowed = set(role_pools[role]["view_allowed_columns"])
        global_view_allowed = allowed if global_view_allowed is None else global_view_allowed & allowed
    global_view_allowed = global_view_allowed or set()

    by_table: dict[str, list[str]] = {}
    for e in entries:
        if e.fq_column not in global_view_allowed:
            continue
        if e.table_category != "S0":
            continue
        if e.column_category != "S0":
            continue
        by_table.setdefault(e.table, []).append(e.fq_column)
    for table in by_table:
        by_table[table] = sorted(set(by_table[table]))
    return by_table


def build_target_role_candidates(
    *,
    roles: list[str],
    role_pools: dict[str, dict[str, Any]],
    by_fq: dict[str, ColumnEntry],
    projection_pool_by_table: dict[str, list[str]],
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for role in roles:
        denied = role_pools[role]["process_denied_columns"]
        role_candidates: list[str] = []
        for fq in denied:
            entry = by_fq.get(fq)
            if not entry:
                continue
            # filter-only requirement: projection must come from S0 table
            if entry.table not in projection_pool_by_table:
                continue
            if projection_pool_by_table[entry.table]:
                role_candidates.append(fq)
        candidates[role] = sorted(set(role_candidates))
    return candidates


def build_role_copy(
    *,
    template_projection: list[str],
    template_process: list[str],
    view_allowed_set: set[str],
    process_allowed_set: set[str],
) -> dict[str, Any]:
    allowed_projection = [c for c in template_projection if c in view_allowed_set]
    dropped_projection = [c for c in template_projection if c not in view_allowed_set]

    allowed_process = [c for c in template_process if c in process_allowed_set]
    dropped_process = [c for c in template_process if c not in process_allowed_set]

    return {
        "projection_columns": allowed_projection,
        "process_columns": allowed_process,
        "dropped_projection": dropped_projection,
        "dropped_process": dropped_process,
        "is_usable": bool(allowed_projection) and bool(allowed_process),
    }


def generate_templates(
    *,
    n_sets: int,
    projection_size: int,
    process_size: int,
    roles: list[str],
    role_pools: dict[str, dict[str, Any]],
    by_fq: dict[str, ColumnEntry],
    by_table: dict[str, list[str]],
    projection_pool_by_table: dict[str, list[str]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str]]:
    templates: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    target_candidates = build_target_role_candidates(
        roles=roles,
        role_pools=role_pools,
        by_fq=by_fq,
        projection_pool_by_table=projection_pool_by_table,
    )
    target_roles = [r for r in roles if target_candidates.get(r)]

    if not projection_pool_by_table:
        warnings.append(
            "No S0 columns from S0 tables are available for projection in this DB."
        )
        return templates, warnings

    if not target_roles:
        warnings.append(
            "No roles have process-restricted columns on tables usable by filter-only projection rules."
        )
        return templates, warnings

    max_attempts = max(200, n_sets * 60)
    attempts = 0

    while len(templates) < n_sets and attempts < max_attempts:
        attempts += 1
        target_role = target_roles[(len(templates) + attempts) % len(target_roles)]
        restricted_candidates = target_candidates[target_role]
        if not restricted_candidates:
            continue

        restricted_process_col = rng.choice(restricted_candidates)
        table = by_fq[restricted_process_col].table

        projection_pool = projection_pool_by_table.get(table, [])
        if not projection_pool:
            continue

        proj_k = min(projection_size, len(projection_pool))
        if proj_k == 0:
            continue
        projection_cols = sorted(rng.sample(projection_pool, proj_k))

        process_pool = [c for c in by_table.get(table, []) if c != restricted_process_col]
        process_cols = [restricted_process_col]
        extra_need = max(0, process_size - 1)
        if extra_need > 0 and process_pool:
            process_cols.extend(rng.sample(process_pool, min(extra_need, len(process_pool))))
        process_cols = sorted(set(process_cols))

        target_restricted = [
            c
            for c in process_cols
            if c in set(role_pools[target_role]["process_denied_columns"])
        ]
        if not target_restricted:
            continue

        signature = (
            target_role,
            table,
            tuple(projection_cols),
            tuple(process_cols),
        )
        if signature in seen:
            continue
        seen.add(signature)

        role_copies: dict[str, Any] = {}
        for role in roles:
            role_copies[role] = build_role_copy(
                template_projection=projection_cols,
                template_process=process_cols,
                view_allowed_set=set(role_pools[role]["view_allowed_columns"]),
                process_allowed_set=set(role_pools[role]["process_allowed_columns"]),
            )

        templates.append(
            {
                "set_id": f"set_{len(templates) + 1:03d}",
                "tables": [table],
                "projection_columns": projection_cols,
                "process_columns": process_cols,
                "target_role": target_role,
                "target_restricted_process_columns": target_restricted,
                "role_copies": role_copies,
            }
        )

    if len(templates) < n_sets:
        warnings.append(
            f"Requested n_sets={n_sets}, created only {len(templates)} due to candidate limits."
        )

    return templates, warnings


def iter_db_dirs(base_dir: Path, selected_db_ids: set[str] | None) -> list[Path]:
    db_dirs: list[Path] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if selected_db_ids and path.name not in selected_db_ids:
            continue
        if (path / "access_control.json").exists() and (path / "schema.json").exists():
            db_dirs.append(path)
    return db_dirs


def build_for_db(db_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    access_control = load_json(db_dir / "access_control.json")
    schema = load_json(db_dir / "schema.json")

    db_id = access_control.get("db_id") or schema.get("db_id") or db_dir.name
    roles = ordered_roles(access_control, args.roles)

    table_map, column_map = get_classification_maps(access_control)
    entries = build_column_entries(schema, table_map, column_map)
    by_fq, by_table = build_entry_maps(entries)
    role_pools = build_role_pools(entries, access_control, roles)
    projection_pool_by_table = build_global_projection_pool(
        entries=entries,
        roles=roles,
        role_pools=role_pools,
    )

    rng = make_rng(args.seed, db_id)
    templates, warnings = generate_templates(
        n_sets=args.n_sets,
        projection_size=args.projection_size,
        process_size=args.process_size,
        roles=roles,
        role_pools=role_pools,
        by_fq=by_fq,
        by_table=by_table,
        projection_pool_by_table=projection_pool_by_table,
        rng=rng,
    )

    return {
        "db_id": db_id,
        "roles": roles,
        "settings": {
            "track": "filter_only_stress",
            "n_sets_requested": args.n_sets,
            "n_sets_created": len(templates),
            "projection_size": args.projection_size,
            "process_size": args.process_size,
            "projection_rule": "S0 columns from S0 tables only",
            "seed": args.seed,
        },
        "column_catalog": [asdict(e) for e in entries],
        "role_pools": role_pools,
        "projection_pool_by_table": projection_pool_by_table,
        "templates": templates,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    selected = set(args.db) if args.db else None
    db_dirs = iter_db_dirs(base_dir, selected)
    if not db_dirs:
        raise ValueError("No DB directories with schema/access_control files were found.")

    combined: dict[str, Any] = {
        "base_dir": str(base_dir),
        "db_count": len(db_dirs),
        "items": [],
    }

    for db_dir in db_dirs:
        cfg = build_for_db(db_dir, args)
        combined["items"].append(cfg)
        warn_count = len(cfg.get("warnings", []))
        print(
            f"[{cfg['db_id']}] sets={cfg['settings']['n_sets_created']} "
            f"warnings={warn_count}"
        )

        if not args.dry_run:
            out_path = db_dir / args.output_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

    if not args.dry_run and args.combined_output:
        combined_path = base_dir / args.combined_output
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"combined output -> {combined_path}")


if __name__ == "__main__":
    main()
