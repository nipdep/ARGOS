#!/usr/bin/env python3
"""
Build view+filter stress-test resource sets from schema + access-control files.

For each DB:
1. Build N template sets with:
   - projection columns
   - process columns
2. Create role-specific copies of each template by filtering columns according to
   access-control policies (view/process, column/table level).
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
        description="Build role-aware column resources for view+filter stress tests."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Root directory that contains one folder per DB.",
    )
    parser.add_argument(
        "--db",
        action="append",
        help="Target DB id(s). Repeat to select multiple DBs. Defaults to all DB folders.",
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=20,
        help="Number of template sets per DB.",
    )
    parser.add_argument(
        "--projection-size",
        type=int,
        default=4,
        help="Number of projection columns per template set.",
    )
    parser.add_argument(
        "--process-size",
        type=int,
        default=3,
        help="Number of process columns per template set.",
    )
    parser.add_argument(
        "--max-tables-per-set",
        type=int,
        default=3,
        help="Sample each set from up to this many tables.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=["view", "both"],
        default="both",
        help=(
            "'view' => projection columns only need view permission; "
            "'both' => projection columns must be allowed for both view and process."
        ),
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow a column to appear in both projection and process sections.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=DEFAULT_ROLES,
        help="Role order to materialize role copies (default: admin analyst staff public).",
    )
    parser.add_argument(
        "--output-name",
        default="qa_config_view_and_filter.json",
        help="Per-DB output filename.",
    )
    parser.add_argument(
        "--combined-output",
        default="qa_config_view_and_filter_all.json",
        help="Combined output filename under base-dir. Set empty string to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic set generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build resources and print summary without writing files.",
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
    for category, tables in table_raw.items():
        for table in tables:
            table_map[table] = category

    column_map: dict[str, str] = {}
    for category, columns in column_raw.items():
        for column in columns:
            column_map[column] = category

    return table_map, column_map


def ordered_roles(access_control: dict[str, Any], requested_roles: list[str]) -> list[str]:
    discovered = set(requested_roles)

    explicit_roles = access_control.get("roles", [])
    if isinstance(explicit_roles, list):
        discovered.update(explicit_roles)

    for policy in access_control.get("policies", []):
        for role in policy.get("roles", []):
            discovered.add(role)

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

    # Legacy format fallback: {scope}_restriction: {category: [roles]}
    legacy_key = f"{scope}_restriction"
    legacy = access_control.get(legacy_key, {})
    if isinstance(legacy, dict):
        for category, denied_roles in legacy.items():
            if isinstance(denied_roles, list) and role in denied_roles:
                denied.add(category)

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
        if not isinstance(column_name, str) or column_name == "*":
            continue
        if table_idx >= len(table_names):
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


def build_role_pools(
    entries: list[ColumnEntry],
    access_control: dict[str, Any],
    roles: list[str],
    projection_mode: str,
) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}

    for role in roles:
        denied_view_col = denied_categories(access_control, role, scope="view", level="column")
        denied_process_col = denied_categories(access_control, role, scope="process", level="column")
        denied_view_table = denied_categories(access_control, role, scope="view", level="table")
        denied_process_table = denied_categories(access_control, role, scope="process", level="table")

        view_allowed = {
            entry.fq_column
            for entry in entries
            if entry.column_category not in denied_view_col
            and entry.table_category not in denied_view_table
        }
        process_allowed = {
            entry.fq_column
            for entry in entries
            if entry.column_category not in denied_process_col
            and entry.table_category not in denied_process_table
        }

        if projection_mode == "both":
            projection_allowed = sorted(view_allowed & process_allowed)
        else:
            projection_allowed = sorted(view_allowed)

        pools[role] = {
            "projection_pool": projection_allowed,
            "process_pool": sorted(process_allowed),
            "projection_pool_size": len(projection_allowed),
            "process_pool_size": len(process_allowed),
            "denied_categories": {
                "view": {
                    "column": sorted(denied_view_col),
                    "table": sorted(denied_view_table),
                },
                "process": {
                    "column": sorted(denied_process_col),
                    "table": sorted(denied_process_table),
                },
            },
        }

    return pools


def generate_templates(
    entries: list[ColumnEntry],
    n_sets: int,
    projection_size: int,
    process_size: int,
    max_tables_per_set: int,
    allow_overlap: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_table: dict[str, list[str]] = {}
    for entry in entries:
        by_table.setdefault(entry.table, []).append(entry.fq_column)
    table_names = sorted(by_table.keys())

    templates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    max_attempts = max(100, n_sets * 40)

    attempts = 0
    while len(templates) < n_sets and attempts < max_attempts:
        attempts += 1
        if not table_names:
            break

        k_tables = rng.randint(1, min(max_tables_per_set, len(table_names)))
        chosen_tables = rng.sample(table_names, k_tables)

        candidate = sorted({c for t in chosen_tables for c in by_table[t]})
        if not candidate:
            continue

        proj_k = min(projection_size, len(candidate))
        if proj_k == 0:
            continue
        projection_cols = sorted(rng.sample(candidate, proj_k))

        if allow_overlap:
            process_candidate = candidate
        else:
            process_candidate = [c for c in candidate if c not in projection_cols]
            if not process_candidate:
                process_candidate = candidate

        proc_k = min(process_size, len(process_candidate))
        process_cols = sorted(rng.sample(process_candidate, proc_k)) if proc_k > 0 else []

        signature = (tuple(projection_cols), tuple(process_cols))
        if signature in seen:
            continue
        seen.add(signature)

        templates.append(
            {
                "set_id": f"set_{len(templates) + 1:03d}",
                "tables": sorted({c.split(".", 1)[0] for c in projection_cols + process_cols}),
                "projection_columns": projection_cols,
                "process_columns": process_cols,
            }
        )

    return templates


def add_role_copies(
    templates: list[dict[str, Any]],
    role_pools: dict[str, dict[str, Any]],
) -> None:
    projection_sets = {
        role: set(pool["projection_pool"]) for role, pool in role_pools.items()
    }
    process_sets = {
        role: set(pool["process_pool"]) for role, pool in role_pools.items()
    }

    for template in templates:
        role_copies: dict[str, Any] = {}
        projection_cols = template["projection_columns"]
        process_cols = template["process_columns"]

        for role in role_pools:
            allowed_projection = [
                col for col in projection_cols if col in projection_sets[role]
            ]
            allowed_process = [col for col in process_cols if col in process_sets[role]]

            dropped_projection = [
                col for col in projection_cols if col not in projection_sets[role]
            ]
            dropped_process = [col for col in process_cols if col not in process_sets[role]]

            role_copies[role] = {
                "projection_columns": allowed_projection,
                "process_columns": allowed_process,
                "dropped_projection": dropped_projection,
                "dropped_process": dropped_process,
                "is_usable": bool(allowed_projection) and (
                    not process_cols or bool(allowed_process)
                ),
            }

        template["role_copies"] = role_copies


def make_rng(global_seed: int, db_id: str) -> random.Random:
    digest = hashlib.sha256(f"{global_seed}:{db_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_for_db(db_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    access_control_path = db_dir / "access_control.json"
    schema_path = db_dir / "schema.json"

    access_control = load_json(access_control_path)
    schema = load_json(schema_path)

    db_id = access_control.get("db_id") or schema.get("db_id") or db_dir.name
    roles = ordered_roles(access_control, args.roles)
    table_category_map, column_category_map = get_classification_maps(access_control)
    entries = build_column_entries(schema, table_category_map, column_category_map)

    role_pools = build_role_pools(
        entries=entries,
        access_control=access_control,
        roles=roles,
        projection_mode=args.projection_mode,
    )

    rng = make_rng(args.seed, db_id)
    templates = generate_templates(
        entries=entries,
        n_sets=args.n_sets,
        projection_size=args.projection_size,
        process_size=args.process_size,
        max_tables_per_set=args.max_tables_per_set,
        allow_overlap=args.allow_overlap,
        rng=rng,
    )
    add_role_copies(templates, role_pools)

    return {
        "db_id": db_id,
        "roles": roles,
        "settings": {
            "track": "view_and_filter_stress",
            "n_sets_requested": args.n_sets,
            "n_sets_created": len(templates),
            "projection_size": args.projection_size,
            "process_size": args.process_size,
            "max_tables_per_set": args.max_tables_per_set,
            "projection_mode": args.projection_mode,
            "allow_overlap": args.allow_overlap,
            "seed": args.seed,
        },
        "column_catalog": [asdict(entry) for entry in entries],
        "role_pools": role_pools,
        "templates": templates,
    }


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
        "resources": [],
    }

    for db_dir in db_dirs:
        resource = build_for_db(db_dir, args)
        combined["resources"].append(resource)

        if not args.dry_run:
            out_path = db_dir / args.output_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(resource, f, indent=2)

        print(
            f"[{resource['db_id']}] sets={resource['settings']['n_sets_created']} "
            f"roles={len(resource['roles'])}"
        )

    if not args.dry_run and args.combined_output:
        combined_path = base_dir / args.combined_output
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"combined output -> {combined_path}")


if __name__ == "__main__":
    main()
