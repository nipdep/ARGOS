#!/usr/bin/env python3
"""
Build view-only stress-test config from access_control + schema.

Goal:
- Generate templates that explicitly try to project restricted view columns.
- No WHERE/aggregation planning here.
- Process-like control is represented only by ORDER BY/LIMIT/DISTINCT metadata.
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
        description="Build qa_config for view-only restricted-column stress tests."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Base dir containing per-DB folders.",
    )
    parser.add_argument(
        "--db",
        action="append",
        help="Target DB id(s). Repeat to select multiple DBs. Defaults to all DBs.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=DEFAULT_ROLES,
        help="Role order for materialization.",
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=20,
        help="Number of templates per DB.",
    )
    parser.add_argument(
        "--projection-size",
        type=int,
        default=4,
        help="Projection size per template (includes at least one restricted column).",
    )
    parser.add_argument(
        "--limit-values",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20],
        help="Candidate LIMIT values for templates.",
    )
    parser.add_argument(
        "--output-name",
        default="qa_config_view_only.json",
        help="Per-DB output file name.",
    )
    parser.add_argument(
        "--combined-output",
        default="qa_config_view_only_all.json",
        help="Combined output name under base-dir. Set empty string to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output files.",
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

    table_map: dict[str, str] = {}
    for cat, tables in table_raw.items():
        for t in tables:
            table_map[t] = cat

    column_map: dict[str, str] = {}
    for cat, cols in column_raw.items():
        for c in cols:
            column_map[c] = cat

    return table_map, column_map


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
    return denied


def ordered_roles(access_control: dict[str, Any], requested_roles: list[str]) -> list[str]:
    discovered = set(requested_roles)
    if isinstance(access_control.get("roles"), list):
        discovered.update(access_control["roles"])
    for p in access_control.get("policies", []):
        discovered.update(p.get("roles", []))

    ordered: list[str] = []
    for r in requested_roles:
        if r in discovered:
            ordered.append(r)
    for r in sorted(discovered):
        if r not in ordered:
            ordered.append(r)
    return ordered


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


def make_rng(seed: int, db_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{db_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_view_permissions(
    entries: list[ColumnEntry],
    access_control: dict[str, Any],
    roles: list[str],
) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    all_cols = {e.fq_column for e in entries}

    for role in roles:
        denied_col_cats = denied_categories(
            access_control, role, scope="view", level="column"
        )
        denied_table_cats = denied_categories(
            access_control, role, scope="view", level="table"
        )
        allowed = {
            e.fq_column
            for e in entries
            if e.column_category not in denied_col_cats
            and e.table_category not in denied_table_cats
        }
        denied = sorted(all_cols - allowed)
        pools[role] = {
            "view_allowed_columns": sorted(allowed),
            "view_denied_columns": denied,
            "view_allowed_count": len(allowed),
            "view_denied_count": len(denied),
            "denied_categories": {
                "column": sorted(denied_col_cats),
                "table": sorted(denied_table_cats),
            },
        }
    return pools


def build_entry_maps(entries: list[ColumnEntry]) -> tuple[dict[str, ColumnEntry], dict[str, list[str]]]:
    by_fq = {e.fq_column: e for e in entries}
    by_table: dict[str, list[str]] = {}
    for e in entries:
        by_table.setdefault(e.table, []).append(e.fq_column)
    for t in by_table:
        by_table[t] = sorted(by_table[t])
    return by_fq, by_table


def choose_target_roles(
    roles: list[str],
    view_pools: dict[str, dict[str, Any]],
) -> list[str]:
    return [r for r in roles if view_pools[r]["view_denied_count"] > 0]


def candidate_restricted_columns_for_role(
    role: str,
    *,
    view_pools: dict[str, dict[str, Any]],
    by_fq: dict[str, ColumnEntry],
    by_table: dict[str, list[str]],
) -> list[str]:
    candidates: list[str] = []
    denied = set(view_pools[role]["view_denied_columns"])
    allowed = set(view_pools[role]["view_allowed_columns"])

    for fq in sorted(denied):
        entry = by_fq.get(fq)
        if not entry:
            continue
        table_cols = by_table.get(entry.table, [])

        # We want this template to be meaningful: keep at least one safe visible column
        # and one S0 ORDER BY candidate in the same table for this role.
        safe_visible = [c for c in table_cols if c in allowed and c != fq]
        orderable_s0 = [
            c
            for c in table_cols
            if c in allowed and by_fq[c].column_category == "S0"
        ]
        if safe_visible and orderable_s0:
            candidates.append(fq)
    return candidates


def build_role_copy(
    *,
    template_projection: list[str],
    template_order_by_column: str,
    view_allowed_set: set[str],
    by_fq: dict[str, ColumnEntry],
) -> dict[str, Any]:
    allowed_projection = [c for c in template_projection if c in view_allowed_set]
    dropped_projection = [c for c in template_projection if c not in view_allowed_set]

    order_by_allowed = template_order_by_column in view_allowed_set
    dropped_order_by = [] if order_by_allowed else [template_order_by_column]

    effective_order_by = template_order_by_column if order_by_allowed else None
    if effective_order_by is None:
        # Fallback: choose an allowed S0 projection column, if available.
        s0_proj = [c for c in allowed_projection if by_fq[c].column_category == "S0"]
        if s0_proj:
            effective_order_by = s0_proj[0]

    return {
        "projection_columns": allowed_projection,
        "dropped_projection": dropped_projection,
        "order_by_column": effective_order_by,
        "dropped_order_by": dropped_order_by,
        "is_usable": bool(allowed_projection) and effective_order_by is not None,
    }


def generate_templates(
    *,
    n_sets: int,
    projection_size: int,
    limit_values: list[int],
    roles: list[str],
    view_pools: dict[str, dict[str, Any]],
    by_fq: dict[str, ColumnEntry],
    by_table: dict[str, list[str]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str]]:
    templates: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    target_roles = choose_target_roles(roles, view_pools)
    if not target_roles:
        warnings.append("No roles have view-denied columns; cannot build restricted view templates.")
        return templates, warnings

    candidate_map = {
        role: candidate_restricted_columns_for_role(
            role,
            view_pools=view_pools,
            by_fq=by_fq,
            by_table=by_table,
        )
        for role in target_roles
    }
    target_roles = [r for r in target_roles if candidate_map.get(r)]
    if not target_roles:
        warnings.append("No restricted-column candidates with usable safe/order-by columns were found.")
        return templates, warnings

    max_attempts = max(200, n_sets * 60)
    attempts = 0

    while len(templates) < n_sets and attempts < max_attempts:
        attempts += 1
        target_role = target_roles[(len(templates) + attempts) % len(target_roles)]
        restricted_candidates = candidate_map[target_role]
        if not restricted_candidates:
            continue

        restricted_col = rng.choice(restricted_candidates)
        restricted_entry = by_fq[restricted_col]
        table = restricted_entry.table
        table_cols = by_table[table]

        # Build projection: restricted + mostly safe S0 columns (same table for simplicity).
        role_allowed_set = set(view_pools[target_role]["view_allowed_columns"])
        safe_s0 = [
            c
            for c in table_cols
            if c in role_allowed_set and by_fq[c].column_category == "S0" and c != restricted_col
        ]
        safe_any = [c for c in table_cols if c in role_allowed_set and c != restricted_col]

        chosen_projection = [restricted_col]
        need = max(0, projection_size - 1)
        pool = safe_s0 + [c for c in safe_any if c not in safe_s0]
        if len(pool) < need:
            pool = safe_any
        if not pool:
            continue
        extra = rng.sample(pool, min(need, len(pool)))
        chosen_projection.extend(extra)
        chosen_projection = sorted(set(chosen_projection))

        # ORDER BY must be S0 by requirement.
        order_by_candidates = [
            c for c in chosen_projection if by_fq[c].column_category == "S0"
        ]
        if not order_by_candidates:
            # Pull one extra S0 allowed column from table if needed.
            fallback = [
                c
                for c in table_cols
                if c in role_allowed_set and by_fq[c].column_category == "S0"
            ]
            if not fallback:
                continue
            chosen_projection.append(fallback[0])
            chosen_projection = sorted(set(chosen_projection))
            order_by_candidates = [fallback[0]]

        order_by_column = rng.choice(order_by_candidates)
        distinct = bool(rng.randint(0, 1))
        limit_value = rng.choice(limit_values)
        sort_direction = "ASC" if rng.randint(0, 1) == 0 else "DESC"

        signature = (
            target_role,
            tuple(chosen_projection),
            order_by_column,
            sort_direction,
            distinct,
            limit_value,
        )
        if signature in seen:
            continue
        seen.add(signature)

        role_copies: dict[str, Any] = {}
        for role in roles:
            role_copy = build_role_copy(
                template_projection=chosen_projection,
                template_order_by_column=order_by_column,
                view_allowed_set=set(view_pools[role]["view_allowed_columns"]),
                by_fq=by_fq,
            )
            role_copies[role] = role_copy

        restricted_for_target = [
            c for c in chosen_projection if c in set(view_pools[target_role]["view_denied_columns"])
        ]
        if not restricted_for_target:
            continue

        templates.append(
            {
                "set_id": f"set_{len(templates) + 1:03d}",
                "tables": [table],
                "projection_columns": chosen_projection,
                "process_columns": [order_by_column],
                "ops": {
                    "distinct": distinct,
                    "order_by": [{"column": order_by_column, "direction": sort_direction}],
                    "limit": limit_value,
                },
                "target_role": target_role,
                "target_restricted_projection_columns": restricted_for_target,
                "role_copies": role_copies,
            }
        )

    if len(templates) < n_sets:
        warnings.append(
            f"Requested n_sets={n_sets}, created only {len(templates)} due to candidate limits."
        )
    return templates, warnings


def iter_db_dirs(base_dir: Path, selected_db_ids: set[str] | None) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if selected_db_ids and p.name not in selected_db_ids:
            continue
        if (p / "access_control.json").exists() and (p / "schema.json").exists():
            dirs.append(p)
    return dirs


def build_for_db(db_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    access_control = load_json(db_dir / "access_control.json")
    schema = load_json(db_dir / "schema.json")

    db_id = access_control.get("db_id") or schema.get("db_id") or db_dir.name
    roles = ordered_roles(access_control, args.roles)
    table_map, column_map = get_classification_maps(access_control)
    entries = build_column_entries(schema, table_map, column_map)
    by_fq, by_table = build_entry_maps(entries)
    view_pools = build_view_permissions(entries, access_control, roles)

    rng = make_rng(args.seed, db_id)
    templates, warnings = generate_templates(
        n_sets=args.n_sets,
        projection_size=args.projection_size,
        limit_values=args.limit_values,
        roles=roles,
        view_pools=view_pools,
        by_fq=by_fq,
        by_table=by_table,
        rng=rng,
    )

    return {
        "db_id": db_id,
        "roles": roles,
        "settings": {
            "track": "view_only_stress",
            "n_sets_requested": args.n_sets,
            "n_sets_created": len(templates),
            "projection_size": args.projection_size,
            "limit_values": args.limit_values,
            "seed": args.seed,
        },
        "column_catalog": [asdict(e) for e in entries],
        "view_role_pools": view_pools,
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
        raise ValueError("No DB directories found with access_control.json + schema.json")

    combined: dict[str, Any] = {
        "base_dir": str(base_dir),
        "db_count": len(db_dirs),
        "items": [],
    }

    for db_dir in db_dirs:
        resource = build_for_db(db_dir, args)
        combined["items"].append(resource)
        warn_count = len(resource.get("warnings", []))
        print(
            f"[{resource['db_id']}] sets={resource['settings']['n_sets_created']} "
            f"warnings={warn_count}"
        )

        if not args.dry_run:
            out_path = db_dir / args.output_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(resource, f, indent=2)

    if not args.dry_run and args.combined_output:
        combined_path = base_dir / args.combined_output
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"combined output -> {combined_path}")


if __name__ == "__main__":
    main()
