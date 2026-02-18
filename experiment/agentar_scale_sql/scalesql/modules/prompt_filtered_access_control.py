from typing import Dict, Sequence, Tuple

from scalesql.modules.access_control_common import (
    get_role_denied_entities,
    load_policy_set,
    resolve_access_control_path,
    resolve_database_description_dir,
    resolve_db_sqlite_path,
)
from scalesql.modules.view_filtered_access_control import (
    build_filtered_column_descriptions,
    build_filtered_light_schema,
    get_role_denied_entities_for_view_filter,
    load_column_description_map,
)


def build_prompt_restriction_hint(
    role: str,
    restricted_tables: Sequence[str],
    restricted_columns: Sequence[str],
) -> str:
    parts = [
        f"Access-control role: {role}.",
        "Do NOT use restricted tables or columns in SQL output.",
        "This prohibition applies to SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, subqueries, and aliases.",
    ]
    if restricted_tables:
        parts.append("Restricted tables: " + ", ".join(sorted(set(restricted_tables))) + ".")
    if restricted_columns:
        parts.append("Restricted columns: " + ", ".join(sorted(set(restricted_columns))) + ".")
    return " ".join(parts)


def build_prompt_filtered_context(
    benchmark_root: str,
    db_id: str,
    role: str,
    sample_rows: int = 3,
) -> Dict[str, object]:
    db_path = resolve_db_sqlite_path(benchmark_root=benchmark_root, db_id=db_id)
    database_description_dir = resolve_database_description_dir(
        benchmark_root=benchmark_root,
        db_id=db_id,
    )

    # Prompt filtering keeps full schema visible but injects explicit restriction text.
    empty_tables, empty_columns = set(), set()
    column_description_map = load_column_description_map(database_description_dir)
    full_schema = build_filtered_light_schema(
        db_path=db_path,
        denied_tables=empty_tables,
        denied_columns=empty_columns,
        sample_rows=sample_rows,
        column_descriptions=column_description_map,
    )
    full_columns_descriptions = build_filtered_column_descriptions(
        database_description_dir=database_description_dir,
        denied_tables=empty_tables,
        denied_columns=empty_columns,
    )

    restricted_tables, restricted_columns = get_role_denied_entities_for_view_filter(
        benchmark_root=benchmark_root,
        db_id=db_id,
        role=role,
        scope="all",
    )
    restriction_hint = build_prompt_restriction_hint(
        role=role,
        restricted_tables=sorted(restricted_tables),
        restricted_columns=sorted(restricted_columns),
    )

    return {
        "db_path": db_path,
        "database_description_dir": database_description_dir,
        "schema": full_schema,
        "columns_descriptions": full_columns_descriptions,
        "restricted_tables": restricted_tables,
        "restricted_columns": restricted_columns,
        "restriction_hint": restriction_hint,
    }


def apply_prompt_filtering_to_evidence(evidence: str, restriction_hint: str) -> str:
    evidence_text = evidence or ""
    if evidence_text.strip():
        return f"{evidence_text}\n{restriction_hint}".strip()
    return restriction_hint


def prepare_prompt_filtered_keyword_extraction_input(
    question: str,
    evidence: str,
    context: Dict[str, object],
) -> Dict[str, str]:
    merged_evidence = apply_prompt_filtering_to_evidence(
        evidence=evidence,
        restriction_hint=str(context.get("restriction_hint", "")),
    )
    return {
        "Database Schema": str(context.get("schema", "")),
        "Question": question,
        "Evidence": merged_evidence,
    }


def get_role_denied_entities_for_prompt_filter(
    benchmark_root: str,
    db_id: str,
    role: str,
) -> Tuple[set[str], set[str]]:
    access_control_path = resolve_access_control_path(benchmark_root=benchmark_root, db_id=db_id)
    policy_set = load_policy_set(access_control_path)
    return get_role_denied_entities(policy_set=policy_set, role=role, scope="all")
