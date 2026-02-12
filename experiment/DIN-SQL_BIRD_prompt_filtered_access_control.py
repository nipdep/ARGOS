import os
from typing import Dict, Optional, Sequence, Set

try:
    from experiment.din_sql_bird_imports import get_base_module, get_dbms_module
except ModuleNotFoundError:
    from din_sql_bird_imports import get_base_module, get_dbms_module

BASE_MODULE = get_base_module()
DBMS_MODULE = get_dbms_module()


def get_role_denied_entities(
    benchmark_root: str,
    db_id: str,
    role: str,
) -> Dict[str, Set[str]]:
    access_control_path = os.path.join(benchmark_root, db_id, "access_control.json")
    policy_set = DBMS_MODULE.load_policy_set(access_control_path)
    role_policy = policy_set.roles.get(role)
    if role_policy is None:
        return {"tables": set(), "columns": set()}
    return {
        "tables": set(role_policy.denied_tables_all),
        "columns": set(role_policy.denied_columns_all),
    }


def build_prompt_filtered_context(
    benchmark_root: str,
    db_id: str,
    role: str,
) -> Dict[str, object]:
    db_uri = os.path.join(benchmark_root, db_id, f"{db_id}.sqlite")
    database_description_dir = os.path.join(benchmark_root, db_id, "database_description")
    denied_entities = get_role_denied_entities(
        benchmark_root=benchmark_root,
        db_id=db_id,
        role=role,
    )
    schema = BASE_MODULE.get_database_schema(db_uri)
    columns_descriptions = BASE_MODULE.table_descriptions_parser(database_description_dir)
    return {
        "db_uri": db_uri,
        "database_description_dir": database_description_dir,
        "schema": schema,
        "columns_descriptions": columns_descriptions,
        "restricted_tables": denied_entities["tables"],
        "restricted_columns": denied_entities["columns"],
    }


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


def run_prompt_filtered_case(
    chat_model,
    question: str,
    hint: str,
    role: str,
    benchmark_root: Optional[str] = None,
    db_id: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    schema: Optional[str] = None,
    columns_descriptions: Optional[str] = None,
    restricted_tables: Optional[Sequence[str]] = None,
    restricted_columns: Optional[Sequence[str]] = None,
    prompt_bundle: Optional[Dict] = None,
) -> Dict:
    """
    Prompt-filtered variant:
    1) detect denied tables/columns for role
    2) keep full schema/descriptions
    3) append explicit restrictions to a common hint used for the full DIN-SQL flow
    """
    if context is not None:
        schema = context["schema"]
        columns_descriptions = context["columns_descriptions"]
        restricted_tables = context["restricted_tables"]
        restricted_columns = context["restricted_columns"]
    elif (
        benchmark_root is not None
        and db_id is not None
        and schema is None
        and columns_descriptions is None
        and restricted_tables is None
        and restricted_columns is None
    ):
        built_context = build_prompt_filtered_context(
            benchmark_root=benchmark_root,
            db_id=db_id,
            role=role,
        )
        schema = built_context["schema"]
        columns_descriptions = built_context["columns_descriptions"]
        restricted_tables = built_context["restricted_tables"]
        restricted_columns = built_context["restricted_columns"]

    if schema is None or columns_descriptions is None:
        raise ValueError(
            "Missing schema context. Pass `context`, or benchmark_root+db_id, or explicit schema/columns_descriptions."
        )

    restriction_hint = build_prompt_restriction_hint(
        role=role,
        restricted_tables=restricted_tables or [],
        restricted_columns=restricted_columns or [],
    )
    common_hint = f"{hint}\n{restriction_hint}".strip()

    result = BASE_MODULE.run_din_sql_case(
        chat_model=chat_model,
        question=question,
        schema=schema,
        hint=common_hint,
        columns_descriptions=columns_descriptions,
        prompt_bundle=prompt_bundle,
    )
    result["restricted_tables"] = sorted(set(restricted_tables or []))
    result["restricted_columns"] = sorted(set(restricted_columns or []))
    result["prompt_restriction_hint"] = restriction_hint
    result["common_hint"] = common_hint
    return result
