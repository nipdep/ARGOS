import os
import re
import sqlite3
from typing import Dict, Optional, Set, Tuple

import pandas as pd

try:
    from experiment import din_sql_base as base_module
    from experiment import dbms_access_control as dbms_module
    from experiment.utils import normalize_name, quote_identifier
except ModuleNotFoundError:
    import din_sql_base as base_module
    import dbms_access_control as dbms_module
    from utils import normalize_name, quote_identifier

def get_role_denied_entities(
    benchmark_root: str,
    db_id: str,
    role: str,
) -> Tuple[Set[str], Set[str]]:
    access_control_path = os.path.join(benchmark_root, db_id, "access_control.json")
    policy_set = dbms_module.load_policy_set(access_control_path)
    role_policy = policy_set.roles.get(role)
    if role_policy is None:
        return set(), set()
    return set(role_policy.denied_tables_all), set(role_policy.denied_columns_all)


def build_filtered_schema(
    db_uri: str,
    denied_tables: Set[str],
    denied_columns: Set[str],
    sample_rows: int = 3,
) -> str:
    chunks = []
    with sqlite3.connect(db_uri) as conn:
        cursor = conn.cursor()
        table_rows = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        for (table_name,) in table_rows:
            if normalize_name(table_name) in denied_tables:
                continue

            table_quoted = quote_identifier(table_name)
            pragma_columns = cursor.execute(f"PRAGMA table_info({table_quoted})").fetchall()
            visible_columns = [
                col for col in pragma_columns if normalize_name(col[1]) not in denied_columns
            ]
            if not visible_columns:
                continue

            column_defs = []
            for col in visible_columns:
                col_name = quote_identifier(col[1])
                col_type = col[2] if col[2] else "TEXT"
                col_not_null = " NOT NULL" if col[3] else ""
                col_pk = " PRIMARY KEY" if col[5] else ""
                column_defs.append(f"        {col_name} {col_type}{col_not_null}{col_pk}")

            fk_rows = cursor.execute(f"PRAGMA foreign_key_list({table_quoted})").fetchall()
            for fk in fk_rows:
                ref_table = fk[2]
                from_col = fk[3]
                to_col = fk[4]
                if normalize_name(ref_table) in denied_tables:
                    continue
                if normalize_name(from_col) in denied_columns:
                    continue
                if normalize_name(to_col) in denied_columns:
                    continue
                column_defs.append(
                    f"        FOREIGN KEY({quote_identifier(from_col)}) REFERENCES {quote_identifier(ref_table)} ({quote_identifier(to_col)})"
                )

            create_stmt = f"CREATE TABLE {table_quoted} (\n" + ",\n".join(column_defs) + "\n)"
            chunks.append(create_stmt)

            selected_cols = ", ".join(quote_identifier(col[1]) for col in visible_columns)
            try:
                sample_data = cursor.execute(
                    f"SELECT {selected_cols} FROM {table_quoted} LIMIT {sample_rows}"
                ).fetchall()
                if sample_data:
                    header = "\t".join(str(col[1]) for col in visible_columns)
                    row_lines = [
                        "\t".join("" if value is None else str(value).replace("\n", " ") for value in row)
                        for row in sample_data
                    ]
                    sample_block = (
                        f"/*\n{sample_rows} rows from {table_name} table:\n"
                        + header
                        + ("\n" + "\n".join(row_lines) if row_lines else "")
                        + "\n*/"
                    )
                    chunks.append(sample_block)
            except sqlite3.Error:
                continue
    return "\n\n".join(chunks)


def build_filtered_column_descriptions(
    database_description_dir: str,
    denied_tables: Set[str],
    denied_columns: Set[str],
) -> str:
    csv_files = sorted(
        os.path.join(database_description_dir, filename)
        for filename in os.listdir(database_description_dir)
        if filename.endswith(".csv")
    )
    db_sections = []
    for file_path in csv_files:
        table_name = os.path.basename(file_path).replace(".csv", "")
        if normalize_name(table_name) in denied_tables:
            continue

        table_lines = [f"Table: {table_name}"]
        table_df = pd.read_csv(file_path, encoding="latin-1")
        for _, row in table_df.iterrows():
            try:
                column_name = str(row.iloc[0]).strip()
                if normalize_name(column_name) in denied_columns:
                    continue
                if pd.notna(row.iloc[2]):
                    col_description = re.sub(r"\s+", " ", str(row.iloc[2]))
                    val_description = re.sub(r"\s+", " ", str(row.iloc[4]))
                    if pd.notna(row.iloc[4]):
                        table_lines.append(
                            f"Column {column_name}: column description -> {col_description}, value description -> {val_description}"
                        )
                    else:
                        table_lines.append(
                            f"Column {column_name}: column description -> {col_description}"
                        )
            except Exception:
                continue

        if len(table_lines) > 1:
            db_sections.append("\n".join(table_lines))
    return "\n\n".join(db_sections)


def build_view_filtered_context(
    benchmark_root: str,
    db_id: str,
    role: str,
    sample_rows: int = 3,
) -> Dict[str, object]:
    db_uri = os.path.join(benchmark_root, db_id, f"{db_id}.sqlite")
    database_description_dir = os.path.join(benchmark_root, db_id, "database_description")
    denied_tables, denied_columns = get_role_denied_entities(
        benchmark_root=benchmark_root,
        db_id=db_id,
        role=role,
    )
    filtered_schema = build_filtered_schema(
        db_uri=db_uri,
        denied_tables=denied_tables,
        denied_columns=denied_columns,
        sample_rows=sample_rows,
    )
    filtered_columns_descriptions = build_filtered_column_descriptions(
        database_description_dir=database_description_dir,
        denied_tables=denied_tables,
        denied_columns=denied_columns,
    )
    return {
        "db_uri": db_uri,
        "database_description_dir": database_description_dir,
        "denied_tables": denied_tables,
        "denied_columns": denied_columns,
        "schema": filtered_schema,
        "columns_descriptions": filtered_columns_descriptions,
    }


def run_view_filtered_case(
    chat_model,
    question: str,
    hint: str,
    benchmark_root: Optional[str] = None,
    db_id: Optional[str] = None,
    role: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    prompt_bundle: Optional[Dict] = None,
    sample_rows: int = 3,
) -> Dict:
    """
    View-filtered variant:
    1) detect denied tables/columns for role from access_control.json
    2) build filtered schema + descriptions
    3) run base DIN-SQL with filtered inputs
    """
    if context is None:
        if not benchmark_root or not db_id or not role:
            raise ValueError(
                "Either pass `context` or provide benchmark_root + db_id + role."
            )
        context = build_view_filtered_context(
            benchmark_root=benchmark_root,
            db_id=db_id,
            role=role,
            sample_rows=sample_rows,
        )
    result = base_module.run_din_sql_case(
        chat_model=chat_model,
        question=question,
        schema=context["schema"],
        hint=hint,
        columns_descriptions=context["columns_descriptions"],
        prompt_bundle=prompt_bundle,
    )
    # result["answer_metadata"]["denied_tables"] = sorted(context["denied_tables"])
    # result["answer_metadata"]["denied_columns"] = sorted(context["denied_columns"])
    return result
