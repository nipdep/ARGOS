import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from scalesql.modules.access_control_common import (
    get_role_denied_entities,
    load_policy_set,
    normalize_name,
    resolve_access_control_path,
    resolve_database_description_dir,
    resolve_db_sqlite_path,
)
from scalesql.modules.light_schema import LightSchema


def load_column_description_map(database_description_dir: str) -> Dict[str, Dict[str, str]]:
    path = Path(database_description_dir)
    if not path.exists():
        return {}

    by_table: Dict[str, Dict[str, str]] = {}
    for csv_path in sorted(path.glob("*.csv")):
        table_name_norm = normalize_name(csv_path.stem)
        table_columns: Dict[str, str] = {}

        try:
            table_df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception:
            continue

        for _, row in table_df.iterrows():
            try:
                column_name = str(row.iloc[0]).strip()
                if not column_name:
                    continue
                column_desc = ""
                if len(row) > 2 and pd.notna(row.iloc[2]):
                    column_desc = re.sub(r"\s+", " ", str(row.iloc[2])).strip()
                if len(row) > 4 and pd.notna(row.iloc[4]):
                    value_desc = re.sub(r"\s+", " ", str(row.iloc[4])).strip()
                    if value_desc:
                        if column_desc:
                            column_desc = f"{column_desc} | value description: {value_desc}"
                        else:
                            column_desc = f"value description: {value_desc}"
                table_columns[normalize_name(column_name)] = column_desc
            except Exception:
                continue

        by_table[table_name_norm] = table_columns
    return by_table


def _sample_column_values(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
    sample_rows: int = 3,
) -> list[Any]:
    query = (
        f"SELECT DISTINCT {column_name} FROM {table_name} "
        f"WHERE {column_name} IS NOT NULL LIMIT {sample_rows}"
    )
    try:
        rows = cursor.execute(query).fetchall()
    except Exception:
        return []

    values: list[Any] = []
    for row in rows:
        if not row:
            continue
        value = row[0]
        if isinstance(value, str) and len(value) > 64:
            values.append(value[:64] + "...")
        else:
            values.append(value)
    return values


def build_filtered_light_schema(
    db_path: str,
    denied_tables: Set[str],
    denied_columns: Set[str],
    sample_rows: int = 3,
    column_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    column_descriptions = column_descriptions or {}
    schema_blocks: list[str] = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        table_rows = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        for (raw_table_name,) in table_rows:
            table_norm = normalize_name(raw_table_name)
            if table_norm in denied_tables:
                continue

            table_quoted = '"' + str(raw_table_name).replace('"', '""') + '"'
            pragma_columns = cursor.execute(f"PRAGMA table_info({table_quoted})").fetchall()

            visible_columns = []
            primary_keys = []
            for col in pragma_columns:
                col_name = str(col[1])
                col_norm = normalize_name(col_name)
                if col_norm in denied_columns:
                    continue
                col_type = col[2] if col[2] else "TEXT"
                if col[5]:
                    primary_keys.append(col_name)
                col_desc = column_descriptions.get(table_norm, {}).get(col_norm, "")

                col_quoted = '"' + col_name.replace('"', '""') + '"'
                samples = _sample_column_values(
                    cursor=cursor,
                    table_name=table_quoted,
                    column_name=col_quoted,
                    sample_rows=sample_rows,
                )
                visible_columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "description": col_desc,
                        "samples": samples,
                    }
                )

            if not visible_columns:
                continue

            fk_rows = cursor.execute(f"PRAGMA foreign_key_list({table_quoted})").fetchall()
            foreign_keys = []
            for fk in fk_rows:
                ref_table = str(fk[2])
                from_col = str(fk[3])
                to_col = str(fk[4])
                if normalize_name(ref_table) in denied_tables:
                    continue
                if normalize_name(from_col) in denied_columns:
                    continue
                if normalize_name(to_col) in denied_columns:
                    continue
                foreign_keys.append(f"{raw_table_name}.{from_col} = {ref_table}.{to_col}")

            schema_block = LightSchema.create_schema(
                database="",
                table=raw_table_name,
                columns=visible_columns,
                table_description="",
                primary_key=primary_keys,
                foreign_key=foreign_keys,
            ).replace("### Table description\n\n", "")
            schema_blocks.append(schema_block)

    return "\n\n".join(schema_blocks)


def build_filtered_column_descriptions(
    database_description_dir: str,
    denied_tables: Set[str],
    denied_columns: Set[str],
) -> str:
    path = Path(database_description_dir)
    if not path.exists():
        return ""

    db_sections: list[str] = []
    for file_path in sorted(path.glob("*.csv")):
        table_name = file_path.stem
        if normalize_name(table_name) in denied_tables:
            continue

        table_lines = [f"Table: {table_name}"]
        try:
            table_df = pd.read_csv(file_path, encoding="latin-1")
        except Exception:
            continue

        for _, row in table_df.iterrows():
            try:
                column_name = str(row.iloc[0]).strip()
                if normalize_name(column_name) in denied_columns:
                    continue
                if len(row) <= 2 or pd.isna(row.iloc[2]):
                    continue

                col_description = re.sub(r"\s+", " ", str(row.iloc[2])).strip()
                val_description = ""
                if len(row) > 4 and pd.notna(row.iloc[4]):
                    val_description = re.sub(r"\s+", " ", str(row.iloc[4])).strip()

                if val_description:
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


def get_role_denied_entities_for_view_filter(
    benchmark_root: str,
    db_id: str,
    role: str,
    scope: str = "all",
) -> Tuple[Set[str], Set[str]]:
    access_control_path = resolve_access_control_path(benchmark_root=benchmark_root, db_id=db_id)
    policy_set = load_policy_set(access_control_path)
    return get_role_denied_entities(policy_set=policy_set, role=role, scope=scope)


def build_view_filtered_context(
    benchmark_root: str,
    db_id: str,
    role: str,
    sample_rows: int = 3,
    scope: str = "all",
) -> Dict[str, object]:
    db_path = resolve_db_sqlite_path(benchmark_root=benchmark_root, db_id=db_id)
    database_description_dir = resolve_database_description_dir(
        benchmark_root=benchmark_root,
        db_id=db_id,
    )
    denied_tables, denied_columns = get_role_denied_entities_for_view_filter(
        benchmark_root=benchmark_root,
        db_id=db_id,
        role=role,
        scope=scope,
    )
    column_description_map = load_column_description_map(database_description_dir)

    filtered_schema = build_filtered_light_schema(
        db_path=db_path,
        denied_tables=denied_tables,
        denied_columns=denied_columns,
        sample_rows=sample_rows,
        column_descriptions=column_description_map,
    )
    filtered_columns_descriptions = build_filtered_column_descriptions(
        database_description_dir=database_description_dir,
        denied_tables=denied_tables,
        denied_columns=denied_columns,
    )
    return {
        "db_path": db_path,
        "database_description_dir": database_description_dir,
        "denied_tables": denied_tables,
        "denied_columns": denied_columns,
        "schema": filtered_schema,
        "columns_descriptions": filtered_columns_descriptions,
    }


def prepare_view_filtered_keyword_extraction_input(
    question: str,
    evidence: str,
    context: Dict[str, object],
) -> Dict[str, str]:
    return {
        "Database Schema": str(context.get("schema", "")),
        "Question": question,
        "Evidence": evidence,
    }
