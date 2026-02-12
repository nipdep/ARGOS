import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


def _normalize_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text[0] in {'"', "'", "`", "["} and text[-1] in {'"', "'", "`", "]"}:
        text = text[1:-1]
    return text.strip().lower()


@dataclass
class RoleAccessPolicy:
    role: str
    denied_tables_view: Set[str] = field(default_factory=set)
    denied_tables_process: Set[str] = field(default_factory=set)
    denied_columns_view: Set[str] = field(default_factory=set)
    denied_columns_process: Set[str] = field(default_factory=set)

    @property
    def denied_tables_all(self) -> Set[str]:
        return self.denied_tables_view | self.denied_tables_process

    @property
    def denied_columns_all(self) -> Set[str]:
        return self.denied_columns_view | self.denied_columns_process


@dataclass
class AccessControlPolicySet:
    db_id: str
    roles: Dict[str, RoleAccessPolicy]


class SQLiteAccessController:
    """
    SQLite access-control runtime using sqlite3 authorizer callbacks.

    SQLite has no built-in user/role ACL model, so this simulates DBMS enforcement
    by binding a role-specific authorizer callback to each connection.
    """

    def __init__(self, db_uri: str, policy_set: AccessControlPolicySet):
        self.db_uri = db_uri
        self.policy_set = policy_set
        self._connections: Dict[str, sqlite3.Connection] = {}

    @classmethod
    def from_benchmark_db(cls, benchmark_root: str, db_id: str) -> "SQLiteAccessController":
        db_uri = os.path.join(benchmark_root, db_id, f"{db_id}.sqlite")
        access_control_path = os.path.join(benchmark_root, db_id, "access_control.json")
        policy_set = load_policy_set(access_control_path)
        return cls(db_uri=db_uri, policy_set=policy_set)

    def warmup_roles(self, roles: Iterable[str]) -> None:
        for role in roles:
            self.get_connection(role)

    def get_connection(self, role: str) -> sqlite3.Connection:
        role_key = str(role)
        if role_key in self._connections:
            return self._connections[role_key]

        conn = sqlite3.connect(self.db_uri)
        conn.set_authorizer(self._build_authorizer(role_key))
        self._connections[role_key] = conn
        return conn

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def evaluate_query(self, role: str, sql_query: str) -> Dict[str, Optional[str]]:
        if not isinstance(sql_query, str) or not sql_query.strip():
            return {
                "status": "deny",
                "error": "Missing SQL query",
                "role": role,
                "db_id": self.policy_set.db_id,
            }

        # DBMS-AC stage only evaluates read queries from the generation phase.
        if not re.match(r"^\s*(SELECT|WITH)\b", sql_query, flags=re.IGNORECASE):
            return {
                "status": "deny",
                "error": "Only SELECT/WITH queries are allowed in DBMS access-control evaluation",
                "role": role,
                "db_id": self.policy_set.db_id,
            }

        conn = self.get_connection(role)
        try:
            cursor = conn.execute(sql_query)
            if cursor.description is not None:
                cursor.fetchone()
            return {
                "status": "allow",
                "error": None,
                "role": role,
                "db_id": self.policy_set.db_id,
            }
        except Exception as exc:
            return {
                "status": "deny",
                "error": str(exc),
                "role": role,
                "db_id": self.policy_set.db_id,
            }

    def _build_authorizer(self, role: str):
        role_policy = self.policy_set.roles.get(role, RoleAccessPolicy(role=role))
        denied_tables = role_policy.denied_tables_all
        denied_columns = role_policy.denied_columns_all

        def _authorizer(action: int, arg1: Optional[str], arg2: Optional[str], dbname, source):
            if action == sqlite3.SQLITE_READ:
                table_name = _normalize_name(arg1)
                column_name = _normalize_name(arg2)
                if table_name and table_name in denied_tables:
                    return sqlite3.SQLITE_DENY
                if column_name and column_name in denied_columns:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        return _authorizer


def build_dbms_access_controller(benchmark_root: str, db_id: str) -> SQLiteAccessController:
    """Build controller once per DB, then reuse it for many query checks."""
    return SQLiteAccessController.from_benchmark_db(benchmark_root=benchmark_root, db_id=db_id)


def load_policy_set(access_control_path: str) -> AccessControlPolicySet:
    with open(access_control_path, "r") as f:
        payload = json.load(f)

    db_id = payload.get("db_id", "")
    classification = payload.get("classification", {})
    table_classes = classification.get("table", {}) if isinstance(classification, dict) else {}
    column_classes = classification.get("column", {}) if isinstance(classification, dict) else {}

    roles: Dict[str, RoleAccessPolicy] = {}

    def _get_role(role: str) -> RoleAccessPolicy:
        if role not in roles:
            roles[role] = RoleAccessPolicy(role=role)
        return roles[role]

    for policy in payload.get("policies", []):
        if policy.get("effect") != "deny":
            continue
        if policy.get("action") != "read":
            continue

        level = policy.get("level")
        scope = policy.get("scope", "view")
        categories = policy.get("categories", [])
        policy_roles = policy.get("roles", [])

        values: List[str] = []
        if level == "table":
            for category in categories:
                values.extend(table_classes.get(category, []))
        elif level == "column":
            for category in categories:
                values.extend(column_classes.get(category, []))
        else:
            continue

        normalized_values = {_normalize_name(v) for v in values if _normalize_name(v)}
        if not normalized_values:
            continue

        for role in policy_roles:
            role_policy = _get_role(role)
            if level == "table":
                if scope == "view":
                    role_policy.denied_tables_view.update(normalized_values)
                elif scope == "process":
                    role_policy.denied_tables_process.update(normalized_values)
                else:
                    role_policy.denied_tables_view.update(normalized_values)
                    role_policy.denied_tables_process.update(normalized_values)
            else:
                if scope == "view":
                    role_policy.denied_columns_view.update(normalized_values)
                elif scope == "process":
                    role_policy.denied_columns_process.update(normalized_values)
                else:
                    role_policy.denied_columns_view.update(normalized_values)
                    role_policy.denied_columns_process.update(normalized_values)

    return AccessControlPolicySet(db_id=db_id, roles=roles)


def execute_with_dbms_access_control(
    db_uri: str,
    sql_query: str,
    role: str = "public",
    access_control_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Compatibility helper for single-query evaluation.
    If access_control_path is provided, enforce role policy; otherwise run plain SQLite execution check.
    """
    if access_control_path:
        policy_set = load_policy_set(access_control_path)
        controller = SQLiteAccessController(db_uri=db_uri, policy_set=policy_set)
        try:
            return controller.evaluate_query(role=role, sql_query=sql_query)
        finally:
            controller.close()

    if not isinstance(sql_query, str) or not sql_query.strip():
        return {"status": "deny", "error": "Missing SQL query", "role": role, "db_id": None}

    try:
        with sqlite3.connect(db_uri) as conn:
            conn.execute(sql_query)
        return {"status": "allow", "error": None, "role": role, "db_id": None}
    except Exception as exc:
        return {"status": "deny", "error": str(exc), "role": role, "db_id": None}


def run_dbms_access_control_for_predictions(
    benchmark_root: str,
    db_id: str,
    prediction_csv: str,
    output_csv: Optional[str] = None,
    default_role: str = "public",
    query_column: str = "final_query",
) -> pd.DataFrame:
    """
    Evaluate generated SQL (usually from type -1 stage) against DBMS access control.
    Assumes one DB per run.
    """
    df = pd.read_csv(prediction_csv)
    if "db_id" in df.columns:
        df = df[df["db_id"] == db_id].copy()

    controller = SQLiteAccessController.from_benchmark_db(benchmark_root=benchmark_root, db_id=db_id)
    if "role" in df.columns:
        role_values = df["role"].fillna(default_role).astype(str)
        roles_to_warm = set(role_values.tolist()) or {default_role}
    else:
        roles_to_warm = {default_role}
    controller.warmup_roles(roles_to_warm)

    results: List[Dict[str, Optional[str]]] = []
    for idx, row in df.iterrows():
        role_raw = row.get("role", default_role)
        role = default_role if pd.isna(role_raw) else str(role_raw)
        sql_query = row.get(query_column, "")
        decision = controller.evaluate_query(role=role, sql_query=sql_query)
        decision.update(
            {
                "index": idx,
                "question_id": row.get("question_id"),
                "query": sql_query,
            }
        )
        results.append(decision)

    controller.close()

    out_df = pd.DataFrame(results)
    if output_csv:
        out_df.to_csv(output_csv, index=False)
    return out_df
