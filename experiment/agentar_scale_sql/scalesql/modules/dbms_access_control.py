import re
import sqlite3
import time
from collections import defaultdict
from typing import Dict, Iterable, Optional

from scalesql.executions.base import (
    BaseDatabaseExecutor,
    QueryExecutionRequest,
    QueryExecutionResponse,
)
from scalesql.modules.access_control_common import (
    AccessControlPolicySet,
    RoleAccessPolicy,
    get_role_denied_entities,
    load_policy_set,
    normalize_name,
    resolve_access_control_path,
    resolve_db_sqlite_path,
)


class SQLiteAccessController:
    """
    SQLite access-control runtime using sqlite3 authorizer callbacks.

    SQLite has no built-in role ACL, so this simulates DBMS-level enforcement
    with role-specific authorizer callbacks.
    """

    def __init__(
        self,
        db_path: str,
        policy_set: AccessControlPolicySet,
        query_timeout_seconds: float = 5.0,
    ):
        self.db_path = db_path
        self.policy_set = policy_set
        self.query_timeout_seconds = float(query_timeout_seconds)
        self._connections: Dict[str, sqlite3.Connection] = {}

    @classmethod
    def from_benchmark_db(
        cls,
        benchmark_root: str,
        db_id: str,
        query_timeout_seconds: float = 5.0,
    ) -> "SQLiteAccessController":
        db_path = resolve_db_sqlite_path(benchmark_root=benchmark_root, db_id=db_id)
        access_control_path = resolve_access_control_path(
            benchmark_root=benchmark_root,
            db_id=db_id,
        )
        policy_set = load_policy_set(access_control_path)
        return cls(
            db_path=db_path,
            policy_set=policy_set,
            query_timeout_seconds=query_timeout_seconds,
        )

    def warmup_roles(self, roles: Iterable[str]) -> None:
        for role in roles:
            self.get_connection(role)

    def get_connection(self, role: str) -> sqlite3.Connection:
        role_key = str(role)
        if role_key in self._connections:
            return self._connections[role_key]

        conn = sqlite3.connect(self.db_path)
        conn.set_authorizer(self._build_authorizer(role_key))
        self._connections[role_key] = conn
        return conn

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def _build_authorizer(self, role: str):
        role_policy = self.policy_set.roles.get(role, RoleAccessPolicy(role=role))
        denied_tables = role_policy.denied_tables_all
        denied_columns = role_policy.denied_columns_all

        def _authorizer(action: int, arg1: Optional[str], arg2: Optional[str], dbname, source):
            if action == sqlite3.SQLITE_READ:
                table_name = normalize_name(arg1)
                column_name = normalize_name(arg2)
                if table_name and table_name in denied_tables:
                    return sqlite3.SQLITE_DENY
                if column_name and column_name in denied_columns:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        return _authorizer

    @staticmethod
    def _format_execution_error(exc: Exception, timeout_seconds: float) -> str:
        error_text = str(exc)
        if timeout_seconds > 0 and "interrupted" in error_text.lower():
            return f"Query timed out after {timeout_seconds:.2f}s"
        return error_text

    def evaluate_query(self, role: str, sql_query: str) -> Dict[str, Optional[str]]:
        if not isinstance(sql_query, str) or not sql_query.strip():
            return {
                "status": "deny",
                "error": "Missing SQL query",
                "role": role,
                "db_id": self.policy_set.db_id,
            }

        if not re.match(r"^\s*(SELECT|WITH)\b", sql_query, flags=re.IGNORECASE):
            return {
                "status": "deny",
                "error": "Only SELECT/WITH queries are allowed in DBMS access-control evaluation",
                "role": role,
                "db_id": self.policy_set.db_id,
            }

        conn = self.get_connection(role)
        timeout_seconds = self.query_timeout_seconds
        progress_step = 10_000
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

        def _progress_handler() -> int:
            if deadline is None:
                return 0
            return 1 if time.monotonic() >= deadline else 0

        try:
            if deadline is not None:
                conn.set_progress_handler(_progress_handler, progress_step)
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
                "error": self._format_execution_error(exc, timeout_seconds),
                "role": role,
                "db_id": self.policy_set.db_id,
            }
        finally:
            if deadline is not None:
                conn.set_progress_handler(None, 0)

    def execute_query(self, role: str, sql_query: str) -> QueryExecutionResponse:
        decision = self.evaluate_query(role=role, sql_query=sql_query)
        if decision["status"] != "allow":
            return QueryExecutionResponse(error_message=decision["error"], results={})

        conn = self.get_connection(role)
        timeout_seconds = self.query_timeout_seconds
        progress_step = 10_000
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

        def _progress_handler() -> int:
            if deadline is None:
                return 0
            return 1 if time.monotonic() >= deadline else 0

        try:
            if deadline is not None:
                conn.set_progress_handler(_progress_handler, progress_step)
            cursor = conn.execute(sql_query)
            results = defaultdict(list)
            if cursor.description:
                for row in cursor.fetchall():
                    for col_name, value in zip((d[0] for d in cursor.description), row):
                        results[col_name].append(value)
            return QueryExecutionResponse(results=dict(results), error_message=None)
        except Exception as exc:
            return QueryExecutionResponse(
                error_message=self._format_execution_error(exc, timeout_seconds),
                results={},
            )
        finally:
            if deadline is not None:
                conn.set_progress_handler(None, 0)


class AccessControlledSQLiteExecutor(BaseDatabaseExecutor):
    """
    BaseDatabaseExecutor-compatible wrapper for role-based DBMS checks.

    Pass role via QueryExecutionRequest.extra_args["role"].
    """

    def __init__(self, db_path: str, policy_set: AccessControlPolicySet):
        super().__init__(connection_string=f"sqlite:///{db_path}")
        self.controller = SQLiteAccessController(db_path=db_path, policy_set=policy_set)

    @classmethod
    def from_benchmark_db(
        cls,
        benchmark_root: str,
        db_id: str,
    ) -> "AccessControlledSQLiteExecutor":
        db_path = resolve_db_sqlite_path(benchmark_root=benchmark_root, db_id=db_id)
        access_control_path = resolve_access_control_path(
            benchmark_root=benchmark_root,
            db_id=db_id,
        )
        policy_set = load_policy_set(access_control_path)
        return cls(db_path=db_path, policy_set=policy_set)

    def execute_query(self, execute_request: QueryExecutionRequest) -> QueryExecutionResponse:
        role = str((execute_request.extra_args or {}).get("role", "public"))
        return self.controller.execute_query(role=role, sql_query=execute_request.query)

    def close(self) -> None:
        self.controller.close()


def build_dbms_access_controller(benchmark_root: str, db_id: str) -> SQLiteAccessController:
    return SQLiteAccessController.from_benchmark_db(
        benchmark_root=benchmark_root,
        db_id=db_id,
    )


def execute_with_dbms_access_control(
    benchmark_root: str,
    db_id: str,
    role: str,
    sql_query: str,
) -> Dict[str, Optional[str]]:
    controller = build_dbms_access_controller(benchmark_root=benchmark_root, db_id=db_id)
    try:
        return controller.evaluate_query(role=role, sql_query=sql_query)
    finally:
        controller.close()


def run_dbms_access_control_case(
    controller: SQLiteAccessController,
    role: str,
    sql_query: Optional[str],
) -> Dict[str, object]:
    sql_text = sql_query or ""
    decision = controller.evaluate_query(role=role, sql_query=sql_text)
    return {
        "final_query": "" if decision["status"] == "deny" else sql_text,
        "answer_metadata": {
            "query": sql_text,
            "dbms_access_control_status": decision["status"],
            "dbms_access_control_error": decision["error"],
            "role": role,
            "db_id": decision["db_id"],
        },
    }


def get_role_denied_entities_for_db(
    benchmark_root: str,
    db_id: str,
    role: str,
    scope: str = "all",
) -> Dict[str, set[str]]:
    policy_set = load_policy_set(resolve_access_control_path(benchmark_root=benchmark_root, db_id=db_id))
    denied_tables, denied_columns = get_role_denied_entities(policy_set=policy_set, role=role, scope=scope)
    return {
        "tables": denied_tables,
        "columns": denied_columns,
    }
