import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


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


def normalize_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text[0] in {'"', "'", "`", "["} and text[-1] in {'"', "'", "`", "]"}:
        text = text[1:-1]
    return text.strip().lower()


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def resolve_access_control_path(benchmark_root: str, db_id: str) -> str:
    return str(Path(benchmark_root) / db_id / "access_control.json")


def resolve_db_sqlite_path(benchmark_root: str, db_id: str) -> str:
    return str(Path(benchmark_root) / db_id / f"{db_id}.sqlite")


def resolve_database_description_dir(benchmark_root: str, db_id: str) -> str:
    return str(Path(benchmark_root) / db_id / "database_description")


def load_policy_set(access_control_path: str) -> AccessControlPolicySet:
    with open(access_control_path, "r", encoding="utf-8") as f:
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

        normalized_values = {normalize_name(value) for value in values if normalize_name(value)}
        if not normalized_values:
            continue

        for role in policy_roles:
            role_policy = _get_role(str(role))
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


def get_role_denied_entities(
    policy_set: AccessControlPolicySet,
    role: str,
    scope: str = "all",
) -> Tuple[Set[str], Set[str]]:
    role_policy = policy_set.roles.get(str(role))
    if role_policy is None:
        return set(), set()

    if scope == "view":
        return set(role_policy.denied_tables_view), set(role_policy.denied_columns_view)
    if scope == "process":
        return set(role_policy.denied_tables_process), set(role_policy.denied_columns_process)
    return set(role_policy.denied_tables_all), set(role_policy.denied_columns_all)
