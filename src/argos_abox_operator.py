from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rdflib import Graph, Literal, Namespace, RDF, URIRef

from src.operators.astObject import SqlglotOperator
from src.operators.astTree import ASTTreeOperator
from src.operators.ontologyInstance import OntologyOperator
from src.prune import ASTPruner


APUF = Namespace("http://www.semanticweb.org/nipun_qk4hy9e/ontologies/2025/5/apuf/")
APUF_HASH = Namespace("http://www.semanticweb.org/nipun_qk4hy9e/ontologies/2025/5/apuf#")


@dataclass
class QueryRefinementResult:
    role: str
    original_query: str
    refined_query: Optional[str]
    table_status: Dict[str, Dict[str, Any]]
    column_status: Dict[str, Dict[str, Any]]
    active_policies: list[str]
    active_rules: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArgosABoxOperator:
    """
    Upper-level ARGOS operator that orchestrates:
    1) DB schema ABOX construction (rdflib)
    2) policy ABOX construction (rdflib)
    3) query instance creation + reasoning + pruning (existing src modules)
    """

    def __init__(self, ontology_path: str | Path):
        self.ontology_path = Path(ontology_path)
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")

        self._tbox_graph = Graph()
        self._tbox_graph.parse(str(self.ontology_path), format="xml")
        self.graph = Graph()
        self.graph += self._tbox_graph

        self.db_id: Optional[str] = None
        self.table_id_by_name: Dict[str, str] = {}
        self.column_id_by_name: Dict[str, str] = {}
        self.role_to_agent_id: Dict[str, str] = {}

        self._runtime_ontology_file: Optional[Path] = None
        self.onto_operator: Optional[OntologyOperator] = None
        self.warmup_table_name: Optional[str] = None

    # ----------------------------- Public API -----------------------------
    def reset(self) -> None:
        self.graph = Graph()
        self.graph += self._tbox_graph
        self.db_id = None
        self.table_id_by_name.clear()
        self.column_id_by_name.clear()
        self.role_to_agent_id.clear()
        self.warmup_table_name = None
        self._teardown_runtime_operator()

    def build_schema_abox(self, schema_json_path: str | Path, db_id: Optional[str] = None) -> None:
        self.reset()
        schema = self._load_json(schema_json_path)
        resolved_db_id = db_id or schema.get("db_id")
        if not resolved_db_id:
            raise ValueError("db_id not found. Provide db_id or ensure it exists in schema.json")

        self.db_id = str(resolved_db_id)
        self._build_database_structure(schema, self.db_id)
        self._ensure_placeholder_entities()
        self._teardown_runtime_operator()

    def build_policy_abox(self, access_control_json_path: str | Path) -> None:
        access_control = self._load_json(access_control_json_path)
        if not self.table_id_by_name:
            raise ValueError("Schema ABOX is not built. Call build_schema_abox(...) before build_policy_abox(...)")
        if self.db_id is None:
            self.db_id = str(access_control.get("db_id", "db"))

        self._build_policy_set(access_control)
        self._teardown_runtime_operator()

    def load_database_context(self, db_dir: str | Path) -> None:
        db_path = Path(db_dir)
        self.build_schema_abox(db_path / "schema.json")
        self.build_policy_abox(db_path / "access_control.json")

    def prepare_reasoner(self) -> None:
        self._materialize_runtime_ontology()
        # self._run_warmup_query()
        self.warmup_table_name = self._pick_warmup_table_name()

    def save_db_abox(
        self,
        output_path: str | Path,
        *,
        include_tbox: bool = False,
    ) -> Path:
        """
        Save the currently built DB ABOX (schema + policy) to disk.

        Args:
            output_path: Destination RDF/TTL file path.
            include_tbox: If True, save full graph (TBOX + ABOX). If False,
                save only ABOX triples introduced for the DB context.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if include_tbox:
            graph_to_save = self.graph
        else:
            graph_to_save = Graph()
            for triple in self.graph:
                if triple not in self._tbox_graph:
                    graph_to_save.add(triple)

        rdf_format = self._infer_rdf_format(path)
        graph_to_save.serialize(destination=str(path), format=rdf_format)
        return path

    def evaluate_query(
        self,
        sql_query: str,
        role: str,
        *,
        reasoned_output_path: Optional[str | Path] = None,
        cleanup_instances: bool = True,
    ) -> QueryRefinementResult:
        if not sql_query or not sql_query.strip():
            raise ValueError("sql_query is empty")
        if self.onto_operator is None:
            self.prepare_reasoner()

        assert self.onto_operator is not None
        agent_id = self._ensure_runtime_agent(role)

        sql_op = SqlglotOperator(sql_query)
        tree_op = ASTTreeOperator(sql_op)

        self.onto_operator.instantiate_ontology(tree_op, agent_id)
        self.onto_operator.reason_and_save(
            output_path=str(reasoned_output_path or self._runtime_ontology_file or ""),
            save=reasoned_output_path is not None,
        )

        table_status = self.onto_operator.get_table_ref_instances()
        column_status = self.onto_operator.get_column_ref_instances()
        active_policies, active_rules = self._extract_active_policy_rule_sets(table_status, column_status)

        pruner = ASTPruner(self.onto_operator)
        pruner.prune()
        refined_query = sql_op.to_sql() if sql_op.ast is not None else None

        if cleanup_instances:
            self.onto_operator.cleanup()

        return QueryRefinementResult(
            role=role,
            original_query=sql_query,
            refined_query=refined_query,
            table_status=table_status,
            column_status=column_status,
            active_policies=active_policies,
            active_rules=active_rules,
        )

    def evaluate_queries(
        self,
        cases: Iterable[Dict[str, Any]],
        *,
        cleanup_instances: bool = True,
    ) -> list[QueryRefinementResult]:
        outputs: list[QueryRefinementResult] = []
        for case in cases:
            sql_query = case.get("query") or case.get("sql_query")
            role = case.get("role", "public")
            if not sql_query:
                continue
            outputs.append(
                self.evaluate_query(
                    sql_query=sql_query,
                    role=role,
                    cleanup_instances=cleanup_instances,
                )
            )
        return outputs

    def close(self) -> None:
        self._teardown_runtime_operator()

    def __enter__(self) -> "ArgosABoxOperator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------------------- Build Phase: DB --------------------------
    def _build_database_structure(self, schema: Dict[str, Any], db_id: str) -> None:
        table_names = schema.get("table_names", [])
        column_names = schema.get("column_names", [])
        if not table_names or not column_names:
            raise ValueError("schema.json must include non-empty table_names and column_names")

        db_uri = URIRef(APUF_HASH[self._safe_id(f"db_{db_id}")])
        self.graph.add((db_uri, RDF.type, APUF.Database))
        self.graph.add((db_uri, RDF.type, APUF.Object))
        self.graph.add((db_uri, APUF.DatabaseName, Literal(db_id)))

        pk_indices = {int(i) - 1 for i in schema.get("primary_keys", []) if isinstance(i, int) and i > 0}
        fk_indices = {
            int(pair[0]) - 1
            for pair in schema.get("foreign_keys", [])
            if isinstance(pair, list) and pair and isinstance(pair[0], int) and pair[0] > 0
        }

        table_uris: Dict[int, URIRef] = {}
        for idx, table_name in enumerate(table_names, start=1):
            table_id = f"t{idx:03d}"
            table_uri = URIRef(APUF_HASH[table_id])
            table_uris[idx - 1] = table_uri

            self.graph.add((table_uri, RDF.type, APUF.Table))
            self.graph.add((table_uri, RDF.type, APUF.Object))
            self.graph.add((table_uri, APUF.TableName, Literal(table_name)))
            self.graph.add((db_uri, APUF.hasTable, table_uri))

            self.table_id_by_name[self._normalize_name(str(table_name))] = table_id

        next_column_idx = 1
        col_uri_by_name: Dict[str, URIRef] = {}
        col_id_by_name: Dict[str, str] = {}
        for col_idx, entry in enumerate(column_names):
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            table_idx, col_name = entry
            if not isinstance(table_idx, int):
                continue

            normalized_col = self._normalize_name(str(col_name))
            if normalized_col not in col_uri_by_name:
                col_id = f"c{next_column_idx:03d}"
                next_column_idx += 1
                col_uri = URIRef(APUF_HASH[col_id])

                col_uri_by_name[normalized_col] = col_uri
                col_id_by_name[normalized_col] = col_id

                self.graph.add((col_uri, RDF.type, APUF.Column))
                self.graph.add((col_uri, RDF.type, APUF.Object))
                self.graph.add((col_uri, APUF.ColumnName, Literal(col_name)))
            else:
                col_uri = col_uri_by_name[normalized_col]
                col_id = col_id_by_name[normalized_col]

            table_uri = table_uris.get(table_idx)
            if table_uri is None:
                continue

            self.graph.add((table_uri, APUF.hasColumn, col_uri))
            self.graph.add((col_uri, APUF.columnOfTable, table_uri))
            if col_idx in pk_indices:
                self.graph.add((table_uri, APUF_HASH.primaryKey, col_uri))
            if col_idx in fk_indices:
                self.graph.add((table_uri, APUF_HASH.foreignKey, col_uri))

            self.column_id_by_name[normalized_col] = col_id

    def _ensure_placeholder_entities(self) -> None:
        table_uri = URIRef(APUF_HASH["t00x"])
        col_uri = URIRef(APUF_HASH["c00x"])

        self.graph.add((table_uri, RDF.type, APUF.Table))
        self.graph.add((table_uri, RDF.type, APUF.Object))
        self.graph.add((table_uri, APUF.TableName, Literal("xxxxx")))
        self.graph.add((table_uri, APUF.hasColumn, col_uri))

        self.graph.add((col_uri, RDF.type, APUF.Column))
        self.graph.add((col_uri, RDF.type, APUF.Object))
        self.graph.add((col_uri, APUF.ColumnName, Literal("ooooo")))
        self.graph.add((col_uri, APUF.columnOfTable, table_uri))

        self.table_id_by_name.setdefault("xxxxx", "t00x")
        self.column_id_by_name.setdefault("ooooo", "c00x")

    # ------------------------ Build Phase: Policies -----------------------
    def _build_policy_set(self, access_control: Dict[str, Any]) -> None:
        classification = access_control.get("classification") or access_control.get("classifiction") or {}
        table_category = classification.get("table", {}) if isinstance(classification, dict) else {}
        column_category = classification.get("column", {}) if isinstance(classification, dict) else {}

        all_roles = set(access_control.get("roles", []) if isinstance(access_control.get("roles"), list) else [])
        for policy in access_control.get("policies", []):
            for role in policy.get("roles", []):
                all_roles.add(role)
        self._ensure_agents(sorted({str(r) for r in all_roles if r}))

        for idx, policy in enumerate(access_control.get("policies", []), start=1):
            policy_name = self._safe_id(str(policy.get("id", f"p{idx:03d}")))
            policy_uri = URIRef(APUF_HASH[policy_name])
            self.graph.add((policy_uri, RDF.type, APUF.Policy))

            action_uri = self._action_uri(policy.get("action"))
            scope_uri = self._scope_uri(policy.get("scope"))
            grant_level_uri = self._grant_level_uri(policy.get("level"))
            grant_type_uri = self._grant_type_uri(policy.get("effect"))

            self.graph.add((policy_uri, APUF.hasAction, action_uri))
            self.graph.add((policy_uri, APUF.hasActionScope, scope_uri))
            self.graph.add((policy_uri, APUF.hasGrantLevel, grant_level_uri))
            self.graph.add((policy_uri, APUF.hasGrantType, grant_type_uri))

            for role in policy.get("roles", []):
                agent_id = self.role_to_agent_id.get(self._normalize_name(str(role)))
                if agent_id:
                    self.graph.add((policy_uri, APUF.hasAgent, URIRef(APUF_HASH[agent_id])))

            for resource_id in self._resolve_policy_resources(policy, table_category, column_category):
                self.graph.add((policy_uri, APUF.controlAccessTo, URIRef(APUF_HASH[resource_id])))

            condition = self._build_condition(policy)
            if condition:
                self.graph.add((policy_uri, APUF_HASH.hasActionCondition, Literal(condition)))

    def _ensure_agents(self, roles: list[str]) -> None:
        next_idx = len(self.role_to_agent_id) + 1
        for role in roles:
            key = self._normalize_name(role)
            if key in self.role_to_agent_id:
                continue
            agent_id = f"a{next_idx:03d}"
            next_idx += 1
            self.role_to_agent_id[key] = agent_id

            agent_uri = URIRef(APUF_HASH[agent_id])
            self.graph.add((agent_uri, RDF.type, APUF.Agent))
            self.graph.add((agent_uri, APUF.AgentID, Literal(role)))

    def _resolve_policy_resources(
        self,
        policy: Dict[str, Any],
        table_category: Dict[str, list[str]],
        column_category: Dict[str, list[str]],
    ) -> list[str]:
        level = self._normalize_name(str(policy.get("level", "")))
        categories = [str(cat) for cat in policy.get("categories", []) if cat]

        resource_ids: set[str] = set()
        if level == "table":
            table_names = self._collect_resources_from_categories(categories, table_category)
            if "table" in policy:
                table_names.append(str(policy["table"]))
            for table_name in table_names:
                table_id = self.table_id_by_name.get(self._normalize_name(table_name))
                if table_id:
                    resource_ids.add(table_id)

        elif level == "column":
            col_names = self._collect_resources_from_categories(categories, column_category)
            if "column" in policy:
                col_names.append(str(policy["column"]))
            for col_name in col_names:
                col_id = self.column_id_by_name.get(self._normalize_name(col_name))
                if col_id:
                    resource_ids.add(col_id)

        elif level == "row":
            candidate_tables: list[str] = []
            if "table" in policy:
                candidate_tables.append(str(policy["table"]))
            predicate = policy.get("predicate")
            if isinstance(predicate, dict) and predicate.get("table"):
                candidate_tables.append(str(predicate["table"]))
            candidate_tables.extend(self._collect_resources_from_categories(categories, table_category))

            for table_name in candidate_tables:
                table_id = self.table_id_by_name.get(self._normalize_name(table_name))
                if table_id:
                    resource_ids.add(table_id)

        return sorted(resource_ids)

    @staticmethod
    def _collect_resources_from_categories(categories: list[str], category_map: Dict[str, list[str]]) -> list[str]:
        outputs: list[str] = []
        for category in categories:
            if category in category_map and isinstance(category_map[category], list):
                outputs.extend([str(x) for x in category_map[category] if x])
            else:
                outputs.append(str(category))
        return outputs

    def _build_condition(self, policy: Dict[str, Any]) -> Optional[str]:
        if policy.get("condition"):
            return str(policy["condition"])

        predicate = policy.get("predicate")
        if isinstance(predicate, str):
            return predicate
        if not isinstance(predicate, dict):
            return None

        if predicate.get("sql"):
            return str(predicate["sql"])

        column = predicate.get("column")
        operation = predicate.get("operation")
        if not column or not operation:
            return None

        value = predicate.get("value")
        if value is None and predicate.get("category") is not None:
            value = predicate["category"]

        op = str(operation).strip()
        if value is None:
            return f"{column} {op}".strip()

        if isinstance(value, str):
            rendered = f"'{value}'"
        else:
            rendered = str(value)

        if "type" in op:
            op = op.replace("type", rendered)
            return f"{column} {op}".strip()
        return f"{column} {op} {rendered}".strip()

    # ------------------------ Runtime / Reasoning -------------------------
    def _materialize_runtime_ontology(self) -> None:
        self._teardown_runtime_operator()

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".rdf", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        self.graph.serialize(destination=str(tmp_path), format="xml")
        self._runtime_ontology_file = tmp_path

        self.onto_operator = OntologyOperator(str(tmp_path))
        self._augment_lookup_case_variants()

    def _run_warmup_query(self) -> None:
        """
        Warm up parser/instantiation/reasoner path once at initialization time.
        """
        if self.onto_operator is None:
            return

        table_name = self._pick_warmup_table_name()
        if not table_name:
            return

        warmup_query = f'SELECT * FROM "{table_name}"'
        agent_id = self._ensure_runtime_agent("public")
        sql_op = SqlglotOperator(warmup_query)
        tree_op = ASTTreeOperator(sql_op)
        self.onto_operator.instantiate_ontology(tree_op, agent_id)
        self.onto_operator.reason_and_save(
            output_path=str(self._runtime_ontology_file or ""),
            save=False,
        )
        self.onto_operator.cleanup()

    def _pick_warmup_table_name(self) -> Optional[str]:
        if self.onto_operator is None:
            return None

        # Prefer real tables over synthetic placeholder entries.
        preferred: list[str] = []
        fallback: list[str] = []
        for table_name in self.onto_operator.table_entities.keys():
            normalized = self._normalize_name(str(table_name))
            if normalized in {"xxxxx", "t00x"}:
                fallback.append(str(table_name))
            else:
                preferred.append(str(table_name))

        if preferred:
            return sorted(preferred)[0]
        if fallback:
            return sorted(fallback)[0]
        return None

    def _teardown_runtime_operator(self) -> None:
        if self.onto_operator is not None:
            try:
                self.onto_operator.close()
            except Exception:
                pass
        self.onto_operator = None

        if self._runtime_ontology_file and self._runtime_ontology_file.exists():
            try:
                self._runtime_ontology_file.unlink()
            except OSError:
                pass
        self._runtime_ontology_file = None

    @staticmethod
    def _infer_rdf_format(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".ttl", ".turtle"}:
            return "turtle"
        if suffix in {".nt"}:
            return "nt"
        if suffix in {".n3"}:
            return "n3"
        return "xml"

    def _augment_lookup_case_variants(self) -> None:
        if self.onto_operator is None:
            return

        table_updates: Dict[str, str] = {}
        for name, table_id in self.onto_operator.table_entities.items():
            for variant in {name, name.lower(), name.upper(), name.capitalize()}:
                table_updates.setdefault(variant, table_id)
        self.onto_operator.table_entities.update(table_updates)

        column_updates: Dict[tuple[str, str], str] = {}
        for (table_name, col_name), col_id in self.onto_operator.column_lookup.items():
            table_variants = {table_name, table_name.lower(), table_name.upper(), table_name.capitalize()}
            col_variants = {col_name, col_name.lower(), col_name.upper(), col_name.capitalize()}
            for t_name in table_variants:
                for c_name in col_variants:
                    column_updates.setdefault((t_name, c_name), col_id)
        self.onto_operator.column_lookup.update(column_updates)

    def _ensure_runtime_agent(self, role: str) -> str:
        role_key = self._normalize_name(role)
        agent_id = self.role_to_agent_id.get(role_key)
        if agent_id:
            return agent_id

        assert self.onto_operator is not None
        next_idx = len(self.role_to_agent_id) + 1
        agent_id = f"a{next_idx:03d}"
        self.role_to_agent_id[role_key] = agent_id

        with self.onto_operator.onto:
            agent = self.onto_operator.onto.Agent(agent_id)
            agent.AgentID.append(role)
        return agent_id

    @staticmethod
    def _extract_active_policy_rule_sets(
        table_status: Dict[str, Dict[str, Any]],
        column_status: Dict[str, Dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        all_status = list(table_status.values()) + list(column_status.values())
        active_policies = sorted({str(x["Policy"]) for x in all_status if x.get("Policy")})
        active_rules = sorted({str(x["Rule"]) for x in all_status if x.get("Rule")})
        return active_policies, active_rules

    # ------------------------------ Helpers -------------------------------
    @staticmethod
    def _load_json(path: str | Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _safe_id(value: str) -> str:
        out = []
        for ch in value:
            if ch.isalnum() or ch in {"_", "-"}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_") or "id"

    @staticmethod
    def _normalize_name(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _action_uri(action: Any) -> URIRef:
        action_value = str(action or "").strip().lower()
        if action_value == "read":
            return URIRef(APUF.ReadAction)
        return URIRef(APUF.ModifyAction)

    @staticmethod
    def _scope_uri(scope: Any) -> URIRef:
        scope_value = str(scope or "").strip().lower()
        mapper = {
            "view": APUF.ViewActionScope,
            "process": APUF.ProcessActionScope,
            "insert": APUF.InsertActionScope,
            "update": APUF.UpdateActionScope,
            "delete": APUF.DeleteActionScope,
        }
        return URIRef(mapper.get(scope_value, APUF.ViewActionScope))

    @staticmethod
    def _grant_level_uri(level: Any) -> URIRef:
        level_value = str(level or "").strip().lower()
        mapper = {
            "table": APUF.TableLevel,
            "column": APUF.ColumnLevel,
            "row": APUF.RowLevel,
        }
        return URIRef(mapper.get(level_value, APUF.TableLevel))

    @staticmethod
    def _grant_type_uri(effect: Any) -> URIRef:
        effect_value = str(effect or "").strip().lower()
        mapper = {
            "deny": APUF.Prohibited,
            "denied": APUF.Prohibited,
            "prohibit": APUF.Prohibited,
            "allow": APUF.Permitted,
            "allowed": APUF.Permitted,
            "permit": APUF.Permitted,
            "conditional": APUF.Conditional,
        }
        return URIRef(mapper.get(effect_value, APUF.Prohibited))


def build_argos_operator_for_db(
    db_id: str,
    *,
    benchmark_root: str | Path = "data/P3T2Q_benchmark/v0",
    ontology_path: str | Path = "data/ontology_file/argos_v2.0.rdf",
) -> ArgosABoxOperator:
    db_dir = Path(benchmark_root) / db_id
    operator = ArgosABoxOperator(ontology_path=ontology_path)
    operator.load_database_context(db_dir)
    operator.prepare_reasoner()
    return operator
